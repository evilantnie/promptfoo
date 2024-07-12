import OpenAI from 'openai';
import { fetchWithCache } from '../cache';
import logger from '../logger';
import type {
  ApiProvider,
  CallApiContextParams,
  CallApiOptionsParams,
  EnvOverrides,
  ProviderResponse,
  TokenUsage,
} from '../types.js';
import { renderVarsInObject } from '../util';
import { safeJsonStringify } from '../util';
import { OpenAiFunction, OpenAiTool } from './openaiUtil';
import { REQUEST_TIMEOUT_MS, parseChatPrompt } from './shared';

const DATABRICKS_CHAT_MODELS = [
  ...['databricks-dbrx-instruct'].map((model) => ({
    id: model,
    cost: {
      input: 0.0008 / 1000,
      output: 0.0024 / 1000,
    },
  })),
  ...['databricks-mixtral-8x7b-instruct'].map((model) => ({
    id: model,
    cost: {
      input: 0.0005 / 1000,
      output: 0.001 / 1000,
    },
  })),
  ...[
    'databricks-meta-llama-3-70b-instruct',
  ].map((model) => ({
    id: model,
    cost: {
      input: 0.0009 / 1000,
      output: 0.0027 / 1000,
    },
  })),
];

interface DatabricksSharedOptions {
  apiKey?: string;
  apiKeyEnvar?: string;
  apiHost?: string;
  apiBaseUrl?: string;
  cost?: number;
  headers?: { [key: string]: string };
}

type DatabricksCompletionOptions = DatabricksSharedOptions & {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  best_of?: number;
  functions?: OpenAiFunction[];
  function_call?: 'none' | 'auto' | { name: string };
  tools?: OpenAiTool[];
  tool_choice?: 'none' | 'auto' | 'required' | { type: 'function'; function?: { name: string } };
  response_format?: { type: 'json_object' };
  stop?: string[];
  passthrough?: object;

  /**
   * If set, automatically call these functions when the assistant activates
   * these function tools.
   */
  functionToolCallbacks?: Record<
    OpenAI.FunctionDefinition['name'],
    (arg: string) => Promise<string>
  >;
};

function failApiCall(err: any) {
  if (err instanceof OpenAI.APIError) {
    return {
      error: `API error: ${err.type} ${err.message}`,
    };
  }
  return {
    error: `API error: ${String(err)}`,
  };
}

function getTokenUsage(data: any, cached: boolean): Partial<TokenUsage> {
  if (data.usage) {
    if (cached) {
      return { cached: data.usage.total_tokens, total: data.usage.total_tokens };
    } else {
      return {
        total: data.usage.total_tokens,
        prompt: data.usage.prompt_tokens || 0,
        completion: data.usage.completion_tokens || 0,
      };
    }
  }
  return {};
}

export class DatabricksGenericProvider implements ApiProvider {
  modelName: string;

  config: DatabricksSharedOptions;
  env?: EnvOverrides;

  constructor(
    modelName: string,
    options: { config?: DatabricksSharedOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    const { config, id, env } = options;
    this.env = env;
    this.modelName = modelName;
    this.config = config || {};
    this.id = id ? () => id : this.id;
  }

  id(): string {
    return this.config.apiHost || this.config.apiBaseUrl
      ? this.modelName
      : `databricks:${this.modelName}`;
  }

  toString(): string {
    return `[Databricks Provider ${this.modelName}]`;
  }

  getApiUrlDefault(): string {
    return 'https://cloud.databricks.com/v1';
  }

  getApiUrl(): string {
    const apiHost = this.config.apiHost || this.env?.DATABRICKS_API_HOST || process.env.DATABRICKS_API_HOST;
    if (apiHost) {
      return `https://${apiHost}`;
    }
    return (
      this.config.apiBaseUrl ||
      this.env?.DATABRICKS_API_BASE_URL ||
      this.env?.DATABRICKS_BASE_URL ||
      process.env.DATABRICKS_API_BASE_URL ||
      process.env.DATABRICKS_BASE_URL ||
      this.getApiUrlDefault()
    );
  }

  getApiKey(): string | undefined {
    return (
      this.config.apiKey ||
      (this.config?.apiKeyEnvar
        ? process.env[this.config.apiKeyEnvar] ||
        this.env?.[this.config.apiKeyEnvar as keyof EnvOverrides]
        : undefined) ||
      this.env?.DATABRICKS_API_KEY ||
      process.env.DATABRICKS_API_KEY
    );
  }

  // @ts-ignore: Params are not used in this implementation
  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    throw new Error('Not implemented');
  }
}

function formatDatabricksError(data: {
  error: { message: string; type?: string; code?: string };
}): string {
  let errorMessage = `API error: ${data.error.message}`;
  if (data.error.type) {
    errorMessage += `, Type: ${data.error.type}`;
  }
  if (data.error.code) {
    errorMessage += `, Code: ${data.error.code}`;
  }
  errorMessage += '\n\n' + safeJsonStringify(data, true /* prettyPrint */);
  return errorMessage;
}

function calculateCost(
  modelName: string,
  config: DatabricksSharedOptions,
  promptTokens?: number,
  completionTokens?: number,
): number | undefined {
  if (!promptTokens || !completionTokens) {
    return undefined;
  }

  const model = [...DATABRICKS_CHAT_MODELS].find(
    (m) => m.id === modelName,
  );
  if (!model || !model.cost) {
    return undefined;
  }

  const inputCost = config.cost ?? model.cost.input;
  const outputCost = config.cost ?? model.cost.output;
  return inputCost * promptTokens + outputCost * completionTokens || undefined;
}

export class DatabricksChatCompletionProvider extends DatabricksGenericProvider {
  static DATABRICKS_CHAT_MODELS = DATABRICKS_CHAT_MODELS;

  static DATABRICKS_CHAT_MODEL_NAMES = DATABRICKS_CHAT_MODELS.map((model) => model.id);

  config: DatabricksCompletionOptions;

  constructor(
    modelName: string,
    options: { config?: DatabricksCompletionOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    if (!DatabricksChatCompletionProvider.DATABRICKS_CHAT_MODEL_NAMES.includes(modelName)) {
      logger.debug(`Using unknown OpenAI chat model: ${modelName}`);
    }
    super(modelName, options);
    this.config = options.config || {};
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    callApiOptions?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    if (!this.getApiKey()) {
      throw new Error(
        'Databricks API key is not set. Set the DATABRICKS_API_KEY environment variable or add `apiKey` to the provider config.',
      );
    }

    const messages = parseChatPrompt(prompt, [{ role: 'user', content: prompt }]);

    const body = {
      model: this.modelName,
      messages: messages,
      max_tokens: this.config.max_tokens ?? parseInt(process.env.DATABRICKS_MAX_TOKENS || '1024'),
      temperature: this.config.temperature ?? parseFloat(process.env.DATABRICKS_TEMPERATURE || '0'),
      top_p: this.config.top_p ?? parseFloat(process.env.DATABRICKS_TOP_P || '1'),
      presence_penalty:
        this.config.presence_penalty ?? parseFloat(process.env.DATABRICKS_PRESENCE_PENALTY || '0'),
      frequency_penalty:
        this.config.frequency_penalty ?? parseFloat(process.env.DATABRICKS_FREQUENCY_PENALTY || '0'),
      ...(this.config.functions
        ? { functions: renderVarsInObject(this.config.functions, context?.vars) }
        : {}),
      ...(this.config.function_call ? { function_call: this.config.function_call } : {}),
      ...(this.config.tools ? { tools: renderVarsInObject(this.config.tools, context?.vars) } : {}),
      ...(this.config.tool_choice ? { tool_choice: this.config.tool_choice } : {}),
      ...(this.config.response_format ? { response_format: this.config.response_format } : {}),
      ...(callApiOptions?.includeLogProbs ? { logprobs: callApiOptions.includeLogProbs } : {}),
      ...(this.config.stop ? { stop: this.config.stop } : {}),
      ...(this.config.passthrough || {}),
    };
    logger.debug(`Calling Databricks API: ${JSON.stringify(body)}`);

    let data,
      cached = false;
    try {
      ({ data, cached } = (await fetchWithCache(
        `${this.getApiUrl()}/chat/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.getApiKey()}`,
            ...this.config.headers,
          },
          body: JSON.stringify(body),
        },
        REQUEST_TIMEOUT_MS,
      )) as unknown as { data: any; cached: boolean });
    } catch (err) {
      return {
        error: `API call error: ${String(err)}`,
      };
    }

    logger.debug(`\tDatabricks chat completions API response: ${JSON.stringify(data)}`);
    if (data.error) {
      return {
        error: formatDatabricksError(data),
      };
    }
    try {
      const message = data.choices[0].message;
      let output = '';
      if (message.content && (message.function_call || message.tool_calls)) {
        output = message;
      } else if (message.content === null) {
        output = message.function_call || message.tool_calls;
      } else {
        output = message.content;
      }
      const logProbs = data.choices[0].logprobs?.content?.map(
        (logProbObj: { token: string; logprob: number }) => logProbObj.logprob,
      );

      // Handle function tool callbacks
      const functionCalls = message.function_call ? [message.function_call] : message.tool_calls;
      if (functionCalls && this.config.functionToolCallbacks) {
        for (const functionCall of functionCalls) {
          const functionName = functionCall.name;
          if (this.config.functionToolCallbacks[functionName]) {
            const functionResult = await this.config.functionToolCallbacks[functionName](
              message.function_call.arguments,
            );
            return {
              output: functionResult,
              tokenUsage: getTokenUsage(data, cached),
              cached,
              logProbs,
              cost: calculateCost(
                this.modelName,
                this.config,
                data.usage?.prompt_tokens,
                data.usage?.completion_tokens,
              ),
            };
          }
        }
      }

      return {
        output,
        tokenUsage: getTokenUsage(data, cached),
        cached,
        logProbs,
        cost: calculateCost(
          this.modelName,
          this.config,
          data.usage?.prompt_tokens,
          data.usage?.completion_tokens,
        ),
      };
    } catch (err) {
      return {
        error: `API error: ${String(err)}: ${JSON.stringify(data)}`,
      };
    }
  }
}

export const DefaultGradingProvider = new DatabricksChatCompletionProvider('databricks-dbrx-instruct');
export const DefaultGradingJsonProvider = new DatabricksChatCompletionProvider('databricks-dbrx-instruct', {
  config: {
    response_format: { type: 'json_object' },
  },
});
export const DefaultSuggestionsProvider = new DatabricksChatCompletionProvider('databricks-dbrx-instruct');


