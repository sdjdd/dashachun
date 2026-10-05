use std::collections::BTreeMap;
use std::time::Duration;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage, ChatCompletionTool,
    ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionCall, FunctionObject,
};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::{ChatItem, Llm, LlmError, LlmEvent, LlmEvents, ToolCall, ToolSpec};

fn parse_reasoning_effort(
    value: &str,
) -> Result<async_openai::types::chat::ReasoningEffort, LlmError> {
    use async_openai::types::chat::ReasoningEffort;
    match value.to_ascii_lowercase().as_str() {
        "none" => Ok(ReasoningEffort::None),
        "low" => Ok(ReasoningEffort::Low),
        "medium" => Ok(ReasoningEffort::Medium),
        "high" => Ok(ReasoningEffort::High),
        other => Err(LlmError::from(format!(
            "invalid reasoning effort {other}: expected none/low/medium/high"
        ))),
    }
}

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful voice assistant. Keep replies short and conversational.";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub system_prompt: String,
    pub max_tokens: Option<u32>,
    pub reasoning_effort: Option<String>,
}

impl OpenAiConfig {
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("LLM_BASE_URL").ok()?;
        let api_key = std::env::var("LLM_API_KEY").ok()?;
        let model = std::env::var("LLM_MODEL").ok()?;
        let system_prompt = std::env::var("LLM_SYSTEM_PROMPT")
            .unwrap_or_else(|_| DEFAULT_SYSTEM_PROMPT.to_string());
        let max_tokens = std::env::var("LLM_MAX_TOKENS")
            .or_else(|_| std::env::var("LLM_MAX_OUTPUT_TOKENS"))
            .ok()
            .and_then(|value| value.parse().ok());
        let reasoning_effort = std::env::var("LLM_REASONING_EFFORT")
            .ok()
            .filter(|value| !value.is_empty());
        Some(Self {
            base_url,
            api_key,
            model,
            system_prompt,
            max_tokens,
            reasoning_effort,
        })
    }
}

pub struct OpenAiLlm {
    client: Client<OpenAIConfig>,
    config: OpenAiConfig,
}

impl OpenAiLlm {
    pub fn new(config: OpenAiConfig) -> Self {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(config.base_url.clone())
                .with_api_key(config.api_key.clone()),
        );
        Self { client, config }
    }

    fn build_request(
        &self,
        history: Vec<ChatItem>,
        tools: Vec<ToolSpec>,
    ) -> Result<async_openai::types::chat::CreateChatCompletionRequest, LlmError> {
        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: self.config.system_prompt.clone().into(),
                name: None,
            },
        ));
        messages.extend(history.into_iter().map(|item| match item {
            ChatItem::User { content } => {
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: content.into(),
                    name: None,
                })
            }
            ChatItem::Assistant {
                content,
                tool_calls,
            } => {
                let tool_calls = if tool_calls.is_empty() {
                    None
                } else {
                    Some(
                        tool_calls
                            .into_iter()
                            .map(|call| {
                                ChatCompletionMessageToolCalls::Function(
                                    ChatCompletionMessageToolCall {
                                        id: call.id,
                                        function: FunctionCall {
                                            name: call.name,
                                            arguments: call.arguments,
                                        },
                                    },
                                )
                            })
                            .collect(),
                    )
                };
                ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                    content: content.map(Into::into),
                    tool_calls,
                    ..Default::default()
                })
            }
            ChatItem::Tool {
                tool_call_id,
                content,
            } => ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
                content: ChatCompletionRequestToolMessageContent::Text(content),
                tool_call_id,
            }),
        }));

        let mut request = CreateChatCompletionRequestArgs::default();
        request
            .model(self.config.model.clone())
            .messages(messages)
            .stream(true);
        if let Some(max_tokens) = self.config.max_tokens {
            request.max_tokens(max_tokens);
        }
        if let Some(effort) = self.config.reasoning_effort.as_deref() {
            request.reasoning_effort(parse_reasoning_effort(effort)?);
        }
        if !tools.is_empty() {
            request.tools(
                tools
                    .into_iter()
                    .map(|spec| {
                        ChatCompletionTools::Function(ChatCompletionTool {
                            function: FunctionObject {
                                name: spec.name,
                                description: Some(spec.description),
                                parameters: Some(spec.parameters),
                                strict: None,
                            },
                        })
                    })
                    .collect::<Vec<_>>(),
            );
        }
        request
            .build()
            .map_err(|err| LlmError::from(format!("build request: {err}")))
    }
}

impl Llm for OpenAiLlm {
    fn chat(
        &self,
        history: Vec<ChatItem>,
        tools: Vec<ToolSpec>,
        cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<LlmEvent, LlmError>>();
        let request = match self.build_request(history, tools) {
            Ok(request) => request,
            Err(err) => {
                let _ = tx.send(Err(err));
                return Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                    rx.recv().await.map(|event| (event, rx))
                }));
            }
        };
        let client = self.client.clone();
        tokio::spawn(async move {
            if let Err(err) = run(client, request, tx.clone(), cancel).await {
                let _ = tx.send(Err(err));
            }
        });
        Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        }))
    }
}

/// Accumulates streamed tool-call fragments, keyed by their `index`.
#[derive(Default)]
struct ToolCallAccumulator {
    calls: BTreeMap<u32, PartialCall>,
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    fn push(&mut self, chunk: async_openai::types::chat::ChatCompletionMessageToolCallChunk) {
        let entry = self.calls.entry(chunk.index).or_default();
        if let Some(id) = chunk.id {
            entry.id = id;
        }
        if let Some(function) = chunk.function {
            if let Some(name) = function.name {
                entry.name.push_str(&name);
            }
            if let Some(arguments) = function.arguments {
                entry.arguments.push_str(&arguments);
            }
        }
    }

    fn finish(self) -> Vec<ToolCall> {
        self.calls
            .into_values()
            .filter(|call| !call.name.is_empty())
            .map(|call| ToolCall {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            })
            .collect()
    }
}

async fn run(
    client: Client<OpenAIConfig>,
    request: async_openai::types::chat::CreateChatCompletionRequest,
    tx: mpsc::UnboundedSender<Result<LlmEvent, LlmError>>,
    cancel: CancellationToken,
) -> Result<(), LlmError> {
    let chat = client.chat();
    let create = tokio::time::timeout(REQUEST_TIMEOUT, chat.create_stream(request));
    let mut stream = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            info!("llm cancelled before stream opened");
            return Ok(());
        }
        result = create => {
            result
                .map_err(|_| LlmError::from("request timeout"))?
                .map_err(|err| LlmError::from(format!("create chat completion: {err}")))?
        }
    };

    let mut tool_calls = ToolCallAccumulator::default();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("llm cancelled, dropping stream");
                return Ok(());
            }
            chunk = stream.next() => {
                let Some(chunk) = chunk else { break };
                let chunk = chunk.map_err(|err| LlmError::from(format!("stream failed: {err}")))?;
                for choice in chunk.choices {
                    if let Some(text) = choice.delta.content
                        && !text.is_empty()
                        && tx.send(Ok(LlmEvent::Delta { text })).is_err()
                    {
                        return Ok(());
                    }
                    if let Some(chunks) = choice.delta.tool_calls {
                        for chunk in chunks {
                            tool_calls.push(chunk);
                        }
                    }
                }
            }
        }
    }

    for call in tool_calls.finish() {
        if tx.send(Ok(LlmEvent::ToolCall(call))).is_err() {
            return Ok(());
        }
    }
    let _ = tx.send(Ok(LlmEvent::Done));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{ChatCompletionMessageToolCallChunk, FunctionCallStream};

    #[test]
    fn parses_reasoning_effort_case_insensitively() {
        assert_eq!(
            parse_reasoning_effort("none").unwrap(),
            async_openai::types::chat::ReasoningEffort::None
        );
        assert_eq!(
            parse_reasoning_effort("HIGH").unwrap(),
            async_openai::types::chat::ReasoningEffort::High
        );
        assert!(parse_reasoning_effort("bogus").is_err());
    }

    fn chunk(
        index: u32,
        id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> ChatCompletionMessageToolCallChunk {
        ChatCompletionMessageToolCallChunk {
            index,
            id: id.map(str::to_string),
            r#type: None,
            function: Some(FunctionCallStream {
                name: name.map(str::to_string),
                arguments: arguments.map(str::to_string),
            }),
        }
    }

    #[test]
    fn accumulates_fragmented_tool_call() {
        let mut acc = ToolCallAccumulator::default();
        acc.push(chunk(0, Some("call_1"), Some("set_"), None));
        acc.push(chunk(0, None, Some("emotion"), Some("{\"emo")));
        acc.push(chunk(0, None, None, Some("tion\":\"happy\"}")));
        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "set_emotion");
        assert_eq!(calls[0].arguments, "{\"emotion\":\"happy\"}");
    }

    #[test]
    fn accumulates_parallel_tool_calls_by_index() {
        let mut acc = ToolCallAccumulator::default();
        acc.push(chunk(0, Some("a"), Some("set_emotion"), Some("{}")));
        acc.push(chunk(1, Some("b"), Some("get_weather"), Some("{\"city\"")));
        acc.push(chunk(1, None, None, Some(":\"beijing\"}")));
        let calls = acc.finish();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "set_emotion");
        assert_eq!(calls[1].name, "get_weather");
        assert_eq!(calls[1].arguments, "{\"city\":\"beijing\"}");
    }
}
