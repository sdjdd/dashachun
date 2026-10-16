use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, ChatCompletionNamedToolChoice,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionStreamOptions, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionTools, CreateChatCompletionRequestArgs, FinishReason, FunctionCall, FunctionName,
    FunctionObject, ToolChoiceOptions,
};
use futures_util::StreamExt;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::agent::{
    ChatItem, Llm, LlmError, LlmEvent, LlmEvents, LlmMeta, LlmResponse, LlmUsage, ReasoningEffort,
    ToolCall, ToolChoice, ToolSpec,
};
use crate::provider::{ProviderRegistry, ProviderSpec};

fn reasoning_wire(effort: ReasoningEffort) -> async_openai::types::chat::ReasoningEffort {
    use async_openai::types::chat::ReasoningEffort as Wire;
    match effort {
        ReasoningEffort::None => Wire::None,
        ReasoningEffort::Low => Wire::Low,
        ReasoningEffort::Medium => Wire::Medium,
        ReasoningEffort::High => Wire::High,
    }
}

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Deserialize)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

impl fmt::Debug for OpenAiConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

pub const KIND: &str = "openai";

pub fn register(registry: &mut ProviderRegistry) {
    registry.register_llm(KIND, build);
}

fn build(spec: &ProviderSpec) -> Result<Arc<dyn Llm>, String> {
    let config: OpenAiConfig = serde_json::from_value(spec.config.clone())
        .map_err(|err| format!("invalid openai llm config: {err}"))?;
    Ok(Arc::new(OpenAiLlm::new(&spec.name, config)))
}

pub struct OpenAiLlm {
    client: Client<OpenAIConfig>,
    config: OpenAiConfig,
    provider: String,
}

impl OpenAiLlm {
    pub fn new(provider: impl Into<String>, config: OpenAiConfig) -> Self {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(config.base_url.clone())
                .with_api_key(config.api_key.clone()),
        );
        Self {
            client,
            config,
            provider: provider.into(),
        }
    }

    fn build_request(
        &self,
        history: Vec<ChatItem>,
        tools: Vec<ToolSpec>,
        tool_choice: Option<ToolChoice<'_>>,
        reasoning: Option<ReasoningEffort>,
    ) -> Result<async_openai::types::chat::CreateChatCompletionRequest, LlmError> {
        let messages = history
            .into_iter()
            .map(|item| match item {
                ChatItem::System { content } => {
                    ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                        content: content.into(),
                        name: None,
                    })
                }
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
            })
            .collect::<Vec<_>>();

        let mut request = CreateChatCompletionRequestArgs::default();
        request
            .model(self.config.model.clone())
            .messages(messages)
            .stream(true);
        // The final usage chunk rides the stream when the server honors
        // `include_usage` — the totals behind the chat span's
        // `gen_ai.usage.*` attributes.
        request.stream_options(ChatCompletionStreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        });
        if let Some(max_tokens) = self.config.max_tokens {
            request.max_tokens(max_tokens);
        }
        // The caller decides the effort per call — with none given, the
        // field is omitted and the server default applies.
        if let Some(effort) = reasoning {
            request.reasoning_effort(reasoning_wire(effort));
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
            // A choice without advertised tools is meaningless; `None`
            // leaves the server default in place.
            if let Some(choice) = tool_choice {
                match choice {
                    ToolChoice::Auto => {
                        request.tool_choice(ChatCompletionToolChoiceOption::Mode(
                            ToolChoiceOptions::Auto,
                        ));
                    }
                    ToolChoice::Required => {
                        request.tool_choice(ChatCompletionToolChoiceOption::Mode(
                            ToolChoiceOptions::Required,
                        ));
                    }
                    ToolChoice::Named(name) => {
                        request.tool_choice(ChatCompletionToolChoiceOption::Function(
                            ChatCompletionNamedToolChoice {
                                function: FunctionName {
                                    name: name.to_string(),
                                },
                            },
                        ));
                    }
                }
            }
        }
        request
            .build()
            .map_err(|err| LlmError::from(format!("build request: {err}")))
    }
}

impl Llm for OpenAiLlm {
    fn meta(&self) -> LlmMeta {
        LlmMeta {
            provider: self.provider.clone(),
            model: self.config.model.clone(),
        }
    }

    fn chat(
        &self,
        history: Vec<ChatItem>,
        tools: Vec<ToolSpec>,
        tool_choice: Option<ToolChoice<'_>>,
        reasoning: Option<ReasoningEffort>,
        cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<LlmEvent, LlmError>>();
        let request = match self.build_request(history, tools, tool_choice, reasoning) {
            Ok(request) => request,
            Err(err) => {
                let _ = tx.send(Err(err));
                return crate::util::recv_unbounded_stream(rx);
            }
        };
        let client = self.client.clone();
        tokio::spawn(async move {
            if let Err(err) = run(client, request, tx.clone(), cancel, IDLE_TIMEOUT).await {
                let _ = tx.send(Err(err));
            }
        });
        crate::util::recv_unbounded_stream(rx)
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

fn finish_reason_wire(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::FunctionCall => "function_call",
    }
}

async fn run(
    client: Client<OpenAIConfig>,
    request: async_openai::types::chat::CreateChatCompletionRequest,
    tx: mpsc::UnboundedSender<Result<LlmEvent, LlmError>>,
    cancel: CancellationToken,
    idle_timeout: Duration,
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
    let mut response = LlmResponse {
        id: None,
        model: None,
        finish_reasons: Vec::new(),
        usage: None,
    };
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("llm cancelled, dropping stream");
                return Ok(());
            }
            chunk = tokio::time::timeout(idle_timeout, stream.next()) => {
                let Some(chunk) = chunk
                    .map_err(|_| LlmError::from(format!("stream idle timeout after {idle_timeout:?}")))?
                else {
                    break;
                };
                let chunk = chunk.map_err(|err| LlmError::from(format!("stream failed: {err}")))?;
                response.id.get_or_insert(chunk.id);
                response.model.get_or_insert(chunk.model);
                if let Some(usage) = chunk.usage {
                    response.usage = Some(LlmUsage {
                        input_tokens: usage.prompt_tokens,
                        output_tokens: usage.completion_tokens,
                    });
                }
                for choice in chunk.choices {
                    if let Some(reason) = choice.finish_reason {
                        response
                            .finish_reasons
                            .push(finish_reason_wire(reason).to_string());
                    }
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
    let _ = tx.send(Ok(LlmEvent::Response(response)));
    let _ = tx.send(Ok(LlmEvent::Done));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{ChatCompletionMessageToolCallChunk, FunctionCallStream};

    fn config() -> OpenAiConfig {
        OpenAiConfig {
            base_url: "http://localhost".into(),
            api_key: "k".into(),
            model: "m".into(),
            max_tokens: None,
        }
    }

    fn tool_specs() -> Vec<ToolSpec> {
        vec![ToolSpec {
            name: "add".into(),
            description: "Add x and y".into(),
            parameters: serde_json::json!({"type": "object"}),
        }]
    }

    #[test]
    fn maps_items_to_request_messages() {
        let llm = OpenAiLlm::new("volcengine", config());
        let request = llm
            .build_request(
                vec![
                    ChatItem::system("be nice"),
                    ChatItem::user("hi"),
                    ChatItem::assistant("hello"),
                ],
                Vec::new(),
                None,
                None,
            )
            .unwrap();
        let value = serde_json::to_value(&request).unwrap();
        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be nice");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hi");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"], "hello");
    }

    #[test]
    fn tool_choice_maps_to_the_wire() {
        let llm = OpenAiLlm::new("volcengine", config());
        let value = serde_json::to_value(
            llm.build_request(
                vec![ChatItem::user("hi")],
                tool_specs(),
                Some(ToolChoice::Named("add")),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            value["tool_choice"],
            serde_json::json!({"type": "function", "function": {"name": "add"}})
        );

        let value = serde_json::to_value(
            llm.build_request(
                vec![ChatItem::user("hi")],
                tool_specs(),
                Some(ToolChoice::Required),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(value["tool_choice"], "required");

        // An explicit `Auto` is sent as `auto` — distinct from `None`, which
        // omits the field entirely.
        let value = serde_json::to_value(
            llm.build_request(
                vec![ChatItem::user("hi")],
                tool_specs(),
                Some(ToolChoice::Auto),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(value["tool_choice"], "auto");
    }

    #[test]
    fn tool_choice_none_is_omitted_and_needs_advertised_tools() {
        let llm = OpenAiLlm::new("volcengine", config());
        let value = serde_json::to_value(
            llm.build_request(vec![ChatItem::user("hi")], tool_specs(), None, None)
                .unwrap(),
        )
        .unwrap();
        assert!(value.get("tool_choice").is_none());

        // A forced choice without any advertised tool is meaningless and is
        // dropped instead of invalidating the request.
        let value = serde_json::to_value(
            llm.build_request(
                vec![ChatItem::user("hi")],
                Vec::new(),
                Some(ToolChoice::Named("add")),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(value.get("tool_choice").is_none());
        assert!(value.get("tools").is_none());
    }

    #[test]
    fn reasoning_maps_to_the_wire() {
        let llm = OpenAiLlm::new("volcengine", config());
        let value = serde_json::to_value(
            llm.build_request(
                vec![ChatItem::user("hi")],
                Vec::new(),
                None,
                Some(ReasoningEffort::Low),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(value["reasoning_effort"], "low");

        // `None` is a real value, not the absent one: it switches reasoning
        // off on the wire.
        let value = serde_json::to_value(
            llm.build_request(
                vec![ChatItem::user("hi")],
                Vec::new(),
                None,
                Some(ReasoningEffort::None),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(value["reasoning_effort"], "none");
    }

    #[test]
    fn reasoning_unset_is_omitted() {
        let llm = OpenAiLlm::new("volcengine", config());
        let value = serde_json::to_value(
            llm.build_request(vec![ChatItem::user("hi")], Vec::new(), None, None)
                .unwrap(),
        )
        .unwrap();
        assert!(value.get("reasoning_effort").is_none());
    }

    #[test]
    fn stream_options_request_the_usage_chunk() {
        let llm = OpenAiLlm::new("volcengine", config());
        let value = serde_json::to_value(
            llm.build_request(vec![ChatItem::user("hi")], Vec::new(), None, None)
                .unwrap(),
        )
        .unwrap();
        // The exact object also proves `include_obfuscation` stays omitted.
        assert_eq!(
            value["stream_options"],
            serde_json::json!({"include_usage": true})
        );
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
        acc.push(chunk(0, Some("call_1"), Some("get_"), None));
        acc.push(chunk(0, None, Some("weather"), Some("{\"ci")));
        acc.push(chunk(0, None, None, Some("ty\":\"beijing\"}")));
        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"beijing\"}");
    }

    #[test]
    fn accumulates_parallel_tool_calls_by_index() {
        let mut acc = ToolCallAccumulator::default();
        acc.push(chunk(0, Some("a"), Some("get_weather"), Some("{}")));
        acc.push(chunk(1, Some("b"), Some("get_weather"), Some("{\"city\"")));
        acc.push(chunk(1, None, None, Some(":\"beijing\"}")));
        let calls = acc.finish();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_weather");
        assert_eq!(calls[1].arguments, "{\"city\":\"beijing\"}");
    }

    #[tokio::test]
    async fn idle_stream_errors_after_timeout() {
        use axum::response::sse::{Event, Sse};

        let chunk = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hi"},
                    "finish_reason": null
                }
            ]
        });
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move || {
                let event = Event::default().data(chunk.to_string());
                async move {
                    Sse::new(
                        futures_util::stream::once(async move {
                            Ok::<_, std::convert::Infallible>(event)
                        })
                        .chain(futures_util::stream::pending()),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(format!("http://{addr}/v1"))
                .with_api_key("k"),
        );
        let request = CreateChatCompletionRequestArgs::default()
            .model("m")
            .messages(vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: "hi".to_string().into(),
                    name: None,
                },
            )])
            .build()
            .unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(
            client,
            request,
            tx,
            CancellationToken::new(),
            Duration::from_millis(200),
        ));

        let first = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            first,
            LlmEvent::Delta {
                text: "hi".to_string()
            }
        );

        let outcome = tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .unwrap()
            .unwrap();
        let message = outcome.unwrap_err().to_string();
        assert!(message.contains("idle"), "unexpected error: {message}");
    }

    /// Serves the given chunks as one SSE chat-completions response, then
    /// closes with `[DONE]`; answers with the base URL to call.
    async fn serve_chunks(chunks: Vec<String>) -> String {
        use axum::response::Sse;
        use axum::response::sse::Event;
        use axum::{Router, routing::post};

        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let body = futures_util::stream::iter(chunks.clone())
                    .map(|data| Ok::<_, std::convert::Infallible>(Event::default().data(data)))
                    .chain(futures_util::stream::once(async {
                        Ok(Event::default().data("[DONE]"))
                    }));
                async move { Sse::new(body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}/v1")
    }

    fn chunk_json(value: serde_json::Value) -> String {
        value.to_string()
    }

    /// Streams one request against `base_url` and collects every event
    /// until the channel closes.
    async fn collect_events(base_url: String) -> Vec<Result<LlmEvent, LlmError>> {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("k"),
        );
        let request = CreateChatCompletionRequestArgs::default()
            .model("m")
            .messages(vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: "hi".to_string().into(),
                    name: None,
                },
            )])
            .build()
            .unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(
            client,
            request,
            tx,
            CancellationToken::new(),
            Duration::from_secs(5),
        ));
        let mut events = Vec::new();
        while let Some(event) = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
        {
            events.push(event);
        }
        handle.await.unwrap().unwrap();
        events
    }

    #[tokio::test]
    async fn usage_chunk_becomes_the_response_event() {
        let base_url = serve_chunks(vec![
            chunk_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "delta": {"content": "hi"}, "finish_reason": null}
                ]
            })),
            chunk_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            })),
            chunk_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            })),
        ])
        .await;
        let events = collect_events(base_url).await;
        assert_eq!(events.len(), 3, "unexpected events: {events:?}");
        assert_eq!(
            events[0].as_ref().unwrap(),
            &LlmEvent::Delta { text: "hi".into() }
        );
        assert_eq!(
            events[1].as_ref().unwrap(),
            &LlmEvent::Response(LlmResponse {
                id: Some("chatcmpl-1".into()),
                model: Some("m".into()),
                finish_reasons: vec!["stop".into()],
                usage: Some(LlmUsage {
                    input_tokens: 11,
                    output_tokens: 7,
                }),
            })
        );
        assert_eq!(events[2].as_ref().unwrap(), &LlmEvent::Done);
    }

    #[tokio::test]
    async fn response_event_tolerates_a_server_without_usage() {
        let base_url = serve_chunks(vec![chunk_json(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": "hi"}, "finish_reason": null}
            ]
        }))])
        .await;
        let events = collect_events(base_url).await;
        assert_eq!(events.len(), 3, "unexpected events: {events:?}");
        assert_eq!(
            events[0].as_ref().unwrap(),
            &LlmEvent::Delta { text: "hi".into() }
        );
        assert_eq!(
            events[1].as_ref().unwrap(),
            &LlmEvent::Response(LlmResponse {
                id: Some("chatcmpl-1".into()),
                model: Some("m".into()),
                finish_reasons: Vec::new(),
                usage: None,
            })
        );
        assert_eq!(events[2].as_ref().unwrap(), &LlmEvent::Done);
    }
}
