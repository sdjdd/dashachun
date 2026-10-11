//! The text generation loop: the one place that knows how to drive an LLM
//! with tools. [`stream_text`] runs the multi-round conversation as a single
//! event stream — each round's deltas stream through as they arrive, the
//! requested tools execute between rounds, and the stream finishes with
//! every message the loop produced; [`generate_text`] is its collecting
//! twin for callers that only want the final result. Both are free
//! functions over any [`Llm`] and the tool registry, so the voice driver,
//! the memory extractor and anything else that wants LLM + tools share this
//! one loop.
//!
//! Layering contract: this module imports only the `llm` and `tool`
//! surfaces. No session, audio, emotion or output concepts may appear here
//! — the loop speaks text and tool calls, nothing else. Cancellation ends
//! the stream silently (no `Done`, no `Err`); the caller distinguishes it
//! through its own token.

use std::pin::Pin;

use futures_util::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use super::llm::{ChatItem, Llm, LlmError, LlmEvent, LlmEvents, ToolCall};
use super::tool::{ToolContext, ToolRegistry, execute_tool};

/// Cap on LLM rounds per generation. A model stuck requesting tools would
/// otherwise loop (and bill) forever; hitting the cap fails the stream.
const MAX_STEPS: usize = 8;

pub type StreamEvents<'a> = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamResult {
    /// Every round's assistant text concatenated — everything the model
    /// "said", in order.
    pub text: String,
    /// The messages the loop produced, in order: the assistant rounds (with
    /// their tool calls) and the tool results. Together with the caller's
    /// input this is the full exchange.
    pub messages: Vec<ChatItem>,
    pub steps: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    Delta { text: String },
    Done(StreamResult),
}

enum Phase<'a> {
    Start,
    Chat(LlmEvents<'a>),
    Tools(std::vec::IntoIter<ToolCall>),
    RoundEnd,
    AfterTools,
    Finished,
}

struct Loop<'a> {
    llm: &'a dyn Llm,
    tools: &'a ToolRegistry,
    ctx: &'a ToolContext,
    cancel: CancellationToken,
    working: Vec<ChatItem>,
    calls: Vec<ToolCall>,
    round_text: String,
    text: String,
    items: Vec<ChatItem>,
    steps: usize,
    spoke: bool,
    needs_reply: bool,
    phase: Phase<'a>,
}

/// Runs the tool loop over `messages` (the caller owns the prompt, including
/// any leading system item) and streams every round's text deltas as they
/// arrive. When the model stops requesting tools — or a round that already
/// spoke was followed only by silent tools — the stream ends with
/// [`StreamEvent::Done`] carrying [`StreamResult`].
pub fn stream_text<'a>(
    llm: &'a dyn Llm,
    messages: Vec<ChatItem>,
    tools: &'a ToolRegistry,
    ctx: &'a ToolContext,
    cancel: CancellationToken,
) -> StreamEvents<'a> {
    let state = Loop {
        llm,
        tools,
        ctx,
        cancel,
        working: messages,
        calls: Vec::new(),
        round_text: String::new(),
        text: String::new(),
        items: Vec::new(),
        steps: 0,
        spoke: false,
        needs_reply: false,
        phase: Phase::Start,
    };
    Box::pin(futures_util::stream::unfold(
        state,
        |mut state| async move {
            loop {
                match std::mem::replace(&mut state.phase, Phase::Finished) {
                    Phase::Start => {
                        if state.cancel.is_cancelled() {
                            return None;
                        }
                        if state.steps >= MAX_STEPS {
                            return Some((
                                Err(LlmError::Failed(format!(
                                    "tool loop exceeded {MAX_STEPS} steps"
                                ))),
                                state,
                            ));
                        }
                        state.steps += 1;
                        state.round_text.clear();
                        state.spoke = false;
                        let chat = state.llm.chat(
                            state.working.clone(),
                            state.tools.specs(),
                            state.cancel.clone(),
                        );
                        state.phase = Phase::Chat(chat);
                    }
                    Phase::Chat(mut chat) => {
                        let item = tokio::select! {
                            item = chat.next() => item,
                            _ = state.cancel.cancelled() => return None,
                        };
                        match item {
                            Some(Ok(LlmEvent::Delta { text })) => {
                                if text.is_empty() {
                                    state.phase = Phase::Chat(chat);
                                    continue;
                                }
                                state.text.push_str(&text);
                                state.round_text.push_str(&text);
                                state.phase = Phase::Chat(chat);
                                return Some((Ok(StreamEvent::Delta { text }), state));
                            }
                            Some(Ok(LlmEvent::ToolCall(call))) => {
                                state.calls.push(call);
                                state.phase = Phase::Chat(chat);
                            }
                            Some(Ok(LlmEvent::Done)) | None => {
                                state.phase = Phase::RoundEnd;
                            }
                            Some(Err(err)) => return Some((Err(err), state)),
                        }
                    }
                    Phase::RoundEnd => {
                        state.spoke = !state.round_text.is_empty();
                        if state.calls.is_empty() {
                            if state.spoke {
                                state.items.push(ChatItem::assistant(std::mem::take(
                                    &mut state.round_text,
                                )));
                            }
                            let result = finish(&mut state);
                            return Some((Ok(StreamEvent::Done(result)), state));
                        }
                        let calls = std::mem::take(&mut state.calls);
                        let round = if state.spoke {
                            ChatItem::assistant_text_tool_calls(
                                std::mem::take(&mut state.round_text),
                                calls.clone(),
                            )
                        } else {
                            ChatItem::assistant_tool_calls(calls.clone())
                        };
                        state.items.push(round.clone());
                        state.working.push(round);
                        state.needs_reply = false;
                        state.phase = Phase::Tools(calls.into_iter());
                    }
                    Phase::Tools(mut calls) => {
                        let Some(call) = calls.next() else {
                            state.phase = Phase::AfterTools;
                            continue;
                        };
                        let outcome = execute_tool(state.tools, state.ctx, &call).await;
                        state.needs_reply |= outcome.needs_reply;
                        let item = ChatItem::tool(call.id.clone(), outcome.content);
                        state.items.push(item.clone());
                        state.working.push(item);
                        state.phase = Phase::Tools(calls);
                    }
                    Phase::AfterTools => {
                        if state.spoke && !state.needs_reply {
                            // The round's spoken text already answered and every
                            // tool was silent plumbing; asking the model again
                            // would only produce a second confirmation of what
                            // was just said.
                            let result = finish(&mut state);
                            return Some((Ok(StreamEvent::Done(result)), state));
                        }
                        state.phase = Phase::Start;
                    }
                    Phase::Finished => return None,
                }
            }
        },
    ))
}

fn finish(state: &mut Loop<'_>) -> StreamResult {
    StreamResult {
        text: std::mem::take(&mut state.text),
        messages: std::mem::take(&mut state.items),
        steps: state.steps,
    }
}

/// The collecting twin of [`stream_text`]: runs the same loop and returns
/// only the final result — the text and every message the loop produced.
pub async fn generate_text(
    llm: &dyn Llm,
    messages: Vec<ChatItem>,
    tools: &ToolRegistry,
    ctx: &ToolContext,
    cancel: CancellationToken,
) -> Result<StreamResult, LlmError> {
    let mut events = stream_text(llm, messages, tools, ctx, cancel);
    let mut result = None;
    while let Some(item) = events.next().await {
        match item {
            Ok(StreamEvent::Done(done)) => result = Some(done),
            Ok(StreamEvent::Delta { .. }) => {}
            Err(err) => return Err(err),
        }
    }
    result.ok_or_else(|| LlmError::Failed("text generation was cancelled".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::sync::Mutex;

    use serde::Deserialize;

    use crate::agent::ToolSpec;
    use crate::agent::tool::{Tool, ToolHandler, ToolOutcome};

    /// Serves one scripted round per `chat` call; a call beyond the script
    /// fails loudly, so a runaway loop can never pass a test silently.
    struct ScriptedLlm {
        rounds: Mutex<VecDeque<Vec<Result<LlmEvent, LlmError>>>>,
    }

    impl ScriptedLlm {
        fn new(rounds: Vec<Vec<Result<LlmEvent, LlmError>>>) -> Self {
            Self {
                rounds: Mutex::new(rounds.into()),
            }
        }
    }

    impl Llm for ScriptedLlm {
        fn chat(
            &self,
            _history: Vec<ChatItem>,
            _tools: Vec<ToolSpec>,
            _cancel: CancellationToken,
        ) -> LlmEvents<'_> {
            let events =
                self.rounds.lock().unwrap().pop_front().unwrap_or_else(|| {
                    vec![Err(LlmError::Failed("scripted llm exhausted".into()))]
                });
            Box::pin(futures_util::stream::iter(events))
        }
    }

    /// Streams one delta and then hangs until cancelled — the deterministic
    /// mid-stream cancellation fixture.
    struct HangingLlm;

    impl Llm for HangingLlm {
        fn chat(
            &self,
            _history: Vec<ChatItem>,
            _tools: Vec<ToolSpec>,
            _cancel: CancellationToken,
        ) -> LlmEvents<'_> {
            Box::pin(futures_util::stream::unfold(
                false,
                |sent: bool| async move {
                    if sent {
                        std::future::pending::<()>().await;
                        None
                    } else {
                        Some((Ok(LlmEvent::Delta { text: "hi".into() }), true))
                    }
                },
            ))
        }
    }

    fn delta(text: &str) -> Result<LlmEvent, LlmError> {
        Ok(LlmEvent::Delta { text: text.into() })
    }

    fn call(name: &str, arguments: &str) -> Result<LlmEvent, LlmError> {
        Ok(LlmEvent::ToolCall(ToolCall {
            id: "call_1".into(),
            name: name.into(),
            arguments: arguments.into(),
        }))
    }

    fn done() -> Result<LlmEvent, LlmError> {
        Ok(LlmEvent::Done)
    }

    fn tool_call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: "call_1".into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct AddArgs {
        x: i64,
        y: i64,
    }

    struct Add;

    #[async_trait::async_trait]
    impl Tool for Add {
        const NAME: &'static str = "add";
        type Args = AddArgs;
        type Output = i64;
        type Error = Infallible;

        fn description(&self) -> String {
            "Add x and y together".to_string()
        }

        async fn call(&self, _ctx: &ToolContext, args: AddArgs) -> Result<i64, Infallible> {
            Ok(args.x + args.y)
        }
    }

    struct SilentAdd;

    #[async_trait::async_trait]
    impl Tool for SilentAdd {
        const NAME: &'static str = "silent_add";
        const NEEDS_REPLY: bool = false;
        type Args = AddArgs;
        type Output = i64;
        type Error = Infallible;

        fn description(&self) -> String {
            "Add x and y together".to_string()
        }

        async fn call(&self, _ctx: &ToolContext, args: AddArgs) -> Result<i64, Infallible> {
            Ok(args.x + args.y)
        }
    }

    struct Failing;

    #[async_trait::async_trait]
    impl Tool for Failing {
        const NAME: &'static str = "fail";
        type Args = AddArgs;
        type Output = i64;
        type Error = String;

        fn description(&self) -> String {
            "Always fails".to_string()
        }

        async fn call(&self, _ctx: &ToolContext, _args: AddArgs) -> Result<i64, String> {
            Err("boom".to_string())
        }
    }

    struct LegacyEcho;

    #[async_trait::async_trait]
    impl ToolHandler for LegacyEcho {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "legacy_echo".to_string(),
                description: "legacy tool".to_string(),
                parameters: serde_json::Value::Null,
            }
        }

        async fn call(&self, _args: &serde_json::Value) -> Result<ToolOutcome, String> {
            Ok(ToolOutcome {
                content: "legacy-ok".to_string(),
                needs_reply: false,
            })
        }
    }

    fn typed_registry() -> ToolRegistry {
        let mut tools = ToolRegistry::new(Vec::new());
        tools.register_tool(Add);
        tools.register_tool(SilentAdd);
        tools.register_tool(Failing);
        tools
    }

    async fn drain(events: &mut StreamEvents<'_>) -> Vec<Result<StreamEvent, LlmError>> {
        let mut seen = Vec::new();
        while let Some(item) = events.next().await {
            seen.push(item);
        }
        seen
    }

    #[tokio::test]
    async fn text_only_turn_streams_deltas_then_done() {
        let llm = ScriptedLlm::new(vec![vec![delta("你"), delta("好"), done()]]);
        let tools = ToolRegistry::new(Vec::new());
        let ctx = ToolContext::default();
        let mut events = stream_text(
            &llm,
            vec![ChatItem::user("hi")],
            &tools,
            &ctx,
            CancellationToken::new(),
        );
        match events.next().await {
            Some(Ok(StreamEvent::Delta { text })) => assert_eq!(text, "你"),
            other => panic!("expected delta, got {other:?}"),
        }
        match events.next().await {
            Some(Ok(StreamEvent::Delta { text })) => assert_eq!(text, "好"),
            other => panic!("expected delta, got {other:?}"),
        }
        match events.next().await {
            Some(Ok(StreamEvent::Done(result))) => {
                assert_eq!(result.text, "你好");
                assert_eq!(result.messages, vec![ChatItem::assistant("你好")]);
                assert_eq!(result.steps, 1);
            }
            other => panic!("expected done, got {other:?}"),
        }
        assert!(events.next().await.is_none());
    }

    #[tokio::test]
    async fn tool_round_is_followed_by_the_reply_round() {
        let llm = ScriptedLlm::new(vec![
            vec![delta("算一下"), call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        let result = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("1+2?")],
            &typed_registry(),
            &ToolContext::default(),
            CancellationToken::new(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        assert_eq!(result.text, "算一下等于 3");
        assert_eq!(result.steps, 2);
        assert_eq!(
            result.messages,
            vec![
                ChatItem::assistant_text_tool_calls(
                    "算一下",
                    vec![tool_call("add", r#"{"x":1,"y":2}"#)]
                ),
                ChatItem::tool("call_1", "3"),
                ChatItem::assistant("等于 3"),
            ]
        );
    }

    #[tokio::test]
    async fn spoken_round_with_silent_tools_closes_without_another_round() {
        let llm = ScriptedLlm::new(vec![vec![
            delta("好的"),
            call("silent_add", r#"{"x":1,"y":2}"#),
            done(),
        ]]);
        let seen = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("1+2?")],
            &typed_registry(),
            &ToolContext::default(),
            CancellationToken::new(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = seen.into_iter().last() else {
            panic!("expected a done event");
        };
        assert_eq!(result.steps, 1);
        assert_eq!(
            result.messages,
            vec![
                ChatItem::assistant_text_tool_calls(
                    "好的",
                    vec![tool_call("silent_add", r#"{"x":1,"y":2}"#)]
                ),
                ChatItem::tool("call_1", "3"),
            ]
        );
    }

    #[tokio::test]
    async fn failing_tool_asks_the_model_again() {
        let llm = ScriptedLlm::new(vec![
            vec![call("fail", r#"{"x":1,"y":2}"#), done()],
            vec![delta("抱歉"), done()],
        ]);
        let result = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("1+2?")],
            &typed_registry(),
            &ToolContext::default(),
            CancellationToken::new(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        assert_eq!(result.steps, 2);
        assert!(
            result
                .messages
                .contains(&ChatItem::tool("call_1", "error: boom"))
        );
    }

    #[tokio::test]
    async fn malformed_typed_arguments_degrade_and_continue() {
        let llm = ScriptedLlm::new(vec![
            vec![call("add", r#"{"x":1}"#), done()],
            vec![delta("重来"), done()],
        ]);
        let result = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("1+2?")],
            &typed_registry(),
            &ToolContext::default(),
            CancellationToken::new(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        assert!(
            result.messages.contains(&ChatItem::tool(
                "call_1",
                "error: invalid arguments: missing field `y`"
            )) || result.messages.iter().any(|item| matches!(
                item,
                ChatItem::Tool { content, .. } if content.starts_with("error: invalid arguments")
            )),
            "messages: {:?}",
            result.messages
        );
        assert_eq!(result.steps, 2);
    }

    #[tokio::test]
    async fn unknown_tool_degrades_and_continues() {
        let llm = ScriptedLlm::new(vec![
            vec![call("nope", "{}"), done()],
            vec![delta("好"), done()],
        ]);
        let result = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("hi")],
            &typed_registry(),
            &ToolContext::default(),
            CancellationToken::new(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        assert!(
            result
                .messages
                .contains(&ChatItem::tool("call_1", "error: unknown tool `nope`"))
        );
        assert_eq!(result.steps, 2);
    }

    #[tokio::test]
    async fn tool_loop_hits_the_step_cap() {
        let rounds = (0..MAX_STEPS)
            .map(|_| vec![call("add", r#"{"x":1,"y":2}"#), done()])
            .collect();
        let llm = ScriptedLlm::new(rounds);
        let seen = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("hi")],
            &typed_registry(),
            &ToolContext::default(),
            CancellationToken::new(),
        ))
        .await;
        match seen.into_iter().last() {
            Some(Err(LlmError::Failed(message))) => {
                assert!(message.contains("exceeded 8 steps"), "{message}");
            }
            other => panic!("expected the step cap error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancelled_before_start_ends_silently() {
        let llm = ScriptedLlm::new(vec![vec![delta("hi"), done()]]);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let seen = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("hi")],
            &ToolRegistry::new(Vec::new()),
            &ToolContext::default(),
            cancel,
        ))
        .await;
        assert!(seen.is_empty(), "{seen:?}");
    }

    #[tokio::test]
    async fn cancelled_mid_stream_ends_silently() {
        let cancel = CancellationToken::new();
        let tools = ToolRegistry::new(Vec::new());
        let ctx = ToolContext::default();
        let mut events = stream_text(
            &HangingLlm,
            vec![ChatItem::user("hi")],
            &tools,
            &ctx,
            cancel.clone(),
        );
        match events.next().await {
            Some(Ok(StreamEvent::Delta { text })) => assert_eq!(text, "hi"),
            other => panic!("expected delta, got {other:?}"),
        }
        cancel.cancel();
        assert!(events.next().await.is_none());
    }

    #[tokio::test]
    async fn chat_error_propagates() {
        let llm = ScriptedLlm::new(vec![vec![Err(LlmError::Failed("provider down".into()))]]);
        let seen = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("hi")],
            &ToolRegistry::new(Vec::new()),
            &ToolContext::default(),
            CancellationToken::new(),
        ))
        .await;
        match seen.into_iter().last() {
            Some(Err(LlmError::Failed(message))) => assert_eq!(message, "provider down"),
            other => panic!("expected the provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn legacy_handler_runs_through_the_same_loop() {
        let llm = ScriptedLlm::new(vec![vec![delta("查"), call("legacy_echo", "{}"), done()]]);
        let result = drain(&mut stream_text(
            &llm,
            vec![ChatItem::user("hi")],
            &ToolRegistry::new(vec![Arc::new(LegacyEcho)]),
            &ToolContext::default(),
            CancellationToken::new(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        assert_eq!(result.steps, 1);
        assert_eq!(
            result.messages,
            vec![
                ChatItem::assistant_text_tool_calls("查", vec![tool_call("legacy_echo", "{}")]),
                ChatItem::tool("call_1", "legacy-ok"),
            ]
        );
    }

    #[tokio::test]
    async fn generate_text_collects_the_stream() {
        let llm = ScriptedLlm::new(vec![
            vec![delta("算一下"), call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        let result = generate_text(
            &llm,
            vec![ChatItem::user("1+2?")],
            &typed_registry(),
            &ToolContext::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(result.text, "算一下等于 3");
        assert_eq!(result.steps, 2);
        assert_eq!(result.messages.len(), 3);
    }
}
