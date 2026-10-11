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
use super::tool::{ToolEntry, execute_tool};

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
    /// Completed LLM rounds.
    pub steps: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    Delta { text: String },
    Done(StreamResult),
}

/// One completed LLM round (step) — what a stop condition sees: the round's
/// tool calls (empty for the final text-only round), whether every call in
/// it succeeded — a degraded call (invalid arguments, unknown tool, tool
/// failure) makes it `false` — and the round's assistant text (empty when
/// the round only requested tools).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepResult {
    pub tool_calls: Vec<ToolCall>,
    pub ok: bool,
    pub text: String,
}

/// Caller-supplied early stop for the loop: `stop_when` is an array of these,
/// checked after each tool round with every completed step; any condition
/// returning `true` ends the stream (AI-SDK `stopWhen` semantics).
pub type StopCondition = Box<dyn Fn(&[StepResult]) -> bool + Send + Sync>;

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
    tools: &'a [ToolEntry],
    cancel: CancellationToken,
    stop_when: Vec<StopCondition>,
    working: Vec<ChatItem>,
    calls: Vec<ToolCall>,
    round_calls: Vec<ToolCall>,
    round_text: String,
    text: String,
    items: Vec<ChatItem>,
    step_results: Vec<StepResult>,
    round_ok: bool,
    phase: Phase<'a>,
}

/// Runs the tool loop over `messages` (the caller owns the prompt, including
/// any leading system item) and streams every round's text deltas as they
/// arrive. When the model stops requesting tools — or a round that already
/// spoke was followed only by silent tools — the stream ends with
/// [`StreamEvent::Done`] carrying [`StreamResult`]. `stop_when` adds
/// caller-supplied early stops: after each tool round every condition sees
/// the completed steps so far, and any condition returning `true` ends the
/// stream the same way. An empty array never stops early; the step cap stays
/// a hard failure guard on top.
/// The single argument of [`stream_text`] and [`generate_text`] — one
/// generation call's parameters. `llm` goes positionally into
/// [`StreamTextParams::builder`]; the rest is optional — `messages` (empty),
/// `tools` (none: no tools are advertised), `stop_when` (empty: never stop
/// early) and `cancel` (a fresh token).
#[derive(bon::Builder)]
pub struct StreamTextParams<'a> {
    #[builder(start_fn)]
    pub llm: &'a dyn Llm,
    #[builder(default)]
    pub messages: Vec<ChatItem>,
    pub tools: Option<&'a [ToolEntry]>,
    #[builder(default)]
    pub stop_when: Vec<StopCondition>,
    #[builder(default = CancellationToken::new())]
    pub cancel: CancellationToken,
}

/// Runs the tool loop over `params.messages` (the caller owns the prompt,
/// including any leading system item) and streams every round's text deltas
/// as they arrive. When the model stops requesting tools, the stream ends
/// with [`StreamEvent::Done`] carrying [`StreamResult`]. `stop_when` owns
/// every earlier end: after each tool round every condition sees the
/// completed steps so far — each step's calls, whether its calls succeeded
/// and the round's assistant text (empty when it only requested tools) — and
/// any condition returning `true` ends the
/// stream the same way. An empty array never stops early; the step cap stays
/// a hard failure guard on top.
pub fn stream_text(params: StreamTextParams<'_>) -> StreamEvents<'_> {
    let StreamTextParams {
        llm,
        messages,
        tools,
        stop_when,
        cancel,
    } = params;
    let tools: &[ToolEntry] = tools.unwrap_or(&[]);
    let state = Loop {
        llm,
        tools,
        cancel,
        stop_when,
        working: messages,
        calls: Vec::new(),
        round_calls: Vec::new(),
        round_text: String::new(),
        text: String::new(),
        items: Vec::new(),
        step_results: Vec::new(),
        round_ok: true,
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
                        if state.step_results.len() >= MAX_STEPS {
                            return Some((
                                Err(LlmError::Failed(format!(
                                    "tool loop exceeded {MAX_STEPS} steps"
                                ))),
                                state,
                            ));
                        }
                        state.round_text.clear();
                        let specs = state
                            .tools
                            .iter()
                            .map(|entry| entry.spec().clone())
                            .collect();
                        let chat =
                            state
                                .llm
                                .chat(state.working.clone(), specs, state.cancel.clone());
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
                        let spoke = !state.round_text.is_empty();
                        let calls = std::mem::take(&mut state.calls);
                        if calls.is_empty() {
                            let text = std::mem::take(&mut state.round_text);
                            if !text.is_empty() {
                                state.items.push(ChatItem::assistant(text.clone()));
                            }
                            state.step_results.push(StepResult {
                                tool_calls: Vec::new(),
                                ok: true,
                                text,
                            });
                            let result = finish(&mut state);
                            return Some((Ok(StreamEvent::Done(result)), state));
                        }
                        let round = if spoke {
                            ChatItem::assistant_text_tool_calls(
                                state.round_text.clone(),
                                calls.clone(),
                            )
                        } else {
                            ChatItem::assistant_tool_calls(calls.clone())
                        };
                        state.items.push(round.clone());
                        state.working.push(round);
                        state.round_calls = calls.clone();
                        state.round_ok = true;
                        state.phase = Phase::Tools(calls.into_iter());
                    }
                    Phase::Tools(mut calls) => {
                        let Some(call) = calls.next() else {
                            state.phase = Phase::AfterTools;
                            continue;
                        };
                        let outcome = execute_tool(state.tools, &call).await;
                        state.round_ok &= outcome.ok;
                        let item = ChatItem::tool(call.id.clone(), outcome.content);
                        state.items.push(item.clone());
                        state.working.push(item);
                        state.phase = Phase::Tools(calls);
                    }
                    Phase::AfterTools => {
                        // The step is complete only now — its `ok` needed
                        // every call's outcome, its text the round's.
                        state.step_results.push(StepResult {
                            tool_calls: std::mem::take(&mut state.round_calls),
                            ok: state.round_ok,
                            text: std::mem::take(&mut state.round_text),
                        });
                        let stop = state
                            .stop_when
                            .iter()
                            .any(|condition| condition(&state.step_results));
                        if stop {
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
        steps: state.step_results.len(),
    }
}

/// The collecting twin of [`stream_text`]: runs the same loop and returns
/// only the final result — the text and every message the loop produced.
pub async fn generate_text(params: StreamTextParams<'_>) -> Result<StreamResult, LlmError> {
    let mut events = stream_text(params);
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
    use crate::agent::tool::{Tool, ToolEntry};

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

        async fn call(&self, args: AddArgs) -> Result<i64, Infallible> {
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

        async fn call(&self, _args: AddArgs) -> Result<i64, String> {
            Err("boom".to_string())
        }
    }

    fn typed_tools() -> Vec<ToolEntry> {
        vec![ToolEntry::new(Add), ToolEntry::new(Failing)]
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
        let mut events = stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .build(),
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
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .build(),
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
    async fn caller_condition_over_speech_and_success_closes_without_another_round() {
        let llm = ScriptedLlm::new(vec![vec![
            delta("好的"),
            call("add", r#"{"x":1,"y":2}"#),
            done(),
        ]]);
        let seen = drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .stop_when(vec![Box::new(|steps: &[StepResult]| {
                    steps.last().is_some_and(|step| {
                        !step.text.is_empty() && step.ok && !step.tool_calls.is_empty()
                    })
                })])
                .build(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = seen.into_iter().last() else {
            panic!("expected a done event");
        };
        // The condition owns the stop: the round spoke, its call succeeded —
        // asking the model again would only re-confirm what was just said.
        assert_eq!(result.text, "好的");
        assert_eq!(result.steps, 1);
        assert_eq!(
            result.messages,
            vec![
                ChatItem::assistant_text_tool_calls(
                    "好的",
                    vec![tool_call("add", r#"{"x":1,"y":2}"#)]
                ),
                ChatItem::tool("call_1", "3"),
            ]
        );
    }

    #[tokio::test]
    async fn caller_condition_does_not_stop_on_a_failed_step() {
        let llm = ScriptedLlm::new(vec![
            vec![delta("好的"), call("fail", r#"{"x":1,"y":2}"#), done()],
            vec![delta("出错了"), done()],
        ]);
        let result = drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .stop_when(vec![Box::new(|steps: &[StepResult]| {
                    steps.last().is_some_and(|step| {
                        !step.text.is_empty() && step.ok && !step.tool_calls.is_empty()
                    })
                })])
                .build(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        // The failed call makes the step not ok, so the condition keeps the
        // loop running for a correction round.
        assert_eq!(result.text, "好的出错了");
        assert_eq!(result.steps, 2);
    }

    #[tokio::test]
    async fn failing_tool_asks_the_model_again() {
        let llm = ScriptedLlm::new(vec![
            vec![call("fail", r#"{"x":1,"y":2}"#), done()],
            vec![delta("抱歉"), done()],
        ]);
        let result = drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .build(),
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
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .build(),
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
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .tools(&typed_tools())
                .build(),
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
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .tools(&typed_tools())
                .build(),
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
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .cancel(cancel)
                .build(),
        ))
        .await;
        assert!(seen.is_empty(), "{seen:?}");
    }

    #[tokio::test]
    async fn cancelled_mid_stream_ends_silently() {
        let cancel = CancellationToken::new();
        let mut events = stream_text(
            StreamTextParams::builder(&HangingLlm)
                .messages(vec![ChatItem::user("hi")])
                .cancel(cancel.clone())
                .build(),
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
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .build(),
        ))
        .await;
        match seen.into_iter().last() {
            Some(Err(LlmError::Failed(message))) => assert_eq!(message, "provider down"),
            other => panic!("expected the provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn generate_text_collects_the_stream() {
        let llm = ScriptedLlm::new(vec![
            vec![delta("算一下"), call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        let result = generate_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(result.text, "算一下等于 3");
        assert_eq!(result.steps, 2);
        assert_eq!(result.messages.len(), 3);
    }

    #[tokio::test]
    async fn stop_when_ends_the_turn_after_the_tool_round() {
        let llm = ScriptedLlm::new(vec![
            vec![delta("算一下"), call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        let result = drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .stop_when(vec![Box::new(|_steps: &[StepResult]| true)])
                .build(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        // The second round never happens: the condition owns the stop.
        assert_eq!(result.text, "算一下");
        assert_eq!(result.steps, 1);
        assert_eq!(
            result.messages,
            vec![
                ChatItem::assistant_text_tool_calls(
                    "算一下",
                    vec![tool_call("add", r#"{"x":1,"y":2}"#)]
                ),
                ChatItem::tool("call_1", "3"),
            ]
        );
    }

    #[tokio::test]
    async fn stop_when_never_firing_keeps_looping() {
        let llm = ScriptedLlm::new(vec![
            vec![delta("算一下"), call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        let result = drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .stop_when(vec![Box::new(|_steps: &[StepResult]| false)])
                .build(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        assert_eq!(result.text, "算一下等于 3");
        assert_eq!(result.steps, 2);
    }

    #[tokio::test]
    async fn stop_conditions_see_the_completed_steps() {
        let seen: Arc<Mutex<Vec<Vec<usize>>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = seen.clone();
        let llm = ScriptedLlm::new(vec![
            vec![call("add", r#"{"x":1,"y":2}"#), done()],
            vec![call("add", r#"{"x":3,"y":4}"#), done()],
            vec![delta("好"), done()],
        ]);
        let result = drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .tools(&typed_tools())
                .stop_when(vec![Box::new(move |steps: &[StepResult]| {
                    recorder
                        .lock()
                        .unwrap()
                        .push(steps.iter().map(|step| step.tool_calls.len()).collect());
                    steps
                        .last()
                        .is_some_and(|step| step.tool_calls.iter().all(|call| call.name == "add"))
                        && steps.len() >= 2
                })])
                .build(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        // First check after round 1, second after round 2 — which fired and
        // kept the third round from ever running.
        assert_eq!(*seen.lock().unwrap(), vec![vec![1], vec![1, 1]]);
        assert_eq!(result.steps, 2);
    }

    #[tokio::test]
    async fn any_matching_condition_stops() {
        let llm = ScriptedLlm::new(vec![
            vec![delta("查"), call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        let result = drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .stop_when(vec![
                    Box::new(|_steps: &[StepResult]| false),
                    Box::new(|steps: &[StepResult]| !steps.is_empty()),
                ])
                .build(),
        ))
        .await;
        let Some(Ok(StreamEvent::Done(result))) = result.into_iter().last() else {
            panic!("expected a done event");
        };
        assert_eq!(result.steps, 1);
    }

    #[tokio::test]
    async fn text_only_round_never_consults_conditions() {
        let llm = ScriptedLlm::new(vec![vec![delta("你"), delta("好"), done()]]);
        let mut events = stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .stop_when(vec![Box::new(|_steps: &[StepResult]| true)])
                .build(),
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
                assert_eq!(result.steps, 1);
                assert_eq!(result.messages, vec![ChatItem::assistant("你好")]);
            }
            other => panic!("expected done, got {other:?}"),
        }
        assert!(events.next().await.is_none());
    }
}
