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
use tracing::Instrument;

use super::llm::{
    ChatItem, Llm, LlmError, LlmEvent, LlmEvents, ReasoningEffort, ToolCall, ToolChoice, ToolSpec,
};
use super::tool::ToolSet;

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
    tools: &'a ToolSet,
    specs: Vec<ToolSpec>,
    tool_choice: Option<ToolChoice<'a>>,
    reasoning: Option<ReasoningEffort>,
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

/// The single argument of [`stream_text`] and [`generate_text`] — one
/// generation call's parameters. `llm` goes positionally into
/// [`StreamTextParams::builder`]; `tools` is the call's [`ToolSet`] — an
/// empty one advertises no tools. The rest is optional — `messages` (empty),
/// `tool_choice`
/// (none: the model decides — the wire field is omitted; `Auto` sends `auto`
/// explicitly, `Required` forces some tool call, `Named(&str)` exactly that
/// tool; it rides every round's request and is ignored when no tools are
/// advertised), `reasoning` (unset: the provider's configured
/// `reasoning_effort`, if any, applies — a set value overrides it for this
/// call's every round, with [`ReasoningEffort::None`] sending `none`
/// explicitly to switch reasoning off), `stop_when` (empty:
/// never stop early) and `cancel` (a fresh token).
#[derive(bon::Builder)]
pub struct StreamTextParams<'a> {
    #[builder(start_fn)]
    pub llm: &'a dyn Llm,
    #[builder(default)]
    pub messages: Vec<ChatItem>,
    pub tools: &'a ToolSet,
    pub tool_choice: Option<ToolChoice<'a>>,
    pub reasoning: Option<ReasoningEffort>,
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
/// any condition returning `true` ends the stream the same way. An empty
/// array never stops early; the step cap stays a hard failure guard on top.
pub fn stream_text(params: StreamTextParams<'_>) -> StreamEvents<'_> {
    let StreamTextParams {
        llm,
        messages,
        tools,
        tool_choice,
        reasoning,
        stop_when,
        cancel,
    } = params;
    // The registry is fixed for the whole call — the spec slice is collected
    // once here, not per round.
    let specs = tools.specs();
    let state = Loop {
        llm,
        tools,
        specs,
        tool_choice,
        reasoning,
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
                        let chat = state.llm.chat(
                            state.working.clone(),
                            state.specs.clone(),
                            state.tool_choice,
                            state.reasoning,
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
                            Some(Ok(LlmEvent::Response(_))) => {
                                // End-of-stream response metadata — usage
                                // and response identity, lifted off the wire
                                // for the round's `chat` span (telemetry.md).
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
                        // One GenAI-conventions span per execution. `otel.name`
                        // carries the composed span name
                        // (`execute_tool {tool.name}`) — `tracing` metadata
                        // names are 'static, so the composed name rides the
                        // field the future tracing-opentelemetry bridge reads
                        // as the exported name; degraded calls record
                        // `error.type` + the error status when the outcome
                        // lands.
                        let span = tracing::info_span!(
                            "execute_tool",
                            otel.name = %format!("execute_tool {}", call.name),
                            otel.kind = "internal",
                            gen_ai.operation.name = "execute_tool",
                            gen_ai.tool.name = %call.name,
                            gen_ai.tool.type = "function",
                            error.type = tracing::field::Empty,
                            otel.status_code = tracing::field::Empty,
                        );
                        let outcome = state.tools.execute(&call).instrument(span.clone()).await;
                        if let Some(error_type) = outcome.error_type {
                            span.record("error.type", error_type);
                            span.record("otel.status_code", "error");
                        }
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
    use std::collections::{BTreeMap, VecDeque};
    use std::convert::Infallible;
    use std::fmt;
    use std::sync::Arc;
    use std::sync::Mutex;

    use serde::Deserialize;
    use tracing::Id;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Record};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;

    use crate::agent::tool::{Tool, ToolSet};
    use crate::agent::{LlmMeta, LlmResponse, LlmUsage, ToolSpec};

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
        fn meta(&self) -> LlmMeta {
            crate::agent::llm::stub_meta()
        }

        fn chat(
            &self,
            _history: Vec<ChatItem>,
            _tools: Vec<ToolSpec>,
            _tool_choice: Option<ToolChoice<'_>>,
            _reasoning: Option<ReasoningEffort>,
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
        fn meta(&self) -> LlmMeta {
            crate::agent::llm::stub_meta()
        }

        fn chat(
            &self,
            _history: Vec<ChatItem>,
            _tools: Vec<ToolSpec>,
            _tool_choice: Option<ToolChoice<'_>>,
            _reasoning: Option<ReasoningEffort>,
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

    fn typed_tools() -> ToolSet {
        ToolSet::new().tool(Add).tool(Failing)
    }

    /// The [`ScriptedLlm`] behavior plus a record of the `tool_choice` and
    /// `reasoning` every round received — the loop-contract probe.
    struct RecordingLlm {
        choices: Mutex<Vec<Option<RecordedChoice>>>,
        efforts: Mutex<Vec<Option<ReasoningEffort>>>,
        rounds: Mutex<VecDeque<Vec<Result<LlmEvent, LlmError>>>>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum RecordedChoice {
        Auto,
        Required,
        Named(String),
    }

    impl RecordingLlm {
        fn new(rounds: Vec<Vec<Result<LlmEvent, LlmError>>>) -> Self {
            Self {
                choices: Mutex::new(Vec::new()),
                efforts: Mutex::new(Vec::new()),
                rounds: Mutex::new(rounds.into()),
            }
        }

        fn choices(&self) -> Vec<Option<RecordedChoice>> {
            self.choices.lock().unwrap().clone()
        }

        fn efforts(&self) -> Vec<Option<ReasoningEffort>> {
            self.efforts.lock().unwrap().clone()
        }
    }

    impl Llm for RecordingLlm {
        fn meta(&self) -> LlmMeta {
            crate::agent::llm::stub_meta()
        }

        fn chat(
            &self,
            _history: Vec<ChatItem>,
            _tools: Vec<ToolSpec>,
            tool_choice: Option<ToolChoice<'_>>,
            reasoning: Option<ReasoningEffort>,
            _cancel: CancellationToken,
        ) -> LlmEvents<'_> {
            self.choices
                .lock()
                .unwrap()
                .push(tool_choice.map(|choice| match choice {
                    ToolChoice::Auto => RecordedChoice::Auto,
                    ToolChoice::Required => RecordedChoice::Required,
                    ToolChoice::Named(name) => RecordedChoice::Named(name.into()),
                }));
            self.efforts.lock().unwrap().push(reasoning);
            let events =
                self.rounds.lock().unwrap().pop_front().unwrap_or_else(|| {
                    vec![Err(LlmError::Failed("scripted llm exhausted".into()))]
                });
            Box::pin(futures_util::stream::iter(events))
        }
    }

    async fn drain(events: &mut StreamEvents<'_>) -> Vec<Result<StreamEvent, LlmError>> {
        let mut seen = Vec::new();
        while let Some(item) = events.next().await {
            seen.push(item);
        }
        seen
    }

    /// Captures span names and fields through a tracing-subscriber layer —
    /// the assertion surface for the conventions spans, no otel SDK.
    #[derive(Clone)]
    struct SpanCapture(Arc<Mutex<Vec<CapturedSpan>>>);

    #[derive(Debug, Clone)]
    struct CapturedSpan {
        id: u64,
        name: String,
        fields: BTreeMap<String, String>,
    }

    impl SpanCapture {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Vec::new())))
        }

        fn spans(&self, name: &str) -> Vec<CapturedSpan> {
            self.0
                .lock()
                .unwrap()
                .iter()
                .filter(|span| span.name == name)
                .cloned()
                .collect()
        }
    }

    /// Installs a bare global `Registry` once per test process. The callsite
    /// interest cache is process-global and a callsite's first registration
    /// is decided by whichever thread reaches it first: a thread without any
    /// subscriber answers `Interest::never`, which poisons the callsite for
    /// every later span — scoped dispatchers never trigger a rebuild. The
    /// bare registry answers `always`, discards everything, and leaves each
    /// test's scoped capture in charge.
    fn install_global_test_registry() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::subscriber::set_global_default(tracing_subscriber::registry())
                .expect("global test registry");
        });
    }

    struct FieldWriter<'a>(&'a mut BTreeMap<String, String>);

    impl Visit for FieldWriter<'_> {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    impl<S> Layer<S> for SpanCapture
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
            let mut fields = BTreeMap::new();
            attrs.record(&mut FieldWriter(&mut fields));
            self.0.lock().unwrap().push(CapturedSpan {
                id: id.into_u64(),
                name: attrs.metadata().name().to_string(),
                fields,
            });
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
            let mut spans = self.0.lock().unwrap();
            if let Some(span) = spans.iter_mut().find(|span| span.id == id.into_u64()) {
                values.record(&mut FieldWriter(&mut span.fields));
            }
        }
    }

    #[tokio::test]
    async fn text_only_turn_streams_deltas_then_done() {
        let llm = ScriptedLlm::new(vec![vec![delta("你"), delta("好"), done()]]);
        let tools = ToolSet::new();
        let mut events = stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .tools(&tools)
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
    async fn response_event_is_inert_to_the_loop() {
        let llm = ScriptedLlm::new(vec![vec![
            delta("你"),
            Ok(LlmEvent::Response(LlmResponse {
                id: Some("chatcmpl-1".into()),
                model: Some("m".into()),
                finish_reasons: vec!["stop".into()],
                usage: Some(LlmUsage {
                    input_tokens: 11,
                    output_tokens: 7,
                }),
            })),
            delta("好"),
            done(),
        ]]);
        let tools = ToolSet::new();
        let mut events = stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .tools(&tools)
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
    async fn tool_execution_opens_a_conventions_span() {
        let llm = ScriptedLlm::new(vec![
            vec![delta("算一下"), call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        install_global_test_registry();
        let capture = SpanCapture::new();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));
        let seen = drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .build(),
        ))
        .await;
        drop(_guard);
        assert!(
            seen.iter()
                .any(|item| matches!(item, Ok(StreamEvent::Done(_))))
        );
        let spans = capture.spans("execute_tool");
        assert_eq!(spans.len(), 1);
        let fields = &spans[0].fields;
        assert_eq!(
            fields.get("otel.name").map(String::as_str),
            Some("execute_tool add")
        );
        assert_eq!(
            fields.get("otel.kind").map(String::as_str),
            Some("internal")
        );
        assert_eq!(
            fields.get("gen_ai.operation.name").map(String::as_str),
            Some("execute_tool")
        );
        assert_eq!(
            fields.get("gen_ai.tool.name").map(String::as_str),
            Some("add")
        );
        assert_eq!(
            fields.get("gen_ai.tool.type").map(String::as_str),
            Some("function")
        );
        assert!(!fields.contains_key("error.type"));
        assert!(!fields.contains_key("otel.status_code"));
    }

    #[tokio::test]
    async fn unknown_tool_span_records_the_error_class() {
        let llm = ScriptedLlm::new(vec![
            vec![call("nope", "{}"), done()],
            vec![delta("好"), done()],
        ]);
        install_global_test_registry();
        let capture = SpanCapture::new();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));
        drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .tools(&typed_tools())
                .build(),
        ))
        .await;
        drop(_guard);
        let spans = capture.spans("execute_tool");
        assert_eq!(spans.len(), 1);
        let fields = &spans[0].fields;
        assert_eq!(
            fields.get("gen_ai.tool.name").map(String::as_str),
            Some("nope")
        );
        assert_eq!(
            fields.get("error.type").map(String::as_str),
            Some("unknown_tool")
        );
        assert_eq!(
            fields.get("otel.status_code").map(String::as_str),
            Some("error")
        );
    }

    #[tokio::test]
    async fn failing_tool_span_records_the_error_class() {
        let llm = ScriptedLlm::new(vec![
            vec![call("fail", r#"{"x":1,"y":2}"#), done()],
            vec![delta("抱歉"), done()],
        ]);
        install_global_test_registry();
        let capture = SpanCapture::new();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));
        drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .build(),
        ))
        .await;
        drop(_guard);
        let spans = capture.spans("execute_tool");
        assert_eq!(spans.len(), 1);
        let fields = &spans[0].fields;
        assert_eq!(
            fields.get("error.type").map(String::as_str),
            Some("tool_error")
        );
    }

    #[tokio::test]
    async fn malformed_arguments_span_records_the_error_class() {
        let llm = ScriptedLlm::new(vec![
            vec![call("add", r#"{"x":1}"#), done()],
            vec![delta("重来"), done()],
        ]);
        install_global_test_registry();
        let capture = SpanCapture::new();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));
        drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .build(),
        ))
        .await;
        drop(_guard);
        let spans = capture.spans("execute_tool");
        assert_eq!(spans.len(), 1);
        let fields = &spans[0].fields;
        assert_eq!(
            fields.get("error.type").map(String::as_str),
            Some("invalid_arguments")
        );
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
                .tools(&ToolSet::new())
                .cancel(cancel)
                .build(),
        ))
        .await;
        assert!(seen.is_empty(), "{seen:?}");
    }

    #[tokio::test]
    async fn cancelled_mid_stream_ends_silently() {
        let cancel = CancellationToken::new();
        let tools = ToolSet::new();
        let mut events = stream_text(
            StreamTextParams::builder(&HangingLlm)
                .messages(vec![ChatItem::user("hi")])
                .tools(&tools)
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
                .tools(&ToolSet::new())
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
    async fn tool_choice_rides_every_round() {
        let llm = RecordingLlm::new(vec![
            vec![call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        let result = generate_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .tool_choice(ToolChoice::Named("add"))
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(result.steps, 2);
        assert_eq!(
            llm.choices(),
            vec![
                Some(RecordedChoice::Named("add".into())),
                Some(RecordedChoice::Named("add".into()))
            ]
        );
    }

    #[tokio::test]
    async fn tool_choice_defaults_to_none_every_round() {
        let llm = RecordingLlm::new(vec![
            vec![call("add", r#"{"x":1,"y":2}"#), done()],
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
        assert_eq!(result.steps, 2);
        assert_eq!(llm.choices(), vec![None, None]);
    }

    #[tokio::test]
    async fn reasoning_rides_every_round() {
        let llm = RecordingLlm::new(vec![
            vec![call("add", r#"{"x":1,"y":2}"#), done()],
            vec![delta("等于 3"), done()],
        ]);
        let result = generate_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("1+2?")])
                .tools(&typed_tools())
                .tool_choice(ToolChoice::Named("add"))
                .reasoning(ReasoningEffort::Low)
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(result.steps, 2);
        assert_eq!(
            llm.efforts(),
            vec![Some(ReasoningEffort::Low), Some(ReasoningEffort::Low)]
        );
    }

    #[tokio::test]
    async fn reasoning_defaults_to_none_every_round() {
        let llm = RecordingLlm::new(vec![vec![delta("好"), done()]]);
        drain(&mut stream_text(
            StreamTextParams::builder(&llm)
                .tools(&ToolSet::new())
                .messages(vec![ChatItem::user("hi")])
                .build(),
        ))
        .await;
        assert_eq!(llm.efforts(), vec![None]);
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
        let tools = ToolSet::new();
        let mut events = stream_text(
            StreamTextParams::builder(&llm)
                .messages(vec![ChatItem::user("hi")])
                .tools(&tools)
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
