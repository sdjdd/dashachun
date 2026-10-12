use super::*;
use crate::agent::InMemEntryMemory;
use crate::agent::InMemMemory;
use crate::agent::collect;
use crate::agent::memory::HISTORY_LIMIT;
use crate::agent::tool::MemoryAdd;
use crate::agent::{
    AgentInput, Asr, AsrError, AsrEvent, AsrEvents, AudioStream, Capture, ChatItem, EntryMemory,
    Llm, LlmEvent, LlmEvents, ReasoningEffort, ReplyCapture, Subtitle, TextStream, ToolCall, Tts,
    TtsError, TtsEvent, TtsEvents, UtteranceCapture,
};
use crate::provider::asr::StubAsr;
use crate::provider::llm::StubLlm;
use crate::provider::tts::StubTts;
use crate::vad::{Vad, VadError, VadEvent, VadFactory};
use futures_util::StreamExt;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

struct ScriptedTts;

impl Tts for ScriptedTts {
    fn synthesize(&self, mut text: TextStream, _cancel: CancellationToken) -> TtsEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<TtsEvent, TtsError>>();
        tokio::spawn(async move {
            let mut acc = String::new();
            while let Some(chunk) = text.next().await {
                acc.push_str(&chunk);
            }
            let _ = tx.send(Ok(TtsEvent::SentenceStart { text: acc.clone() }));
            let _ = tx.send(Ok(TtsEvent::Subtitle(Subtitle {
                text: acc,
                start_ms: 0,
                end_ms: 120,
            })));
            let _ = tx.send(Ok(TtsEvent::Audio(vec![0.25; 960])));
            let _ = tx.send(Ok(TtsEvent::Done));
        });
        crate::util::recv_unbounded_stream(rx)
    }
}

struct ScriptedLlm {
    reply: String,
    calls: Arc<Mutex<Vec<Vec<ChatItem>>>>,
}
impl Llm for ScriptedLlm {
    fn chat(
        &self,
        history: Vec<ChatItem>,
        _tools: Vec<crate::agent::ToolSpec>,
        _tool_choice: Option<crate::agent::ToolChoice>,
        _reasoning: Option<crate::agent::ReasoningEffort>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        self.calls.lock().unwrap().push(history);
        let text = self.reply.clone();
        Box::pin(futures_util::stream::iter([
            Ok(LlmEvent::Delta { text }),
            Ok(LlmEvent::Done),
        ]))
    }
}

/// Records the `reasoning` every call received — the voice turn's
/// reasoning-off pin is a latency guarantee, this is its guard.
#[derive(Default)]
struct EffortProbe {
    efforts: Mutex<Vec<Option<ReasoningEffort>>>,
}

impl Llm for EffortProbe {
    fn chat(
        &self,
        _history: Vec<ChatItem>,
        _tools: Vec<crate::agent::ToolSpec>,
        _tool_choice: Option<crate::agent::ToolChoice>,
        reasoning: Option<crate::agent::ReasoningEffort>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        self.efforts.lock().unwrap().push(reasoning);
        Box::pin(futures_util::stream::iter([
            Ok(LlmEvent::Delta { text: "hi".into() }),
            Ok(LlmEvent::Done),
        ]))
    }
}

struct ToolCallingLlm {
    calls: Arc<Mutex<Vec<Vec<ChatItem>>>>,
    tool_calls: Vec<ToolCall>,
    reply: String,
}

impl Llm for ToolCallingLlm {
    fn chat(
        &self,
        history: Vec<ChatItem>,
        _tools: Vec<crate::agent::ToolSpec>,
        _tool_choice: Option<crate::agent::ToolChoice>,
        _reasoning: Option<crate::agent::ReasoningEffort>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        let round = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(history);
            calls.len()
        };
        if round == 1 {
            let mut events: Vec<Result<LlmEvent, crate::agent::LlmError>> = self
                .tool_calls
                .iter()
                .cloned()
                .map(|call| Ok(LlmEvent::ToolCall(call)))
                .collect();
            events.push(Ok(LlmEvent::Done));
            Box::pin(futures_util::stream::iter(events))
        } else {
            let text = self.reply.clone();
            Box::pin(futures_util::stream::iter([
                Ok(LlmEvent::Delta { text }),
                Ok(LlmEvent::Done),
            ]))
        }
    }
}

/// Streams a spoken preamble before the tool call in round 1, then
/// replies in round 2.
struct PreambleToolLlm {
    calls: Arc<Mutex<Vec<Vec<ChatItem>>>>,
    preamble: String,
    reply: String,
}

impl Llm for PreambleToolLlm {
    fn chat(
        &self,
        history: Vec<ChatItem>,
        _tools: Vec<crate::agent::ToolSpec>,
        _tool_choice: Option<crate::agent::ToolChoice>,
        _reasoning: Option<crate::agent::ReasoningEffort>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        let round = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(history);
            calls.len()
        };
        if round == 1 {
            let events: Vec<Result<LlmEvent, crate::agent::LlmError>> = vec![
                Ok(LlmEvent::Delta {
                    text: self.preamble.clone(),
                }),
                Ok(LlmEvent::ToolCall(ToolCall {
                    id: "call_x".into(),
                    name: "nonexistent".into(),
                    arguments: "{}".into(),
                })),
                Ok(LlmEvent::Done),
            ];
            Box::pin(futures_util::stream::iter(events))
        } else {
            let text = self.reply.clone();
            Box::pin(futures_util::stream::iter([
                Ok(LlmEvent::Delta { text }),
                Ok(LlmEvent::Done),
            ]))
        }
    }
}

struct ScriptedVad {
    events: std::collections::VecDeque<VadEvent>,
}
impl Vad for ScriptedVad {
    fn push(&mut self, samples: &[f32]) -> Vec<VadEvent> {
        if samples.iter().any(|s| *s != 0.0) {
            let mut events = Vec::new();
            while let Some(event) = self.events.pop_front() {
                events.push(event);
                if matches!(events.last(), Some(VadEvent::SpeechEnd { .. })) {
                    break;
                }
            }
            events
        } else {
            Vec::new()
        }
    }

    fn flush(&mut self) -> Vec<VadEvent> {
        Vec::new()
    }
}

struct ScriptedVadFactory {
    events: Vec<VadEvent>,
}

impl VadFactory for ScriptedVadFactory {
    fn build(&self, _sample_rate: u32) -> Result<Box<dyn Vad>, VadError> {
        Ok(Box::new(ScriptedVad {
            events: self.events.iter().cloned().collect(),
        }))
    }
}

impl ScriptedVadFactory {
    /// One utterance per entry: SpeechStart, one Speech per sample count,
    /// then SpeechEnd, with `at_ms` advancing 100 ms per event (no test
    /// asserts on it).
    fn utterances(utterances: &[&[usize]]) -> Self {
        let mut events = Vec::new();
        let mut at_ms = 0;
        for samples in utterances {
            events.push(VadEvent::SpeechStart { at_ms });
            at_ms += 100;
            for count in *samples {
                events.push(VadEvent::Speech {
                    samples: vec![0.5; *count],
                });
            }
            events.push(VadEvent::SpeechEnd { at_ms });
            at_ms += 100;
        }
        Self { events }
    }
}

fn input_channel() -> (mpsc::Sender<AgentInput>, AgentInputStream) {
    let (tx, rx) = mpsc::channel::<AgentInput>(8);
    let stream: AgentInputStream = crate::util::recv_stream(rx);
    (tx, stream)
}

fn session() -> AgentSession {
    AgentSession {
        id: "test".into(),
        sample_rate: 16000,
        channels: 1,
        frame_duration_ms: 60,
    }
}

fn empty_tools() -> Arc<ToolSet> {
    Arc::new(ToolSet::new())
}

const OUTPUT_TIMEOUT: Duration = Duration::from_secs(2);

/// The all-stub agent: every field a quiet default, overridden per test
/// through struct-update syntax (`..stub_agent()`).
fn stub_agent() -> CompositeAgent {
    CompositeAgent {
        asr: Arc::new(StubAsr::new("hello")),
        llm: Arc::new(StubLlm::default()),
        tts: Arc::new(StubTts),
        vad: Arc::new(ScriptedVadFactory { events: Vec::new() }),
        memory: Arc::new(InMemMemory::default()),
        entries: Arc::new(InMemEntryMemory::default()),
        tools: empty_tools(),
        system_prompt: SystemPrompt::default(),
        capture: None,
    }
}

/// Opens the input channel and runs the agent on the shared test session.
fn run(agent: CompositeAgent) -> (mpsc::Sender<AgentInput>, AgentOutputStream) {
    let (tx, input) = input_channel();
    (tx, agent.run(session(), input))
}

/// Drains the output until an item matches, panicking on timeout (or an
/// ended stream) with everything seen so far.
async fn next_matching(
    output: &mut AgentOutputStream,
    pred: impl Fn(&AgentOutput) -> bool,
) -> AgentOutput {
    next_matching_within(output, OUTPUT_TIMEOUT, pred).await
}

async fn next_matching_within(
    output: &mut AgentOutputStream,
    timeout: Duration,
    pred: impl Fn(&AgentOutput) -> bool,
) -> AgentOutput {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut seen = Vec::new();
    loop {
        match tokio::time::timeout_at(deadline, output.next()).await {
            Ok(Some(item)) if pred(&item) => return item,
            Ok(Some(item)) => seen.push(item),
            Ok(None) => panic!("output ended without a match; seen: {seen:?}"),
            Err(_) => panic!("no matching item within {timeout:?}; seen: {seen:?}"),
        }
    }
}

/// Drains the output up to and including the reply's `TtsStop`, returning
/// every item in order.
async fn collect_reply(output: &mut AgentOutputStream) -> Vec<AgentOutput> {
    let deadline = tokio::time::Instant::now() + OUTPUT_TIMEOUT;
    let mut seen = Vec::new();
    loop {
        match tokio::time::timeout_at(deadline, output.next()).await {
            Ok(Some(item)) => {
                let done = matches!(item, AgentOutput::TtsStop);
                seen.push(item);
                if done {
                    return seen;
                }
            }
            Ok(None) => panic!("output ended before tts stop; seen: {seen:?}"),
            Err(_) => panic!("no tts stop within {OUTPUT_TIMEOUT:?}; seen: {seen:?}"),
        }
    }
}

/// Polls `f` until it yields `Some`, panicking after the deadline.
async fn wait_until<T>(what: &str, f: impl Fn() -> Option<T>) -> T {
    let deadline = tokio::time::Instant::now() + OUTPUT_TIMEOUT;
    loop {
        if let Some(value) = f() {
            return value;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} not observed"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn wait_for_calls(calls: &Arc<Mutex<Vec<Vec<ChatItem>>>>, len: usize) -> Vec<Vec<ChatItem>> {
    wait_until("llm calls", || {
        let snapshot = calls.lock().unwrap().clone();
        (snapshot.len() >= len).then_some(snapshot)
    })
    .await
}

async fn wait_for_replies(replies: &StoredReplies, len: usize) -> Vec<FakeReply> {
    wait_until("fake replies", || {
        let snapshot = replies.lock().unwrap().clone();
        (snapshot.len() >= len).then_some(snapshot)
    })
    .await
}

async fn wait_for_utterances(utterances: &StoredUtterances, len: usize) -> Vec<Vec<Vec<f32>>> {
    wait_until("fake utterances", || {
        let snapshot = utterances.lock().unwrap().clone();
        (snapshot.len() >= len).then_some(snapshot)
    })
    .await
}

#[tokio::test]
async fn audio_drives_vad_asr_to_stt() {
    let agent = CompositeAgent {
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let item = next_matching(&mut output, |item| {
        matches!(item, AgentOutput::Stt { is_final: true, .. })
    })
    .await;
    let AgentOutput::Stt {
        text,
        is_final: true,
    } = item
    else {
        panic!("matched item was not a final stt: {item:?}");
    };
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn interrupt_emits_tts_stop() {
    let agent = stub_agent();
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::Interrupt { reason: None })
        .await
        .unwrap();

    next_matching(&mut output, |item| matches!(item, AgentOutput::TtsAbort)).await;
}

/// Holds the synthesis open (never emits `Done`) until cancelled, so a
/// reply is still "speaking" when the next input arrives.
struct HeldTts;

impl Tts for HeldTts {
    fn synthesize(&self, mut text: TextStream, cancel: CancellationToken) -> TtsEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<TtsEvent, TtsError>>();
        tokio::spawn(async move {
            while let Some(chunk) = text.next().await {
                if !chunk.is_empty() {
                    let _ = tx.send(Ok(TtsEvent::Audio(vec![0.25; 960])));
                    break;
                }
            }
            cancel.cancelled().await;
        });
        crate::util::recv_unbounded_stream(rx)
    }
}

#[tokio::test]
async fn new_utterance_final_aborts_in_flight_tts() {
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }),
        tts: Arc::new(HeldTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[], &[]])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    next_matching(&mut output, |item| matches!(item, AgentOutput::TtsStart)).await;

    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    next_matching(&mut output, |item| matches!(item, AgentOutput::TtsAbort)).await;
}

/// Models a provider whose task never exits, even when cancelled: the
/// cancel handshake can only resolve through the `CANCEL_TIMEOUT` abort,
/// so anything the driver does while waiting must happen detached.
struct ImmortalTts;

impl Tts for ImmortalTts {
    fn synthesize(&self, mut text: TextStream, _cancel: CancellationToken) -> TtsEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<TtsEvent, TtsError>>();
        tokio::spawn(async move {
            while let Some(chunk) = text.next().await {
                if !chunk.is_empty() {
                    let _ = tx.send(Ok(TtsEvent::Audio(vec![0.25; 960])));
                }
            }
            futures_util::future::pending::<()>().await;
        });
        crate::util::recv_unbounded_stream(rx)
    }
}

#[tokio::test]
async fn stalled_tts_cancel_does_not_delay_next_reply() {
    let agent = CompositeAgent {
        llm: Arc::new(HangingLlm),
        tts: Arc::new(ImmortalTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[], &[]])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    next_matching(&mut output, |item| matches!(item, AgentOutput::TtsStart)).await;

    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    // The next reply must start without waiting for the stalled tts:
    // the abort path may only block for the short cancel timeout.
    next_matching_within(&mut output, Duration::from_secs(1), |item| {
        matches!(item, AgentOutput::TtsStart)
    })
    .await;
}

#[tokio::test]
async fn session_history_is_capped_to_the_last_messages() {
    // Enough plain user/assistant turns for the cap to bind: the history
    // seen by call t has 2t-1 items (the user item is appended before the
    // LLM runs, its reply after).
    let turns = HISTORY_LIMIT / 2 + 5;
    let calls = Arc::new(Mutex::new(Vec::new()));
    let silent: &[usize] = &[];
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: calls.clone(),
        }),
        tts: Arc::new(ScriptedTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&vec![silent; turns])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    for turn in 1..=turns {
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
        wait_for_calls(&calls, turn).await;
        // Wait for the turn to complete so its assistant item is in the
        // memory before the next turn's audio can barge in.
        next_matching(&mut output, |item| matches!(item, AgentOutput::TtsStop)).await;
    }

    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.len(), turns);
    for call in &calls {
        assert!(
            call.len() <= HISTORY_LIMIT,
            "history grew past the cap: {}",
            call.len()
        );
    }
    assert_eq!(calls[calls.len() - 2].len(), HISTORY_LIMIT);
    assert_eq!(calls[calls.len() - 1].len(), HISTORY_LIMIT);
    assert!(matches!(
        calls[calls.len() - 1].first(),
        Some(ChatItem::Assistant { .. })
    ));
}

/// Emits `SpeechStart` + `Speech` on every push and `SpeechEnd` on flush,
/// so one utterance spans any number of audio chunks.
struct StreamVad {
    started: bool,
}

impl Vad for StreamVad {
    fn push(&mut self, samples: &[f32]) -> Vec<VadEvent> {
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(VadEvent::SpeechStart { at_ms: 0 });
        }
        events.push(VadEvent::Speech {
            samples: samples.to_vec(),
        });
        events
    }

    fn flush(&mut self) -> Vec<VadEvent> {
        self.started = false;
        vec![VadEvent::SpeechEnd { at_ms: 0 }]
    }
}

struct StreamVadFactory;

impl VadFactory for StreamVadFactory {
    fn build(&self, _sample_rate: u32) -> Result<Box<dyn Vad>, VadError> {
        Ok(Box::new(StreamVad { started: false }))
    }
}

/// Consumes audio slower than the driver produces it and records the
/// chunks it actually received.
struct SlowAsr {
    delay: Duration,
    chunks: Arc<Mutex<Vec<usize>>>,
}

impl Asr for SlowAsr {
    fn transcribe(&self, audio: AudioStream, _cancel: CancellationToken) -> AsrEvents<'_> {
        let delay = self.delay;
        let chunks = self.chunks.clone();
        let (tx, rx) = mpsc::unbounded_channel::<Result<AsrEvent, AsrError>>();
        tokio::spawn(async move {
            let mut audio = audio;
            while let Some(samples) = audio.next().await {
                tokio::time::sleep(delay).await;
                chunks.lock().unwrap().push(samples.len());
                let _ = tx.send(Ok(AsrEvent::Partial {
                    text: String::new(),
                }));
            }
            let _ = tx.send(Ok(AsrEvent::Final {
                text: "done".to_string(),
            }));
        });
        crate::util::recv_unbounded_stream(rx)
    }
}

/// Splits the reply into one delta per char.
struct ChunkedLlm {
    reply: String,
}

impl Llm for ChunkedLlm {
    fn chat(
        &self,
        _history: Vec<ChatItem>,
        _tools: Vec<crate::agent::ToolSpec>,
        _tool_choice: Option<crate::agent::ToolChoice>,
        _reasoning: Option<crate::agent::ReasoningEffort>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        let events = self
            .reply
            .chars()
            .map(|c| {
                Ok(LlmEvent::Delta {
                    text: c.to_string(),
                })
            })
            .chain(std::iter::once(Ok(LlmEvent::Done)))
            .collect::<Vec<_>>();
        Box::pin(futures_util::stream::iter(events))
    }
}

/// Consumes text slower than the driver produces it.
struct SlowTts {
    delay: Duration,
}

impl Tts for SlowTts {
    fn synthesize(&self, mut text: TextStream, _cancel: CancellationToken) -> TtsEvents<'_> {
        let delay = self.delay;
        let (tx, rx) = mpsc::unbounded_channel::<Result<TtsEvent, TtsError>>();
        tokio::spawn(async move {
            let mut acc = String::new();
            while let Some(chunk) = text.next().await {
                tokio::time::sleep(delay).await;
                acc.push_str(&chunk);
            }
            let _ = tx.send(Ok(TtsEvent::SentenceStart { text: acc }));
            let _ = tx.send(Ok(TtsEvent::Done));
        });
        crate::util::recv_unbounded_stream(rx)
    }
}

#[tokio::test]
async fn slow_tts_receives_every_delta() {
    let reply = "字".repeat(200);
    let agent = CompositeAgent {
        llm: Arc::new(ChunkedLlm {
            reply: reply.clone(),
        }),
        tts: Arc::new(SlowTts {
            delay: Duration::from_millis(1),
        }),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    // 200 one-char deltas through a 1 ms-per-chunk consumer: the
    // sentence must arrive whole, never truncated by backpressure.
    let item = next_matching_within(&mut output, Duration::from_secs(5), |item| {
        matches!(item, AgentOutput::TtsSentence { .. })
    })
    .await;
    let AgentOutput::TtsSentence { text } = item else {
        panic!("matched item was not a tts sentence: {item:?}");
    };
    assert_eq!(text, reply);
}

#[tokio::test]
async fn slow_asr_receives_every_audio_chunk() {
    let chunks = Arc::new(Mutex::new(Vec::new()));
    let agent = CompositeAgent {
        asr: Arc::new(SlowAsr {
            delay: Duration::from_millis(1),
            chunks: chunks.clone(),
        }),
        vad: Arc::new(StreamVadFactory),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    for _ in 0..80 {
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
    }
    tx.send(AgentInput::ListenStop).await.unwrap();

    next_matching_within(&mut output, Duration::from_secs(5), |item| {
        matches!(item, AgentOutput::Stt { is_final: true, .. })
    })
    .await;
    assert_eq!(chunks.lock().unwrap().len(), 80);
}

#[tokio::test]
async fn llm_receives_history_and_reply_accumulates() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: calls.clone(),
        }),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[], &[]])),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let first = wait_for_calls(&calls, 1).await;
    assert_eq!(first[0], vec![ChatItem::user("hello")]);

    tokio::time::sleep(Duration::from_millis(50)).await;

    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let second = wait_for_calls(&calls, 2).await;
    assert_eq!(
        second[1],
        vec![
            ChatItem::user("hello"),
            ChatItem::assistant("hi"),
            ChatItem::user("hello"),
        ]
    );
}

#[tokio::test]
async fn the_voice_turn_pins_reasoning_off() {
    let probe = Arc::new(EffortProbe::default());
    let agent = CompositeAgent {
        llm: probe.clone(),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[], &[]])),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let efforts = wait_until("the llm call", || {
        let seen = probe.efforts.lock().unwrap().clone();
        (!seen.is_empty()).then_some(seen)
    })
    .await;
    assert_eq!(efforts, vec![Some(ReasoningEffort::None)]);
}

#[tokio::test]
async fn emoji_prefix_emits_emotion_and_strips_tts_text() {
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "🙂你好呀".into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }),
        tts: Arc::new(ScriptedTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let seen = collect_reply(&mut output).await;

    let emotion_pos = seen
        .iter()
        .position(|item| matches!(item, AgentOutput::Emotion { emotion } if emotion == "happy"));
    let tts_start_pos = seen
        .iter()
        .position(|item| matches!(item, AgentOutput::TtsStart));
    assert_eq!(emotion_pos, Some(1), "emotion should follow the final stt");
    assert!(tts_start_pos.is_some(), "tts should start");
    assert!(
        emotion_pos < tts_start_pos,
        "emotion must precede tts start"
    );
    assert!(
        seen.iter()
            .any(|item| matches!(item, AgentOutput::TtsSentence { text } if text == "你好呀")),
        "tts text should have the emoji stripped"
    );
    assert!(
        seen.iter().any(|item| matches!(
            item,
            AgentOutput::TtsSubtitle { subtitle } if subtitle.text == "你好呀"
        )),
        "subtitle should have the emoji stripped"
    );
}

#[tokio::test]
async fn unsupported_emoji_is_stripped_but_emits_no_emotion() {
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "🦄你好".into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }),
        tts: Arc::new(ScriptedTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let seen = collect_reply(&mut output).await;

    assert!(
        !seen
            .iter()
            .any(|item| matches!(item, AgentOutput::Emotion { .. })),
        "unsupported emoji must not emit emotion"
    );
    assert!(
        seen.iter()
            .any(|item| matches!(item, AgentOutput::TtsSentence { text } if text == "你好")),
        "unsupported emoji should still be stripped from tts text"
    );
}

#[tokio::test]
async fn emotion_is_kept_in_history() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "🙂你好".into(),
            calls: calls.clone(),
        }),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[], &[]])),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
    let _ = wait_for_calls(&calls, 1).await;

    tokio::time::sleep(Duration::from_millis(50)).await;
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let second = wait_for_calls(&calls, 2).await;
    assert_eq!(
        second[1],
        vec![
            ChatItem::user("hello"),
            ChatItem::assistant("🙂你好"),
            ChatItem::user("hello"),
        ]
    );
}

#[tokio::test]
async fn system_prompt_is_prepended_to_llm_history() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: calls.clone(),
        }),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        system_prompt: SystemPrompt::new("Be a helpful assistant."),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let first = wait_for_calls(&calls, 1).await;
    let prompt = SystemPrompt::new("Be a helpful assistant.");
    assert_eq!(
        first[0],
        vec![prompt.chat_item().unwrap(), ChatItem::user("hello")]
    );
}

#[tokio::test]
async fn system_prompt_carries_the_user_memory() {
    let entries = InMemEntryMemory::default();
    entries.add("likes tea").await.unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: calls.clone(),
        }),
        entries: Arc::new(entries),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        system_prompt: SystemPrompt::new("Be helpful."),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let first = wait_for_calls(&calls, 1).await;
    assert!(matches!(
        first[0].first(),
        Some(ChatItem::System { content }) if content.contains("[mem_01] likes tea")
    ));
}

/// Turn 1 speaks a preamble and stores a fact through the memory tool;
/// the loop closes the turn right there instead of asking again. Turn 2
/// must see the stored entry in its system prompt.
struct MemoryLlm {
    calls: Arc<Mutex<Vec<Vec<ChatItem>>>>,
}

impl Llm for MemoryLlm {
    fn chat(
        &self,
        history: Vec<ChatItem>,
        _tools: Vec<crate::agent::ToolSpec>,
        _tool_choice: Option<crate::agent::ToolChoice>,
        _reasoning: Option<crate::agent::ReasoningEffort>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        let round = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(history);
            calls.len()
        };
        let events: Vec<Result<LlmEvent, crate::agent::LlmError>> = match round {
            1 => vec![
                Ok(LlmEvent::Delta {
                    text: "好的，我记一下".into(),
                }),
                Ok(LlmEvent::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "memory_add".into(),
                    arguments: "{\"content\":\"likes tea\"}".into(),
                })),
                Ok(LlmEvent::Done),
            ],
            _ => vec![
                Ok(LlmEvent::Delta {
                    text: "记好了".into(),
                }),
                Ok(LlmEvent::Done),
            ],
        };
        Box::pin(futures_util::stream::iter(events))
    }
}

#[tokio::test]
async fn memory_tool_stores_an_entry_the_next_turn_sees() {
    let entries = Arc::new(InMemEntryMemory::default());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let tools = Arc::new(ToolSet::new().tool(MemoryAdd::new(entries.clone())));
    let agent = CompositeAgent {
        llm: Arc::new(MemoryLlm {
            calls: calls.clone(),
        }),
        tools,
        entries,
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[], &[]])),
        system_prompt: SystemPrompt::new("Be helpful."),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
    wait_for_calls(&calls, 1).await;

    // The tool result needs no reply, so turn 1 closed after its single
    // round instead of asking the model again.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        calls.lock().unwrap().len(),
        1,
        "a spoken round with a silent tool must not be followed by another round"
    );

    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let second = wait_for_calls(&calls, 2).await;
    assert!(matches!(
        second[1].first(),
        Some(ChatItem::System { content }) if content.contains("[mem_01] likes tea")
    ));
}

#[tokio::test]
async fn unknown_tool_degrades_gracefully() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let agent = CompositeAgent {
        llm: Arc::new(ToolCallingLlm {
            calls: calls.clone(),
            tool_calls: vec![ToolCall {
                id: "call_x".into(),
                name: "nonexistent".into(),
                arguments: "{}".into(),
            }],
            reply: "ok".into(),
        }),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let rounds = wait_for_calls(&calls, 2).await;
    assert_eq!(
        rounds[1],
        vec![
            ChatItem::user("hello"),
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "nonexistent".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_x", "error: unknown tool `nonexistent`"),
        ]
    );
}

#[tokio::test]
async fn tts_streams_sentence_audio_then_stop() {
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }),
        tts: Arc::new(ScriptedTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let seen = collect_reply(&mut output).await;

    assert!(matches!(seen.first(), Some(AgentOutput::Stt { .. })));
    assert!(
        seen.iter()
            .any(|item| matches!(item, AgentOutput::TtsStart))
    );
    assert!(
        seen.iter()
            .any(|item| matches!(item, AgentOutput::TtsSentence { text } if text == "hi"))
    );
    assert!(seen.iter().any(|item| matches!(
        item,
        AgentOutput::TtsSubtitle { subtitle }
            if subtitle.text == "hi" && subtitle.end_ms == 120
    )));
    assert!(
        seen.iter()
            .any(|item| matches!(item, AgentOutput::Audio(s) if s.len() == 960))
    );
    assert!(matches!(seen.last(), Some(AgentOutput::TtsStop)));
}

struct CancelAwareTts {
    cancelled: Arc<Mutex<bool>>,
}

impl Tts for CancelAwareTts {
    fn synthesize(&self, _text: TextStream, cancel: CancellationToken) -> TtsEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<TtsEvent, TtsError>>();
        let cancelled = self.cancelled.clone();
        tokio::spawn(async move {
            let _ = tx.send(Ok(TtsEvent::SentenceStart { text: "hi".into() }));
            let _ = tx.send(Ok(TtsEvent::Audio(vec![0.25; 960])));
            cancel.cancelled().await;
            *cancelled.lock().unwrap() = true;
            let _ = tx.send(Ok(TtsEvent::Done));
        });
        crate::util::recv_unbounded_stream(rx)
    }
}

struct HangingLlm;

impl Llm for HangingLlm {
    fn chat(
        &self,
        _history: Vec<ChatItem>,
        _tools: Vec<crate::agent::ToolSpec>,
        _tool_choice: Option<crate::agent::ToolChoice>,
        _reasoning: Option<crate::agent::ReasoningEffort>,
        cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        Box::pin(futures_util::stream::unfold(false, move |delivered| {
            let cancel = cancel.clone();
            async move {
                if delivered {
                    cancel.cancelled().await;
                    return None;
                }
                Some((Ok(LlmEvent::Delta { text: "hi".into() }), true))
            }
        }))
    }
}

#[tokio::test]
async fn interrupt_cancels_in_flight_tts_instead_of_aborting() {
    let cancelled = Arc::new(Mutex::new(false));
    let agent = CompositeAgent {
        llm: Arc::new(HangingLlm),
        tts: Arc::new(CancelAwareTts {
            cancelled: cancelled.clone(),
        }),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    next_matching(&mut output, |item| matches!(item, AgentOutput::Audio(_))).await;

    tx.send(AgentInput::Interrupt { reason: None })
        .await
        .unwrap();
    let item = output.next().await;
    assert!(matches!(item, Some(AgentOutput::TtsAbort)), "got {item:?}");
    wait_until("tts cancelled", || cancelled.lock().unwrap().then_some(())).await;
}

/// In-memory capture: records which utterances and replies finished,
/// with their chunks; dropping a capture records a discard.
struct FakeCapture {
    utterances: StoredUtterances,
    replies: StoredReplies,
}

/// How a fake reply capture ended, mirroring the driver's outcomes.
#[derive(Debug, Clone, PartialEq)]
enum FakeReply {
    Finished { chunks: Vec<Vec<f32>> },
    Discarded,
}

impl Capture for FakeCapture {
    fn start_utterance(&self, _session: &AgentSession) -> UtteranceCapture {
        let (recording, mut rx) = UtteranceCapture::channel();
        let utterances = self.utterances.clone();
        tokio::spawn(async move {
            if let Some(chunks) = collect(&mut rx).await {
                utterances.lock().unwrap().push(chunks);
            }
        });
        recording
    }

    fn start_reply(&self, _session: &AgentSession) -> ReplyCapture {
        let (recording, mut rx) = ReplyCapture::channel();
        let replies = self.replies.clone();
        tokio::spawn(async move {
            let Some(chunks) = collect(&mut rx).await else {
                replies.lock().unwrap().push(FakeReply::Discarded);
                return;
            };
            replies.lock().unwrap().push(FakeReply::Finished { chunks });
        });
        recording
    }
}

type StoredUtterances = Arc<Mutex<Vec<Vec<Vec<f32>>>>>;
type StoredReplies = Arc<Mutex<Vec<FakeReply>>>;

fn fake_capture() -> (Arc<FakeCapture>, StoredUtterances, StoredReplies) {
    let utterances = Arc::new(Mutex::new(Vec::new()));
    let replies = Arc::new(Mutex::new(Vec::new()));
    let capture = Arc::new(FakeCapture {
        utterances: utterances.clone(),
        replies: replies.clone(),
    });
    (capture, utterances, replies)
}

#[tokio::test]
async fn finalized_utterance_stores_text_and_audio() {
    let memory = Arc::new(InMemMemory::default());
    let (capture, utterances, _replies) = fake_capture();
    let agent = CompositeAgent {
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[960]])),
        memory: memory.clone(),
        capture: Some(capture),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    next_matching(&mut output, |item| {
        matches!(item, AgentOutput::Stt { is_final: true, .. })
    })
    .await;

    // The memory row is written (and the hook fired) before the capture
    // is finished, so the text is recorded by the time the audio lands.
    assert_eq!(
        wait_for_utterances(&utterances, 1).await,
        vec![vec![vec![0.5; 960]]]
    );
    assert_eq!(memory.stored_utterances(), vec!["hello"]);
}

#[tokio::test]
async fn interrupted_utterance_is_discarded() {
    let memory = Arc::new(InMemMemory::default());
    let (capture, utterances, _replies) = fake_capture();
    let agent = CompositeAgent {
        // No SpeechEnd: the interrupt must cancel the live utterance.
        vad: Arc::new(ScriptedVadFactory {
            events: vec![
                VadEvent::SpeechStart { at_ms: 0 },
                VadEvent::Speech {
                    samples: vec![0.5; 960],
                },
            ],
        }),
        memory: memory.clone(),
        capture: Some(capture),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    // Let the driver spawn the utterance before cancelling it: the token
    // is cancelled before the ASR input closes, so the late final from
    // StubAsr must not finish the recording.
    tokio::time::sleep(Duration::from_millis(50)).await;
    tx.send(AgentInput::Interrupt { reason: None })
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(utterances.lock().unwrap().is_empty());
    assert!(memory.stored_utterances().is_empty());
}

#[tokio::test]
async fn empty_utterance_stores_nothing() {
    let memory = Arc::new(InMemMemory::default());
    let (capture, utterances, _replies) = fake_capture();
    let agent = CompositeAgent {
        asr: Arc::new(StubAsr::new("")),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        memory: memory.clone(),
        capture: Some(capture),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    // StubAsr still emits its (empty) final; the driver answers nothing.
    next_matching(&mut output, |item| {
        matches!(item, AgentOutput::Stt { is_final: true, .. })
    })
    .await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(utterances.lock().unwrap().is_empty());
    assert!(memory.stored_utterances().is_empty());
}

#[tokio::test]
async fn completed_turn_logs_reply_and_tool_items() {
    let memory = Arc::new(InMemMemory::default());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let agent = CompositeAgent {
        llm: Arc::new(ToolCallingLlm {
            calls: calls.clone(),
            tool_calls: vec![ToolCall {
                id: "call_x".into(),
                name: "nonexistent".into(),
                arguments: "{}".into(),
            }],
            reply: "ok".into(),
        }),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        memory: memory.clone(),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
    wait_for_calls(&calls, 2).await;

    let logged = wait_until("turn items", || memory.logged_turns().first().cloned()).await;
    assert_eq!(
        logged,
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "nonexistent".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_x", "error: unknown tool `nonexistent`"),
            ChatItem::assistant("ok"),
        ]
    );
}

#[tokio::test]
async fn spoken_preamble_before_tool_call_is_kept() {
    let memory = Arc::new(InMemMemory::default());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let agent = CompositeAgent {
        llm: Arc::new(PreambleToolLlm {
            calls: calls.clone(),
            preamble: "好的，我查一下".into(),
            reply: "答复".into(),
        }),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        memory: memory.clone(),
        ..stub_agent()
    };
    let (tx, _output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
    let rounds = wait_for_calls(&calls, 2).await;

    let call = || ToolCall {
        id: "call_x".into(),
        name: "nonexistent".into(),
        arguments: "{}".into(),
    };
    // The preamble rides on the round item, so the next LLM round sees it.
    assert_eq!(
        rounds[1],
        vec![
            ChatItem::user("hello"),
            ChatItem::assistant_text_tool_calls("好的，我查一下", vec![call()]),
            ChatItem::tool("call_x", "error: unknown tool `nonexistent`"),
        ]
    );
    // And it is stored with the turn's tool exchange.
    let logged = wait_until("turn items", || memory.logged_turns().first().cloned()).await;
    assert_eq!(
        logged,
        vec![
            ChatItem::assistant_text_tool_calls("好的，我查一下", vec![call()]),
            ChatItem::tool("call_x", "error: unknown tool `nonexistent`"),
            ChatItem::assistant("答复"),
        ]
    );
}

#[tokio::test]
async fn completed_reply_audio_is_stored() {
    let (capture, _utterances, replies) = fake_capture();
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }),
        tts: Arc::new(ScriptedTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        capture: Some(capture),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    // The natural tts stop finishes the capture; the audio attaches to
    // the turn row through the memory's hook.
    next_matching(&mut output, |item| matches!(item, AgentOutput::TtsStop)).await;

    let stored = wait_for_replies(&replies, 1).await;
    assert_eq!(
        stored,
        vec![FakeReply::Finished {
            chunks: vec![vec![0.25; 960]],
        }]
    );
}

#[tokio::test]
async fn aborted_reply_after_done_keeps_the_full_row_attachment() {
    let (capture, _utterances, replies) = fake_capture();
    let agent = CompositeAgent {
        llm: Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }),
        tts: Arc::new(HeldTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[], &[]])),
        capture: Some(capture),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    // Wait until the reply is playing and its turn has completed (the
    // hook has fired), then start the next utterance: the barge-in must
    // attach the partial audio to the full reply row.
    next_matching(&mut output, |item| matches!(item, AgentOutput::Audio(_))).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    let stored = wait_for_replies(&replies, 1).await;
    assert_eq!(
        stored,
        vec![FakeReply::Finished {
            chunks: vec![vec![0.25; 960]],
        }]
    );
}

/// Streams deltas then fails: the reply is cut before the model finished.
struct FailingLlm {
    reply: String,
}

impl Llm for FailingLlm {
    fn chat(
        &self,
        _history: Vec<ChatItem>,
        _tools: Vec<crate::agent::ToolSpec>,
        _tool_choice: Option<crate::agent::ToolChoice>,
        _reasoning: Option<crate::agent::ReasoningEffort>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        // The pause before the failure lets the driver process the first
        // synthesized chunk, so the abort settles an open capture.
        let deltas = futures_util::stream::iter(self.reply.chars().map(|c| {
            Ok(LlmEvent::Delta {
                text: c.to_string(),
            })
        }));
        let failure = futures_util::stream::once(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Err(crate::agent::LlmError::Failed("boom".into()))
        });
        Box::pin(deltas.chain(failure))
    }
}

#[tokio::test]
async fn aborted_reply_before_done_stores_the_truncated_row() {
    let memory = Arc::new(InMemMemory::default());
    let (capture, _utterances, replies) = fake_capture();
    let agent = CompositeAgent {
        llm: Arc::new(FailingLlm { reply: "hi".into() }),
        tts: Arc::new(HeldTts),
        vad: Arc::new(ScriptedVadFactory::utterances(&[&[]])),
        memory: memory.clone(),
        capture: Some(capture),
        ..stub_agent()
    };
    let (tx, mut output) = run(agent);

    tx.send(AgentInput::ListenStart { mode: None })
        .await
        .unwrap();
    tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

    // The llm error aborts the reply mid-flight: the driver stores the
    // partially streamed text as a truncated assistant row and the
    // capture attaches the partial audio through the hook.
    next_matching(&mut output, |item| {
        matches!(item, AgentOutput::Error { .. })
    })
    .await;

    let stored = wait_for_replies(&replies, 1).await;
    assert_eq!(
        stored,
        vec![FakeReply::Finished {
            chunks: vec![vec![0.25; 960]],
        }]
    );
    assert_eq!(memory.stored_partials(), vec!["hi"]);
    assert!(memory.logged_turns().is_empty());
}
