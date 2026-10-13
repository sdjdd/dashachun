//! The reply leg: one turn per utterance. The turn spawns the speak stage
//! and runs the think stage inline; the stage owns the reply-side
//! bookkeeping — the live turn slot, the speaking flag and the partially
//! streamed reply — and settles it at every cut.

use std::sync::Arc;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, info, trace, warn};

use super::{Cancellable, CompositeAgent, ListenEvent, TTS_TEXT_CHANNEL_CAPACITY, stop_task};
use crate::agent::tool::stop_after_memory_writes;
use crate::agent::{
    AgentOutput, AgentSession, Capture, ChatItem, Llm, Memory, ReasoningEffort, ReplyCapture,
    StreamEvent, StreamTextParams, Subtitle, SystemPrompt, TextStream, ToolSet, Tts, TtsEvent,
    emotion, labelled_utterance, stream_text,
};
use crate::util::recv_stream;

struct TurnDeps {
    llm: Arc<dyn Llm>,
    tools: Arc<ToolSet>,
    tts: Arc<dyn Tts>,
    session_id: String,
}

enum TurnOutcome {
    Done,
    Failed,
    Cancelled,
}

/// One turn's progress, tagged with the turn id so the reply stage can drop
/// everything belonging to a superseded turn.
pub(super) enum ReplyEvent {
    TtsStart,
    SentenceStart { text: String },
    Subtitle { subtitle: Subtitle },
    Audio { samples: Vec<f32> },
    TtsStop,
    SpeakError { message: String },
    Delta { text: String },
    Emotion { emotion: emotion::Emotion },
    ReplyDone { reply: String, items: Vec<ChatItem> },
    Failed { message: String },
}

/// Why the live reply is being cut. The four cuts are the TtsAbort rules:
/// a device interrupt and a TTS failure abort even when idle; a reset, a
/// barge-in and an LLM failure abort only a reply that is currently
/// speaking. The listener-driven cuts cancel the turn; the failure cuts
/// need no cancel — the turn is already ending on its own.
enum Cut {
    Interrupt,
    Superseded,
    LlmFailed { message: String },
    TtsFailed { message: String },
}

impl Cut {
    fn cancels_turn(&self) -> bool {
        matches!(self, Cut::Interrupt | Cut::Superseded)
    }

    fn aborts_always(&self) -> bool {
        matches!(self, Cut::Interrupt | Cut::TtsFailed { .. })
    }

    fn error(&self) -> Option<String> {
        match self {
            Cut::LlmFailed { message } | Cut::TtsFailed { message } => Some(message.clone()),
            Cut::Interrupt | Cut::Superseded => None,
        }
    }
}

/// The live reply: the current turn slot, the speaking flag and the
/// partially streamed reply's bookkeeping — everything a cut touches.
struct LiveReply {
    turn: Option<Turn>,
    speaking: bool,
    capture: Option<ReplyCapture>,
    turn_logged: bool,
    streamed: String,
}

impl LiveReply {
    fn new() -> Self {
        Self {
            turn: None,
            speaking: false,
            capture: None,
            turn_logged: false,
            streamed: String::new(),
        }
    }

    /// Records one synthesized chunk, lazily opening the reply capture.
    fn record(&mut self, capture: &dyn Capture, session: &AgentSession, samples: &[f32]) {
        let recording = self
            .capture
            .get_or_insert_with(|| capture.start_reply(session));
        recording.push(samples);
    }

    /// The natural `TtsStop`: finish any open capture — the audio attaches
    /// to the logged turn row — and clear the per-turn bookkeeping.
    fn stopped(&mut self) {
        self.speaking = false;
        if let Some(recording) = self.capture.take() {
            recording.finish();
        }
        self.turn_logged = false;
        self.streamed.clear();
    }

    /// Ends the reply at a cut, in one uniform order: settle the
    /// partial-reply bookkeeping, cancel the turn for the listener-driven
    /// cuts, forward an optional error and emit `TtsAbort` per the cut's
    /// policy, then clear the speaking flag. `Err` means the output closed
    /// and the stage stops driving.
    async fn cut(
        &mut self,
        memory: &Arc<dyn Memory>,
        session: &AgentSession,
        out_tx: &mpsc::Sender<AgentOutput>,
        cut: Cut,
    ) -> Result<(), ()> {
        self.settle(memory, session);
        if cut.cancels_turn()
            && let Some(current) = self.turn.take()
        {
            current.cancel();
        }
        if let Some(message) = cut.error()
            && out_tx.send(AgentOutput::Error { message }).await.is_err()
        {
            return Err(());
        }
        if cut.aborts_always() || self.speaking {
            out_tx.send(AgentOutput::TtsAbort).await.map_err(|_| ())?;
        }
        self.speaking = false;
        Ok(())
    }

    /// Ends the reply audio capture at a cut (barge-in, provider error):
    /// when the turn completed the audio attaches to the stored reply row,
    /// otherwise the partially streamed text is stored as a truncated
    /// assistant row and the partial audio attaches to it — both through
    /// the memory's hooks. A reply that never produced text (and no turn
    /// row) just discards the capture. A session teardown drops the
    /// handles, which discards it too.
    fn settle(&mut self, memory: &Arc<dyn Memory>, session: &AgentSession) {
        if let Some(recording) = self.capture.take() {
            if self.turn_logged {
                recording.finish();
            } else {
                let text = std::mem::take(&mut self.streamed);
                if text.is_empty() {
                    drop(recording);
                } else {
                    memory.store_partial_reply(session, &text);
                    recording.finish();
                }
            }
        }
        self.turn_logged = false;
        self.streamed.clear();
    }
}

/// A spawned reply turn: its id plus the cancellable task.
struct Turn {
    id: u64,
    task: Cancellable,
}

impl Turn {
    fn cancel(self) {
        self.task.cancel();
    }

    fn cancelled(&self) -> bool {
        self.task.cancelled()
    }
}

/// The reply leg. One `Turn` per utterance; forwards the live turn's events,
/// drops everything belonging to a superseded one, and owns the reply-side
/// bookkeeping through `LiveReply`.
pub(super) async fn reply_stage(
    agent: CompositeAgent,
    session: AgentSession,
    mut utterances: mpsc::Receiver<ListenEvent>,
    events_tx: mpsc::Sender<(u64, ReplyEvent)>,
    mut events_rx: mpsc::Receiver<(u64, ReplyEvent)>,
    out_tx: mpsc::Sender<AgentOutput>,
) {
    let mut reply = LiveReply::new();
    let mut turn_id: u64 = 0;
    loop {
        tokio::select! {
            item = utterances.recv() => {
                let Some(item) = item else { break };
                match item {
                    ListenEvent::Interrupt => {
                        if reply
                            .cut(&agent.memory, &session, &out_tx, Cut::Interrupt)
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    ListenEvent::Reset => {
                        if reply
                            .cut(&agent.memory, &session, &out_tx, Cut::Superseded)
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    ListenEvent::Utterance {
                        text,
                        speaker,
                        turn: token,
                    } => {
                        if reply
                            .cut(&agent.memory, &session, &out_tx, Cut::Superseded)
                            .await
                            .is_err()
                        {
                            break;
                        }
                        agent
                            .memory
                            .append(vec![ChatItem::user(labelled_utterance(
                                text,
                                speaker,
                                &agent.speakers,
                            ))])
                            .await;
                        let entries = agent.entries.list().await;
                        let prompt = agent.system_prompt.for_turn(&entries, &agent.speakers);
                        let history = agent.memory.history().await;
                        turn_id += 1;
                        reply.turn = Some(spawn_turn(
                            &agent,
                            &session,
                            history,
                            prompt,
                            turn_id,
                            token,
                            events_tx.clone(),
                        ));
                    }
                }
            }
            Some((id, event)) = events_rx.recv() => {
                let Some(current) = reply.turn.as_ref() else { continue };
                if current.id != id || current.cancelled() {
                    continue;
                }
                match event {
                    ReplyEvent::TtsStart => {
                        reply.speaking = true;
                        if out_tx.send(AgentOutput::TtsStart).await.is_err() {
                            break;
                        }
                    }
                    ReplyEvent::SentenceStart { text } => {
                        if out_tx
                            .send(AgentOutput::TtsSentence { text })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    ReplyEvent::Subtitle { subtitle } => {
                        if out_tx
                            .send(AgentOutput::TtsSubtitle { subtitle })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    ReplyEvent::Audio { samples } => {
                        if let Some(capture) = agent.capture.as_deref() {
                            reply.record(capture, &session, &samples);
                        }
                        if out_tx.send(AgentOutput::Audio(samples)).await.is_err() {
                            break;
                        }
                    }
                    ReplyEvent::TtsStop => {
                        reply.stopped();
                        if out_tx.send(AgentOutput::TtsStop).await.is_err() {
                            break;
                        }
                    }
                    ReplyEvent::SpeakError { message } => {
                        warn!(session_id = %session.id, %message, "tts failed");
                        if reply
                            .cut(&agent.memory, &session, &out_tx, Cut::TtsFailed { message })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    ReplyEvent::Delta { text } => {
                        reply.streamed.push_str(&text);
                    }
                    ReplyEvent::Emotion { emotion } => {
                        if out_tx
                            .send(AgentOutput::Emotion {
                                emotion: emotion.to_string(),
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    ReplyEvent::ReplyDone { reply: text, items } => {
                        info!(session_id = %session.id, %text, "llm result");
                        agent.memory.log_items(&session, items.clone());
                        reply.turn_logged = true;
                        agent.memory.append(items).await;
                    }
                    ReplyEvent::Failed { message } => {
                        warn!(session_id = %session.id, %message, "llm failed");
                        if reply
                            .cut(&agent.memory, &session, &out_tx, Cut::LlmFailed { message })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        }
    }
    if let Some(current) = reply.turn.take() {
        current.cancel();
    }
}

fn spawn_turn(
    agent: &CompositeAgent,
    session: &AgentSession,
    history: Vec<ChatItem>,
    prompt: SystemPrompt,
    turn_id: u64,
    token: CancellationToken,
    events: mpsc::Sender<(u64, ReplyEvent)>,
) -> Turn {
    let deps = TurnDeps {
        llm: agent.llm.clone(),
        tools: agent.tools.clone(),
        tts: agent.tts.clone(),
        session_id: session.id.clone(),
    };
    let handle = tokio::spawn(run_turn(
        deps,
        history,
        prompt,
        turn_id,
        events,
        token.clone(),
    ));
    Turn {
        id: turn_id,
        task: Cancellable::new(token, handle),
    }
}

async fn run_turn(
    deps: TurnDeps,
    history: Vec<ChatItem>,
    prompt: SystemPrompt,
    turn_id: u64,
    events: mpsc::Sender<(u64, ReplyEvent)>,
    token: CancellationToken,
) {
    let (speak_tx, speak_rx) = mpsc::channel::<String>(TTS_TEXT_CHANNEL_CAPACITY);
    // The speak leg runs on a child token: a failed turn silences it with
    // the same polite stop as a barge-in while the turn token itself stays
    // live, so the reply stage still drains the Failed mapping.
    let speak_token = token.child_token();
    let mut speak = tokio::spawn(speak_stage(
        deps.tts.clone(),
        speak_rx,
        events.clone(),
        turn_id,
        speak_token.clone(),
        deps.session_id.clone(),
    ));
    let outcome = think_stage(&deps, history, prompt, &speak_tx, &events, turn_id, &token)
        .instrument(tracing::info_span!("think", session_id = %deps.session_id))
        .await;
    drop(speak_tx);
    match outcome {
        TurnOutcome::Done => {
            let _ = speak.await;
        }
        TurnOutcome::Failed | TurnOutcome::Cancelled => {
            speak_token.cancel();
            stop_task(&mut speak).await;
        }
    }
}

/// The thinking leg: one `stream_text` call per turn. The tool sub-loop
/// lives entirely in the generation layer; this stage maps its events onto
/// the reply pipeline — raw deltas (the partial-reply text), the turn's
/// leading emoji, the emoji-stripped text for the speak leg, and the
/// produced messages into `ReplyDone` for the memory.
async fn think_stage(
    deps: &TurnDeps,
    history: Vec<ChatItem>,
    prompt: SystemPrompt,
    speak_tx: &mpsc::Sender<String>,
    events: &mpsc::Sender<(u64, ReplyEvent)>,
    turn_id: u64,
    cancel: &CancellationToken,
) -> TurnOutcome {
    let mut messages = Vec::new();
    if let Some(system) = prompt.chat_item() {
        messages.push(system);
    }
    messages.extend(history);
    let mut stream = stream_text(
        StreamTextParams::builder(deps.llm.as_ref())
            .messages(messages)
            .tools(&deps.tools)
            // Thinking models buffer reasoning before any content — a spoken
            // reply must start immediately, so the turn pins reasoning off
            // no matter what the provider config says.
            .reasoning(ReasoningEffort::None)
            .stop_when(vec![stop_after_memory_writes()])
            .cancel(cancel.clone())
            .build(),
    );
    let mut stripper = emotion::Stripper::new();
    let mut emotion_pending = true;
    let mut speak_started = false;
    loop {
        let ev = tokio::select! {
            item = stream.next() => match item {
                Some(item) => item,
                // A stream that ends without `Done` was cancelled.
                None => return TurnOutcome::Cancelled,
            },
            _ = cancel.cancelled() => return TurnOutcome::Cancelled,
        };
        match ev {
            Ok(StreamEvent::Delta { text }) => {
                trace!(session_id = %deps.session_id, text = %text, "llm delta");
                if emit(
                    events,
                    turn_id,
                    ReplyEvent::Delta { text: text.clone() },
                    cancel,
                )
                .await
                .is_err()
                {
                    return TurnOutcome::Cancelled;
                }
                if emotion_pending && let Some(found) = emotion::detect(&text) {
                    emotion_pending = false;
                    if emit(
                        events,
                        turn_id,
                        ReplyEvent::Emotion { emotion: found },
                        cancel,
                    )
                    .await
                    .is_err()
                    {
                        return TurnOutcome::Cancelled;
                    }
                }
                let spoken = stripper.push(&text);
                if speak_chunk(
                    spoken,
                    &mut speak_started,
                    speak_tx,
                    events,
                    turn_id,
                    cancel,
                )
                .await
                .is_err()
                {
                    return TurnOutcome::Cancelled;
                }
            }
            Ok(StreamEvent::Done(result)) => {
                let tail = stripper.finish();
                if speak_chunk(tail, &mut speak_started, speak_tx, events, turn_id, cancel)
                    .await
                    .is_err()
                {
                    return TurnOutcome::Cancelled;
                }
                if emit(
                    events,
                    turn_id,
                    ReplyEvent::ReplyDone {
                        reply: result.text,
                        items: result.messages,
                    },
                    cancel,
                )
                .await
                .is_err()
                {
                    return TurnOutcome::Cancelled;
                }
                return TurnOutcome::Done;
            }
            Err(err) => {
                let _ = emit(
                    events,
                    turn_id,
                    ReplyEvent::Failed {
                        message: err.to_string(),
                    },
                    cancel,
                )
                .await;
                return TurnOutcome::Failed;
            }
        }
    }
}

async fn speak_chunk(
    chunk: String,
    speak_started: &mut bool,
    speak_tx: &mpsc::Sender<String>,
    events: &mpsc::Sender<(u64, ReplyEvent)>,
    turn_id: u64,
    cancel: &CancellationToken,
) -> Result<(), ()> {
    if chunk.is_empty() {
        return Ok(());
    }
    if !*speak_started {
        *speak_started = true;
        emit(events, turn_id, ReplyEvent::TtsStart, cancel).await?;
    }
    if speak_tx.send(chunk).await.is_err() {
        debug!("tts input closed, dropping delta");
    }
    Ok(())
}

async fn emit(
    events: &mpsc::Sender<(u64, ReplyEvent)>,
    turn_id: u64,
    event: ReplyEvent,
    cancel: &CancellationToken,
) -> Result<(), ()> {
    tokio::select! {
        result = events.send((turn_id, event)) => result.map_err(|_| ()),
        _ = cancel.cancelled() => Err(()),
    }
}

/// The speaking leg: maps one `tts.synthesize` call's events into the turn
/// event stream. A cancelled token exits silently (no TtsStop); a natural
/// `Done` emits `TtsStop`, an error emits `SpeakError`.
async fn speak_stage(
    tts: Arc<dyn Tts>,
    text_rx: mpsc::Receiver<String>,
    events: mpsc::Sender<(u64, ReplyEvent)>,
    turn_id: u64,
    cancel: CancellationToken,
    session_id: String,
) {
    let text: TextStream = recv_stream(text_rx);
    let mut synth = tts.synthesize(text, cancel.clone());
    loop {
        let item = tokio::select! {
            item = synth.next() => match item {
                Some(item) => item,
                None => return,
            },
            _ = cancel.cancelled() => return,
        };
        let event = match item {
            Ok(TtsEvent::SentenceStart { text }) => ReplyEvent::SentenceStart { text },
            Ok(TtsEvent::Subtitle(subtitle)) => ReplyEvent::Subtitle { subtitle },
            Ok(TtsEvent::Audio(samples)) => ReplyEvent::Audio { samples },
            Ok(TtsEvent::Done) => {
                let _ = events.send((turn_id, ReplyEvent::TtsStop)).await;
                return;
            }
            Err(err) => {
                let _ = events
                    .send((
                        turn_id,
                        ReplyEvent::SpeakError {
                            message: err.to_string(),
                        },
                    ))
                    .await;
                return;
            }
        };
        if events.send((turn_id, event)).await.is_err() {
            debug!(session_id, "reply output closed");
            return;
        }
    }
}
