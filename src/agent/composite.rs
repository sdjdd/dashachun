use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, trace, warn};

use crate::agent::{
    Agent, AgentInput, AgentInputStream, AgentOutput, AgentOutputStream, AgentSession, Capture,
    Memory, ReplyCapture, SystemPrompt, ToolRegistry, UtteranceCapture, emotion,
};
use crate::asr::{Asr, AsrEvent, AudioStream};
use crate::llm::{ChatItem, Llm, LlmEvent, ToolCall};
use crate::tts::{Subtitle, TextStream, Tts, TtsEvent};
use crate::vad::{Vad, VadEvent, VadFactory};

const ASR_CHANNEL_CAPACITY: usize = 64;
const LLM_CHANNEL_CAPACITY: usize = 64;
const TTS_CHANNEL_CAPACITY: usize = 64;
const TTS_TEXT_CHANNEL_CAPACITY: usize = 64;
const OUTPUT_CHANNEL_CAPACITY: usize = 64;
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone)]
pub struct CompositeAgent {
    pub asr: Arc<dyn Asr>,
    pub llm: Arc<dyn Llm>,
    pub tts: Arc<dyn Tts>,
    pub vad: Arc<dyn VadFactory>,
    pub memory: Arc<dyn Memory>,
    pub tools: Arc<ToolRegistry>,
    pub system_prompt: SystemPrompt,
    pub capture: Option<Arc<dyn Capture>>,
}

impl Agent for CompositeAgent {
    fn run(&self, session: AgentSession, input: AgentInputStream) -> AgentOutputStream {
        let (out_tx, out_rx) = mpsc::channel::<AgentOutput>(OUTPUT_CHANNEL_CAPACITY);
        let session_id = session.id.clone();
        info!(session_id, "agent started");
        tokio::spawn({
            let agent = self.clone();
            async move {
                drive(agent, session, input, out_tx).await;
                info!(session_id, "agent stopped");
            }
        });
        Box::pin(futures_util::stream::unfold(out_rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        }))
    }
}

enum AsrMessage {
    Partial { generation: u64, text: String },
    Final { generation: u64, text: String },
    Error { generation: u64, message: String },
}

enum LlmMessage {
    Delta {
        generation: u64,
        text: String,
    },
    Emotion {
        generation: u64,
        emotion: emotion::Emotion,
    },
    Output {
        generation: u64,
        output: AgentOutput,
    },
    Done {
        generation: u64,
        reply: String,
        history: Vec<ChatItem>,
    },
    Error {
        generation: u64,
        message: String,
    },
}

enum TtsMessage {
    SentenceStart { generation: u64, text: String },
    Subtitle { generation: u64, subtitle: Subtitle },
    Audio { generation: u64, samples: Vec<f32> },
    Done { generation: u64 },
    Error { generation: u64, message: String },
}

async fn drive(
    agent: CompositeAgent,
    session: AgentSession,
    mut input: AgentInputStream,
    out_tx: mpsc::Sender<AgentOutput>,
) {
    let mut vad: Option<Box<dyn Vad>> = None;
    let (asr_tx, mut asr_rx) = mpsc::channel::<AsrMessage>(ASR_CHANNEL_CAPACITY);
    let (llm_tx, mut llm_rx) = mpsc::channel::<LlmMessage>(LLM_CHANNEL_CAPACITY);
    let (tts_tx, mut tts_rx) = mpsc::channel::<TtsMessage>(TTS_CHANNEL_CAPACITY);
    let mut utterance: Option<Utterance> = None;
    let mut completion: Option<Completion> = None;
    let mut synthesis: Option<Synthesis> = None;
    let mut generation: u64 = 0;
    let mut emotion_pending = false;
    let mut stripper = emotion::Stripper::new();
    let mut speaking = false;
    let mut reply_capture: Option<ReplyCapture> = None;
    let mut reply_turn_logged = false;
    let mut streamed_reply = String::new();

    loop {
        tokio::select! {
            item = input.next() => {
                let Some(item) = item else { break };
                match item {
                    AgentInput::ListenStart { .. } => {
                        settle_reply(
                            &agent.memory,
                            &session,
                            &mut reply_capture,
                            &mut reply_turn_logged,
                            &mut streamed_reply,
                        );
                        if cancel_in_flight(
                            &mut utterance,
                            &mut completion,
                            &mut synthesis,
                            &mut speaking,
                        ) && out_tx.send(AgentOutput::TtsAbort).await.is_err()
                        {
                            break;
                        }
                        vad = match agent.vad.build(session.sample_rate) {
                            Ok(vad) => Some(vad),
                            Err(err) => {
                                warn!(session_id = %session.id, %err, "failed to build vad");
                                None
                            }
                        };
                        generation += 1;
                        utterance = None;
                        completion = None;
                        synthesis = None;
                        emotion_pending = false;
                    }
                    AgentInput::ListenStop => {
                        info!(session_id = %session.id, "listen stop");
                        if let Some(vad) = vad.as_mut() {
                            let events = vad.flush();
                            handle_vad_events(
                                events,
                                &session,
                                &agent.asr,
                                &asr_tx,
                                &mut generation,
                                &mut utterance,
                                &agent.memory,
                                agent.capture.as_deref(),
                            )
                            .await;
                        }
                        utterance = None;
                    }
                    AgentInput::Audio(samples) => {
                        let Some(vad) = vad.as_mut() else { continue };
                        let events = vad.push(&samples);
                        handle_vad_events(
                            events,
                            &session,
                            &agent.asr,
                            &asr_tx,
                            &mut generation,
                            &mut utterance,
                            &agent.memory,
                            agent.capture.as_deref(),
                        )
                        .await;
                    }
                    AgentInput::Interrupt { reason } => {
                        info!(session_id = %session.id, reason = ?reason, "interrupt");
                        settle_reply(
                            &agent.memory,
                            &session,
                            &mut reply_capture,
                            &mut reply_turn_logged,
                            &mut streamed_reply,
                        );
                        cancel_in_flight(
                            &mut utterance,
                            &mut completion,
                            &mut synthesis,
                            &mut speaking,
                        );
                        generation += 1;
                        utterance = None;
                        completion = None;
                        synthesis = None;
                        emotion_pending = false;
                        if out_tx.send(AgentOutput::TtsAbort).await.is_err() {
                            break;
                        }
                    }
                    AgentInput::Mcp(payload) => {
                        if out_tx.send(AgentOutput::Mcp(payload)).await.is_err() {
                            break;
                        }
                    }
                }
            }
            Some(message) = asr_rx.recv() => {
                let current = match &message {
                    AsrMessage::Partial { generation, .. }
                    | AsrMessage::Final { generation, .. }
                    | AsrMessage::Error { generation, .. } => *generation,
                };
                if current != generation {
                    continue;
                }
                match message {
                    AsrMessage::Partial { text, .. } => {
                        if out_tx
                            .send(AgentOutput::Stt {
                                text,
                                is_final: false,
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    AsrMessage::Final { text, .. } => {
                        info!(session_id = %session.id, %text, "asr result");
                        if out_tx
                            .send(AgentOutput::Stt {
                                text: text.clone(),
                                is_final: true,
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                        if text.is_empty() {
                            continue;
                        }
                        if let Some(current) = completion.take() {
                            current.cancel();
                        }
                        if let Some(current) = synthesis.take() {
                            current.cancel();
                        }
                        settle_reply(
                            &agent.memory,
                            &session,
                            &mut reply_capture,
                            &mut reply_turn_logged,
                            &mut streamed_reply,
                        );
                        if speaking
                            && out_tx.send(AgentOutput::TtsAbort).await.is_err()
                        {
                            break;
                        }
                        speaking = false;
                        agent
                            .memory
                            .append(vec![ChatItem::user(text)])
                            .await;
                        emotion_pending = true;
                        stripper = emotion::Stripper::new();
                        completion = Some(spawn_llm(
                            &agent.llm,
                            &agent.tools,
                            &llm_tx,
                            session.id.clone(),
                            generation,
                            agent.memory.history().await,
                            agent.system_prompt.clone(),
                        ));
                    }
                    AsrMessage::Error { message, .. } => {
                        warn!(session_id = %session.id, %message, "asr failed");
                        if out_tx.send(AgentOutput::Error { message }).await.is_err() {
                            break;
                        }
                    }
                }
            }
            Some(message) = llm_rx.recv() => {
                let current = match &message {
                    LlmMessage::Delta { generation, .. }
                    | LlmMessage::Emotion { generation, .. }
                    | LlmMessage::Output { generation, .. }
                    | LlmMessage::Done { generation, .. }
                    | LlmMessage::Error { generation, .. } => *generation,
                };
                if current != generation {
                    continue;
                }
                match message {
                    LlmMessage::Delta { text, .. } => {
                        trace!(session_id = %session.id, %text, "llm delta");
                        if text.is_empty() {
                            continue;
                        }
                        streamed_reply.push_str(&text);
                        if emotion_pending
                            && let Some(emotion) = emotion::detect(&text)
                        {
                            emotion_pending = false;
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
                        let text = stripper.push(&text);
                        if !feed_tts(
                            text,
                            &mut synthesis,
                            &mut speaking,
                            &agent.tts,
                            &tts_tx,
                            &out_tx,
                            &session.id,
                            generation,
                        )
                        .await
                        {
                            break;
                        }
                    }
                    LlmMessage::Emotion { emotion, .. } => {
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
                    LlmMessage::Done { reply, history: items, .. } => {
                        completion = None;
                        info!(session_id = %session.id, %reply, "llm result");
                        agent.memory.log_items(&session, items.clone());
                        reply_turn_logged = true;
                        agent.memory.append(items).await;
                        let tail = stripper.finish();
                        if !feed_tts(
                            tail,
                            &mut synthesis,
                            &mut speaking,
                            &agent.tts,
                            &tts_tx,
                            &out_tx,
                            &session.id,
                            generation,
                        )
                        .await
                        {
                            break;
                        }
                        if let Some(current) = synthesis.take() {
                            current.detach();
                        }
                    }
                    LlmMessage::Output { output, .. } => {
                        if out_tx.send(output).await.is_err() {
                            break;
                        }
                    }
                    LlmMessage::Error { message, .. } => {
                        completion = None;
                        settle_reply(
                            &agent.memory,
                            &session,
                            &mut reply_capture,
                            &mut reply_turn_logged,
                            &mut streamed_reply,
                        );
                        let tts_active = synthesis.take().is_some();
                        warn!(session_id = %session.id, %message, "llm failed");
                        if out_tx.send(AgentOutput::Error { message }).await.is_err() {
                            break;
                        }
                        if tts_active && out_tx.send(AgentOutput::TtsAbort).await.is_err() {
                            break;
                        }
                    }
                }
            }
            Some(message) = tts_rx.recv() => {
                let current = match &message {
                    TtsMessage::SentenceStart { generation, .. }
                    | TtsMessage::Subtitle { generation, .. }
                    | TtsMessage::Audio { generation, .. }
                    | TtsMessage::Done { generation, .. }
                    | TtsMessage::Error { generation, .. } => *generation,
                };
                if current != generation {
                    continue;
                }
                match message {
                    TtsMessage::SentenceStart { text, .. } => {
                        if out_tx.send(AgentOutput::TtsSentence { text }).await.is_err() {
                            break;
                        }
                    }
                    TtsMessage::Subtitle { subtitle, .. } => {
                        if out_tx
                            .send(AgentOutput::TtsSubtitle { subtitle })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    TtsMessage::Audio { samples, .. } => {
                        if let Some(capture) = agent.capture.as_deref() {
                            let recording =
                                reply_capture.get_or_insert_with(|| capture.start_reply(&session));
                            recording.push(&samples);
                        }
                        if out_tx.send(AgentOutput::Audio(samples)).await.is_err() {
                            break;
                        }
                    }
                    TtsMessage::Done { .. } => {
                        synthesis = None;
                        speaking = false;
                        if let Some(recording) = reply_capture.take() {
                            recording.finish();
                        }
                        reply_turn_logged = false;
                        streamed_reply.clear();
                        if out_tx.send(AgentOutput::TtsStop).await.is_err() {
                            break;
                        }
                    }
                    TtsMessage::Error { message, .. } => {
                        synthesis = None;
                        speaking = false;
                        settle_reply(
                            &agent.memory,
                            &session,
                            &mut reply_capture,
                            &mut reply_turn_logged,
                            &mut streamed_reply,
                        );
                        warn!(session_id = %session.id, %message, "tts failed");
                        if out_tx.send(AgentOutput::Error { message }).await.is_err()
                            || out_tx.send(AgentOutput::TtsAbort).await.is_err()
                        {
                            break;
                        }
                    }
                }
            }
        }
    }

    cancel_in_flight(
        &mut utterance,
        &mut completion,
        &mut synthesis,
        &mut speaking,
    );
}

/// Cancels every in-flight stage without waiting and reports whether a spoken
/// reply was still playing, so the caller can tell the gateway to drop its
/// buffered playback. Each provider handshake runs detached: generation gating
/// already filters stale events, so a stalled provider must never be allowed
/// to freeze the driver loop.
fn cancel_in_flight(
    utterance: &mut Option<Utterance>,
    completion: &mut Option<Completion>,
    synthesis: &mut Option<Synthesis>,
    speaking: &mut bool,
) -> bool {
    if let Some(current) = utterance.take() {
        current.cancel();
    }
    if let Some(current) = completion.take() {
        current.cancel();
    }
    if let Some(current) = synthesis.take() {
        current.cancel();
    }
    std::mem::take(speaking)
}

/// Ends the reply audio capture at a cut (barge-in, provider error): when
/// the turn completed the audio attaches to the stored reply row, otherwise
/// the partially streamed text is stored as a truncated assistant row and
/// the partial audio attaches to it — both through the memory's hooks. A
/// reply that never produced text (and no turn row) just discards the
/// capture. A session teardown drops the handles, which discards it too.
fn settle_reply(
    memory: &Arc<dyn Memory>,
    session: &AgentSession,
    capture: &mut Option<ReplyCapture>,
    turn_logged: &mut bool,
    streamed: &mut String,
) {
    if let Some(recording) = capture.take() {
        if *turn_logged {
            recording.finish();
        } else {
            let text = std::mem::take(streamed);
            if text.is_empty() {
                drop(recording);
            } else {
                memory.store_partial_reply(session, &text);
                recording.finish();
            }
        }
    }
    *turn_logged = false;
    streamed.clear();
}

struct Utterance {
    tx: mpsc::Sender<Vec<f32>>,
    handle: Option<JoinHandle<()>>,
    cancel: CancellationToken,
    capture: Option<UtteranceCapture>,
}

impl Utterance {
    fn detach(mut self) {
        self.handle.take();
    }

    fn cancel(mut self) {
        let handle = self.handle.take();
        self.cancel.cancel();
        drop(self);
        tokio::spawn(async move {
            if let Some(mut handle) = handle {
                stop_task(&mut handle).await;
            }
        });
    }
}

impl Drop for Utterance {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

struct Completion {
    handle: JoinHandle<()>,
    cancel: CancellationToken,
}

impl Completion {
    fn cancel(self) {
        self.cancel.cancel();
        tokio::spawn(async move {
            let mut this = self;
            stop_task(&mut this.handle).await;
        });
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct Synthesis {
    tx: mpsc::Sender<String>,
    handle: Option<JoinHandle<()>>,
    cancel: CancellationToken,
}

impl Synthesis {
    fn detach(mut self) {
        self.handle.take();
    }

    fn cancel(mut self) {
        let handle = self.handle.take();
        self.cancel.cancel();
        drop(self);
        tokio::spawn(async move {
            if let Some(mut handle) = handle {
                stop_task(&mut handle).await;
            }
        });
    }
}

impl Drop for Synthesis {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

async fn stop_task(handle: &mut JoinHandle<()>) {
    if tokio::time::timeout(CANCEL_TIMEOUT, &mut *handle)
        .await
        .is_err()
    {
        warn!("provider did not cancel in time, aborting");
        handle.abort();
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_vad_events(
    events: Vec<VadEvent>,
    session: &AgentSession,
    asr: &Arc<dyn Asr>,
    asr_tx: &mpsc::Sender<AsrMessage>,
    generation: &mut u64,
    utterance: &mut Option<Utterance>,
    memory: &Arc<dyn Memory>,
    capture: Option<&dyn Capture>,
) {
    for event in events {
        match event {
            VadEvent::SpeechStart { at_ms } => {
                info!(session_id = %session.id, at_ms, "speech start");
                *generation += 1;
                let capture = capture.map(|capture| capture.start_utterance(session));
                *utterance = Some(spawn_asr(
                    asr,
                    asr_tx,
                    session.clone(),
                    *generation,
                    memory.clone(),
                    capture,
                ));
            }
            VadEvent::Speech { samples } => {
                if let Some(current) = utterance.as_ref() {
                    if let Some(capture) = &current.capture {
                        capture.push(&samples);
                    }
                    if current.tx.send(samples).await.is_err() {
                        debug!(session_id = %session.id, "asr input closed, dropping frame");
                    }
                }
            }
            VadEvent::SpeechEnd { .. } => {
                if let Some(current) = utterance.take() {
                    current.detach();
                }
            }
        }
    }
}

fn spawn_asr(
    asr: &Arc<dyn Asr>,
    asr_tx: &mpsc::Sender<AsrMessage>,
    session: AgentSession,
    generation: u64,
    memory: Arc<dyn Memory>,
    capture: Option<UtteranceCapture>,
) -> Utterance {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(ASR_CHANNEL_CAPACITY);
    let audio: AudioStream = Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    }));
    let asr = asr.clone();
    let asr_tx = asr_tx.clone();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let push = capture.clone();
    let handle = tokio::spawn(async move {
        run_asr(
            asr,
            audio,
            asr_tx,
            session,
            generation,
            task_cancel,
            memory,
            capture,
        )
        .await;
    });
    Utterance {
        tx,
        handle: Some(handle),
        cancel,
        capture: push,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_asr(
    asr: Arc<dyn Asr>,
    audio: AudioStream,
    asr_tx: mpsc::Sender<AsrMessage>,
    session: AgentSession,
    generation: u64,
    cancel: CancellationToken,
    memory: Arc<dyn Memory>,
    capture: Option<UtteranceCapture>,
) {
    let mut events = asr.transcribe(audio, cancel.clone());
    while let Some(result) = events.next().await {
        let message = match result {
            Ok(AsrEvent::Partial { text }) => AsrMessage::Partial { generation, text },
            Ok(AsrEvent::Final { text }) => {
                // The utterance is real speech even when the turn was
                // superseded, so the row is stored regardless of the cancel
                // token; the capture learns its message id through the
                // memory's hook, not from here.
                if !cancel.is_cancelled() && !text.is_empty() {
                    match memory.store_utterance(&session, &text).await {
                        Ok(()) => {
                            if let Some(capture) = capture {
                                capture.finish();
                            }
                        }
                        Err(err) => {
                            warn!(session_id = %session.id, %err, "failed to store utterance");
                        }
                    }
                }
                let _ = asr_tx.send(AsrMessage::Final { generation, text }).await;
                return;
            }
            Err(err) => AsrMessage::Error {
                generation,
                message: err.to_string(),
            },
        };
        if asr_tx.send(message).await.is_err() {
            debug!(session_id = %session.id, "asr output closed");
            return;
        }
    }
}

fn spawn_llm(
    llm: &Arc<dyn Llm>,
    tools: &Arc<ToolRegistry>,
    llm_tx: &mpsc::Sender<LlmMessage>,
    session_id: String,
    generation: u64,
    history: Vec<ChatItem>,
    system_prompt: SystemPrompt,
) -> Completion {
    let llm = llm.clone();
    let tools = tools.clone();
    let llm_tx = llm_tx.clone();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let handle = tokio::spawn(async move {
        run_llm(
            llm,
            tools,
            history,
            llm_tx,
            session_id,
            generation,
            system_prompt,
            task_cancel,
        )
        .await;
    });
    Completion { handle, cancel }
}

#[allow(clippy::too_many_arguments)]
async fn run_llm(
    llm: Arc<dyn Llm>,
    tools: Arc<ToolRegistry>,
    history: Vec<ChatItem>,
    llm_tx: mpsc::Sender<LlmMessage>,
    session_id: String,
    generation: u64,
    system_prompt: SystemPrompt,
    cancel: CancellationToken,
) {
    let mut working = history;
    let mut new_items: Vec<ChatItem> = Vec::new();
    let mut reply = String::new();

    loop {
        if cancel.is_cancelled() {
            return;
        }
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut text = String::new();
        let mut messages = Vec::with_capacity(working.len() + 1);
        if let Some(system) = system_prompt.chat_item() {
            messages.push(system);
        }
        messages.extend(working.iter().cloned());
        let mut events = llm.chat(messages, tools.specs(), cancel.clone());
        while let Some(result) = events.next().await {
            match result {
                Ok(LlmEvent::Delta { text: delta }) => {
                    if delta.is_empty() {
                        continue;
                    }
                    text.push_str(&delta);
                    if llm_tx
                        .send(LlmMessage::Delta {
                            generation,
                            text: delta,
                        })
                        .await
                        .is_err()
                    {
                        debug!(session_id, "llm output closed");
                        return;
                    }
                }
                Ok(LlmEvent::ToolCall(call)) => calls.push(call),
                Ok(LlmEvent::Done) => break,
                Err(err) => {
                    let _ = llm_tx
                        .send(LlmMessage::Error {
                            generation,
                            message: err.to_string(),
                        })
                        .await;
                    return;
                }
            }
        }

        if calls.is_empty() {
            if !text.is_empty() {
                reply.push_str(&text);
                new_items.push(ChatItem::assistant(text));
            }
            let _ = llm_tx
                .send(LlmMessage::Done {
                    generation,
                    reply,
                    history: new_items,
                })
                .await;
            return;
        }

        if llm_tx
            .send(LlmMessage::Emotion {
                generation,
                emotion: emotion::Emotion::Thinking,
            })
            .await
            .is_err()
        {
            debug!(session_id, "llm output closed");
            return;
        }
        new_items.push(ChatItem::assistant_tool_calls(calls.clone()));
        working.push(ChatItem::assistant_tool_calls(calls.clone()));
        for call in calls {
            let (content, output) = execute_tool(&tools, &call, &session_id).await;
            if let Some(output) = output
                && llm_tx
                    .send(LlmMessage::Output { generation, output })
                    .await
                    .is_err()
            {
                debug!(session_id, "llm output closed");
                return;
            }
            new_items.push(ChatItem::tool(call.id.clone(), content.clone()));
            working.push(ChatItem::tool(call.id, content));
        }
        if llm_tx
            .send(LlmMessage::Emotion {
                generation,
                emotion: emotion::Emotion::Neutral,
            })
            .await
            .is_err()
        {
            debug!(session_id, "llm output closed");
            return;
        }
    }
}

async fn execute_tool(
    tools: &ToolRegistry,
    call: &ToolCall,
    session_id: &str,
) -> (String, Option<AgentOutput>) {
    debug!(
        session_id,
        tool = %call.name,
        args = %call.arguments,
        "tool call"
    );
    let Some(handler) = tools.get(&call.name) else {
        warn!(session_id, tool = %call.name, "unknown tool requested");
        return (format!("error: unknown tool `{}`", call.name), None);
    };
    let args = if call.arguments.trim().is_empty() {
        serde_json::Value::Null
    } else {
        match serde_json::from_str(&call.arguments) {
            Ok(args) => args,
            Err(err) => {
                warn!(session_id, tool = %call.name, %err, "invalid tool arguments");
                return (format!("error: invalid arguments: {err}"), None);
            }
        }
    };
    let result = match handler.call(&args).await {
        Ok(outcome) => (outcome.content, outcome.output),
        Err(err) => {
            warn!(session_id, tool = %call.name, %err, "tool failed");
            (format!("error: {err}"), None)
        }
    };
    debug!(
        session_id,
        tool = %call.name,
        result = %result.0,
        "tool result"
    );
    result
}

#[allow(clippy::too_many_arguments)]
async fn feed_tts(
    text: String,
    synthesis: &mut Option<Synthesis>,
    speaking: &mut bool,
    tts: &Arc<dyn Tts>,
    tts_tx: &mpsc::Sender<TtsMessage>,
    out_tx: &mpsc::Sender<AgentOutput>,
    session_id: &str,
    generation: u64,
) -> bool {
    if text.is_empty() {
        return true;
    }
    match synthesis.as_mut() {
        Some(current) => {
            if current.tx.send(text).await.is_err() {
                debug!(session_id, "tts input closed, dropping delta");
            }
        }
        None => {
            if out_tx.send(AgentOutput::TtsStart).await.is_err() {
                return false;
            }
            let current = spawn_tts(tts, tts_tx, session_id.to_string(), generation);
            if current.tx.send(text).await.is_err() {
                debug!(session_id, "tts input closed, dropping delta");
            }
            *synthesis = Some(current);
            *speaking = true;
        }
    }
    true
}

fn spawn_tts(
    tts: &Arc<dyn Tts>,
    tts_tx: &mpsc::Sender<TtsMessage>,
    session_id: String,
    generation: u64,
) -> Synthesis {
    let (tx, rx) = mpsc::channel::<String>(TTS_TEXT_CHANNEL_CAPACITY);
    let text: TextStream = Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    }));
    let tts = tts.clone();
    let tts_tx = tts_tx.clone();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let handle = tokio::spawn(async move {
        run_tts(tts, text, tts_tx, session_id, generation, task_cancel).await;
    });
    Synthesis {
        tx,
        handle: Some(handle),
        cancel,
    }
}

async fn run_tts(
    tts: Arc<dyn Tts>,
    text: TextStream,
    tts_tx: mpsc::Sender<TtsMessage>,
    session_id: String,
    generation: u64,
    cancel: CancellationToken,
) {
    let mut events = tts.synthesize(text, cancel);
    while let Some(result) = events.next().await {
        let message = match result {
            Ok(TtsEvent::SentenceStart { text }) => TtsMessage::SentenceStart { generation, text },
            Ok(TtsEvent::Subtitle(subtitle)) => TtsMessage::Subtitle {
                generation,
                subtitle,
            },
            Ok(TtsEvent::Audio(samples)) => TtsMessage::Audio {
                generation,
                samples,
            },
            Ok(TtsEvent::Done) => {
                let _ = tts_tx.send(TtsMessage::Done { generation }).await;
                return;
            }
            Err(err) => {
                let _ = tts_tx
                    .send(TtsMessage::Error {
                        generation,
                        message: err.to_string(),
                    })
                    .await;
                return;
            }
        };
        if tts_tx.send(message).await.is_err() {
            debug!(session_id, "tts output closed");
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::InMemMemory;
    use crate::agent::collect;
    use crate::agent::memory::HISTORY_LIMIT;
    use crate::asr::{AsrError, AsrEvents, StubAsr};
    use crate::llm::{LlmEvents, StubLlm};
    use crate::tts::{StubTts, Tts, TtsError, TtsEvent, TtsEvents};
    use crate::vad::{Vad, VadError, VadEvent};
    use std::sync::Mutex;
    use std::time::Duration;

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
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|event| (event, rx))
            }))
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
            _tools: Vec<crate::llm::ToolSpec>,
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

    struct ToolCallingLlm {
        calls: Arc<Mutex<Vec<Vec<ChatItem>>>>,
        tool_calls: Vec<ToolCall>,
        reply: String,
    }

    impl Llm for ToolCallingLlm {
        fn chat(
            &self,
            history: Vec<ChatItem>,
            _tools: Vec<crate::llm::ToolSpec>,
            _cancel: CancellationToken,
        ) -> LlmEvents<'_> {
            let round = {
                let mut calls = self.calls.lock().unwrap();
                calls.push(history);
                calls.len()
            };
            if round == 1 {
                let mut events: Vec<Result<LlmEvent, crate::llm::LlmError>> = self
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
        let stream: AgentInputStream =
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            }));
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

    fn empty_tools() -> Arc<ToolRegistry> {
        Arc::new(ToolRegistry::new(Vec::new()))
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

    async fn wait_for_calls(
        calls: &Arc<Mutex<Vec<Vec<ChatItem>>>>,
        len: usize,
    ) -> Vec<Vec<ChatItem>> {
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
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|event| (event, rx))
            }))
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
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|event| (event, rx))
            }))
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
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|event| (event, rx))
            }))
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
            _tools: Vec<crate::llm::ToolSpec>,
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
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|event| (event, rx))
            }))
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

        let emotion_pos = seen.iter().position(
            |item| matches!(item, AgentOutput::Emotion { emotion } if emotion == "happy"),
        );
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
    async fn tool_call_emits_thinking_emotion() {
        let agent = CompositeAgent {
            llm: Arc::new(ToolCallingLlm {
                calls: Arc::new(Mutex::new(Vec::new())),
                tool_calls: vec![ToolCall {
                    id: "call_x".into(),
                    name: "nonexistent".into(),
                    arguments: "{}".into(),
                }],
                reply: "🙂搞定".into(),
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

        let thinking = seen.iter().position(
            |item| matches!(item, AgentOutput::Emotion { emotion } if emotion == "thinking"),
        );
        assert!(
            thinking.is_some(),
            "tool call should emit thinking: {seen:?}"
        );
        let reply_emotion = seen.iter().position(
            |item| matches!(item, AgentOutput::Emotion { emotion } if emotion == "happy"),
        );
        assert!(
            thinking < reply_emotion,
            "thinking must precede the reply emotion"
        );
    }

    #[tokio::test]
    async fn tool_call_without_reply_emoji_resets_to_neutral() {
        let agent = CompositeAgent {
            llm: Arc::new(ToolCallingLlm {
                calls: Arc::new(Mutex::new(Vec::new())),
                tool_calls: vec![ToolCall {
                    id: "call_x".into(),
                    name: "nonexistent".into(),
                    arguments: "{}".into(),
                }],
                reply: "搞定了".into(),
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

        let thinking = seen.iter().position(
            |item| matches!(item, AgentOutput::Emotion { emotion } if emotion == "thinking"),
        );
        let neutral = seen.iter().position(
            |item| matches!(item, AgentOutput::Emotion { emotion } if emotion == "neutral"),
        );
        assert!(
            thinking.is_some(),
            "tool call should emit thinking: {seen:?}"
        );
        assert!(
            thinking < neutral,
            "reply without emoji should reset thinking to neutral: {seen:?}"
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
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|event| (event, rx))
            }))
        }
    }

    struct HangingLlm;

    impl Llm for HangingLlm {
        fn chat(
            &self,
            _history: Vec<ChatItem>,
            _tools: Vec<crate::llm::ToolSpec>,
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
            _tools: Vec<crate::llm::ToolSpec>,
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
                Err(crate::llm::LlmError::Failed("boom".into()))
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
}
