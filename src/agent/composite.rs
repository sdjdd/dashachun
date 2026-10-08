use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, trace, warn};

use crate::agent::{
    Agent, AgentInput, AgentInputStream, AgentOutput, AgentOutputStream, AgentSession, Memory,
    ReplyReceipt, ReplyRecording, SystemPrompt, ToolRegistry, UtteranceRecording, emotion,
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
    pub transcript: Option<Arc<dyn crate::agent::TranscriptSink>>,
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
    let mut reply_recording: Option<ReplyRecording> = None;
    let mut reply_receipt: Option<ReplyReceipt> = None;
    let mut streamed_reply = String::new();

    loop {
        tokio::select! {
            item = input.next() => {
                let Some(item) = item else { break };
                match item {
                    AgentInput::ListenStart { .. } => {
                        settle_reply(
                            &mut reply_recording,
                            &mut reply_receipt,
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
                                agent.transcript.as_deref(),
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
                            agent.transcript.as_deref(),
                        )
                        .await;
                    }
                    AgentInput::Interrupt { reason } => {
                        info!(session_id = %session.id, reason = ?reason, "interrupt");
                        settle_reply(
                            &mut reply_recording,
                            &mut reply_receipt,
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
                            &mut reply_recording,
                            &mut reply_receipt,
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
                        if let Some(sink) = agent.transcript.as_deref() {
                            reply_receipt = Some(sink.log_items(&session, items.clone()));
                        }
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
                            &mut reply_recording,
                            &mut reply_receipt,
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
                        if let Some(sink) = agent.transcript.as_deref() {
                            let recording =
                                reply_recording.get_or_insert_with(|| sink.start_reply(&session));
                            recording.push(&samples);
                        }
                        if out_tx.send(AgentOutput::Audio(samples)).await.is_err() {
                            break;
                        }
                    }
                    TtsMessage::Done { .. } => {
                        synthesis = None;
                        speaking = false;
                        if let Some(recording) = reply_recording.take() {
                            match reply_receipt.take() {
                                Some(receipt) => recording.finish(receipt),
                                None => drop(recording),
                            }
                        }
                        streamed_reply.clear();
                        if out_tx.send(AgentOutput::TtsStop).await.is_err() {
                            break;
                        }
                    }
                    TtsMessage::Error { message, .. } => {
                        synthesis = None;
                        speaking = false;
                        settle_reply(
                            &mut reply_recording,
                            &mut reply_receipt,
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

/// Ends the reply audio capture at a cut (barge-in, provider error): with the
/// receipt when the turn completed, so the partial audio attaches to the full
/// reply row, otherwise with the partially streamed text, so the collector
/// writes the truncated assistant row itself. A session teardown just drops
/// the handles, which discards the capture.
fn settle_reply(
    recording: &mut Option<ReplyRecording>,
    receipt: &mut Option<ReplyReceipt>,
    streamed: &mut String,
) {
    if let Some(recording) = recording.take() {
        match receipt.take() {
            Some(receipt) => recording.finish(receipt),
            None => {
                let text = std::mem::take(streamed);
                if !text.is_empty() {
                    recording.finish_partial(text);
                }
            }
        }
    }
    receipt.take();
    streamed.clear();
}

struct Utterance {
    tx: mpsc::Sender<Vec<f32>>,
    handle: Option<JoinHandle<()>>,
    cancel: CancellationToken,
    recording: Option<UtteranceRecording>,
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

async fn handle_vad_events(
    events: Vec<VadEvent>,
    session: &AgentSession,
    asr: &Arc<dyn Asr>,
    asr_tx: &mpsc::Sender<AsrMessage>,
    generation: &mut u64,
    utterance: &mut Option<Utterance>,
    transcript: Option<&dyn crate::agent::TranscriptSink>,
) {
    for event in events {
        match event {
            VadEvent::SpeechStart { at_ms } => {
                info!(session_id = %session.id, at_ms, "speech start");
                *generation += 1;
                let recording = transcript.map(|sink| sink.start_utterance(session));
                *utterance = Some(spawn_asr(
                    asr,
                    asr_tx,
                    session.id.clone(),
                    *generation,
                    recording,
                ));
            }
            VadEvent::Speech { samples } => {
                if let Some(current) = utterance.as_ref() {
                    if let Some(recording) = &current.recording {
                        recording.push(&samples);
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
    session_id: String,
    generation: u64,
    recording: Option<UtteranceRecording>,
) -> Utterance {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(ASR_CHANNEL_CAPACITY);
    let audio: AudioStream = Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    }));
    let asr = asr.clone();
    let asr_tx = asr_tx.clone();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let push = recording.clone();
    let handle = tokio::spawn(async move {
        run_asr(
            asr,
            audio,
            asr_tx,
            session_id,
            generation,
            task_cancel,
            recording,
        )
        .await;
    });
    Utterance {
        tx,
        handle: Some(handle),
        cancel,
        recording: push,
    }
}

async fn run_asr(
    asr: Arc<dyn Asr>,
    audio: AudioStream,
    asr_tx: mpsc::Sender<AsrMessage>,
    session_id: String,
    generation: u64,
    cancel: CancellationToken,
    recording: Option<UtteranceRecording>,
) {
    let mut events = asr.transcribe(audio, cancel.clone());
    while let Some(result) = events.next().await {
        let message = match result {
            Ok(AsrEvent::Partial { text }) => AsrMessage::Partial { generation, text },
            Ok(AsrEvent::Final { text }) => {
                if !cancel.is_cancelled()
                    && !text.is_empty()
                    && let Some(recording) = recording
                {
                    recording.finish(text.clone());
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
            debug!(session_id, "asr output closed");
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

    #[tokio::test]
    async fn audio_drives_vad_asr_to_stt() {
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(StubLlm::default()),
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let mut finals = Vec::new();
        while let Some(item) = output.next().await {
            if let AgentOutput::Stt { text, is_final } = item
                && is_final
            {
                finals.push(text);
                break;
            }
        }
        assert_eq!(finals, vec!["hello".to_string()]);
    }

    #[tokio::test]
    async fn interrupt_emits_tts_stop() {
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(StubLlm::default()),
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory { events: vec![] }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::Interrupt { reason: None })
            .await
            .unwrap();

        let item = output.next().await;
        assert!(matches!(item, Some(AgentOutput::TtsAbort)), "got {item:?}");
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
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(ScriptedLlm {
                reply: "hi".into(),
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
            tts: Arc::new(HeldTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                    VadEvent::SpeechStart { at_ms: 200 },
                    VadEvent::SpeechEnd { at_ms: 300 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while let Some(item) = tokio::time::timeout_at(deadline, output.next())
            .await
            .ok()
            .flatten()
        {
            if matches!(item, AgentOutput::TtsStart) {
                break;
            }
        }

        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let mut aborted = false;
        while let Some(item) = tokio::time::timeout_at(deadline, output.next())
            .await
            .ok()
            .flatten()
        {
            if matches!(item, AgentOutput::TtsAbort) {
                aborted = true;
                break;
            }
        }
        assert!(aborted, "second final must abort the in-flight reply");
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
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(HangingLlm),
            tts: Arc::new(ImmortalTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                    VadEvent::SpeechStart { at_ms: 200 },
                    VadEvent::SpeechEnd { at_ms: 300 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while let Some(item) = tokio::time::timeout_at(deadline, output.next())
            .await
            .ok()
            .flatten()
        {
            if matches!(item, AgentOutput::TtsStart) {
                break;
            }
        }

        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let quick = tokio::time::Instant::now() + Duration::from_secs(1);
        let mut restarted = false;
        while let Some(item) = tokio::time::timeout_at(quick, output.next())
            .await
            .ok()
            .flatten()
        {
            if matches!(item, AgentOutput::TtsStart) {
                restarted = true;
                break;
            }
        }
        assert!(
            restarted,
            "next reply must start without waiting for the stalled tts"
        );
    }

    #[tokio::test]
    async fn session_history_is_capped_to_the_last_messages() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(ScriptedLlm {
                reply: "hi".into(),
                calls: calls.clone(),
            }),
            tts: Arc::new(ScriptedTts),
            vad: Arc::new(ScriptedVadFactory {
                events: (0..7)
                    .flat_map(|k| {
                        vec![
                            VadEvent::SpeechStart { at_ms: k * 100 },
                            VadEvent::SpeechEnd {
                                at_ms: k * 100 + 50,
                            },
                        ]
                    })
                    .collect(),
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        for turn in 1..=7 {
            tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
            wait_for_calls(&calls, turn).await;
            // Wait for the turn to complete so its assistant item is in the
            // memory before the next turn's audio can barge in.
            loop {
                match output.next().await {
                    Some(AgentOutput::TtsStop) => break,
                    Some(_) => {}
                    None => panic!("agent output ended before turn {turn} finished"),
                }
            }
        }

        let calls = calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 7);
        for call in &calls {
            assert!(
                call.len() <= 10,
                "history grew past the cap: {}",
                call.len()
            );
        }
        assert_eq!(calls[5].len(), 10);
        assert_eq!(calls[6].len(), 10);
        assert!(matches!(calls[6].first(), Some(ChatItem::Assistant { .. })));
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
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(ChunkedLlm {
                reply: reply.clone(),
            }),
            tts: Arc::new(SlowTts {
                delay: Duration::from_millis(1),
            }),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut sentence = None;
        while let Some(item) = tokio::time::timeout_at(deadline, output.next())
            .await
            .ok()
            .flatten()
        {
            if let AgentOutput::TtsSentence { text } = item {
                sentence = Some(text);
                break;
            }
        }
        assert_eq!(sentence.as_deref(), Some(reply.as_str()));
    }

    #[tokio::test]
    async fn slow_asr_receives_every_audio_chunk() {
        let chunks = Arc::new(Mutex::new(Vec::new()));
        let agent = CompositeAgent {
            asr: Arc::new(SlowAsr {
                delay: Duration::from_millis(1),
                chunks: chunks.clone(),
            }),
            llm: Arc::new(StubLlm::default()),
            tts: Arc::new(StubTts),
            vad: Arc::new(StreamVadFactory),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        for _ in 0..80 {
            tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
        }
        tx.send(AgentInput::ListenStop).await.unwrap();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let item = tokio::time::timeout_at(deadline, output.next())
                .await
                .ok()
                .flatten();
            match item {
                Some(AgentOutput::Stt { is_final: true, .. }) => break,
                Some(_) => continue,
                None => panic!("asr never produced a final transcription"),
            }
        }
        assert_eq!(chunks.lock().unwrap().len(), 80);
    }

    async fn wait_for_calls(
        calls: &Arc<Mutex<Vec<Vec<ChatItem>>>>,
        len: usize,
    ) -> Vec<Vec<ChatItem>> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = calls.lock().unwrap().clone();
            if snapshot.len() >= len {
                return snapshot;
            }
            assert!(tokio::time::Instant::now() < deadline, "llm not called");
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn llm_receives_history_and_reply_accumulates() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: calls.clone(),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                    VadEvent::SpeechStart { at_ms: 200 },
                    VadEvent::SpeechEnd { at_ms: 300 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let _output = agent.run(session(), input);

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
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            reply: "🙂你好呀".into(),
            calls: calls.clone(),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(ScriptedTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let mut seen = Vec::new();
        while let Some(item) = output.next().await {
            let done = matches!(item, AgentOutput::TtsStop);
            seen.push(item);
            if done {
                break;
            }
        }

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
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            reply: "🦄你好".into(),
            calls: calls.clone(),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(ScriptedTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let mut seen = Vec::new();
        while let Some(item) = output.next().await {
            let done = matches!(item, AgentOutput::TtsStop);
            seen.push(item);
            if done {
                break;
            }
        }

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
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            reply: "🙂你好".into(),
            calls: calls.clone(),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                    VadEvent::SpeechStart { at_ms: 200 },
                    VadEvent::SpeechEnd { at_ms: 300 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let _output = agent.run(session(), input);

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
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: calls.clone(),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::new("Be a helpful assistant."),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let _output = agent.run(session(), input);

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
        let llm: Arc<dyn Llm> = Arc::new(ToolCallingLlm {
            calls: calls.clone(),
            tool_calls: vec![ToolCall {
                id: "call_x".into(),
                name: "nonexistent".into(),
                arguments: "{}".into(),
            }],
            reply: "ok".into(),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let _output = agent.run(session(), input);

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
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn Llm> = Arc::new(ToolCallingLlm {
            calls: calls.clone(),
            tool_calls: vec![ToolCall {
                id: "call_x".into(),
                name: "nonexistent".into(),
                arguments: "{}".into(),
            }],
            reply: "🙂搞定".into(),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(ScriptedTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let mut seen = Vec::new();
        while let Some(item) = output.next().await {
            let done = matches!(item, AgentOutput::TtsStop);
            seen.push(item);
            if done {
                break;
            }
        }

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
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn Llm> = Arc::new(ToolCallingLlm {
            calls: calls.clone(),
            tool_calls: vec![ToolCall {
                id: "call_x".into(),
                name: "nonexistent".into(),
                arguments: "{}".into(),
            }],
            reply: "搞定了".into(),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(ScriptedTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let mut seen = Vec::new();
        while let Some(item) = output.next().await {
            let done = matches!(item, AgentOutput::TtsStop);
            seen.push(item);
            if done {
                break;
            }
        }

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
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm,
            tts: Arc::new(ScriptedTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let mut seen = Vec::new();
        while let Some(item) = output.next().await {
            let done = matches!(item, AgentOutput::TtsStop);
            seen.push(item);
            if done {
                break;
            }
        }

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
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(HangingLlm),
            tts: Arc::new(CancelAwareTts {
                cancelled: cancelled.clone(),
            }),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: None,
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        loop {
            let item = output.next().await;
            if matches!(item, Some(AgentOutput::Audio(_))) {
                break;
            }
        }

        tx.send(AgentInput::Interrupt { reason: None })
            .await
            .unwrap();
        let item = output.next().await;
        assert!(matches!(item, Some(AgentOutput::TtsAbort)), "got {item:?}");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while !*cancelled.lock().unwrap() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "tts was not cancelled"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// In-memory transcript sink: pairs each finished utterance's chunks with
    /// its transcript text and records every logged item batch.
    struct FakeTranscript {
        utterances: StoredUtterances,
        logged: LoggedItems,
        replies: StoredReplies,
    }

    /// How a fake reply capture ended, mirroring the collector's outcomes.
    #[derive(Debug, Clone, PartialEq)]
    enum FakeReply {
        Attached {
            chunks: Vec<Vec<f32>>,
            message_id: i64,
        },
        Unlinked,
        Partial {
            chunks: Vec<Vec<f32>>,
            text: String,
        },
        Discarded,
    }

    impl crate::agent::TranscriptSink for FakeTranscript {
        fn start_utterance(&self, _session: &AgentSession) -> crate::agent::UtteranceRecording {
            let (recording, mut rx) = crate::agent::UtteranceRecording::channel();
            let utterances = self.utterances.clone();
            tokio::spawn(async move {
                if let Some((chunks, text)) = crate::agent::transcript::collect(&mut rx).await {
                    utterances.lock().unwrap().push((chunks, text));
                }
            });
            recording
        }

        fn log_items(
            &self,
            _session: &AgentSession,
            items: Vec<ChatItem>,
        ) -> crate::agent::ReplyReceipt {
            self.logged.lock().unwrap().push(items.clone());
            let has_reply = items.iter().any(|item| {
                matches!(item, ChatItem::Assistant { content: Some(_), tool_calls } if tool_calls.is_empty())
            });
            crate::agent::ReplyReceipt::ready(has_reply.then_some(42))
        }

        fn start_reply(&self, _session: &AgentSession) -> crate::agent::ReplyRecording {
            let (recording, mut rx) = crate::agent::ReplyRecording::channel();
            let replies = self.replies.clone();
            tokio::spawn(async move {
                let Some((chunks, finish)) = crate::agent::transcript::collect_reply(&mut rx).await
                else {
                    replies.lock().unwrap().push(FakeReply::Discarded);
                    return;
                };
                let reply = match finish {
                    crate::agent::ReplyFinish::Receipt(receipt) => match receipt.resolve().await {
                        Some(message_id) => FakeReply::Attached { chunks, message_id },
                        None => FakeReply::Unlinked,
                    },
                    crate::agent::ReplyFinish::Partial(text) => FakeReply::Partial { chunks, text },
                };
                replies.lock().unwrap().push(reply);
            });
            recording
        }
    }

    type StoredUtterances = Arc<Mutex<Vec<(Vec<Vec<f32>>, String)>>>;
    type LoggedItems = Arc<Mutex<Vec<Vec<ChatItem>>>>;
    type StoredReplies = Arc<Mutex<Vec<FakeReply>>>;

    fn fake_transcript() -> (
        Arc<FakeTranscript>,
        StoredUtterances,
        LoggedItems,
        StoredReplies,
    ) {
        let utterances = Arc::new(Mutex::new(Vec::new()));
        let logged = Arc::new(Mutex::new(Vec::new()));
        let replies = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::new(FakeTranscript {
            utterances: utterances.clone(),
            logged: logged.clone(),
            replies: replies.clone(),
        });
        (sink, utterances, logged, replies)
    }

    async fn wait_for_replies(replies: &StoredReplies, len: usize) -> Vec<FakeReply> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = replies.lock().unwrap().clone();
            if snapshot.len() >= len {
                return snapshot;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "replies not stored: {snapshot:?}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn wait_for_utterances(
        utterances: &StoredUtterances,
        len: usize,
    ) -> Vec<(Vec<Vec<f32>>, String)> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = utterances.lock().unwrap().clone();
            if snapshot.len() >= len {
                return snapshot;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "utterances not stored: {snapshot:?}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn finalized_utterance_is_stored_with_transcript_and_audio() {
        let (transcript, utterances, _logged, _replies) = fake_transcript();
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(StubLlm::default()),
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::Speech {
                        samples: vec![0.5; 960],
                    },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: Some(transcript),
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        while let Some(item) = output.next().await {
            if matches!(item, AgentOutput::Stt { is_final: true, .. }) {
                break;
            }
        }

        let stored = wait_for_utterances(&utterances, 1).await;
        assert_eq!(stored[0].1, "hello");
        assert_eq!(stored[0].0, vec![vec![0.5; 960]]);
    }

    #[tokio::test]
    async fn interrupted_utterance_is_discarded() {
        let (transcript, utterances, logged, _replies) = fake_transcript();
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(StubLlm::default()),
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::Speech {
                        samples: vec![0.5; 960],
                    },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: Some(transcript),
        };
        let (tx, input) = input_channel();
        let _output = agent.run(session(), input);

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
        assert!(logged.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn empty_transcript_stores_nothing() {
        let (transcript, utterances, logged, _replies) = fake_transcript();
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("")),
            llm: Arc::new(StubLlm::default()),
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: Some(transcript),
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        // StubAsr still emits its (empty) final; the driver answers nothing.
        while let Some(item) = output.next().await {
            if matches!(item, AgentOutput::Stt { is_final: true, .. }) {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(utterances.lock().unwrap().is_empty());
        assert!(logged.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn completed_turn_logs_reply_and_tool_items() {
        let (transcript, _utterances, logged, _replies) = fake_transcript();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(ToolCallingLlm {
                calls: calls.clone(),
                tool_calls: vec![ToolCall {
                    id: "call_x".into(),
                    name: "nonexistent".into(),
                    arguments: "{}".into(),
                }],
                reply: "ok".into(),
            }),
            tts: Arc::new(StubTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: Some(transcript),
        };
        let (tx, input) = input_channel();
        let _output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();
        wait_for_calls(&calls, 2).await;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let logged = loop {
            let snapshot = logged.lock().unwrap().clone();
            if let Some(items) = snapshot.first() {
                break items.clone();
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "turn items never logged"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        };
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
    async fn completed_reply_audio_is_stored_with_receipt() {
        let (transcript, _utterances, _logged, replies) = fake_transcript();
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(ScriptedLlm {
                reply: "hi".into(),
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
            tts: Arc::new(ScriptedTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: Some(transcript),
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        // The natural tts stop finishes the capture with the receipt.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while let Some(item) = tokio::time::timeout_at(deadline, output.next())
            .await
            .ok()
            .flatten()
        {
            if matches!(item, AgentOutput::TtsStop) {
                break;
            }
        }

        let stored = wait_for_replies(&replies, 1).await;
        assert_eq!(
            stored,
            vec![FakeReply::Attached {
                chunks: vec![vec![0.25; 960]],
                message_id: 42,
            }]
        );
    }

    #[tokio::test]
    async fn aborted_reply_after_done_attaches_partial_audio_to_the_reply_row() {
        let (transcript, _utterances, _logged, replies) = fake_transcript();
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(ScriptedLlm {
                reply: "hi".into(),
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
            tts: Arc::new(HeldTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                    VadEvent::SpeechStart { at_ms: 200 },
                    VadEvent::SpeechEnd { at_ms: 300 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: Some(transcript),
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        // Wait until the reply is playing and its turn has completed (the
        // receipt is in the driver's hands), then start the next utterance:
        // the barge-in must attach the partial audio to the full reply row.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while let Some(item) = tokio::time::timeout_at(deadline, output.next())
            .await
            .ok()
            .flatten()
        {
            if matches!(item, AgentOutput::Audio(_)) {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let stored = wait_for_replies(&replies, 1).await;
        assert_eq!(
            stored,
            vec![FakeReply::Attached {
                chunks: vec![vec![0.25; 960]],
                message_id: 42,
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
        let (transcript, _utterances, logged, replies) = fake_transcript();
        let agent = CompositeAgent {
            asr: Arc::new(StubAsr::new("hello")),
            llm: Arc::new(FailingLlm { reply: "hi".into() }),
            tts: Arc::new(HeldTts),
            vad: Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
            memory: Arc::new(InMemMemory::default()),
            tools: empty_tools(),
            system_prompt: SystemPrompt::default(),
            transcript: Some(transcript),
        };
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        // The llm error aborts the reply mid-flight; the partial audio goes
        // to the collector together with the partially streamed text.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while let Some(item) = tokio::time::timeout_at(deadline, output.next())
            .await
            .ok()
            .flatten()
        {
            if matches!(item, AgentOutput::Error { .. }) {
                break;
            }
        }

        let stored = wait_for_replies(&replies, 1).await;
        assert_eq!(
            stored,
            vec![FakeReply::Partial {
                chunks: vec![vec![0.25; 960]],
                text: "hi".into(),
            }]
        );
        assert!(logged.lock().unwrap().is_empty());
    }
}
