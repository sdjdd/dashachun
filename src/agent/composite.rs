use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, trace, warn};

use crate::agent::{
    Agent, AgentInput, AgentInputStream, AgentOutput, AgentOutputStream, AgentSession,
};
use crate::asr::{Asr, AsrEvent, AudioStream};
use crate::llm::{ChatMessage, Llm, LlmEvent};
use crate::tts::{Subtitle, TextStream, Tts, TtsEvent};
use crate::vad::{Vad, VadEvent, VadFactory};

const ASR_CHANNEL_CAPACITY: usize = 64;
const LLM_CHANNEL_CAPACITY: usize = 64;
const TTS_CHANNEL_CAPACITY: usize = 64;
const TTS_TEXT_CHANNEL_CAPACITY: usize = 64;
const OUTPUT_CHANNEL_CAPACITY: usize = 64;
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);

pub struct CompositeAgent {
    asr: Arc<dyn Asr>,
    llm: Option<Arc<dyn Llm>>,
    tts: Option<Arc<dyn Tts>>,
    vad: Arc<dyn VadFactory>,
}

impl CompositeAgent {
    pub fn new(
        asr: Arc<dyn Asr>,
        llm: Option<Arc<dyn Llm>>,
        tts: Option<Arc<dyn Tts>>,
        vad: Arc<dyn VadFactory>,
    ) -> Self {
        Self { asr, llm, tts, vad }
    }

    pub fn llm(&self) -> Option<&Arc<dyn Llm>> {
        self.llm.as_ref()
    }

    pub fn tts(&self) -> Option<&Arc<dyn Tts>> {
        self.tts.as_ref()
    }
}

impl Agent for CompositeAgent {
    fn run(&self, session: AgentSession, input: AgentInputStream) -> AgentOutputStream {
        let (out_tx, out_rx) = mpsc::channel::<AgentOutput>(OUTPUT_CHANNEL_CAPACITY);
        let asr = self.asr.clone();
        let llm = self.llm.clone();
        let tts = self.tts.clone();
        let vad_factory = self.vad.clone();
        let session_id = session.id.clone();
        info!(session_id, "agent started");
        tokio::spawn(async move {
            drive(&session, input, asr, llm, tts, vad_factory, out_tx).await;
            info!(session_id, "agent stopped");
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
    Delta { generation: u64, text: String },
    Done { generation: u64, text: String },
    Error { generation: u64, message: String },
}

enum TtsMessage {
    SentenceStart { generation: u64, text: String },
    Subtitle { generation: u64, subtitle: Subtitle },
    Audio { generation: u64, samples: Vec<f32> },
    Done { generation: u64 },
    Error { generation: u64, message: String },
}

async fn drive(
    session: &AgentSession,
    mut input: AgentInputStream,
    asr: Arc<dyn Asr>,
    llm: Option<Arc<dyn Llm>>,
    tts: Option<Arc<dyn Tts>>,
    vad_factory: Arc<dyn VadFactory>,
    out_tx: mpsc::Sender<AgentOutput>,
) {
    let mut vad: Option<Box<dyn Vad>> = None;
    let (asr_tx, mut asr_rx) = mpsc::channel::<AsrMessage>(ASR_CHANNEL_CAPACITY);
    let (llm_tx, mut llm_rx) = mpsc::channel::<LlmMessage>(LLM_CHANNEL_CAPACITY);
    let (tts_tx, mut tts_rx) = mpsc::channel::<TtsMessage>(TTS_CHANNEL_CAPACITY);
    let mut utterance: Option<Utterance> = None;
    let mut completion: Option<Completion> = None;
    let mut synthesis: Option<Synthesis> = None;
    let mut history: Vec<ChatMessage> = Vec::new();
    let mut generation: u64 = 0;

    loop {
        tokio::select! {
            item = input.next() => {
                let Some(item) = item else { break };
                match item {
                    AgentInput::ListenStart { .. } => {
                        cancel_in_flight(&mut utterance, &mut completion, &mut synthesis).await;
                        vad = match vad_factory.build(session.sample_rate) {
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
                    }
                    AgentInput::ListenStop => {
                        info!(session_id = %session.id, "listen stop");
                        if let Some(vad) = vad.as_mut() {
                            let events = vad.flush();
                            handle_vad_events(
                                events,
                                &session.id,
                                &asr,
                                &asr_tx,
                                &mut generation,
                                &mut utterance,
                            );
                        }
                        utterance = None;
                    }
                    AgentInput::Audio(samples) => {
                        let Some(vad) = vad.as_mut() else { continue };
                        let events = vad.push(&samples);
                        handle_vad_events(
                            events,
                            &session.id,
                            &asr,
                            &asr_tx,
                            &mut generation,
                            &mut utterance,
                        );
                    }
                    AgentInput::Interrupt { reason } => {
                        info!(session_id = %session.id, reason = ?reason, "interrupt");
                        cancel_in_flight(&mut utterance, &mut completion, &mut synthesis).await;
                        generation += 1;
                        utterance = None;
                        completion = None;
                        synthesis = None;
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
                        match llm.as_ref() {
                            Some(llm) => {
                                if let Some(current) = completion.take() {
                                    current.cancel().await;
                                }
                                if let Some(current) = synthesis.take() {
                                    current.cancel().await;
                                }
                                history.push(ChatMessage::user(text));
                                completion = Some(spawn_llm(
                                    llm,
                                    &llm_tx,
                                    session.id.clone(),
                                    generation,
                                    history.clone(),
                                ));
                            }
                            None => {
                                for output in response(&text) {
                                    if out_tx.send(output).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
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
                        match synthesis.as_mut() {
                            Some(current) => {
                                if current.tx.try_send(text).is_err() {
                                    trace!(session_id = %session.id, "tts input full, dropping delta");
                                }
                            }
                            None => {
                                if let Some(tts) = tts.as_ref() {
                                    if out_tx.send(AgentOutput::TtsStart).await.is_err() {
                                        break;
                                    }
                                    let current =
                                        spawn_tts(tts, &tts_tx, session.id.clone(), generation);
                                    if current.tx.try_send(text).is_err() {
                                        trace!(session_id = %session.id, "tts input full, dropping delta");
                                    }
                                    synthesis = Some(current);
                                }
                            }
                        }
                    }
                    LlmMessage::Done { text, .. } => {
                        completion = None;
                        info!(session_id = %session.id, %text, "llm result");
                        if !text.is_empty() {
                            history.push(ChatMessage::assistant(text));
                        }
                        if let Some(current) = synthesis.take() {
                            current.detach();
                        }
                    }
                    LlmMessage::Error { message, .. } => {
                        completion = None;
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
                        if out_tx.send(AgentOutput::Audio(samples)).await.is_err() {
                            break;
                        }
                    }
                    TtsMessage::Done { .. } => {
                        synthesis = None;
                        if out_tx.send(AgentOutput::TtsStop).await.is_err() {
                            break;
                        }
                    }
                    TtsMessage::Error { message, .. } => {
                        synthesis = None;
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

    cancel_in_flight(&mut utterance, &mut completion, &mut synthesis).await;
}

async fn cancel_in_flight(
    utterance: &mut Option<Utterance>,
    completion: &mut Option<Completion>,
    synthesis: &mut Option<Synthesis>,
) {
    tokio::join!(
        async {
            if let Some(current) = utterance.take() {
                current.cancel().await;
            }
        },
        async {
            if let Some(current) = completion.take() {
                current.cancel().await;
            }
        },
        async {
            if let Some(current) = synthesis.take() {
                current.cancel().await;
            }
        },
    );
}

struct Utterance {
    tx: mpsc::Sender<Vec<f32>>,
    handle: Option<JoinHandle<()>>,
    cancel: CancellationToken,
}

impl Utterance {
    fn detach(mut self) {
        self.handle.take();
    }

    async fn cancel(mut self) {
        self.cancel.cancel();
        if let Some(mut handle) = self.handle.take() {
            stop_task(&mut handle).await;
        }
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
    async fn cancel(mut self) {
        self.cancel.cancel();
        stop_task(&mut self.handle).await;
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

    async fn cancel(mut self) {
        self.cancel.cancel();
        if let Some(mut handle) = self.handle.take() {
            stop_task(&mut handle).await;
        }
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

fn handle_vad_events(
    events: Vec<VadEvent>,
    session_id: &str,
    asr: &Arc<dyn Asr>,
    asr_tx: &mpsc::Sender<AsrMessage>,
    generation: &mut u64,
    utterance: &mut Option<Utterance>,
) {
    for event in events {
        match event {
            VadEvent::SpeechStart { at_ms } => {
                info!(session_id, at_ms, "speech start");
                *generation += 1;
                *utterance = Some(spawn_asr(asr, asr_tx, session_id.to_string(), *generation));
            }
            VadEvent::Speech { samples } => {
                if let Some(current) = utterance.as_ref()
                    && current.tx.try_send(samples).is_err()
                {
                    trace!(session_id, "asr input full, dropping frame");
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
) -> Utterance {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(ASR_CHANNEL_CAPACITY);
    let audio: AudioStream = Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    }));
    let asr = asr.clone();
    let asr_tx = asr_tx.clone();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let handle = tokio::spawn(async move {
        run_asr(asr, audio, asr_tx, session_id, generation, task_cancel).await;
    });
    Utterance {
        tx,
        handle: Some(handle),
        cancel,
    }
}

async fn run_asr(
    asr: Arc<dyn Asr>,
    audio: AudioStream,
    asr_tx: mpsc::Sender<AsrMessage>,
    session_id: String,
    generation: u64,
    cancel: CancellationToken,
) {
    let mut events = asr.transcribe(audio, cancel);
    while let Some(result) = events.next().await {
        let message = match result {
            Ok(AsrEvent::Partial { text }) => AsrMessage::Partial { generation, text },
            Ok(AsrEvent::Final { text }) => {
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
    llm_tx: &mpsc::Sender<LlmMessage>,
    session_id: String,
    generation: u64,
    history: Vec<ChatMessage>,
) -> Completion {
    let llm = llm.clone();
    let llm_tx = llm_tx.clone();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let handle = tokio::spawn(async move {
        run_llm(llm, history, llm_tx, session_id, generation, task_cancel).await;
    });
    Completion { handle, cancel }
}

async fn run_llm(
    llm: Arc<dyn Llm>,
    history: Vec<ChatMessage>,
    llm_tx: mpsc::Sender<LlmMessage>,
    session_id: String,
    generation: u64,
    cancel: CancellationToken,
) {
    let mut events = llm.chat(history, cancel);
    let mut text = String::new();
    while let Some(result) = events.next().await {
        let message = match result {
            Ok(LlmEvent::Delta { text: delta }) => {
                text.push_str(&delta);
                LlmMessage::Delta {
                    generation,
                    text: delta,
                }
            }
            Ok(LlmEvent::Done) => {
                let _ = llm_tx.send(LlmMessage::Done { generation, text }).await;
                return;
            }
            Err(err) => {
                let _ = llm_tx
                    .send(LlmMessage::Error {
                        generation,
                        message: err.to_string(),
                    })
                    .await;
                return;
            }
        };
        if llm_tx.send(message).await.is_err() {
            debug!(session_id, "llm output closed");
            return;
        }
    }
    let _ = llm_tx.send(LlmMessage::Done { generation, text }).await;
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

fn response(text: &str) -> Vec<AgentOutput> {
    vec![
        AgentOutput::TtsStart,
        AgentOutput::TtsSentence {
            text: text.to_string(),
        },
        AgentOutput::TtsStop,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::StubAsr;
    use crate::llm::LlmEvents;
    use crate::tts::{Tts, TtsError, TtsEvent, TtsEvents};
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
        calls: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
    }
    impl Llm for ScriptedLlm {
        fn chat(&self, history: Vec<ChatMessage>, _cancel: CancellationToken) -> LlmEvents<'_> {
            self.calls.lock().unwrap().push(history);
            let text = self.reply.clone();
            Box::pin(futures_util::stream::iter([
                Ok(LlmEvent::Delta { text }),
                Ok(LlmEvent::Done),
            ]))
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

    #[tokio::test]
    async fn audio_drives_vad_asr_to_stt() {
        let agent = CompositeAgent::new(
            Arc::new(StubAsr::new("hello")),
            None,
            None,
            Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
        );
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
        let agent = CompositeAgent::new(
            Arc::new(StubAsr::new("hello")),
            None,
            None,
            Arc::new(ScriptedVadFactory { events: vec![] }),
        );
        let (tx, input) = input_channel();
        let mut output = agent.run(session(), input);

        tx.send(AgentInput::Interrupt { reason: None })
            .await
            .unwrap();

        let item = output.next().await;
        assert!(matches!(item, Some(AgentOutput::TtsAbort)), "got {item:?}");
    }

    async fn wait_for_calls(
        calls: &Arc<Mutex<Vec<Vec<ChatMessage>>>>,
        len: usize,
    ) -> Vec<Vec<ChatMessage>> {
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
        let agent = CompositeAgent::new(
            Arc::new(StubAsr::new("hello")),
            Some(llm),
            None,
            Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                    VadEvent::SpeechStart { at_ms: 200 },
                    VadEvent::SpeechEnd { at_ms: 300 },
                ],
            }),
        );
        let (tx, input) = input_channel();
        let _output = agent.run(session(), input);

        tx.send(AgentInput::ListenStart { mode: None })
            .await
            .unwrap();
        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let first = wait_for_calls(&calls, 1).await;
        assert_eq!(first[0], vec![ChatMessage::user("hello")]);

        tokio::time::sleep(Duration::from_millis(50)).await;

        tx.send(AgentInput::Audio(vec![0.5; 960])).await.unwrap();

        let second = wait_for_calls(&calls, 2).await;
        assert_eq!(
            second[1],
            vec![
                ChatMessage::user("hello"),
                ChatMessage::assistant("hi"),
                ChatMessage::user("hello"),
            ]
        );
    }

    #[tokio::test]
    async fn tts_streams_sentence_audio_then_stop() {
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            reply: "hi".into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let agent = CompositeAgent::new(
            Arc::new(StubAsr::new("hello")),
            Some(llm),
            Some(Arc::new(ScriptedTts)),
            Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
        );
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
        fn chat(&self, _history: Vec<ChatMessage>, cancel: CancellationToken) -> LlmEvents<'_> {
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
        let agent = CompositeAgent::new(
            Arc::new(StubAsr::new("hello")),
            Some(Arc::new(HangingLlm)),
            Some(Arc::new(CancelAwareTts {
                cancelled: cancelled.clone(),
            })),
            Arc::new(ScriptedVadFactory {
                events: vec![
                    VadEvent::SpeechStart { at_ms: 0 },
                    VadEvent::SpeechEnd { at_ms: 100 },
                ],
            }),
        );
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
        assert!(*cancelled.lock().unwrap(), "tts was not cancelled");
    }
}
