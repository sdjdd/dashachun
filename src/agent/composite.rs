use std::sync::Arc;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, trace, warn};

use crate::agent::{
    Agent, AgentInput, AgentInputStream, AgentOutput, AgentOutputStream, AgentSession,
};
use crate::asr::{Asr, AsrEvent, AudioStream};
use crate::llm::Llm;
use crate::tts::Tts;
use crate::vad::{Vad, VadEvent, VadFactory};

const ASR_CHANNEL_CAPACITY: usize = 64;
const OUTPUT_CHANNEL_CAPACITY: usize = 64;

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
        let vad_factory = self.vad.clone();
        let session_id = session.id.clone();
        info!(session_id, "agent started");
        tokio::spawn(async move {
            drive(&session, input, asr, vad_factory, out_tx).await;
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

async fn drive(
    session: &AgentSession,
    mut input: AgentInputStream,
    asr: Arc<dyn Asr>,
    vad_factory: Arc<dyn VadFactory>,
    out_tx: mpsc::Sender<AgentOutput>,
) {
    let mut vad: Option<Box<dyn Vad>> = None;
    let (asr_tx, mut asr_rx) = mpsc::channel::<AsrMessage>(ASR_CHANNEL_CAPACITY);
    let mut utterance: Option<Utterance> = None;
    let mut generation: u64 = 0;

    loop {
        tokio::select! {
            item = input.next() => {
                let Some(item) = item else { break };
                match item {
                    AgentInput::ListenStart { mode } => {
                        info!(session_id = %session.id, mode = ?mode, "listen start");
                        vad = match vad_factory.build(session.sample_rate) {
                            Ok(vad) => Some(vad),
                            Err(err) => {
                                warn!(session_id = %session.id, %err, "failed to build vad");
                                None
                            }
                        };
                        generation += 1;
                        utterance = None;
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
                        generation += 1;
                        utterance = None;
                        if out_tx.send(AgentOutput::TtsStop).await.is_err() {
                            break;
                        }
                    }
                    AgentInput::Mcp(payload) => {
                        debug!(session_id = %session.id, %payload, "mcp message");
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
                        debug!(session_id = %session.id, %text, "asr partial");
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
                        if !text.is_empty() {
                            for output in response(&text) {
                                if out_tx.send(output).await.is_err() {
                                    return;
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
        }
    }

    drop(utterance.take());
}

struct Utterance {
    tx: mpsc::Sender<Vec<f32>>,
    handle: Option<JoinHandle<()>>,
}

impl Utterance {
    fn detach(mut self) {
        self.handle.take();
    }
}

impl Drop for Utterance {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
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
            VadEvent::SpeechEnd { at_ms } => {
                info!(session_id, at_ms, "speech end");
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
    let handle = tokio::spawn(async move {
        run_asr(asr, audio, asr_tx, session_id, generation).await;
    });
    Utterance {
        tx,
        handle: Some(handle),
    }
}

async fn run_asr(
    asr: Arc<dyn Asr>,
    audio: AudioStream,
    asr_tx: mpsc::Sender<AsrMessage>,
    session_id: String,
    generation: u64,
) {
    let mut events = asr.transcribe(audio);
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
    use crate::vad::{Vad, VadError, VadEvent};

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
        assert!(matches!(item, Some(AgentOutput::TtsStop)), "got {item:?}");
    }
}
