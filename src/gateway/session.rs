use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::Message;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::agent::{Agent, AgentInput, AgentOutputStream, AgentSession};
use crate::audio::{OpusDecoder, OpusEncoder};
use crate::dto::ws::{
    Abort, AudioParams, ClientHello, InboundMessage, Listen, LlmMessage, Mcp, ServerHello,
    SttMessage, TtsMessage,
};

use super::player::{Player, spawn_player};
use super::send_json;

const AGENT_CHANNEL_CAPACITY: usize = 64;

pub(super) struct Session {
    id: Option<String>,
    tx: mpsc::Sender<Message>,
    decoder: Option<OpusDecoder>,
    agent: Arc<dyn Agent>,
    playback_prebuffer_ms: u32,
    agent_tx: Option<mpsc::Sender<AgentInput>>,
    agent_task: Option<JoinHandle<()>>,
    player: Option<Player>,
}

impl Session {
    pub(super) fn new(
        tx: mpsc::Sender<Message>,
        agent: Arc<dyn Agent>,
        playback_prebuffer_ms: u32,
    ) -> Self {
        Self {
            id: None,
            tx,
            decoder: None,
            agent,
            playback_prebuffer_ms,
            agent_tx: None,
            agent_task: None,
            player: None,
        }
    }

    pub(super) async fn handle_text(&mut self, text: &str) {
        match serde_json::from_str::<InboundMessage>(text) {
            Ok(message) => self.handle(message).await,
            Err(err) => warn!(%err, %text, "invalid message"),
        }
    }

    async fn handle(&mut self, message: InboundMessage) {
        match message {
            InboundMessage::Hello(hello) => self.on_hello(hello).await,
            InboundMessage::Listen(listen) => self.on_listen(listen),
            InboundMessage::Abort(abort) => self.on_abort(abort).await,
            InboundMessage::Mcp(mcp) => self.on_mcp(mcp),
        }
    }

    async fn on_hello(&mut self, hello: ClientHello) {
        if self.id.is_some() {
            warn!("duplicate hello, ignoring");
            return;
        }

        let id = Uuid::new_v4().to_string();
        let audio_params = hello.audio_params.unwrap_or_default();
        info!(
            session_id = %id,
            version = hello.version,
            transport = %hello.transport,
            mcp = hello.features.mcp,
            aec = hello.features.aec,
            format = %audio_params.format,
            sample_rate = audio_params.sample_rate,
            channels = audio_params.channels,
            frame_duration = audio_params.frame_duration,
            "device hello"
        );
        if hello.version != 1 {
            warn!(
                version = hello.version,
                "only protocol v1 raw opus frames are supported; audio will be garbled"
            );
        }
        self.decoder =
            match OpusDecoder::new(audio_params.sample_rate, audio_params.channels as u16) {
                Ok(decoder) => Some(decoder),
                Err(err) => {
                    warn!(%err, "failed to create opus decoder");
                    None
                }
            };
        send_json(
            &self.tx,
            &ServerHello::new(id.clone(), audio_params.clone()),
        )
        .await;
        self.id = Some(id.clone());
        self.start_agent(id, audio_params);
    }

    fn start_agent(&mut self, session_id: String, audio_params: AudioParams) {
        let (agent_tx, agent_rx) = mpsc::channel::<AgentInput>(AGENT_CHANNEL_CAPACITY);
        let input: crate::agent::AgentInputStream = Box::pin(futures_util::stream::unfold(
            agent_rx,
            |mut rx| async move { rx.recv().await.map(|item| (item, rx)) },
        ));
        let session = AgentSession {
            id: session_id,
            sample_rate: audio_params.sample_rate,
            channels: audio_params.channels as u16,
            frame_duration_ms: audio_params.frame_duration,
        };
        let output = self.agent.run(session, input);
        let tx = self.tx.clone();
        let session_id = self.id.clone().unwrap_or_default();
        let encoder = OpusEncoder::new(
            audio_params.sample_rate,
            audio_params.channels as u16,
            audio_params.frame_duration,
        )
        .map_err(|err| warn!(%err, "failed to create opus encoder"))
        .ok();
        let player = spawn_player(
            tx.clone(),
            audio_params.frame_duration,
            self.playback_prebuffer_ms,
            session_id.clone(),
        );
        self.player = Some(player.clone());
        self.agent_task = Some(tokio::spawn(run_agent(
            output, tx, session_id, encoder, player,
        )));
        self.agent_tx = Some(agent_tx);
    }

    fn on_listen(&mut self, listen: Listen) {
        let Some(session_id) = self.id.clone() else {
            warn!("listen message before hello, ignoring");
            return;
        };

        match listen.state.as_str() {
            "start" => {
                self.send_agent(AgentInput::ListenStart { mode: listen.mode });
            }
            "detect" => {
                info!(session_id, text = ?listen.text, "wake word detected");
            }
            "stop" => {
                self.send_agent(AgentInput::ListenStop);
            }
            other => warn!(%other, "unknown listen state"),
        }
    }

    async fn on_abort(&mut self, abort: Abort) {
        let reason = abort.reason;
        info!(session_id = ?self.id, reason = ?reason, "abort requested");
        if let Some(player) = self.player.as_ref() {
            player.abort().await;
        }
        self.send_agent(AgentInput::Interrupt { reason });
    }

    fn on_mcp(&mut self, mcp: Mcp) {
        debug!(session_id = ?self.id, payload = %mcp.payload, "mcp message");
        self.send_agent(AgentInput::Mcp(mcp.payload));
    }

    pub(super) fn handle_binary(&mut self, data: &[u8]) {
        let Some(decoder) = self.decoder.as_mut() else {
            debug!(len = data.len(), "binary frame before decoder ready");
            return;
        };
        let samples = match decoder.decode(data) {
            Ok(samples) => samples.to_vec(),
            Err(err) => {
                warn!(%err, len = data.len(), "failed to decode opus frame");
                return;
            }
        };
        tracing::trace!(samples = samples.len(), "decoded opus frame");
        self.send_agent(AgentInput::Audio(samples));
    }

    fn send_agent(&self, input: AgentInput) {
        if let Some(tx) = self.agent_tx.as_ref()
            && tx.try_send(input).is_err()
        {
            tracing::trace!("agent input full or closed, dropping message");
        }
    }

    pub(super) async fn shutdown(&mut self, grace: Duration) {
        self.agent_tx = None;
        if let Some(mut handle) = self.agent_task.take()
            && tokio::time::timeout(grace, &mut handle).await.is_err()
        {
            warn!("agent task did not stop in time, aborting");
            handle.abort();
        }
    }
}

async fn run_agent(
    mut output: AgentOutputStream,
    tx: mpsc::Sender<Message>,
    session_id: String,
    mut encoder: Option<OpusEncoder>,
    player: Player,
) {
    use crate::agent::AgentOutput;
    use futures_util::StreamExt;
    while let Some(item) = output.next().await {
        match item {
            AgentOutput::Stt { text, is_final } => {
                if is_final {
                    debug!(%text, "stt final");
                }
                send_json(&tx, &SttMessage::new(session_id.clone(), text)).await;
            }
            AgentOutput::TtsStart => {
                send_json(&tx, &TtsMessage::start(session_id.clone())).await;
                player.start().await;
            }
            AgentOutput::TtsSentence { text } => {
                send_json(&tx, &TtsMessage::sentence_start(session_id.clone(), text)).await;
            }
            AgentOutput::TtsSubtitle { subtitle } => {
                tracing::trace!(
                    text = %subtitle.text,
                    start_ms = subtitle.start_ms,
                    end_ms = subtitle.end_ms,
                    "tts subtitle"
                );
                player.subtitle(subtitle.start_ms, subtitle.text).await;
            }
            AgentOutput::TtsStop => {
                if let Some(encoder) = encoder.as_mut()
                    && let Some(packet) = encoder.flush()
                {
                    player.push(packet).await;
                }
                player.finish().await;
                send_json(&tx, &TtsMessage::stop(session_id.clone())).await;
            }
            AgentOutput::TtsAbort => {
                if let Some(encoder) = encoder.as_mut() {
                    encoder.reset();
                }
                player.abort().await;
                send_json(&tx, &TtsMessage::stop(session_id.clone())).await;
            }
            AgentOutput::Audio(samples) => {
                if let Some(encoder) = encoder.as_mut() {
                    for packet in encoder.push(&samples) {
                        player.push(packet).await;
                    }
                }
            }
            AgentOutput::Emotion { emotion } => {
                send_json(&tx, &LlmMessage::new(session_id.clone(), emotion)).await;
            }
            AgentOutput::Mcp(_) => {}
            AgentOutput::Error { message } => {
                warn!(%message, "agent error");
            }
        }
    }
}
