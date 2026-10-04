use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use crate::agent::{Agent, AgentInput, AgentOutputStream, AgentSession};
use crate::audio::OpusDecoder;
use crate::dto::ws::{
    Abort, AudioParams, ClientHello, InboundMessage, Listen, Mcp, ServerHello, SttMessage,
    TtsMessage,
};
use crate::state::AppState;

const AGENT_CHANNEL_CAPACITY: usize = 64;

pub fn routes() -> Router<AppState> {
    Router::new().route("/gateway", get(handle_device_connect))
}

async fn handle_device_connect(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    info!(
        device_id = header_str(&headers, "device-id"),
        client_id = header_str(&headers, "client-id"),
        user_agent = header_str(&headers, "user-agent"),
        "gateway upgrade"
    );
    ws.on_upgrade(move |socket| handle_device_socket(socket, state))
}

fn header_str<'a>(headers: &'a HeaderMap, name: &'static str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

async fn handle_device_socket(socket: WebSocket, state: AppState) {
    info!("device connected");
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<Message>(32);

    let writer = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if SinkExt::send(&mut sink, msg).await.is_err() {
                break;
            }
        }
    });

    let mut session = Session::new(tx.clone(), state.agent);
    while let Some(result) = stream.next().await {
        let msg = match result {
            Ok(Message::Close(_)) => break,
            Ok(msg) => msg,
            Err(err) => {
                debug!(%err, "websocket stream ended");
                break;
            }
        };

        match msg {
            Message::Text(txt) => {
                trace!(%txt, "inbound text");
                match serde_json::from_str::<InboundMessage>(txt.as_str()) {
                    Ok(message) => session.handle(message).await,
                    Err(err) => warn!(%err, %txt, "invalid message"),
                }
            }
            Message::Binary(data) => {
                trace!(len = data.len(), "inbound binary");
                session.handle_binary(&data);
            }
            _ => {}
        }
    }

    session.shutdown();
    drop(session);
    drop(tx);
    let _ = writer.await;
    info!("device disconnected");
}

struct Session {
    id: Option<String>,
    tx: mpsc::Sender<Message>,
    decoder: Option<OpusDecoder>,
    agent: Arc<dyn Agent>,
    agent_tx: Option<mpsc::Sender<AgentInput>>,
    agent_task: Option<JoinHandle<()>>,
}

impl Session {
    fn new(tx: mpsc::Sender<Message>, agent: Arc<dyn Agent>) -> Self {
        Self {
            id: None,
            tx,
            decoder: None,
            agent,
            agent_tx: None,
            agent_task: None,
        }
    }

    async fn handle(&mut self, message: InboundMessage) {
        match message {
            InboundMessage::Hello(hello) => self.on_hello(hello).await,
            InboundMessage::Listen(listen) => self.on_listen(listen),
            InboundMessage::Abort(abort) => self.on_abort(abort),
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
        self.agent_task = Some(tokio::spawn(run_agent(output, tx, session_id)));
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
                info!(session_id, "device started listening");
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

    fn on_abort(&mut self, abort: Abort) {
        let reason = abort.reason;
        info!(session_id = ?self.id, reason = ?reason, "abort requested");
        self.send_agent(AgentInput::Interrupt { reason });
    }

    fn on_mcp(&mut self, mcp: Mcp) {
        debug!(session_id = ?self.id, payload = %mcp.payload, "mcp message");
        self.send_agent(AgentInput::Mcp(mcp.payload));
    }

    fn handle_binary(&mut self, data: &[u8]) {
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
        trace!(samples = samples.len(), "decoded opus frame");
        self.send_agent(AgentInput::Audio(samples));
    }

    fn send_agent(&self, input: AgentInput) {
        if let Some(tx) = self.agent_tx.as_ref()
            && tx.try_send(input).is_err()
        {
            trace!("agent input full or closed, dropping message");
        }
    }

    fn shutdown(&mut self) {
        self.agent_tx = None;
        if let Some(handle) = self.agent_task.take() {
            handle.abort();
        }
    }
}

async fn run_agent(mut output: AgentOutputStream, tx: mpsc::Sender<Message>, session_id: String) {
    use crate::agent::AgentOutput;
    while let Some(item) = output.next().await {
        match item {
            AgentOutput::Stt { text, is_final } => {
                if is_final {
                    info!(%text, "stt final");
                } else {
                    debug!(%text, "stt partial");
                }
                send_json(&tx, &SttMessage::new(session_id.clone(), text)).await;
            }
            AgentOutput::TtsStart => {
                send_json(&tx, &TtsMessage::start(session_id.clone())).await;
            }
            AgentOutput::TtsSentence { text } => {
                send_json(&tx, &TtsMessage::sentence_start(session_id.clone(), text)).await;
            }
            AgentOutput::TtsStop => {
                send_json(&tx, &TtsMessage::stop(session_id.clone())).await;
            }
            AgentOutput::Audio(samples) => {
                debug!(samples = samples.len(), "agent audio (codec not wired yet)");
            }
            AgentOutput::Mcp(payload) => {
                debug!(%payload, "agent mcp output");
            }
            AgentOutput::Error { message } => {
                warn!(%message, "agent error");
            }
        }
    }
}

async fn send_json<T: serde::Serialize>(tx: &mpsc::Sender<Message>, value: &T) {
    match serde_json::to_string(value) {
        Ok(payload) => {
            let _ = tx.send(Message::text(payload)).await;
        }
        Err(err) => error!(%err, "failed to serialize message"),
    }
}
