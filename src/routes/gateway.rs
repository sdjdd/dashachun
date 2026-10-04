use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use crate::audio::OpusDecoder;
use crate::dto::ws::{
    Abort, ClientHello, InboundMessage, Listen, Mcp, ServerHello, SttMessage, TtsMessage,
};
use crate::state::AppState;

pub fn routes() -> Router<AppState> {
    Router::new().route("/gateway", get(handle_device_connect))
}

async fn handle_device_connect(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_device_socket(socket, state))
}

async fn handle_device_socket(socket: WebSocket, _state: AppState) {
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

    let mut session = Session::new(tx.clone());
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
            Message::Text(txt) => match serde_json::from_str::<InboundMessage>(txt.as_str()) {
                Ok(message) => session.handle(message).await,
                Err(err) => warn!(%err, "invalid message"),
            },
            Message::Binary(data) => session.handle_binary(&data),
            _ => {}
        }
    }

    session.shutdown();
    drop(session);
    drop(tx);
    let _ = writer.await;
    info!("device disconnected");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    Idle,
    Listening,
    Speaking,
}

struct Session {
    id: Option<String>,
    state: SessionState,
    tx: mpsc::Sender<Message>,
    pipeline: Option<JoinHandle<()>>,
    decoder: Option<OpusDecoder>,
}

impl Session {
    fn new(tx: mpsc::Sender<Message>) -> Self {
        Self {
            id: None,
            state: SessionState::Idle,
            tx,
            pipeline: None,
            decoder: None,
        }
    }

    async fn handle(&mut self, message: InboundMessage) {
        match message {
            InboundMessage::Hello(hello) => self.on_hello(hello).await,
            InboundMessage::Listen(listen) => self.on_listen(listen).await,
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
        self.decoder =
            match OpusDecoder::new(audio_params.sample_rate, audio_params.channels as u16) {
                Ok(decoder) => Some(decoder),
                Err(err) => {
                    warn!(%err, "failed to create opus decoder");
                    None
                }
            };
        send_json(&self.tx, &ServerHello::new(id.clone(), audio_params)).await;
        self.id = Some(id);
    }

    async fn on_listen(&mut self, listen: Listen) {
        let Some(session_id) = self.id.clone() else {
            warn!("listen message before hello, ignoring");
            return;
        };

        match listen.state.as_str() {
            "start" => {
                self.stop_pipeline();
                self.state = SessionState::Listening;
                info!(session_id, mode = ?listen.mode, "device started listening");
            }
            "detect" => {
                info!(session_id, text = ?listen.text, "wake word detected");
            }
            "stop" => {
                info!(session_id, "device stopped listening");
                self.start_pipeline(session_id);
            }
            other => warn!(%other, "unknown listen state"),
        }
    }

    fn on_abort(&mut self, abort: Abort) {
        self.stop_pipeline();
        self.state = SessionState::Idle;
        info!(session_id = ?self.id, reason = ?abort.reason, "abort requested");
    }

    fn on_mcp(&self, mcp: Mcp) {
        debug!(session_id = ?self.id, payload = %mcp.payload, "mcp message");
    }

    fn handle_binary(&mut self, data: &[u8]) {
        if self.state != SessionState::Listening {
            let state = self.state;
            debug!(len = data.len(), ?state, "unexpected binary frame");
            return;
        }
        let Some(decoder) = self.decoder.as_mut() else {
            debug!(len = data.len(), "binary frame before decoder ready");
            return;
        };
        match decoder.decode(data) {
            Ok(samples) => trace!(samples = samples.len(), "decoded opus frame"),
            Err(err) => warn!(%err, "failed to decode opus frame"),
        }
    }

    fn start_pipeline(&mut self, session_id: String) {
        self.stop_pipeline();
        self.state = SessionState::Speaking;
        let tx = self.tx.clone();
        self.pipeline = Some(tokio::spawn(async move {
            run_stub_pipeline(&session_id, &tx).await;
        }));
    }

    fn stop_pipeline(&mut self) {
        if let Some(handle) = self.pipeline.take() {
            handle.abort();
        }
    }

    fn shutdown(&mut self) {
        self.stop_pipeline();
    }
}

async fn run_stub_pipeline(session_id: &str, tx: &mpsc::Sender<Message>) {
    info!(session_id, "stub pipeline: stt -> tts start/stop");
    let session = session_id.to_string();
    send_json(
        tx,
        &SttMessage::new(session_id.to_string(), "你好".to_string()),
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    send_json(tx, &TtsMessage::start(session.clone())).await;
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    send_json(
        tx,
        &TtsMessage::sentence_start(session.clone(), "你好".to_string()),
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    send_json(tx, &TtsMessage::stop(session)).await;
}

async fn send_json<T: serde::Serialize>(tx: &mpsc::Sender<Message>, value: &T) {
    match serde_json::to_string(value) {
        Ok(payload) => {
            let _ = tx.send(Message::text(payload)).await;
        }
        Err(err) => error!(%err, "failed to serialize message"),
    }
}
