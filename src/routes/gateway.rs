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

use crate::asr::{Asr, AsrEvent, AudioStream};
use crate::audio::OpusDecoder;
use crate::dto::ws::{
    Abort, AudioParams, ClientHello, InboundMessage, Listen, Mcp, ServerHello, SttMessage,
    TtsMessage,
};
use crate::state::AppState;
use crate::vad::{Vad, VadConfig, VadEvent, VadEvents};

const ASR_CHANNEL_CAPACITY: usize = 64;

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

    let mut session = Session::new(tx.clone(), state.asr);
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
    decoder: Option<OpusDecoder>,
    audio_params: AudioParams,
    asr: Arc<dyn Asr>,
    vad_task: Option<JoinHandle<()>>,
    audio_tx: Option<mpsc::Sender<Vec<f32>>>,
}

impl Session {
    fn new(tx: mpsc::Sender<Message>, asr: Arc<dyn Asr>) -> Self {
        Self {
            id: None,
            state: SessionState::Idle,
            tx,
            decoder: None,
            audio_params: AudioParams::default(),
            asr,
            vad_task: None,
            audio_tx: None,
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
        let vad_config = VadConfig::from_env();
        info!(
            session_id = %id,
            decoder = self.decoder.is_some(),
            speech_threshold = vad_config.speech_threshold,
            silence_threshold = vad_config.silence_threshold,
            min_speech_ms = vad_config.min_speech_ms,
            min_silence_ms = vad_config.min_silence_ms,
            pre_padding_ms = vad_config.pre_padding_ms,
            "audio pipeline ready"
        );
        send_json(
            &self.tx,
            &ServerHello::new(id.clone(), audio_params.clone()),
        )
        .await;
        self.audio_params = audio_params;
        self.id = Some(id);
    }

    async fn on_listen(&mut self, listen: Listen) {
        let Some(session_id) = self.id.clone() else {
            warn!("listen message before hello, ignoring");
            return;
        };

        match listen.state.as_str() {
            "start" => {
                self.start_listening(session_id, listen.mode);
            }
            "detect" => {
                info!(session_id, text = ?listen.text, "wake word detected");
            }
            "stop" => {
                self.stop_listening();
            }
            other => warn!(%other, "unknown listen state"),
        }
    }

    fn on_abort(&mut self, abort: Abort) {
        self.stop_vad();
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
        let Ok(samples) = decoder.decode(data) else {
            warn!(len = data.len(), "failed to decode opus frame");
            return;
        };
        trace!(samples = samples.len(), "decoded opus frame");

        if let Some(tx) = self.audio_tx.as_ref()
            && tx.try_send(samples.to_vec()).is_err()
        {
            trace!("vad input full, dropping frame");
        }
    }

    fn start_listening(&mut self, session_id: String, mode: Option<String>) {
        self.stop_vad();
        if self.decoder.is_none() {
            self.state = SessionState::Listening;
            info!(session_id, mode = ?mode, "device started listening");
            return;
        }
        match Vad::new(self.audio_params.sample_rate, VadConfig::from_env()) {
            Ok(vad) => {
                let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(ASR_CHANNEL_CAPACITY);
                let audio: AudioStream = Box::pin(futures_util::stream::unfold(
                    audio_rx,
                    |mut rx| async move { rx.recv().await.map(|chunk| (chunk, rx)) },
                ));
                let events = vad.segment(audio);
                let tx = self.tx.clone();
                let asr = self.asr.clone();
                self.vad_task = Some(tokio::spawn(run_vad(session_id.clone(), tx, asr, events)));
                self.audio_tx = Some(audio_tx);
                self.state = SessionState::Listening;
                info!(session_id, mode = ?mode, "device started listening");
            }
            Err(err) => {
                warn!(%err, "failed to create vad");
                self.state = SessionState::Listening;
            }
        }
    }

    fn stop_listening(&mut self) {
        self.audio_tx = None;
        self.state = SessionState::Speaking;
    }

    fn stop_vad(&mut self) {
        self.audio_tx = None;
        if let Some(handle) = self.vad_task.take() {
            handle.abort();
        }
    }

    fn shutdown(&mut self) {
        self.stop_vad();
    }
}

async fn run_vad(
    session_id: String,
    tx: mpsc::Sender<Message>,
    asr: Arc<dyn Asr>,
    mut events: VadEvents,
) {
    let mut utterance: Option<(mpsc::Sender<Vec<f32>>, UtteranceTask)> = None;

    while let Some(event) = events.next().await {
        match event {
            VadEvent::SpeechStart { at_ms } => {
                info!(session_id, at_ms, "speech start");
                let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(ASR_CHANNEL_CAPACITY);
                let audio: AudioStream = Box::pin(futures_util::stream::unfold(
                    audio_rx,
                    |mut rx| async move { rx.recv().await.map(|chunk| (chunk, rx)) },
                ));
                let handle =
                    tokio::spawn(run_asr(session_id.clone(), tx.clone(), asr.clone(), audio));
                utterance = Some((audio_tx, UtteranceTask(Some(handle))));
            }
            VadEvent::Speech { samples } => {
                if let Some((tx, _)) = utterance.as_ref()
                    && tx.try_send(samples).is_err()
                {
                    trace!("asr input full, dropping frame");
                }
            }
            VadEvent::SpeechEnd { at_ms } => {
                info!(session_id, at_ms, "speech end");
                if let Some((_, task)) = utterance.take() {
                    task.detach();
                }
            }
        }
    }
}

struct UtteranceTask(Option<JoinHandle<()>>);

impl UtteranceTask {
    fn detach(mut self) {
        if let Some(handle) = self.0.take() {
            drop(handle);
        }
    }
}

impl Drop for UtteranceTask {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

async fn run_asr(
    session_id: String,
    tx: mpsc::Sender<Message>,
    asr: Arc<dyn Asr>,
    audio: AudioStream,
) {
    let mut events = asr.transcribe(audio);
    while let Some(result) = events.next().await {
        match result {
            Ok(AsrEvent::Partial { text }) => {
                debug!(session_id, %text, "asr partial");
            }
            Ok(AsrEvent::Final { text }) => {
                info!(session_id, %text, "asr result");
                if !text.is_empty() {
                    run_response(&session_id, &tx, text).await;
                }
            }
            Err(err) => {
                warn!(session_id, %err, "asr failed");
                return;
            }
        }
    }
}

async fn run_response(session_id: &str, tx: &mpsc::Sender<Message>, text: String) {
    let session = session_id.to_string();
    send_json(tx, &SttMessage::new(session.clone(), text.clone())).await;
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    send_json(tx, &TtsMessage::start(session.clone())).await;
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    send_json(tx, &TtsMessage::sentence_start(session.clone(), text)).await;
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
