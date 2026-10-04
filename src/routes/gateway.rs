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
    Abort, ClientHello, InboundMessage, Listen, Mcp, ServerHello, SttMessage, TtsMessage,
};
use crate::state::AppState;
use crate::vad::{Vad, VadConfig, VadEvent};

const AUDIO_STATS_INTERVAL: u64 = 50;
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

#[derive(Debug, Default)]
struct AudioStats {
    frames: u64,
    samples: u64,
    peak: f32,
    rms_sum: f64,
    vad_peak: f32,
}

impl AudioStats {
    fn push(&mut self, samples: &[f32]) {
        self.frames += 1;
        self.samples += samples.len() as u64;
        for &sample in samples {
            let magnitude = sample.abs();
            if magnitude > self.peak {
                self.peak = magnitude;
            }
            self.rms_sum += (sample as f64) * (sample as f64);
        }
    }

    fn rms(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            (self.rms_sum / self.samples as f64).sqrt() as f32
        }
    }
}

struct Session {
    id: Option<String>,
    state: SessionState,
    tx: mpsc::Sender<Message>,
    asr_task: Option<JoinHandle<()>>,
    decoder: Option<OpusDecoder>,
    vad: Option<Vad>,
    audio: AudioStats,
    asr: Arc<dyn Asr>,
    asr_tx: Option<mpsc::Sender<Option<Vec<f32>>>>,
}

impl Session {
    fn new(tx: mpsc::Sender<Message>, asr: Arc<dyn Asr>) -> Self {
        Self {
            id: None,
            state: SessionState::Idle,
            tx,
            asr_task: None,
            decoder: None,
            vad: None,
            audio: AudioStats::default(),
            asr,
            asr_tx: None,
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
        self.vad = match Vad::new(audio_params.sample_rate, vad_config) {
            Ok(vad) => Some(vad),
            Err(err) => {
                warn!(%err, "failed to create vad");
                None
            }
        };
        info!(
            session_id = %id,
            decoder = self.decoder.is_some(),
            vad = self.vad.is_some(),
            speech_threshold = vad_config.speech_threshold,
            silence_threshold = vad_config.silence_threshold,
            min_speech_ms = vad_config.min_speech_ms,
            min_silence_ms = vad_config.min_silence_ms,
            "audio pipeline ready"
        );
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
        self.stop_asr();
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

        let Session {
            id,
            decoder,
            vad,
            audio,
            asr_tx,
            ..
        } = self;

        let Some(decoder) = decoder.as_mut() else {
            debug!(len = data.len(), "binary frame before decoder ready");
            return;
        };
        let Ok(samples) = decoder.decode(data) else {
            warn!(len = data.len(), "failed to decode opus frame");
            return;
        };
        trace!(samples = samples.len(), "decoded opus frame");

        audio.push(samples);
        if let Some(vad) = vad.as_mut() {
            let mut events = Vec::new();
            let probability = vad.push(samples, &mut events);
            audio.vad_peak = audio.vad_peak.max(probability);
            log_vad_events(id.as_deref(), events);
        }
        if let Some(asr_tx) = asr_tx.as_mut()
            && asr_tx.try_send(Some(samples.to_vec())).is_err()
        {
            trace!(len = samples.len(), "asr input full, dropping frame");
        }
        if audio.frames.is_multiple_of(AUDIO_STATS_INTERVAL) {
            log_audio_stats(id.as_deref(), audio);
        }
    }

    fn flush_vad(&mut self) {
        let Some(vad) = self.vad.as_mut() else {
            return;
        };
        let mut events = Vec::new();
        vad.flush(&mut events);
        log_vad_events(self.id.as_deref(), events);
    }

    fn start_listening(&mut self, session_id: String, mode: Option<String>) {
        self.stop_asr();
        self.reset_vad();
        self.audio = AudioStats::default();

        let (audio_tx, audio_rx) = mpsc::channel::<Option<Vec<f32>>>(ASR_CHANNEL_CAPACITY);
        let tx = self.tx.clone();
        let asr = self.asr.clone();
        self.asr_task = Some(tokio::spawn(run_asr(
            session_id.clone(),
            tx,
            asr,
            asr_stream(audio_rx),
        )));
        self.asr_tx = Some(audio_tx);
        self.state = SessionState::Listening;
        info!(session_id, mode = ?mode, "device started listening");
    }

    fn stop_listening(&mut self) {
        self.flush_vad();
        let duration_ms = self.audio.samples * 1000 / crate::audio::SAMPLE_RATE as u64;
        info!(session_id = ?self.id, duration_ms, "device stopped listening");
        log_audio_stats(self.id.as_deref(), &self.audio);

        if let Some(asr_tx) = self.asr_tx.as_mut() {
            let _ = asr_tx.try_send(None);
        }
        self.asr_tx = None;
        self.state = SessionState::Speaking;
    }

    fn stop_asr(&mut self) {
        self.asr_tx = None;
        if let Some(handle) = self.asr_task.take() {
            handle.abort();
        }
    }

    fn shutdown(&mut self) {
        self.flush_vad();
        self.stop_asr();
    }

    fn reset_vad(&mut self) {
        if let Some(vad) = self.vad.as_mut() {
            vad.reset();
        }
    }
}

fn log_audio_stats(session_id: Option<&str>, stats: &AudioStats) {
    info!(
        session_id,
        frames = stats.frames,
        samples = stats.samples,
        peak = stats.peak,
        rms = stats.rms(),
        vad_peak = stats.vad_peak,
        "audio stats"
    );
}

fn log_vad_events(session_id: Option<&str>, events: Vec<VadEvent>) {
    for event in events {
        match event {
            VadEvent::SpeechStart { at_ms } => {
                info!(session_id, at_ms, "vad: speech start");
            }
            VadEvent::SpeechEnd { at_ms } => {
                info!(session_id, at_ms, "vad: speech end");
            }
        }
    }
}

fn asr_stream(rx: mpsc::Receiver<Option<Vec<f32>>>) -> AudioStream {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Some(Some(chunk)) => Some((chunk, rx)),
            _ => None,
        }
    }))
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
                info!(session_id, %text, "asr partial");
            }
            Ok(AsrEvent::Final { text }) => {
                info!(session_id, %text, "asr final");
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
