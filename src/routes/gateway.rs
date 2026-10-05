use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use crate::agent::{Agent, AgentInput, AgentOutputStream, AgentSession};
use crate::audio::{OpusDecoder, OpusEncoder};
use crate::dto::ws::{
    Abort, AudioParams, ClientHello, InboundMessage, Listen, Mcp, ServerHello, SttMessage,
    TtsMessage,
};
use crate::state::AppState;

const AGENT_CHANNEL_CAPACITY: usize = 64;
const PLAYER_CHANNEL_CAPACITY: usize = 64;
const MAX_PREBUFFER_FRAMES: u32 = 8;

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
    let shutdown = state.shutdown_signal();
    ws.on_upgrade(move |socket| handle_device_socket(socket, state, shutdown))
}

fn header_str<'a>(headers: &'a HeaderMap, name: &'static str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

async fn handle_device_socket(
    socket: WebSocket,
    state: AppState,
    shutdown: tokio::sync::watch::Receiver<bool>,
) {
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

    let mut session = Session::new(
        tx.clone(),
        state.agent,
        state.config.server.playback_prebuffer_ms,
    );
    let grace = Duration::from_millis(state.config.server.shutdown_grace_ms);
    let mut shutdown = shutdown;

    let mut shutdown_requested = false;
    loop {
        tokio::select! {
            biased;
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    shutdown_requested = true;
                    break;
                }
            }
            result = stream.next() => {
                let Some(result) = result else { break };
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
        }
    }

    session.shutdown(grace).await;
    if shutdown_requested {
        let _ = session.tx.send(Message::Close(None)).await;
    }
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
    playback_prebuffer_ms: u32,
    agent_tx: Option<mpsc::Sender<AgentInput>>,
    agent_task: Option<JoinHandle<()>>,
    player: Option<Player>,
}

impl Session {
    fn new(tx: mpsc::Sender<Message>, agent: Arc<dyn Agent>, playback_prebuffer_ms: u32) -> Self {
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

    async fn shutdown(&mut self, grace: Duration) {
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
                player.start().await;
            }
            AgentOutput::TtsSentence { text } => {
                send_json(&tx, &TtsMessage::sentence_start(session_id.clone(), text)).await;
            }
            AgentOutput::TtsSubtitle { subtitle } => {
                trace!(
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
            AgentOutput::Mcp(payload) => {
                debug!(%payload, "agent mcp output");
            }
            AgentOutput::Error { message } => {
                warn!(%message, "agent error");
            }
        }
    }
}

enum PlayerCommand {
    Packet(Vec<u8>),
    Subtitle { start_ms: u64, text: String },
    Start,
    Finish(oneshot::Sender<()>),
    Abort,
}

struct Player {
    tx: mpsc::Sender<PlayerCommand>,
}

impl Clone for Player {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl Player {
    async fn push(&self, packet: Vec<u8>) {
        if self.tx.send(PlayerCommand::Packet(packet)).await.is_err() {
            trace!("player closed, dropping audio packet");
        }
    }

    async fn subtitle(&self, start_ms: u64, text: String) {
        if self
            .tx
            .send(PlayerCommand::Subtitle { start_ms, text })
            .await
            .is_err()
        {
            trace!("player closed, dropping subtitle");
        }
    }

    async fn start(&self) {
        let _ = self.tx.send(PlayerCommand::Start).await;
    }

    async fn abort(&self) {
        let _ = self.tx.send(PlayerCommand::Abort).await;
    }

    async fn finish(&self) {
        let (done_tx, done_rx) = oneshot::channel();
        if self.tx.send(PlayerCommand::Finish(done_tx)).await.is_ok() {
            let _ = done_rx.await;
        }
    }
}

fn spawn_player(
    tx: mpsc::Sender<Message>,
    frame_duration_ms: u32,
    prebuffer_ms: u32,
    session_id: String,
) -> Player {
    let (player_tx, player_rx) = mpsc::channel::<PlayerCommand>(PLAYER_CHANNEL_CAPACITY);
    let frame_ms = frame_duration_ms.max(1);
    let frame = Duration::from_millis(u64::from(frame_ms));
    let burst = (prebuffer_ms / frame_ms).clamp(1, MAX_PREBUFFER_FRAMES);
    let offset_ms = u64::from(burst) * u64::from(frame_ms);
    tokio::spawn(run_player(
        player_rx, tx, frame, frame_ms, burst, offset_ms, session_id,
    ));
    Player { tx: player_tx }
}

struct PlayerState {
    queue: VecDeque<Vec<u8>>,
    ack: Option<oneshot::Sender<()>>,
    credits: f64,
    capacity: f64,
    frame_ms: u64,
    offset_ms: u64,
    sent_ms: u64,
    pending: VecDeque<(u64, String)>,
}

impl PlayerState {
    fn reset(&mut self) {
        self.sent_ms = 0;
        self.pending.clear();
    }

    fn enqueue_subtitle(&mut self, start_ms: u64, text: String) {
        let at_ms = start_ms.saturating_add(self.offset_ms);
        self.pending.push_back((at_ms, text));
    }
}

async fn flush_due_subtitles(
    state: &mut PlayerState,
    tx: &mpsc::Sender<Message>,
    session_id: &str,
) {
    while let Some((at_ms, _)) = state.pending.front() {
        if *at_ms > state.sent_ms {
            break;
        }
        let Some((_, text)) = state.pending.pop_front() else {
            break;
        };
        debug!(text = %text, sent_ms = state.sent_ms, "subtitle sent");
        send_json(
            tx,
            &TtsMessage::sentence_start(session_id.to_string(), text),
        )
        .await;
    }
}

async fn run_player(
    mut rx: mpsc::Receiver<PlayerCommand>,
    tx: mpsc::Sender<Message>,
    frame: Duration,
    frame_ms: u32,
    burst: u32,
    offset_ms: u64,
    session_id: String,
) {
    let capacity = f64::from(burst);
    let mut state = PlayerState {
        queue: VecDeque::new(),
        ack: None,
        credits: capacity,
        capacity,
        frame_ms: u64::from(frame_ms),
        offset_ms,
        sent_ms: 0,
        pending: VecDeque::new(),
    };
    let mut last: Option<Instant> = None;
    loop {
        let now = Instant::now();
        if let Some(previous) = last {
            state.credits = (state.credits
                + now.duration_since(previous).as_secs_f64() / frame.as_secs_f64())
            .min(state.capacity);
        }
        last = Some(now);

        if !state.queue.is_empty() {
            if state.credits >= 1.0 {
                let Some(packet) = state.queue.pop_front() else {
                    continue;
                };
                if tx.send(Message::Binary(packet.into())).await.is_err() {
                    return;
                }
                state.credits -= 1.0;
                state.sent_ms = state.sent_ms.saturating_add(state.frame_ms);
                last = Some(Instant::now());
                flush_due_subtitles(&mut state, &tx, &session_id).await;
                continue;
            }
            let wait = frame.mul_f64(1.0 - state.credits);
            tokio::select! {
                _ = tokio::time::sleep(wait) => {}
                command = rx.recv() => {
                    if !handle_player_command(command, &mut state, &tx, &session_id).await {
                        return;
                    }
                }
            }
            continue;
        }

        if state.ack.is_some() {
            drain_pending_subtitles(&mut state, &tx, &session_id).await;
            if let Some(done) = state.ack.take() {
                let _ = done.send(());
            }
        }
        if !handle_player_command(rx.recv().await, &mut state, &tx, &session_id).await {
            return;
        }
    }
}

async fn drain_pending_subtitles(
    state: &mut PlayerState,
    tx: &mpsc::Sender<Message>,
    session_id: &str,
) {
    while let Some((_, text)) = state.pending.pop_front() {
        debug!(text = %text, "subtitle flushed");
        send_json(
            tx,
            &TtsMessage::sentence_start(session_id.to_string(), text),
        )
        .await;
    }
}

async fn handle_player_command(
    command: Option<PlayerCommand>,
    state: &mut PlayerState,
    tx: &mpsc::Sender<Message>,
    session_id: &str,
) -> bool {
    match command {
        Some(PlayerCommand::Packet(packet)) => state.queue.push_back(packet),
        Some(PlayerCommand::Subtitle { start_ms, text }) => {
            state.enqueue_subtitle(start_ms, text);
            flush_due_subtitles(state, tx, session_id).await;
        }
        Some(PlayerCommand::Start) => state.reset(),
        Some(PlayerCommand::Finish(done)) => state.ack = Some(done),
        Some(PlayerCommand::Abort) => {
            state.queue.clear();
            state.credits = state.capacity;
            state.reset();
        }
        None => return false,
    }
    true
}

async fn send_json<T: serde::Serialize>(tx: &mpsc::Sender<Message>, value: &T) {
    match serde_json::to_string(value) {
        Ok(payload) => {
            let _ = tx.send(Message::text(payload)).await;
        }
        Err(err) => error!(%err, "failed to serialize message"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence_start(message: &Message) -> Option<String> {
        let Message::Text(text) = message else {
            return None;
        };
        let value: serde_json::Value = serde_json::from_str(text.as_str()).ok()?;
        (value["type"] == "tts" && value["state"] == "sentence_start")
            .then(|| value["text"].as_str().unwrap_or_default().to_string())
    }

    #[tokio::test]
    async fn player_releases_one_packet_per_frame() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        for i in 0..3u8 {
            player.push(vec![i]).await;
        }
        let finished = tokio::spawn(async move {
            player.finish().await;
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        let first = rx.try_recv().expect("first packet should be released");
        assert!(matches!(first, Message::Binary(data) if data.as_ref() == [0u8]));
        assert!(
            rx.try_recv().is_err(),
            "packets released ahead of frame pacing"
        );

        finished.await.unwrap();
        let mut rest = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            if let Message::Binary(data) = msg {
                rest.push(data[0]);
            }
        }
        assert_eq!(rest, vec![1, 2]);
    }

    #[tokio::test]
    async fn player_primes_device_queue_with_prebuffer() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 100, "test".into());
        for i in 0..4u8 {
            player.push(vec![i]).await;
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
        let mut primed = Vec::new();
        while let Ok(Message::Binary(data)) = rx.try_recv() {
            primed.push(data[0]);
        }
        assert_eq!(primed, vec![0, 1], "expected a two-frame prebuffer burst");
        assert!(rx.try_recv().is_err(), "rest must stay paced");

        let finished = tokio::spawn(async move {
            player.finish().await;
        });
        finished.await.unwrap();
        let mut rest = Vec::new();
        while let Ok(Message::Binary(data)) = rx.try_recv() {
            rest.push(data[0]);
        }
        assert_eq!(rest, vec![2, 3]);
    }

    #[tokio::test]
    async fn player_abort_drops_pending_audio() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        for i in 0..3u8 {
            player.push(vec![i]).await;
        }
        player.abort().await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        let first = rx.try_recv().expect("first packet should be released");
        assert!(matches!(first, Message::Binary(data) if data.as_ref() == [0u8]));

        let finished = tokio::spawn(async move {
            player.finish().await;
        });
        finished.await.unwrap();
        assert!(
            rx.try_recv().is_err(),
            "buffered packets should be dropped after abort"
        );
    }

    #[tokio::test]
    async fn player_emits_subtitle_after_reaching_playback_position() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 100, "test".into());
        player.push(vec![0]).await;
        player.push(vec![1]).await;
        player.push(vec![2]).await;
        player.subtitle(100, "hello".into()).await;

        tokio::time::sleep(Duration::from_millis(10)).await;
        let mut seen = Vec::new();
        while let Ok(message) = rx.try_recv() {
            seen.push(message);
        }
        assert_eq!(seen.len(), 2, "prebuffer burst must precede the subtitle");
        assert!(seen.iter().all(|m| matches!(m, Message::Binary(_))));

        let finished = tokio::spawn(async move {
            player.finish().await;
        });
        finished.await.unwrap();

        let mut subtitles = Vec::new();
        while let Ok(message) = rx.try_recv() {
            if let Some(text) = sentence_start(&message) {
                subtitles.push(text);
            }
        }
        assert_eq!(subtitles, vec!["hello".to_string()]);
    }

    #[tokio::test]
    async fn player_flushes_subtitle_before_finish_ack() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        player.push(vec![0]).await;
        player.subtitle(10_000, "late".into()).await;

        player.finish().await;

        let mut subtitles = Vec::new();
        while let Ok(message) = rx.try_recv() {
            if let Some(text) = sentence_start(&message) {
                subtitles.push(text);
            }
        }
        assert_eq!(subtitles, vec!["late".to_string()]);
    }

    #[tokio::test]
    async fn player_abort_drops_pending_subtitle() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        player.push(vec![0]).await;
        player.subtitle(10_000, "gone".into()).await;
        player.abort().await;

        player.finish().await;

        while let Ok(message) = rx.try_recv() {
            assert!(
                sentence_start(&message).is_none(),
                "subtitle should be dropped"
            );
        }
    }

    #[tokio::test]
    async fn cloned_player_handle_clears_pending_subtitle() {
        let (tx, mut rx) = mpsc::channel::<Message>(16);
        let player = spawn_player(tx, 50, 0, "test".into());
        let session_handle = player.clone();

        player.subtitle(10_000, "gone".into()).await;
        session_handle.abort().await;

        player.finish().await;
        while let Ok(message) = rx.try_recv() {
            assert!(
                sentence_start(&message).is_none(),
                "abort via a cloned handle must drop pending subtitles"
            );
        }
    }
}
