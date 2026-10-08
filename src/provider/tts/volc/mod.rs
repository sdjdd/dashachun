use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::handshake::client::Request;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

mod pool;
mod protocol;

use crate::agent::{Subtitle, TextStream, Tts, TtsError, TtsEvent, TtsEvents};
use crate::audio::DownlinkAudio;
use pool::{ConnPool, Parked};
use protocol::Message as TtsMessage;

pub const DEFAULT_RESOURCE_ID: &str = "seed-tts-2.0";

/// Sample rates the Volcengine bidirectional streaming TTS accepts.
const SUPPORTED_SAMPLE_RATES: &[u32] = &[8000, 16000, 22050, 24000, 32000, 44100, 48000];
const DEFAULT_FORMAT: &str = "pcm";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);
const CONN_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const IDLE_SWEEP_INTERVAL: Duration = Duration::from_secs(5);
const MAX_IDLE_CONNS: usize = 16;

type Connection =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Clone)]
struct SessionConfig {
    speaker: String,
    format: String,
    sample_rate: u32,
}

impl SessionConfig {
    fn payload(&self) -> serde_json::Value {
        serde_json::json!({
            "user": { "uid": "xiaozhi" },
            "event": protocol::EVENT_START_SESSION,
            "namespace": "BidirectionalTTS",
            "req_params": {
                "speaker": self.speaker,
                "audio_params": {
                    "format": self.format,
                    "sample_rate": self.sample_rate,
                    "enable_subtitle": true,
                }
            }
        })
    }
}

#[derive(Clone)]
struct ConnSpec {
    endpoint: String,
    api_key: String,
    resource_id: String,
}

impl ConnSpec {
    fn build_request(&self) -> Result<(Request, String), TtsError> {
        let mut request = self
            .endpoint
            .as_str()
            .into_client_request()
            .map_err(|err| TtsError::from(format!("invalid endpoint: {err}")))?;
        let connect_id = uuid::Uuid::new_v4().to_string();
        let headers = request.headers_mut();
        for (name, value) in [
            ("X-Api-Key", self.api_key.as_str()),
            ("X-Api-Resource-Id", self.resource_id.as_str()),
            ("X-Api-Connect-Id", connect_id.as_str()),
            ("X-Control-Require-Usage-Tokens-Return", "*"),
        ] {
            let value = value
                .parse()
                .map_err(|err| TtsError::from(format!("invalid header {name}: {err}")))?;
            headers.insert(name, value);
        }
        Ok((request, connect_id))
    }
}

struct PoolTiming {
    idle_timeout: Duration,
    sweep_interval: Duration,
    max_idle: usize,
}

pub struct VolcTts {
    spec: ConnSpec,
    session: SessionConfig,
    pool: Arc<ConnPool>,
}

impl VolcTts {
    /// Errors on a downlink format Volc TTS cannot produce.
    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        resource_id: impl Into<String>,
        speaker: impl Into<String>,
        downlink: DownlinkAudio,
    ) -> Result<Self, TtsError> {
        Self::build(
            endpoint,
            api_key,
            resource_id,
            speaker,
            downlink,
            PoolTiming {
                idle_timeout: CONN_IDLE_TIMEOUT,
                sweep_interval: IDLE_SWEEP_INTERVAL,
                max_idle: MAX_IDLE_CONNS,
            },
        )
    }

    fn build(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        resource_id: impl Into<String>,
        speaker: impl Into<String>,
        downlink: DownlinkAudio,
        timing: PoolTiming,
    ) -> Result<Self, TtsError> {
        if !SUPPORTED_SAMPLE_RATES.contains(&downlink.sample_rate) {
            return Err(TtsError::from(format!(
                "unsupported volc tts sample rate {}: expected one of {SUPPORTED_SAMPLE_RATES:?}",
                downlink.sample_rate
            )));
        }
        if downlink.channels != 1 {
            return Err(TtsError::from(format!(
                "unsupported volc tts channel count {}: only mono is supported",
                downlink.channels
            )));
        }
        let pool = Arc::new(ConnPool::new(timing.max_idle));
        spawn_idle_sweeper(pool.clone(), timing.idle_timeout, timing.sweep_interval);
        Ok(Self {
            spec: ConnSpec {
                endpoint: endpoint.into(),
                api_key: api_key.into(),
                resource_id: resource_id.into(),
            },
            session: SessionConfig {
                speaker: speaker.into(),
                format: DEFAULT_FORMAT.to_string(),
                sample_rate: downlink.sample_rate,
            },
            pool,
        })
    }

    #[cfg(test)]
    fn with_pool_timing(
        endpoint: &str,
        idle_timeout: Duration,
        sweep_interval: Duration,
        max_idle: usize,
    ) -> Self {
        Self::build(
            endpoint,
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            crate::audio::DOWNLINK,
            PoolTiming {
                idle_timeout,
                sweep_interval,
                max_idle,
            },
        )
        .unwrap()
    }

    pub fn from_env(downlink: DownlinkAudio) -> Result<Option<Self>, TtsError> {
        let Some(api_key) = std::env::var("VOLC_TTS_API_KEY").ok() else {
            return Ok(None);
        };
        let Some(endpoint) = std::env::var("VOLC_TTS_BASE_URL").ok() else {
            return Ok(None);
        };
        let Some(speaker) = std::env::var("VOLC_TTS_SPEAKER").ok() else {
            return Ok(None);
        };
        let resource_id = std::env::var("VOLC_TTS_RESOURCE_ID")
            .unwrap_or_else(|_| DEFAULT_RESOURCE_ID.to_string());
        Ok(Some(Self::new(
            endpoint,
            api_key,
            resource_id,
            speaker,
            downlink,
        )?))
    }
}

impl Tts for VolcTts {
    fn synthesize(&self, text: TextStream, cancel: CancellationToken) -> TtsEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<TtsEvent, TtsError>>();
        let pool = self.pool.clone();
        let spec = self.spec.clone();
        let session = self.session.clone();
        tokio::spawn(async move {
            if let Err(err) = run(pool, spec, session, text, tx.clone(), cancel).await {
                let _ = tx.send(Err(err));
            }
        });
        Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        }))
    }
}

async fn run(
    pool: Arc<ConnPool>,
    spec: ConnSpec,
    session: SessionConfig,
    mut text: TextStream,
    tx: mpsc::UnboundedSender<Result<TtsEvent, TtsError>>,
    cancel: CancellationToken,
) -> Result<(), TtsError> {
    let mut first = None;
    while first.is_none() {
        tokio::select! {
            _ = cancel.cancelled() => {
                let _ = tx.send(Ok(TtsEvent::Done));
                return Ok(());
            }
            chunk = text.next() => match chunk {
                Some(chunk) if !chunk.is_empty() => first = Some(chunk),
                Some(_) => {}
                None => {
                    let _ = tx.send(Ok(TtsEvent::Done));
                    return Ok(());
                }
            }
        }
    }
    let Some(first) = first else {
        let _ = tx.send(Ok(TtsEvent::Done));
        return Ok(());
    };

    let mut parked = match select_cancel(cancel.clone(), acquire(&pool, &spec)).await {
        SelectOutcome::Cancelled => {
            let _ = tx.send(Ok(TtsEvent::Done));
            return Ok(());
        }
        SelectOutcome::Ready(result) => result?,
    };
    if cancel.is_cancelled() {
        park(&pool, parked);
        let _ = tx.send(Ok(TtsEvent::Done));
        return Ok(());
    }

    let session_id = uuid::Uuid::new_v4().to_string();
    let frame = protocol::start_session(&session_id, &session.payload());
    let start_session = async {
        parked
            .sink
            .send(Message::Binary(frame.into()))
            .await
            .map_err(|err| TtsError::from(format!("send start session: {err}")))?;
        next_matching(
            &mut parked.stream,
            protocol::MSG_FULL_SERVER_RESPONSE,
            protocol::EVENT_SESSION_STARTED,
        )
        .await
    };
    match select_cancel(cancel.clone(), start_session).await {
        SelectOutcome::Cancelled => {
            if cancel_session(&mut parked.sink, &mut parked.stream, &session_id).await {
                park(&pool, parked);
            }
            let _ = tx.send(Ok(TtsEvent::Done));
            return Ok(());
        }
        SelectOutcome::Ready(result) => {
            result?;
        }
    }
    debug!(%session_id, "tts session started");

    let frame = protocol::task_request(&session_id, &first);
    parked
        .sink
        .send(Message::Binary(frame.into()))
        .await
        .map_err(|err| TtsError::from(format!("send first task request: {err}")))?;

    if cancel.is_cancelled() {
        if cancel_session(&mut parked.sink, &mut parked.stream, &session_id).await {
            park(&pool, parked);
        }
        let _ = tx.send(Ok(TtsEvent::Done));
        return Ok(());
    }

    let mut text_done = false;
    let mut canceling = false;
    let mut chunks_sent = 1usize;
    let mut audio_frames = 0usize;
    let mut cancel_deadline = tokio::time::Instant::now() + CANCEL_TIMEOUT;

    let end = loop {
        tokio::select! {
            _ = cancel.cancelled(), if !text_done && !canceling => {
                debug!(%session_id, "tts cancelled, sending cancel session");
                text_done = true;
                canceling = true;
                cancel_deadline = tokio::time::Instant::now() + CANCEL_TIMEOUT;
                parked.sink.send(Message::Binary(protocol::cancel_session(&session_id).into()))
                    .await
                    .map_err(|err| TtsError::from(format!("send cancel session: {err}")))?;
            }
            _ = tokio::time::sleep_until(cancel_deadline), if canceling => {
                warn!(%session_id, "tts cancel not confirmed in time, dropping connection");
                break SessionEnd::Abandoned;
            }
            chunk = text.next(), if !text_done => {
                match chunk {
                    Some(chunk) if !chunk.is_empty() => {
                        let frame = protocol::task_request(&session_id, &chunk);
                        parked.sink.send(Message::Binary(frame.into()))
                            .await
                            .map_err(|err| TtsError::from(format!("send task request: {err}")))?;
                        chunks_sent += 1;
                    }
                    Some(_) => {}
                    None => {
                        text_done = true;
                        debug!(%session_id, chunks = chunks_sent, "tts text stream ended, finishing session");
                        parked.sink.send(Message::Binary(protocol::finish_session(&session_id).into()))
                            .await
                            .map_err(|err| TtsError::from(format!("send finish session: {err}")))?;
                    }
                }
            }
            incoming = parked.stream.next() => {
                let data = match incoming {
                    Some(Ok(Message::Binary(data))) => data,
                    Some(Ok(Message::Close(_))) | None => break SessionEnd::Abandoned,
                    Some(Err(err)) => {
                        return Err(TtsError::from(format!("receive failed: {err}")));
                    }
                    Some(Ok(_)) => continue,
                };
                let message = match protocol::parse(&data) {
                    Ok(message) => message,
                    Err(err) => {
                        warn!(%err, "failed to parse tts response");
                        continue;
                    }
                };
                if message.message_type == protocol::MSG_SERVER_ERROR_RESPONSE {
                    return Err(TtsError::from(format!(
                        "tts error {:?}: {}",
                        message.error_code,
                        String::from_utf8_lossy(message.audio())
                    )));
                }
                match message.event {
                    protocol::EVENT_TTS_RESPONSE => {
                        if message.message_type != protocol::MSG_AUDIO_ONLY_SERVER {
                            return Err(TtsError::from(format!(
                                "unexpected tts audio message type {}",
                                message.message_type
                            )));
                        }
                        let samples = pcm16_to_f32(message.audio());
                        audio_frames += 1;
                        if tx.send(Ok(TtsEvent::Audio(samples))).is_err() {
                            return Ok(());
                        }
                    }
                    protocol::EVENT_TTS_SUBTITLE => {
                        if let Some(subtitle) = subtitle(&message)
                            && tx.send(Ok(TtsEvent::Subtitle(subtitle))).is_err()
                        {
                            return Ok(());
                        }
                    }
                    protocol::EVENT_SESSION_CANCELED => {
                        if canceling {
                            debug!(%session_id, "session canceled");
                            break SessionEnd::Canceled;
                        }
                    }
                    protocol::EVENT_SESSION_FINISHED => {
                        debug!(%session_id, "session finished");
                        break SessionEnd::Finished;
                    }
                    protocol::EVENT_CONNECTION_FINISHED => {
                        debug!(
                            %session_id,
                            chunks = chunks_sent,
                            audio_frames,
                            "connection finished"
                        );
                        break SessionEnd::Abandoned;
                    }
                    protocol::EVENT_CONNECTION_FAILED | protocol::EVENT_SESSION_FAILED => {
                        return Err(TtsError::from(format!("tts session failed: {}", message.event)));
                    }
                    _ => {}
                }
            }
        }
    };

    match end {
        SessionEnd::Finished | SessionEnd::Canceled => {
            debug!(connect_id = %parked.connect_id, "tts parking connection");
            park(&pool, parked);
        }
        SessionEnd::Abandoned => {}
    }
    let _ = tx.send(Ok(TtsEvent::Done));
    Ok(())
}

enum SessionEnd {
    Finished,
    Canceled,
    Abandoned,
}

enum SelectOutcome<T> {
    Ready(T),
    Cancelled,
}

async fn select_cancel<F>(cancel: CancellationToken, future: F) -> SelectOutcome<F::Output>
where
    F: std::future::Future,
{
    tokio::select! {
        biased;
        _ = cancel.cancelled() => SelectOutcome::Cancelled,
        output = future => SelectOutcome::Ready(output),
    }
}

async fn acquire(pool: &ConnPool, spec: &ConnSpec) -> Result<Parked, TtsError> {
    loop {
        let Some(mut parked) = pool.checkout() else {
            return open_connection(spec).await;
        };
        if connection_alive(&mut parked) {
            debug!(connect_id = %parked.connect_id, "tts reusing pooled connection");
            return Ok(parked);
        }
        debug!(connect_id = %parked.connect_id, "tts pooled connection stale, dropping");
    }
}

async fn open_connection(spec: &ConnSpec) -> Result<Parked, TtsError> {
    let (request, connect_id) = spec.build_request()?;
    let socket = connect(request).await?;
    let (sink, stream) = socket.split();
    let mut parked = Parked {
        sink,
        stream,
        connect_id,
    };
    parked
        .sink
        .send(Message::Binary(protocol::start_connection().into()))
        .await
        .map_err(|err| TtsError::from(format!("send start connection: {err}")))?;
    next_matching(
        &mut parked.stream,
        protocol::MSG_FULL_SERVER_RESPONSE,
        protocol::EVENT_CONNECTION_STARTED,
    )
    .await?;
    debug!(connect_id = %parked.connect_id, "tts connection started");
    Ok(parked)
}

async fn connect(request: Request) -> Result<Connection, TtsError> {
    let (socket, _response) = tokio::time::timeout(CONNECT_TIMEOUT, async {
        tokio_tungstenite::connect_async(request).await
    })
    .await
    .map_err(|_| TtsError::from("connect timeout"))?
    .map_err(|err| TtsError::from(format!("connect failed: {err}")))?;

    debug!("tts websocket connected");
    Ok(socket)
}

/// Checks a parked connection for a server-side close while it sat idle: any
/// frames already delivered (stale session leftovers, keepalives) are drained
/// and discarded. `false` means the connection died and must not be reused.
fn connection_alive(conn: &mut Parked) -> bool {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        match Pin::new(&mut conn.stream).poll_next(&mut cx) {
            Poll::Pending => return true,
            Poll::Ready(None)
            | Poll::Ready(Some(Err(_)))
            | Poll::Ready(Some(Ok(Message::Close(_)))) => return false,
            Poll::Ready(Some(Ok(_))) => continue,
        }
    }
}

fn park(pool: &ConnPool, conn: Parked) {
    if let Some(conn) = pool.checkin(conn) {
        debug!(connect_id = %conn.connect_id, "tts pool full, closing connection");
        tokio::spawn(close_conn(conn));
    }
}

fn spawn_idle_sweeper(pool: Arc<ConnPool>, idle_timeout: Duration, sweep_interval: Duration) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(sweep_interval);
        loop {
            tick.tick().await;
            for conn in pool.expire(idle_timeout) {
                debug!(
                    connect_id = %conn.connect_id,
                    "tts idle connection expired, closing"
                );
                close_conn(conn).await;
            }
        }
    });
}

async fn close_conn(conn: Parked) {
    let Parked {
        mut sink,
        mut stream,
        ..
    } = conn;
    if sink
        .send(Message::Binary(protocol::finish_connection().into()))
        .await
        .is_err()
    {
        return;
    }
    let _ = tokio::time::timeout(
        CANCEL_TIMEOUT,
        next_matching(
            &mut stream,
            protocol::MSG_FULL_SERVER_RESPONSE,
            protocol::EVENT_CONNECTION_FINISHED,
        ),
    )
    .await;
}

/// Sends CancelSession and reports whether the server confirmed it; a confirmed
/// cancel leaves the connection session-free and reusable.
async fn cancel_session(
    sink: &mut SplitSink<Connection, Message>,
    stream: &mut SplitStream<Connection>,
    session_id: &str,
) -> bool {
    if sink
        .send(Message::Binary(protocol::cancel_session(session_id).into()))
        .await
        .is_err()
    {
        return false;
    }
    matches!(
        tokio::time::timeout(
            CANCEL_TIMEOUT,
            next_matching(
                stream,
                protocol::MSG_FULL_SERVER_RESPONSE,
                protocol::EVENT_SESSION_CANCELED,
            ),
        )
        .await,
        Ok(Ok(_))
    )
}

async fn next_matching<S>(
    stream: &mut S,
    message_type: u8,
    event: i32,
) -> Result<TtsMessage, TtsError>
where
    S: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        match stream.next().await {
            Some(Ok(Message::Binary(data))) => {
                let message =
                    protocol::parse(&data).map_err(|err| TtsError::from(err.to_string()))?;
                if message.message_type == protocol::MSG_SERVER_ERROR_RESPONSE {
                    return Err(TtsError::from(format!(
                        "tts error {:?}: {}",
                        message.error_code,
                        String::from_utf8_lossy(message.audio())
                    )));
                }
                if message.message_type == message_type && message.event == event {
                    return Ok(message);
                }
            }
            Some(Ok(Message::Close(_))) | None => {
                return Err(TtsError::from("connection closed"));
            }
            Some(Err(err)) => {
                return Err(TtsError::from(format!("receive failed: {err}")));
            }
            Some(Ok(_)) => {}
        }
    }
}

fn subtitle(message: &TtsMessage) -> Option<Subtitle> {
    let json = message.json()?;
    let text = json
        .get("text")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut start_ms: Option<u64> = None;
    let mut end_ms = 0;
    if let Some(words) = json.get("words").and_then(serde_json::Value::as_array) {
        for word in words {
            let Some(start) = word.get("startTime").and_then(seconds_to_ms) else {
                continue;
            };
            let Some(end) = word.get("endTime").and_then(seconds_to_ms) else {
                continue;
            };
            start_ms = Some(start_ms.map_or(start, |current| current.min(start)));
            end_ms = end_ms.max(end);
        }
    }
    if text.is_empty() && start_ms.is_none() {
        return None;
    }
    Some(Subtitle {
        text,
        start_ms: start_ms.unwrap_or(0),
        end_ms,
    })
}

fn seconds_to_ms(value: &serde_json::Value) -> Option<u64> {
    let seconds = value.as_f64()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    Some((seconds * 1000.0).round() as u64)
}

fn pcm16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| i16::from_le_bytes(*chunk) as f32 / 32768.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::DOWNLINK;
    use std::sync::{Arc, Mutex};

    #[test]
    fn session_payload_has_required_fields() {
        let session = SessionConfig {
            speaker: "zh_female_gaolengyujie_uranus_bigtts".into(),
            format: "pcm".into(),
            sample_rate: 16000,
        };
        let payload = session.payload();
        assert_eq!(
            payload["req_params"]["speaker"],
            "zh_female_gaolengyujie_uranus_bigtts"
        );
        assert_eq!(payload["req_params"]["audio_params"]["format"], "pcm");
        assert_eq!(payload["req_params"]["audio_params"]["sample_rate"], 16000);
        assert_eq!(
            payload["req_params"]["audio_params"]["enable_subtitle"],
            true
        );
        assert!(payload["req_params"].get("text").is_none());
    }

    #[test]
    fn rejects_unsupported_downlink_sample_rate() {
        let downlink = DownlinkAudio {
            sample_rate: 12000,
            ..DOWNLINK
        };
        let Err(err) = VolcTts::new(
            "ws://localhost",
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            downlink,
        ) else {
            panic!("unsupported downlink sample rate must be rejected");
        };
        assert!(err.to_string().contains("sample rate"), "{err}");
    }

    #[test]
    fn rejects_unsupported_downlink_channels() {
        let downlink = DownlinkAudio {
            channels: 2,
            ..DOWNLINK
        };
        let Err(err) = VolcTts::new(
            "ws://localhost",
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            downlink,
        ) else {
            panic!("unsupported downlink channel count must be rejected");
        };
        assert!(err.to_string().contains("channel"), "{err}");
    }

    fn message(payload: &str) -> TtsMessage {
        TtsMessage {
            payload: payload.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn pcm16_decodes_little_endian() {
        assert_eq!(
            pcm16_to_f32(&[0, 0, 0xff, 0x7f]),
            vec![0.0, 32767.0 / 32768.0]
        );
    }

    #[test]
    fn subtitle_spans_text_and_words() {
        let subtitle = subtitle(&message(
            r#"{"phonemes":[],"text":"你好。","words":[{"confidence":0.9,"endTime":0.615,"startTime":0.585,"word":"你"},{"confidence":0.8,"endTime":1.0,"startTime":0.615,"word":"好。"}]}"#,
        ))
        .unwrap();
        assert_eq!(subtitle.text, "你好。");
        assert_eq!(subtitle.start_ms, 585);
        assert_eq!(subtitle.end_ms, 1000);
    }

    #[test]
    fn subtitle_skips_empty() {
        assert!(subtitle(&message(r#"{"phonemes":[],"text":"","words":[]}"#)).is_none());
    }

    #[test]
    fn subtitle_keeps_text_without_words() {
        let subtitle = subtitle(&message(r#"{"phonemes":[],"text":"你好。","words":[]}"#)).unwrap();
        assert_eq!(subtitle.text, "你好。");
        assert_eq!(subtitle.start_ms, 0);
        assert_eq!(subtitle.end_ms, 0);
    }

    fn frame_event(data: &[u8]) -> i32 {
        i32::from_be_bytes(data[4..8].try_into().unwrap())
    }

    fn frame_session_id(data: &[u8]) -> String {
        let len = u32::from_be_bytes(data[8..12].try_into().unwrap()) as usize;
        String::from_utf8_lossy(&data[12..12 + len]).into_owned()
    }

    fn text_stream() -> (mpsc::Sender<String>, TextStream) {
        let (tx, rx) = mpsc::channel::<String>(4);
        let stream: TextStream = Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|chunk| (chunk, rx))
        }));
        (tx, stream)
    }

    async fn collect_until_done(events: &mut TtsEvents<'_>) {
        while let Some(event) = events.next().await {
            if matches!(event, Ok(TtsEvent::Done)) {
                return;
            }
        }
        panic!("event stream ended before Done");
    }

    type ServerSocket = tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>;

    async fn reply(socket: &mut ServerSocket, event: i32) {
        let _ = socket
            .send(Message::Binary(
                protocol::server_event_frame(event, "sid").into(),
            ))
            .await;
    }

    fn spawn_mock_tts_server(
        listener: tokio::net::TcpListener,
        order: Arc<Mutex<Vec<i32>>>,
        session_ids: Arc<Mutex<Vec<String>>>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            drop(listener);
            let Ok(mut socket) = tokio_tungstenite::accept_async(stream).await else {
                return;
            };
            while let Some(Ok(message)) = socket.next().await {
                let Message::Binary(data) = message else {
                    continue;
                };
                match frame_event(&data) {
                    protocol::EVENT_START_CONNECTION => {
                        reply(&mut socket, protocol::EVENT_CONNECTION_STARTED).await;
                    }
                    protocol::EVENT_START_SESSION => {
                        order.lock().unwrap().push(protocol::EVENT_START_SESSION);
                        session_ids.lock().unwrap().push(frame_session_id(&data));
                        reply(&mut socket, protocol::EVENT_SESSION_STARTED).await;
                    }
                    protocol::EVENT_FINISH_SESSION => {
                        order.lock().unwrap().push(protocol::EVENT_FINISH_SESSION);
                        reply(&mut socket, protocol::EVENT_SESSION_FINISHED).await;
                    }
                    protocol::EVENT_CANCEL_SESSION => {
                        order.lock().unwrap().push(protocol::EVENT_CANCEL_SESSION);
                        reply(&mut socket, protocol::EVENT_SESSION_CANCELED).await;
                    }
                    protocol::EVENT_FINISH_CONNECTION => {
                        order
                            .lock()
                            .unwrap()
                            .push(protocol::EVENT_FINISH_CONNECTION);
                        reply(&mut socket, protocol::EVENT_CONNECTION_FINISHED).await;
                    }
                    _ => {}
                }
            }
        })
    }

    #[tokio::test]
    async fn cancel_parks_connection_for_reuse() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let order = Arc::new(Mutex::new(Vec::<i32>::new()));
        let session_ids = Arc::new(Mutex::new(Vec::<String>::new()));
        let server = spawn_mock_tts_server(listener, order.clone(), session_ids.clone());

        let tts = VolcTts::new(
            format!("ws://{addr}"),
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            DOWNLINK,
        )
        .unwrap();
        let (text_tx, text) = text_stream();
        let cancel = CancellationToken::new();
        let mut events = tts.synthesize(text, cancel.clone());

        text_tx.send("你好".to_string()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        collect_until_done(&mut events).await;
        assert_eq!(
            *order.lock().unwrap(),
            vec![
                protocol::EVENT_START_SESSION,
                protocol::EVENT_CANCEL_SESSION
            ],
            "cancel must not finish the connection"
        );
        assert_eq!(
            tts.pool.parked_count(),
            1,
            "canceled connection is reusable"
        );
        server.abort();
    }

    #[tokio::test]
    async fn cancel_before_text_opens_no_connection() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let tts = VolcTts::new(
            format!("ws://{addr}"),
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            DOWNLINK,
        )
        .unwrap();
        let (_text_tx, text) = text_stream();
        let cancel = CancellationToken::new();
        let mut events = tts.synthesize(text, cancel.clone());

        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        collect_until_done(&mut events).await;
        assert!(
            tokio::time::timeout(Duration::from_millis(300), listener.accept())
                .await
                .is_err(),
            "no connection should be opened before the first text chunk"
        );
    }

    #[tokio::test]
    async fn empty_reply_opens_no_connection() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let tts = VolcTts::new(
            format!("ws://{addr}"),
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            DOWNLINK,
        )
        .unwrap();
        let (text_tx, text) = text_stream();
        let mut events = tts.synthesize(text, CancellationToken::new());
        drop(text_tx);

        collect_until_done(&mut events).await;
        assert!(
            tokio::time::timeout(Duration::from_millis(300), listener.accept())
                .await
                .is_err(),
            "an empty reply must not open a connection"
        );
    }

    #[tokio::test]
    async fn second_reply_reuses_parked_connection() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let order = Arc::new(Mutex::new(Vec::<i32>::new()));
        let session_ids = Arc::new(Mutex::new(Vec::<String>::new()));
        let server = spawn_mock_tts_server(listener, order.clone(), session_ids.clone());

        let tts = VolcTts::new(
            format!("ws://{addr}"),
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            DOWNLINK,
        )
        .unwrap();

        for chunk in ["你好", "世界"] {
            let (text_tx, text) = text_stream();
            let mut events = tts.synthesize(text, CancellationToken::new());
            text_tx.send(chunk.to_string()).await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            drop(text_tx);
            collect_until_done(&mut events).await;
        }

        assert_eq!(
            *order.lock().unwrap(),
            vec![
                protocol::EVENT_START_SESSION,
                protocol::EVENT_FINISH_SESSION,
                protocol::EVENT_START_SESSION,
                protocol::EVENT_FINISH_SESSION,
            ],
            "both replies must run on one connection without finishing it"
        );
        let session_ids = session_ids.lock().unwrap().clone();
        assert_eq!(session_ids.len(), 2);
        assert_ne!(session_ids[0], session_ids[1], "sessions get fresh ids");
        assert_eq!(tts.pool.parked_count(), 1);
        server.abort();
    }

    #[tokio::test]
    async fn idle_parked_connection_is_closed_by_sweeper() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let order = Arc::new(Mutex::new(Vec::<i32>::new()));
        let session_ids = Arc::new(Mutex::new(Vec::<String>::new()));
        let server = spawn_mock_tts_server(listener, order.clone(), session_ids.clone());

        let tts = VolcTts::with_pool_timing(
            &format!("ws://{addr}"),
            Duration::from_millis(300),
            Duration::from_millis(50),
            4,
        );
        let (text_tx, text) = text_stream();
        let mut events = tts.synthesize(text, CancellationToken::new());
        text_tx.send("你好".to_string()).await.unwrap();
        drop(text_tx);

        collect_until_done(&mut events).await;
        assert_eq!(tts.pool.parked_count(), 1);

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while !order
            .lock()
            .unwrap()
            .contains(&protocol::EVENT_FINISH_CONNECTION)
        {
            assert!(
                tokio::time::Instant::now() < deadline,
                "watchdog did not close the idle connection"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            *order.lock().unwrap(),
            vec![
                protocol::EVENT_START_SESSION,
                protocol::EVENT_FINISH_SESSION,
                protocol::EVENT_FINISH_CONNECTION,
            ]
        );
        assert_eq!(tts.pool.parked_count(), 0);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn reuse_resets_the_idle_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let order = Arc::new(Mutex::new(Vec::<i32>::new()));
        let session_ids = Arc::new(Mutex::new(Vec::<String>::new()));
        let server = spawn_mock_tts_server(listener, order.clone(), session_ids.clone());

        let tts = VolcTts::with_pool_timing(
            &format!("ws://{addr}"),
            Duration::from_millis(300),
            Duration::from_millis(50),
            4,
        );

        for (index, chunk) in ["你好", "世界"].into_iter().enumerate() {
            if index > 0 {
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            let (text_tx, text) = text_stream();
            let mut events = tts.synthesize(text, CancellationToken::new());
            text_tx.send(chunk.to_string()).await.unwrap();
            drop(text_tx);
            collect_until_done(&mut events).await;
        }

        // The first park's deadline has passed, but reply 2 re-parked the
        // connection, so the sweeper must not have closed it yet.
        tokio::time::sleep(Duration::from_millis(220)).await;
        assert_eq!(
            tts.pool.parked_count(),
            1,
            "reuse must re-arm the idle timeout"
        );
        assert!(
            !order
                .lock()
                .unwrap()
                .contains(&protocol::EVENT_FINISH_CONNECTION),
            "the reused connection must not be closed by the stale deadline"
        );

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while !order
            .lock()
            .unwrap()
            .contains(&protocol::EVENT_FINISH_CONNECTION)
        {
            assert!(
                tokio::time::Instant::now() < deadline,
                "sweeper did not close the idle connection"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(tts.pool.parked_count(), 0);
        assert_eq!(
            *order.lock().unwrap(),
            vec![
                protocol::EVENT_START_SESSION,
                protocol::EVENT_FINISH_SESSION,
                protocol::EVENT_START_SESSION,
                protocol::EVENT_FINISH_SESSION,
                protocol::EVENT_FINISH_CONNECTION,
            ]
        );
        server.abort();
    }
}
