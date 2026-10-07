use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

mod protocol;

use super::{Subtitle, TextStream, Tts, TtsError, TtsEvent, TtsEvents};
use crate::audio::DownlinkAudio;
use protocol::Message as TtsMessage;

pub const DEFAULT_RESOURCE_ID: &str = "seed-tts-2.0";

/// Sample rates the Volcengine bidirectional streaming TTS accepts.
const SUPPORTED_SAMPLE_RATES: &[u32] = &[8000, 16000, 22050, 24000, 32000, 44100, 48000];
const DEFAULT_FORMAT: &str = "pcm";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);

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

pub struct VolcTts {
    endpoint: String,
    api_key: String,
    resource_id: String,
    session: SessionConfig,
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
        Ok(Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            resource_id: resource_id.into(),
            session: SessionConfig {
                speaker: speaker.into(),
                format: DEFAULT_FORMAT.to_string(),
                sample_rate: downlink.sample_rate,
            },
        })
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

    fn build_request(
        &self,
    ) -> Result<tokio_tungstenite::tungstenite::handshake::client::Request, TtsError> {
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
        Ok(request)
    }

    async fn connect(
        request: tokio_tungstenite::tungstenite::handshake::client::Request,
    ) -> Result<Connection, TtsError> {
        let (socket, _response) = tokio::time::timeout(CONNECT_TIMEOUT, async {
            tokio_tungstenite::connect_async(request).await
        })
        .await
        .map_err(|_| TtsError::from("connect timeout"))?
        .map_err(|err| TtsError::from(format!("connect failed: {err}")))?;

        debug!("tts websocket connected");
        Ok(socket)
    }
}

impl Tts for VolcTts {
    fn synthesize(&self, text: TextStream, cancel: CancellationToken) -> TtsEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<TtsEvent, TtsError>>();
        let request = self.build_request();
        let session = self.session.clone();
        tokio::spawn(async move {
            let socket = match request {
                Ok(request) => match VolcTts::connect(request).await {
                    Ok(socket) => socket,
                    Err(err) => {
                        let _ = tx.send(Err(err));
                        return;
                    }
                },
                Err(err) => {
                    let _ = tx.send(Err(err));
                    return;
                }
            };
            if let Err(err) = run(socket, text, tx.clone(), session, cancel).await {
                let _ = tx.send(Err(err));
            }
        });
        Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        }))
    }
}

async fn run(
    socket: Connection,
    mut text: TextStream,
    tx: mpsc::UnboundedSender<Result<TtsEvent, TtsError>>,
    session: SessionConfig,
    cancel: CancellationToken,
) -> Result<(), TtsError> {
    let (mut sink, mut stream) = socket.split();

    let start_connection = async {
        sink.send(Message::Binary(protocol::start_connection().into()))
            .await
            .map_err(|err| TtsError::from(format!("send start connection: {err}")))?;
        next_matching(
            &mut stream,
            protocol::MSG_FULL_SERVER_RESPONSE,
            protocol::EVENT_CONNECTION_STARTED,
        )
        .await
    };
    match select_cancel(cancel.clone(), start_connection).await {
        SelectOutcome::Cancelled => return cancel_before_session(&mut sink, &tx).await,
        SelectOutcome::Ready(result) => {
            result?;
        }
    }
    debug!("tts connection started");

    let mut first = None;
    while first.is_none() {
        tokio::select! {
            _ = cancel.cancelled() => {
                return cancel_before_session(&mut sink, &tx).await;
            }
            chunk = text.next() => match chunk {
                Some(chunk) if !chunk.is_empty() => first = Some(chunk),
                Some(_) => continue,
                None => break,
            }
        }
    }
    let Some(first) = first else {
        sink.send(Message::Binary(protocol::finish_connection().into()))
            .await
            .map_err(|err| TtsError::from(format!("send finish connection: {err}")))?;
        let _ = tx.send(Ok(TtsEvent::Done));
        return Ok(());
    };

    let session_id = uuid::Uuid::new_v4().to_string();
    if cancel.is_cancelled() {
        return cancel_before_session(&mut sink, &tx).await;
    }
    let frame = protocol::start_session(&session_id, &session.payload());
    let start_session = async {
        sink.send(Message::Binary(frame.into()))
            .await
            .map_err(|err| TtsError::from(format!("send start session: {err}")))?;
        next_matching(
            &mut stream,
            protocol::MSG_FULL_SERVER_RESPONSE,
            protocol::EVENT_SESSION_STARTED,
        )
        .await
    };
    match select_cancel(cancel.clone(), start_session).await {
        SelectOutcome::Cancelled => {
            return cancel_session_and_finish(&mut sink, &mut stream, &tx, &session_id).await;
        }
        SelectOutcome::Ready(result) => {
            result?;
        }
    }
    debug!(%session_id, "tts session started");

    let frame = protocol::task_request(&session_id, &first);
    sink.send(Message::Binary(frame.into()))
        .await
        .map_err(|err| TtsError::from(format!("send first task request: {err}")))?;

    if cancel.is_cancelled() {
        return cancel_session_and_finish(&mut sink, &mut stream, &tx, &session_id).await;
    }

    let mut text_done = false;
    let mut finish_sent = false;
    let mut canceling = false;
    let mut chunks_sent = 1usize;
    let mut audio_frames = 0usize;
    let mut cancel_deadline = tokio::time::Instant::now() + CANCEL_TIMEOUT;

    loop {
        tokio::select! {
            _ = cancel.cancelled(), if !text_done && !finish_sent && !canceling => {
                debug!(%session_id, "tts cancelled, sending cancel session");
                text_done = true;
                canceling = true;
                cancel_deadline = tokio::time::Instant::now() + CANCEL_TIMEOUT;
                sink.send(Message::Binary(protocol::cancel_session(&session_id).into()))
                    .await
                    .map_err(|err| TtsError::from(format!("send cancel session: {err}")))?;
            }
            _ = tokio::time::sleep_until(cancel_deadline), if canceling && !finish_sent => {
                warn!(%session_id, "tts cancel not confirmed in time, finishing connection");
                finish_sent = true;
                sink.send(Message::Binary(protocol::finish_connection().into()))
                    .await
                    .map_err(|err| TtsError::from(format!("send finish connection: {err}")))?;
            }
            chunk = text.next(), if !text_done => {
                match chunk {
                    Some(chunk) if !chunk.is_empty() => {
                        let frame = protocol::task_request(&session_id, &chunk);
                        sink.send(Message::Binary(frame.into()))
                            .await
                            .map_err(|err| TtsError::from(format!("send task request: {err}")))?;
                        chunks_sent += 1;
                    }
                    Some(_) => {}
                    None => {
                        text_done = true;
                        debug!(%session_id, chunks = chunks_sent, "tts text stream ended, finishing session");
                        sink.send(Message::Binary(protocol::finish_session(&session_id).into()))
                            .await
                            .map_err(|err| TtsError::from(format!("send finish session: {err}")))?;
                    }
                }
            }
            incoming = stream.next() => {
                let data = match incoming {
                    Some(Ok(Message::Binary(data))) => data,
                    Some(Ok(Message::Close(_))) | None => break,
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
                        if canceling && !finish_sent {
                            finish_sent = true;
                            sink.send(Message::Binary(protocol::finish_connection().into()))
                                .await
                                .map_err(|err| TtsError::from(format!("send finish connection: {err}")))?;
                            debug!(%session_id, "session canceled");
                        }
                    }
                    protocol::EVENT_SESSION_FINISHED => {
                        if !finish_sent {
                            finish_sent = true;
                            sink.send(Message::Binary(protocol::finish_connection().into()))
                                .await
                                .map_err(|err| TtsError::from(format!("send finish connection: {err}")))?;
                            debug!(%session_id, "session finished");
                        }
                    }
                    protocol::EVENT_CONNECTION_FINISHED => {
                        debug!(
                            %session_id,
                            chunks = chunks_sent,
                            audio_frames,
                            "connection finished"
                        );
                        let _ = tx.send(Ok(TtsEvent::Done));
                        return Ok(());
                    }
                    protocol::EVENT_CONNECTION_FAILED | protocol::EVENT_SESSION_FAILED => {
                        return Err(TtsError::from(format!("tts session failed: {}", message.event)));
                    }
                    _ => {}
                }
            }
        }
    }

    let _ = tx.send(Ok(TtsEvent::Done));
    Ok(())
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

async fn await_event<S>(stream: &mut S, event: i32) -> Result<(), TtsError>
where
    S: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let deadline = tokio::time::Instant::now() + CANCEL_TIMEOUT;
    loop {
        match tokio::time::timeout_at(deadline, stream.next()).await {
            Err(_) => return Ok(()),
            Ok(None) | Ok(Some(Ok(Message::Close(_)))) => return Ok(()),
            Ok(Some(Err(err))) => return Err(TtsError::from(format!("receive failed: {err}"))),
            Ok(Some(Ok(Message::Binary(data)))) => {
                if let Ok(message) = protocol::parse(&data) {
                    if message.message_type == protocol::MSG_SERVER_ERROR_RESPONSE {
                        return Err(TtsError::from(format!(
                            "tts error {:?}: {}",
                            message.error_code,
                            String::from_utf8_lossy(message.audio())
                        )));
                    }
                    if message.event == event {
                        return Ok(());
                    }
                }
            }
            Ok(Some(Ok(_))) => {}
        }
    }
}

async fn cancel_before_session<W>(
    sink: &mut W,
    tx: &mpsc::UnboundedSender<Result<TtsEvent, TtsError>>,
) -> Result<(), TtsError>
where
    W: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
{
    debug!("tts cancelled before session start, finishing connection");
    sink.send(Message::Binary(protocol::finish_connection().into()))
        .await
        .map_err(|err| TtsError::from(format!("send finish connection: {err}")))?;
    let _ = tx.send(Ok(TtsEvent::Done));
    Ok(())
}

async fn cancel_session_and_finish<W, S>(
    sink: &mut W,
    stream: &mut S,
    tx: &mpsc::UnboundedSender<Result<TtsEvent, TtsError>>,
    session_id: &str,
) -> Result<(), TtsError>
where
    W: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    S: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    debug!(%session_id, "tts cancelled during handshake, sending cancel session");
    sink.send(Message::Binary(protocol::cancel_session(session_id).into()))
        .await
        .map_err(|err| TtsError::from(format!("send cancel session: {err}")))?;
    if await_event(stream, protocol::EVENT_SESSION_CANCELED)
        .await
        .is_err()
    {
        warn!(%session_id, "tts cancel not confirmed in time, finishing connection");
    }
    sink.send(Message::Binary(protocol::finish_connection().into()))
        .await
        .map_err(|err| TtsError::from(format!("send finish connection: {err}")))?;
    let _ = await_event(stream, protocol::EVENT_CONNECTION_FINISHED).await;
    let _ = tx.send(Ok(TtsEvent::Done));
    Ok(())
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
    use std::sync::Arc;

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

    #[tokio::test]
    async fn cancel_sends_cancel_then_finish_after_session_canceled() {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let order = Arc::new(std::sync::Mutex::new(Vec::<i32>::new()));
        let order_server = order.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            while let Some(Ok(message)) = ws.next().await {
                let Message::Binary(data) = message else {
                    continue;
                };
                match frame_event(&data) {
                    protocol::EVENT_START_CONNECTION => {
                        ws.send(Message::Binary(
                            server_event_frame(protocol::EVENT_CONNECTION_STARTED, "cid").into(),
                        ))
                        .await
                        .unwrap();
                    }
                    protocol::EVENT_START_SESSION => {
                        ws.send(Message::Binary(
                            server_event_frame(protocol::EVENT_SESSION_STARTED, "sid").into(),
                        ))
                        .await
                        .unwrap();
                    }
                    protocol::EVENT_CANCEL_SESSION => {
                        order_server.lock().unwrap().push(101);
                        ws.send(Message::Binary(
                            server_event_frame(protocol::EVENT_SESSION_CANCELED, "sid").into(),
                        ))
                        .await
                        .unwrap();
                    }
                    protocol::EVENT_FINISH_CONNECTION => {
                        order_server.lock().unwrap().push(2);
                        ws.send(Message::Binary(
                            server_event_frame(protocol::EVENT_CONNECTION_FINISHED, "cid").into(),
                        ))
                        .await
                        .unwrap();
                    }
                    _ => {}
                }
            }
        });

        fn server_event_frame(event: i32, session_id: &str) -> Vec<u8> {
            protocol::server_event_frame(event, session_id)
        }

        let tts = VolcTts::new(
            format!("ws://{addr}"),
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            DOWNLINK,
        )
        .unwrap();
        let (text_tx, text_rx) = mpsc::channel::<String>(4);
        let text: TextStream =
            Box::pin(futures_util::stream::unfold(text_rx, |mut rx| async move {
                rx.recv().await.map(|chunk| (chunk, rx))
            }));
        let cancel = CancellationToken::new();
        let mut events = tts.synthesize(text, cancel.clone());

        text_tx.send("你好".to_string()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        let mut saw_done = false;
        while let Some(event) = events.next().await {
            if matches!(event, Ok(TtsEvent::Done)) {
                saw_done = true;
                break;
            }
        }
        assert!(saw_done, "expected Done after cancel handshake");
        drop(text_tx);
        server.await.unwrap();
        assert_eq!(*order.lock().unwrap(), vec![101, 2]);
    }

    #[tokio::test]
    async fn cancel_during_connection_handshake_finishes_connection() {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let order = Arc::new(std::sync::Mutex::new(Vec::<i32>::new()));
        let order_server = order.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            while let Some(Ok(message)) = ws.next().await {
                let Message::Binary(data) = message else {
                    continue;
                };
                if frame_event(&data) == protocol::EVENT_FINISH_CONNECTION {
                    order_server.lock().unwrap().push(2);
                }
            }
        });

        let tts = VolcTts::new(
            format!("ws://{addr}"),
            "key",
            DEFAULT_RESOURCE_ID,
            "speaker",
            DOWNLINK,
        )
        .unwrap();
        let (_text_tx, text_rx) = mpsc::channel::<String>(4);
        let text: TextStream =
            Box::pin(futures_util::stream::unfold(text_rx, |mut rx| async move {
                rx.recv().await.map(|chunk| (chunk, rx))
            }));
        let cancel = CancellationToken::new();
        let mut events = tts.synthesize(text, cancel.clone());

        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        let mut saw_done = false;
        while let Some(event) = events.next().await {
            if matches!(event, Ok(TtsEvent::Done)) {
                saw_done = true;
                break;
            }
        }
        assert!(saw_done, "expected Done before session start");
        server.await.unwrap();
        assert_eq!(*order.lock().unwrap(), vec![2]);
    }
}
