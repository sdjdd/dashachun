use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tracing::{info, warn};

mod protocol;

use super::{Subtitle, TextStream, Tts, TtsError, TtsEvent, TtsEvents};
use protocol::Message as TtsMessage;

pub const DEFAULT_RESOURCE_ID: &str = "seed-tts-2.0";

const DEFAULT_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_FORMAT: &str = "pcm";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

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
    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        resource_id: impl Into<String>,
        speaker: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            resource_id: resource_id.into(),
            session: SessionConfig {
                speaker: speaker.into(),
                format: DEFAULT_FORMAT.to_string(),
                sample_rate: DEFAULT_SAMPLE_RATE,
            },
        }
    }

    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("VOLC_TTS_API_KEY").ok()?;
        let endpoint = std::env::var("VOLC_TTS_BASE_URL").ok()?;
        let speaker = std::env::var("VOLC_TTS_SPEAKER").ok()?;
        let resource_id = std::env::var("VOLC_TTS_RESOURCE_ID")
            .unwrap_or_else(|_| DEFAULT_RESOURCE_ID.to_string());
        Some(Self::new(endpoint, api_key, resource_id, speaker))
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

        info!("tts websocket connected");
        Ok(socket)
    }
}

impl Tts for VolcTts {
    fn synthesize(&self, text: TextStream) -> TtsEvents<'_> {
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
            if let Err(err) = run(socket, text, tx.clone(), session).await {
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
) -> Result<(), TtsError> {
    let (mut sink, mut stream) = socket.split();

    sink.send(Message::Binary(protocol::start_connection().into()))
        .await
        .map_err(|err| TtsError::from(format!("send start connection: {err}")))?;
    next_matching(
        &mut stream,
        protocol::MSG_FULL_SERVER_RESPONSE,
        protocol::EVENT_CONNECTION_STARTED,
    )
    .await?;

    let mut first = None;
    while first.is_none() {
        match text.next().await {
            Some(chunk) if !chunk.is_empty() => first = Some(chunk),
            Some(_) => continue,
            None => break,
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
    let frame = protocol::start_session(&session_id, &session.payload());
    sink.send(Message::Binary(frame.into()))
        .await
        .map_err(|err| TtsError::from(format!("send start session: {err}")))?;
    next_matching(
        &mut stream,
        protocol::MSG_FULL_SERVER_RESPONSE,
        protocol::EVENT_SESSION_STARTED,
    )
    .await?;

    let frame = protocol::task_request(&session_id, &first);
    sink.send(Message::Binary(frame.into()))
        .await
        .map_err(|err| TtsError::from(format!("send first task request: {err}")))?;

    let mut text_done = false;
    let mut finish_sent = false;

    loop {
        tokio::select! {
            chunk = text.next(), if !text_done => {
                match chunk {
                    Some(chunk) if !chunk.is_empty() => {
                        let frame = protocol::task_request(&session_id, &chunk);
                        sink.send(Message::Binary(frame.into()))
                            .await
                            .map_err(|err| TtsError::from(format!("send task request: {err}")))?;
                    }
                    Some(_) => {}
                    None => {
                        text_done = true;
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
                    protocol::EVENT_SESSION_FINISHED => {
                        if !finish_sent {
                            finish_sent = true;
                            sink.send(Message::Binary(protocol::finish_connection().into()))
                                .await
                                .map_err(|err| TtsError::from(format!("send finish connection: {err}")))?;
                        }
                    }
                    protocol::EVENT_CONNECTION_FINISHED => {
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
}
