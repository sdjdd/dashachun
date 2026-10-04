use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tracing::{info, warn};

mod protocol;

use super::{Asr, AsrError, AsrEvent, AsrEvents, AudioStream};
use protocol::Response;

pub const DEFAULT_RESOURCE_ID: &str = "volc.seedasr.sauc.duration";

const SAMPLE_RATE: usize = 16_000;
const BYTES_PER_SAMPLE: usize = 2;
const SEGMENT_MS: usize = 200;
const SEGMENT_BYTES: usize = SAMPLE_RATE * BYTES_PER_SAMPLE * SEGMENT_MS / 1000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct VolcAsr {
    endpoint: String,
    api_key: String,
    resource_id: String,
}

impl VolcAsr {
    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        resource_id: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            resource_id: resource_id.into(),
        }
    }

    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("VOLC_ASR_API_KEY").ok()?;
        let endpoint = std::env::var("VOLC_ASR_BASE_URL").ok()?;
        let resource_id = std::env::var("VOLC_ASR_RESOURCE_ID")
            .unwrap_or_else(|_| DEFAULT_RESOURCE_ID.to_string());
        Some(Self::new(endpoint, api_key, resource_id))
    }

    fn build_request(
        &self,
    ) -> Result<tokio_tungstenite::tungstenite::handshake::client::Request, AsrError> {
        let mut request = self
            .endpoint
            .as_str()
            .into_client_request()
            .map_err(|err| AsrError::from(format!("invalid endpoint: {err}")))?;
        let headers = request.headers_mut();
        for (name, value) in [
            ("X-Api-Key", self.api_key.as_str()),
            ("X-Api-Resource-Id", self.resource_id.as_str()),
            ("X-Api-Sequence", "-1"),
        ] {
            let value = value
                .parse()
                .map_err(|err| AsrError::from(format!("invalid header {name}: {err}")))?;
            headers.insert(name, value);
        }
        let request_id = uuid::Uuid::new_v4().to_string();
        for name in ["X-Api-Request-Id", "X-Api-Connect-Id"] {
            let value = request_id
                .parse()
                .map_err(|err| AsrError::from(format!("invalid header {name}: {err}")))?;
            headers.insert(name, value);
        }
        Ok(request)
    }

    async fn connect(
        request: tokio_tungstenite::tungstenite::handshake::client::Request,
    ) -> Result<Connection, AsrError> {
        let (socket, _response) = tokio::time::timeout(CONNECT_TIMEOUT, async {
            tokio_tungstenite::connect_async(request).await
        })
        .await
        .map_err(|_| AsrError::from("connect timeout"))?
        .map_err(|err| AsrError::from(format!("connect failed: {err}")))?;

        info!("asr websocket connected");
        Ok(socket)
    }
}

type Connection =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

impl Asr for VolcAsr {
    fn transcribe(&self, audio: AudioStream) -> AsrEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<AsrEvent, AsrError>>();
        let request = self.build_request();
        tokio::spawn(async move {
            let socket = match request {
                Ok(request) => match Self::connect(request).await {
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
            if let Err(err) = run(socket, audio, tx.clone()).await {
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
    audio: AudioStream,
    tx: mpsc::UnboundedSender<Result<AsrEvent, AsrError>>,
) -> Result<(), AsrError> {
    let (mut sink, mut stream) = socket.split();

    let full_request = protocol::build_full_client_request(1, &request_payload());
    sink.send(Message::Binary(full_request.into()))
        .await
        .map_err(|err| AsrError::from(format!("send full request: {err}")))?;

    let mut seq = 2i32;
    let mut pending: Vec<u8> = Vec::new();
    let mut audio = audio;
    let mut audio_done = false;
    let mut text = String::new();

    loop {
        tokio::select! {
            chunk = audio.next(), if !audio_done => {
                match chunk {
                    Some(chunk) => {
                        pending.extend_from_slice(&protocol::audio_to_pcm16(&chunk));
                        while pending.len() >= SEGMENT_BYTES {
                            let segment: Vec<u8> = pending.drain(..SEGMENT_BYTES).collect();
                            let frame = protocol::build_audio_request(seq, &segment, false);
                            sink.send(Message::Binary(frame.into()))
                                .await
                                .map_err(|err| AsrError::from(format!("send audio: {err}")))?;
                            seq += 1;
                        }
                    }
                    None => {
                        audio_done = true;
                        let last = protocol::build_audio_request(seq, &pending, true);
                        sink.send(Message::Binary(last.into()))
                            .await
                            .map_err(|err| AsrError::from(format!("send last audio: {err}")))?;
                    }
                }
            }
            message = stream.next() => {
                let data = match message {
                    Some(Ok(Message::Binary(data))) => data,
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(err)) => {
                        return Err(AsrError::from(format!("receive failed: {err}")));
                    }
                    Some(Ok(_)) => continue,
                };
                let response = match protocol::parse_response(&data) {
                    Ok(response) => response,
                    Err(err) => {
                        warn!(%err, "failed to parse asr response");
                        continue;
                    }
                };
                if response.code != 0 && response.code != 20_000_000 {
                    return Err(AsrError::from(format!(
                        "asr error code {}: {:?}",
                        response.code, response.payload
                    )));
                }
                if let Some(update) = response_text(&response)
                    && update != text
                {
                    text = update;
                    let _ = tx.send(Ok(AsrEvent::Partial { text: text.clone() }));
                }
                if response.is_last {
                    break;
                }
            }
        }
    }

    let _ = tx.send(Ok(AsrEvent::Final { text }));
    Ok(())
}

fn response_text(response: &Response) -> Option<String> {
    response
        .payload
        .as_ref()?
        .get("result")?
        .get("text")?
        .as_str()
        .map(str::to_string)
}

fn request_payload() -> serde_json::Value {
    serde_json::json!({
        "user": { "uid": "xiaozhi" },
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": 16000,
            "bits": 16,
            "channel": 1
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": true,
            "enable_punc": true,
            "show_utterances": true
        }
    })
}
