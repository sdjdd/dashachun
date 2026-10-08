use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

mod protocol;

use crate::agent::{Asr, AsrError, AsrEvent, AsrEvents, AudioStream};
use protocol::Response;

pub const DEFAULT_RESOURCE_ID: &str = "volc.seedasr.sauc.duration";

const SAMPLE_RATE: usize = 16_000;
const BYTES_PER_SAMPLE: usize = 2;
const SEGMENT_MS: usize = 200;
const SEGMENT_BYTES: usize = SAMPLE_RATE * BYTES_PER_SAMPLE * SEGMENT_MS / 1000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);

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

        debug!("asr websocket connected");
        Ok(socket)
    }
}

type Connection =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

impl Asr for VolcAsr {
    fn transcribe(&self, audio: AudioStream, cancel: CancellationToken) -> AsrEvents<'_> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<AsrEvent, AsrError>>();
        let request = self.build_request();
        tokio::spawn(async move {
            let socket = match request {
                Ok(request) => match tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return,
                    result = Self::connect(request) => result,
                } {
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
            if let Err(err) = run(socket, audio, tx.clone(), cancel).await {
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
    cancel: CancellationToken,
) -> Result<(), AsrError> {
    let (mut sink, mut stream) = socket.split();

    debug!("asr session started, sending config frame");
    let full_request = protocol::build_full_client_request(1, &request_payload());
    sink.send(Message::Binary(full_request.into()))
        .await
        .map_err(|err| AsrError::from(format!("send full request: {err}")))?;

    let mut seq = 2i32;
    let mut pending: Vec<u8> = Vec::new();
    let mut audio = audio;
    let mut audio_done = false;
    let mut text = String::new();
    let mut sent_segments = 0usize;

    loop {
        tokio::select! {
            _ = cancel.cancelled(), if !audio_done => {
                debug!("asr cancelled, sending final frame");
                let last = protocol::build_audio_request(seq, &pending, true);
                sink.send(Message::Binary(last.into()))
                    .await
                    .map_err(|err| AsrError::from(format!("send last audio: {err}")))?;
                if wait_for_final(&mut stream).await {
                    break;
                }
                warn!("asr did not confirm cancellation in time");
                break;
            }
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
                            sent_segments += 1;
                        }
                    }
                    None => {
                        audio_done = true;
                        debug!(
                            seq,
                            segments = sent_segments,
                            remaining_bytes = pending.len(),
                            "asr audio stream ended, sending last frame"
                        );
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

    debug!(
        seq,
        segments = sent_segments,
        chars = text.chars().count(),
        "asr session closed"
    );
    let _ = tx.send(Ok(AsrEvent::Final { text }));
    Ok(())
}

async fn wait_for_final(
    stream: &mut (
             impl futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>>
             + Unpin
         ),
) -> bool {
    let deadline = tokio::time::Instant::now() + CANCEL_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return false;
        }
        match tokio::time::timeout(remaining, stream.next()).await {
            Err(_) => return false,
            Ok(Some(Ok(Message::Binary(data)))) => {
                if let Ok(response) = protocol::parse_response(&data)
                    && response.is_last
                {
                    return true;
                }
            }
            Ok(Some(Ok(Message::Close(_))) | None) => return true,
            Ok(Some(Err(_))) => return false,
            Ok(Some(Ok(_))) => {}
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    fn final_response_frame(seq: i32, text: &str) -> Vec<u8> {
        let body = protocol::gzip_compress(
            serde_json::json!({ "result": { "text": text } })
                .to_string()
                .as_bytes(),
        );
        let mut frame = vec![
            0x11,
            (protocol::MSG_FULL_SERVER_RESPONSE << 4) | protocol::FLAG_NEG_WITH_SEQUENCE,
            (protocol::SERIALIZATION_JSON << 4) | protocol::COMPRESSION_GZIP,
            0x00,
        ];
        frame.extend_from_slice(&seq.to_be_bytes());
        frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
        frame.extend_from_slice(&body);
        frame
    }

    #[tokio::test]
    async fn cancel_sends_negative_frame_and_waits_for_final() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            let mut saw_last = false;
            while let Some(Ok(message)) = ws.next().await {
                let Message::Binary(data) = message else {
                    continue;
                };
                if data.get(1).map(|b| b & 0x0f) == Some(protocol::FLAG_NEG_WITH_SEQUENCE) {
                    saw_last = true;
                    let seq = i32::from_be_bytes(data[4..8].try_into().unwrap());
                    ws.send(Message::Binary(
                        final_response_frame(seq, "cancelled").into(),
                    ))
                    .await
                    .unwrap();
                }
            }
            saw_last
        });

        let asr = VolcAsr::new(format!("ws://{addr}"), "key", DEFAULT_RESOURCE_ID);
        let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(4);
        let audio: AudioStream = Box::pin(futures_util::stream::poll_fn(move |cx| {
            audio_rx.poll_recv(cx)
        }));
        let cancel = CancellationToken::new();
        let mut events = asr.transcribe(audio, cancel.clone());

        audio_tx.send(vec![0.5; 960]).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        let mut final_text = None;
        while let Some(event) = events.next().await {
            if let Ok(AsrEvent::Final { text }) = event {
                final_text = Some(text);
                break;
            }
        }
        assert!(final_text.is_some(), "expected a final result");
        drop(audio_tx);
        assert!(server.await.unwrap(), "server never saw a negative frame");
    }
}
