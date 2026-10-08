//! Durable conversation transcript: the `messages` log plus per-utterance
//! FLAC captures attached to the user message and per-reply FLAC captures
//! attached to the assistant message.
//!
//! All storage runs on detached tasks fed through unbounded channels — the
//! agent driver only does non-blocking sends (see [`UtteranceRecording`] and
//! [`ReplyRecording`]) and never waits on Postgres or the encoder.

use sqlx::PgPool;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::debug;
use tracing::warn;
use uuid::Uuid;

use super::AgentSession;
use crate::audio::DOWNLINK;
use crate::llm::ChatItem;

const BITS_PER_SAMPLE: usize = 16;

/// Device-side identity stamped onto every transcript row, taken from the
/// verified device record at gateway upgrade.
#[derive(Debug, Clone)]
pub struct TranscriptOwner {
    pub user_id: i64,
    pub agent_id: i64,
    pub client_id: Uuid,
    pub device_id: Option<String>,
}

pub trait TranscriptSink: Send + Sync {
    /// Starts capturing the upcoming utterance. Finishing the recording with
    /// the transcript text also writes the user message row the audio is
    /// attached to; dropping it without finishing discards the capture.
    fn start_utterance(&self, session: &AgentSession) -> UtteranceRecording;

    /// Persists one turn's conversation items — the tool exchange plus the
    /// final reply — as a single row in the background and returns a receipt
    /// resolving to that row's id (`None` when the turn produced nothing to
    /// store). Fire-and-forget: failures are logged, never surfaced.
    fn log_items(&self, session: &AgentSession, items: Vec<ChatItem>) -> ReplyReceipt;

    /// Starts capturing the synthesized reply audio (the server's own
    /// downlink format). Finishing the recording with the reply receipt
    /// attaches the audio to the assistant message row; finishing with
    /// partial text (a reply cut before the model finished) writes the
    /// truncated assistant row itself; dropping it discards the capture.
    fn start_reply(&self, session: &AgentSession) -> ReplyRecording;
}

/// Handle over one utterance's audio capture. [`Clone`]: the driver keeps a
/// push side while the ASR task holds the finishing side; the channel closes
/// once every handle is gone, which is what marks a capture as discarded.
#[derive(Clone)]
pub struct UtteranceRecording {
    tx: mpsc::UnboundedSender<Capture>,
}

/// One capture message on an utterance's channel: a decoded uplink chunk or
/// the transcript that ends the capture.
pub enum Capture {
    Chunk(Vec<f32>),
    Finish { text: String },
}

impl UtteranceRecording {
    /// Opens a capture channel for custom sink implementations. The consumer
    /// receives chunks in order and at most one finish, then the channel
    /// closes.
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<Capture>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    /// Feeds one decoded uplink chunk. Non-blocking by contract: the caller
    /// is the agent driver, which must never wait on storage.
    pub fn push(&self, samples: &[f32]) {
        let _ = self.tx.send(Capture::Chunk(samples.to_vec()));
    }

    /// Ends the capture with the transcript text and stores it. An empty text
    /// discards the capture: audio without a transcript is useless for
    /// playback and voice cloning.
    pub fn finish(self, text: String) {
        let _ = self.tx.send(Capture::Finish { text });
    }
}

/// Resolves to the turn's message row id once the turn's items are stored;
/// `None` when the turn produced nothing to store.
pub struct ReplyReceipt {
    rx: oneshot::Receiver<Option<i64>>,
}

impl ReplyReceipt {
    /// Opens a receipt channel for custom sink implementations: the sink
    /// resolves it with the reply row's id (or `None`) after its insert.
    pub fn channel() -> (oneshot::Sender<Option<i64>>, Self) {
        let (tx, rx) = oneshot::channel();
        (tx, Self { rx })
    }

    /// An already-resolved receipt, for fakes and tests.
    pub fn ready(id: Option<i64>) -> Self {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(id);
        Self { rx }
    }

    pub async fn resolve(self) -> Option<i64> {
        self.rx.await.ok().flatten()
    }
}

/// Handle over one reply's audio capture: the synthesized reply in the
/// server's downlink format. Finishing attaches the audio to the assistant
/// message row; dropping it discards the capture.
pub struct ReplyRecording {
    tx: mpsc::UnboundedSender<ReplyCapture>,
}

/// One capture message on a reply's channel: a synthesized chunk, the reply
/// receipt that ends a completed reply, or the partial text that ends a reply
/// cut before the model finished.
pub enum ReplyCapture {
    Chunk(Vec<f32>),
    Finish(ReplyReceipt),
    Partial(String),
}

/// How a reply capture ended: with the reply receipt (the turn completed, the
/// audio attaches to the stored assistant row), or with the partially
/// streamed text (the reply was cut before the model finished, the collector
/// writes the truncated assistant row itself).
pub enum ReplyFinish {
    Receipt(ReplyReceipt),
    Partial(String),
}

impl ReplyRecording {
    /// Opens a capture channel for custom sink implementations. The consumer
    /// receives chunks in order and at most one finish, then the channel
    /// closes.
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<ReplyCapture>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    /// Feeds one synthesized chunk. Non-blocking by contract: the caller is
    /// the agent driver, which must never wait on storage.
    pub fn push(&self, samples: &[f32]) {
        let _ = self.tx.send(ReplyCapture::Chunk(samples.to_vec()));
    }

    /// Ends the capture of a completed reply: the audio attaches to the
    /// assistant row the receipt resolves to. `None` (or a lost insert)
    /// discards the audio.
    pub fn finish(self, receipt: ReplyReceipt) {
        let _ = self.tx.send(ReplyCapture::Finish(receipt));
    }

    /// Ends the capture of a reply cut before the model finished: the
    /// collector writes the truncated assistant row itself and attaches the
    /// partial audio to it. An empty text discards the whole capture.
    pub fn finish_partial(self, text: String) {
        let _ = self.tx.send(ReplyCapture::Partial(text));
    }
}

/// Postgres-backed [`TranscriptSink`].
pub struct DbTranscriptSink {
    pool: PgPool,
    owner: TranscriptOwner,
}

impl DbTranscriptSink {
    pub fn new(pool: PgPool, owner: TranscriptOwner) -> Self {
        Self { pool, owner }
    }
}

impl TranscriptSink for DbTranscriptSink {
    fn start_utterance(&self, session: &AgentSession) -> UtteranceRecording {
        let (recording, mut rx) = UtteranceRecording::channel();
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let session_id = session.id.clone();
        let sample_rate = session.sample_rate;
        let channels = session.channels;
        tokio::spawn(async move {
            let Some((chunks, text)) = collect(&mut rx).await else {
                return;
            };
            match store_utterance(
                &pool,
                &owner,
                &session_id,
                sample_rate,
                channels,
                &chunks,
                &text,
            )
            .await
            {
                Ok(message_id) => {
                    debug!(session_id, message_id, "utterance stored");
                }
                Err(err) => warn!(session_id, %err, "failed to store utterance"),
            }
        });
        recording
    }

    fn log_items(&self, session: &AgentSession, items: Vec<ChatItem>) -> ReplyReceipt {
        let (tx, rx) = oneshot::channel();
        if items.is_empty() {
            let _ = tx.send(None);
            return ReplyReceipt { rx };
        }
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let session_id = session.id.clone();
        tokio::spawn(async move {
            let reply = match store_items(&pool, &owner, &session_id, &items).await {
                Ok(reply_id) => reply_id,
                Err(err) => {
                    warn!(session_id, %err, "failed to store conversation items");
                    None
                }
            };
            let _ = tx.send(reply);
        });
        ReplyReceipt { rx }
    }

    fn start_reply(&self, session: &AgentSession) -> ReplyRecording {
        let (recording, mut rx) = ReplyRecording::channel();
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let session_id = session.id.clone();
        tokio::spawn(async move {
            let Some((chunks, finish)) = collect_reply(&mut rx).await else {
                return;
            };
            match finish {
                ReplyFinish::Receipt(receipt) => {
                    let Some(message_id) = receipt.resolve().await else {
                        warn!(
                            session_id,
                            "reply audio discarded: no assistant message row"
                        );
                        return;
                    };
                    match store_reply_audio(&pool, message_id, &chunks).await {
                        Ok(()) => debug!(session_id, message_id, "reply audio stored"),
                        Err(err) => {
                            warn!(session_id, message_id, %err, "failed to store reply audio")
                        }
                    }
                }
                ReplyFinish::Partial(text) => {
                    match store_partial_reply(&pool, &owner, &session_id, &text, &chunks).await {
                        Ok(message_id) => {
                            debug!(session_id, message_id, "partial reply stored");
                        }
                        Err(err) => warn!(session_id, %err, "failed to store partial reply"),
                    }
                }
            }
        });
        recording
    }
}

/// Drains a capture channel into `(chunks, text)`. `None` when the channel
/// closed without a finish (interrupted utterance) or the transcript is empty
/// (VAD false trigger, silent ASR) — both discard the capture.
pub async fn collect(rx: &mut mpsc::UnboundedReceiver<Capture>) -> Option<(Vec<Vec<f32>>, String)> {
    let mut chunks = Vec::new();
    while let Some(capture) = rx.recv().await {
        match capture {
            Capture::Chunk(chunk) => chunks.push(chunk),
            Capture::Finish { text } => {
                return (!text.is_empty()).then_some((chunks, text));
            }
        }
    }
    None
}

/// Drains a reply capture channel into `(chunks, finish)`. `None` when the
/// channel closed without a finish (session torn down mid-reply) — the
/// capture is discarded.
pub async fn collect_reply(
    rx: &mut mpsc::UnboundedReceiver<ReplyCapture>,
) -> Option<(Vec<Vec<f32>>, ReplyFinish)> {
    let mut chunks = Vec::new();
    while let Some(capture) = rx.recv().await {
        match capture {
            ReplyCapture::Chunk(chunk) => chunks.push(chunk),
            ReplyCapture::Finish(receipt) => {
                return Some((chunks, ReplyFinish::Receipt(receipt)));
            }
            ReplyCapture::Partial(text) => {
                return Some((chunks, ReplyFinish::Partial(text)));
            }
        }
    }
    None
}

struct TurnRow {
    content: Option<String>,
    parts: Option<serde_json::Value>,
}

/// Folds one turn's conversation items — the tool exchange plus the final
/// reply — into a single transcript row, in execution order. Arguments and
/// results stay verbatim strings; round boundaries are recoverable from the
/// part order (a `tool_call` following a `tool_result` opens the next round).
/// `System`/`User` items are never stored here: the user message is written
/// by the utterance capture, the system prompt is prepended every turn.
fn turn_row(items: &[ChatItem]) -> Option<TurnRow> {
    let mut parts = Vec::new();
    let mut content = None;
    for item in items {
        match item {
            ChatItem::System { .. } | ChatItem::User { .. } => {}
            ChatItem::Assistant {
                content: text,
                tool_calls,
            } => {
                for call in tool_calls {
                    parts.push(serde_json::json!({
                        "type": "tool_call",
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }));
                }
                if text.as_deref().is_some_and(|text| !text.is_empty()) {
                    content = text.clone();
                }
            }
            ChatItem::Tool {
                tool_call_id,
                content,
            } => {
                parts.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }));
            }
        }
    }
    if parts.is_empty() && content.is_none() {
        return None;
    }
    Some(TurnRow {
        content,
        parts: (!parts.is_empty()).then_some(serde_json::Value::Array(parts)),
    })
}

async fn store_items(
    pool: &PgPool,
    owner: &TranscriptOwner,
    session_id: &str,
    items: &[ChatItem],
) -> Result<Option<i64>, String> {
    let Some(row) = turn_row(items) else {
        return Ok(None);
    };
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO messages (session_id, user_id, agent_id, client_id, device_id, \
         role, content, parts) \
         VALUES ($1, $2, $3, $4, $5, 'assistant', $6, $7) RETURNING id",
    )
    .bind(session_id)
    .bind(owner.user_id)
    .bind(owner.agent_id)
    .bind(owner.client_id)
    .bind(&owner.device_id)
    .bind(&row.content)
    .bind(&row.parts)
    .fetch_one(pool)
    .await
    .map_err(|err| err.to_string())?;
    Ok(Some(id))
}

async fn store_utterance(
    pool: &PgPool,
    owner: &TranscriptOwner,
    session_id: &str,
    sample_rate: u32,
    channels: u16,
    chunks: &[Vec<f32>],
    text: &str,
) -> Result<i64, String> {
    let total_samples: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    if total_samples == 0 {
        return Err("utterance has no audio".into());
    }
    let duration_ms = (total_samples as i64 / i64::from(channels)) * 1000 / i64::from(sample_rate);
    let audio = encode_flac(channels, sample_rate, chunks)?;

    let mut tx = pool.begin().await.map_err(|err| err.to_string())?;
    let message_id: i64 = sqlx::query_scalar(
        "INSERT INTO messages (session_id, user_id, agent_id, client_id, device_id, role, content) \
         VALUES ($1, $2, $3, $4, $5, 'user', $6) RETURNING id",
    )
    .bind(session_id)
    .bind(owner.user_id)
    .bind(owner.agent_id)
    .bind(owner.client_id)
    .bind(&owner.device_id)
    .bind(text)
    .fetch_one(&mut *tx)
    .await
    .map_err(|err| err.to_string())?;
    insert_audio(
        &mut *tx,
        message_id,
        sample_rate,
        channels,
        duration_ms,
        &audio,
    )
    .await?;
    tx.commit().await.map_err(|err| err.to_string())?;
    Ok(message_id)
}

async fn insert_audio<'e, E>(
    executor: E,
    message_id: i64,
    sample_rate: u32,
    channels: u16,
    duration_ms: i64,
    audio: &[u8],
) -> Result<(), String>
where
    E: sqlx::PgExecutor<'e>,
{
    sqlx::query(
        "INSERT INTO message_audios (message_id, audio, sample_rate, channels, duration_ms) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(message_id)
    .bind(audio)
    .bind(i32::try_from(sample_rate).unwrap_or_default())
    .bind(i32::from(channels))
    .bind(i32::try_from(duration_ms).unwrap_or_default())
    .execute(executor)
    .await
    .map_err(|err| err.to_string())?;
    Ok(())
}

/// Stores a completed reply's audio against the assistant row the receipt
/// resolved to. No audio chunks (a reply the user never heard) store nothing.
async fn store_reply_audio(
    pool: &PgPool,
    message_id: i64,
    chunks: &[Vec<f32>],
) -> Result<(), String> {
    let total_samples: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    if total_samples == 0 {
        return Ok(());
    }
    let duration_ms = (total_samples as i64 / i64::from(DOWNLINK.channels)) * 1000
        / i64::from(DOWNLINK.sample_rate);
    let audio = encode_flac(DOWNLINK.channels, DOWNLINK.sample_rate, chunks)?;
    insert_audio(
        pool,
        message_id,
        DOWNLINK.sample_rate,
        DOWNLINK.channels,
        duration_ms,
        &audio,
    )
    .await
}

/// Stores a reply cut before the model finished: the truncated assistant row
/// plus the partial audio the device heard, in one transaction. An empty
/// partial text stores nothing.
async fn store_partial_reply(
    pool: &PgPool,
    owner: &TranscriptOwner,
    session_id: &str,
    text: &str,
    chunks: &[Vec<f32>],
) -> Result<i64, String> {
    if text.is_empty() {
        return Err("partial reply has no text".into());
    }
    let total_samples: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    let audio = if total_samples > 0 {
        Some(encode_flac(
            DOWNLINK.channels,
            DOWNLINK.sample_rate,
            chunks,
        )?)
    } else {
        None
    };
    let duration_ms = (total_samples as i64 / i64::from(DOWNLINK.channels)) * 1000
        / i64::from(DOWNLINK.sample_rate);

    let mut tx = pool.begin().await.map_err(|err| err.to_string())?;
    let message_id: i64 = sqlx::query_scalar(
        "INSERT INTO messages (session_id, user_id, agent_id, client_id, device_id, role, content) \
         VALUES ($1, $2, $3, $4, $5, 'assistant', $6) RETURNING id",
    )
    .bind(session_id)
    .bind(owner.user_id)
    .bind(owner.agent_id)
    .bind(owner.client_id)
    .bind(&owner.device_id)
    .bind(text)
    .fetch_one(&mut *tx)
    .await
    .map_err(|err| err.to_string())?;
    if let Some(audio) = audio {
        insert_audio(
            &mut *tx,
            message_id,
            DOWNLINK.sample_rate,
            DOWNLINK.channels,
            duration_ms,
            &audio,
        )
        .await?;
    }
    tx.commit().await.map_err(|err| err.to_string())?;
    Ok(message_id)
}

/// Encodes the utterance as FLAC: 16-bit PCM, the device's uplink sample rate
/// and channels. The uplink arrives as lossy Opus already, so this is the
/// lossless storage of exactly what we received — 16-bit quantization adds
/// nothing beyond the decode.
fn encode_flac(channels: u16, sample_rate: u32, chunks: &[Vec<f32>]) -> Result<Vec<u8>, String> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let total: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut samples = Vec::with_capacity(total);
    for chunk in chunks {
        samples.extend(chunk.iter().map(|&sample| pcm16_sample(sample)));
    }
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|err| format!("flac config: {err:?}"))?;
    // The encoder emits the tail as a short frame; a block size below 16
    // samples is an illegal FLAC frame that decoders reject, so pad the tail
    // up to the minimum (at most 15 samples of silence).
    let remainder = samples.len() % config.block_size;
    if remainder != 0 && remainder < MIN_FLAC_BLOCK {
        samples.extend(std::iter::repeat_n(0, MIN_FLAC_BLOCK - remainder));
    }
    let source = flacenc::source::MemSource::from_samples(
        &samples,
        usize::from(channels),
        BITS_PER_SAMPLE,
        sample_rate as usize,
    );
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|err| format!("flac encode: {err:?}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|err| format!("flac write: {err:?}"))?;
    Ok(sink.as_slice().to_vec())
}

/// Smallest legal FLAC frame block size.
const MIN_FLAC_BLOCK: usize = 16;

/// The uplink was decoded from 16-bit Opus, so rounding back to PCM16 keeps
/// the received fidelity.
fn pcm16_sample(sample: f32) -> i32 {
    (sample * 32767.0).round().clamp(-32768.0, 32767.0) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_items_fold_into_one_turn_row() {
        use crate::llm::ToolCall;

        let row = turn_row(&[
            ChatItem::system("system prompt"),
            ChatItem::user("hello"),
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"hangzhou\"}".into(),
            }]),
            ChatItem::tool("call_x", "sunny 26 degrees"),
            ChatItem::assistant("🙂sunny today"),
        ])
        .expect("the turn produced items");

        assert_eq!(row.content.as_deref(), Some("🙂sunny today"));
        let parts = row.parts.unwrap().as_array().unwrap().clone();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "tool_call");
        assert_eq!(parts[0]["id"], "call_x");
        assert_eq!(parts[0]["name"], "get_weather");
        assert_eq!(parts[0]["arguments"], "{\"city\":\"hangzhou\"}");
        assert_eq!(parts[1]["type"], "tool_result");
        assert_eq!(parts[1]["tool_call_id"], "call_x");
        assert_eq!(parts[1]["content"], "sunny 26 degrees");
    }

    #[test]
    fn turn_row_stores_nothing_without_parts_or_text() {
        use crate::llm::ToolCall;

        assert!(turn_row(&[]).is_none());
        assert!(turn_row(&[ChatItem::system("system prompt")]).is_none());
        // An empty final reply over no tool exchange stores nothing either.
        assert!(turn_row(&[ChatItem::assistant("")]).is_none());
        // A tool round without a final reply still stores.
        let row = turn_row(&[
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_x", "result"),
        ])
        .expect("tool parts are stored");
        assert_eq!(row.content, None);
        assert_eq!(row.parts.unwrap().as_array().unwrap().len(), 2);
    }

    #[test]
    fn flac_encoding_roundtrips_through_claxon() {
        // Six samples force the tail-padding path: the encoder would emit an
        // illegal short frame, the padding keeps the stream decodable.
        let chunks = vec![vec![0.0, 0.25, -0.5], vec![0.75, -1.0, 1.0]];
        let bytes = encode_flac(1, 16000, &chunks).unwrap();
        assert!(bytes.starts_with(b"fLaC"));

        let expected: Vec<i32> = chunks.iter().flatten().map(|&s| pcm16_sample(s)).collect();
        let mut reader = claxon::FlacReader::new(&bytes[..]).unwrap();
        let streaminfo = reader.streaminfo();
        assert_eq!(streaminfo.sample_rate, 16000);
        assert_eq!(streaminfo.channels, 1);
        assert_eq!(streaminfo.bits_per_sample, 16);

        let decoded: Vec<i32> = reader.samples().map(|sample| sample.unwrap()).collect();
        assert_eq!(&decoded[..expected.len()], &expected[..]);
        assert!(decoded[expected.len()..].iter().all(|&sample| sample == 0));
    }

    #[test]
    fn flac_encoding_survives_a_block_boundary_overshoot() {
        // 4097 samples: the final frame would hold a single sample, the
        // smallest regression class for the stored audio.
        let chunk: Vec<f32> = (0..4097).map(|i| (i % 97) as f32 / 97.0 - 0.5).collect();
        let bytes = encode_flac(1, 16000, &[chunk]).unwrap();

        let mut reader = claxon::FlacReader::new(&bytes[..]).unwrap();
        let decoded: Vec<i32> = reader.samples().map(|sample| sample.unwrap()).collect();
        let expected: Vec<i32> = (0..4097)
            .map(|i| pcm16_sample((i % 97) as f32 / 97.0 - 0.5))
            .collect();
        assert_eq!(&decoded[..expected.len()], &expected[..]);
        assert!(decoded[expected.len()..].iter().all(|&sample| sample == 0));
    }

    #[tokio::test]
    async fn dropped_recording_collects_nothing() {
        let (recording, mut rx) = UtteranceRecording::channel();
        recording.push(&[0.5]);
        drop(recording);
        assert!(collect(&mut rx).await.is_none());
    }

    #[tokio::test]
    async fn empty_transcript_discards_the_capture() {
        let (recording, mut rx) = UtteranceRecording::channel();
        recording.push(&[0.5]);
        recording.finish(String::new());
        assert!(collect(&mut rx).await.is_none());
    }

    #[tokio::test]
    async fn dropped_reply_recording_collects_nothing() {
        let (recording, mut rx) = ReplyRecording::channel();
        recording.push(&[0.5]);
        drop(recording);
        assert!(collect_reply(&mut rx).await.is_none());
    }

    #[tokio::test]
    async fn reply_recording_delivers_receipt_and_partial() {
        let (recording, mut rx) = ReplyRecording::channel();
        recording.push(&[0.5, 0.25]);
        recording.finish(ReplyReceipt::ready(Some(7)));
        let (chunks, finish) = collect_reply(&mut rx).await.unwrap();
        assert_eq!(chunks, vec![vec![0.5, 0.25]]);
        let ReplyFinish::Receipt(receipt) = finish else {
            panic!("expected a receipt finish");
        };
        assert_eq!(receipt.resolve().await, Some(7));

        let (recording, mut rx) = ReplyRecording::channel();
        recording.finish_partial("partial".into());
        let (chunks, finish) = collect_reply(&mut rx).await.unwrap();
        assert!(chunks.is_empty());
        assert!(matches!(finish, ReplyFinish::Partial(text) if text == "partial"));
    }
}
