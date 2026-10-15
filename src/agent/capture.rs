//! Audio storage: a byproduct feature that attaches captured audio to the
//! message rows the memory commits. Captured FLAC goes to an object store
//! through the [`AudioStorage`] port and the `message_audios` row records its
//! object key. The driver drives the session lifecycle (speech and tts
//! boundaries) through the [`Capture`] handles and binds each capture to its
//! row at the store call site (`attach`/`attach_sink` with the id the store
//! call returned or hands to its sink), so audio code never writes a message
//! row and never receives a row that is not its own.

use std::sync::Arc;

use sqlx::PgPool;
use time::OffsetDateTime;
use tokio::sync::mpsc;
use tracing::debug;
use tracing::warn;
use uuid::Uuid;

use crate::agent::AgentSession;
use crate::agent::memory::OnCommit;
use crate::audio::DOWNLINK;

const BITS_PER_SAMPLE: usize = 16;

/// The storage port of the audio feature: where finished FLAC captures go.
/// One put per capture under a fresh object key; a failed put discards that
/// audio while the message row stays the primary record. `put` returns the
/// bucket the object landed in — the inserted row records bucket and key,
/// so a later bucket switch can never orphan a reference, and `get` reads
/// back through the bucket a row recorded, never the current one. The port
/// deliberately has no delete: objects outlive their rows by design (failed
/// puts, agent deletes) and are reaped by the storage's periodic retention,
/// never app-side.
#[async_trait::async_trait]
pub trait AudioStorage: Send + Sync {
    async fn put(&self, key: &str, bytes: Vec<u8>) -> Result<String, String>;
    async fn get(&self, bucket: &str, key: &str) -> Result<Vec<u8>, String>;
}

/// The audio-storage feature: captures the device's uplink and the
/// synthesized reply, and stores them as `message_audios` rows linked by
/// `message_id` to the rows the memory commits. One instance lives per
/// connection; hand it to the driver as the [`Capture`] implementation and
/// bind each capture to its row through the store call's id (the attach
/// sink or the returned id). A `None` storage disables storage entirely:
/// captures still flow but store no rows and no objects.
pub struct AudioCapture {
    pool: PgPool,
    client_id: Uuid,
    audio_storage: Option<Arc<dyn AudioStorage>>,
}

impl AudioCapture {
    pub fn new(
        pool: PgPool,
        client_id: Uuid,
        audio_storage: Option<Arc<dyn AudioStorage>>,
    ) -> Self {
        Self {
            pool,
            client_id,
            audio_storage,
        }
    }
}

impl Capture for AudioCapture {
    fn start_utterance(&self, session: &AgentSession) -> UtteranceCapture {
        let (capture, mut rx, mut id_rx) = UtteranceCapture::channel();
        let pool = self.pool.clone();
        let client_id = self.client_id;
        let audio_storage = self.audio_storage.clone();
        let conversation_id = session.conversation_id;
        let sample_rate = session.sample_rate;
        let channels = session.channels;
        tokio::spawn(async move {
            let Some(chunks) = collect(&mut rx).await else {
                return;
            };
            let Some(message_id) = id_rx.recv().await else {
                debug!(conversation_id, "utterance audio discarded: no message row");
                return;
            };
            match store_audio(
                &pool,
                audio_storage.as_ref(),
                client_id,
                message_id,
                sample_rate,
                channels,
                &chunks,
            )
            .await
            {
                Ok(()) => debug!(conversation_id, message_id, "utterance audio stored"),
                Err(err) => {
                    warn!(conversation_id, message_id, %err, "failed to store utterance audio")
                }
            }
        });
        capture
    }

    fn start_reply(&self, session: &AgentSession) -> ReplyCapture {
        let (capture, mut rx, mut id_rx) = ReplyCapture::channel();
        let pool = self.pool.clone();
        let client_id = self.client_id;
        let audio_storage = self.audio_storage.clone();
        let conversation_id = session.conversation_id;
        tokio::spawn(async move {
            let Some(chunks) = collect(&mut rx).await else {
                return;
            };
            let Some(message_id) = id_rx.recv().await else {
                debug!(conversation_id, "reply audio discarded: no message row");
                return;
            };
            match store_audio(
                &pool,
                audio_storage.as_ref(),
                client_id,
                message_id,
                DOWNLINK.sample_rate,
                DOWNLINK.channels,
                &chunks,
            )
            .await
            {
                Ok(()) => debug!(conversation_id, message_id, "reply audio stored"),
                Err(err) => warn!(conversation_id, message_id, %err, "failed to store reply audio"),
            }
        });
        capture
    }
}

/// The session-lifecycle surface the driver drives: one capture per
/// utterance and per spoken reply. Starting a capture is cheap and
/// synchronous; the storage work happens in the capture's own task.
pub trait Capture: Send + Sync {
    /// Starts capturing the upcoming utterance (the device's uplink
    /// format). Finishing it attaches the audio to the user message row the
    /// memory commits; dropping the capture discards it.
    fn start_utterance(&self, session: &AgentSession) -> UtteranceCapture;

    /// Starts capturing the synthesized reply audio (the server's own
    /// downlink format). Finishing it attaches the audio to the last
    /// message row the memory commits for the turn — full or truncated
    /// partial alike; dropping the capture discards it.
    fn start_reply(&self, session: &AgentSession) -> ReplyCapture;
}

/// One capture message on a capture channel: a decoded chunk or the end of
/// the capture.
pub enum Frame {
    Chunk(Vec<f32>),
    Finish,
}

/// Handle over one utterance's audio capture. [`Clone`]: the driver keeps a
/// push side while the ASR task holds the finishing side; the channel closes
/// once every handle is gone, which is what marks a capture as discarded.
#[derive(Clone)]
pub struct UtteranceCapture {
    tx: mpsc::UnboundedSender<Frame>,
    id_tx: mpsc::Sender<i64>,
}

impl UtteranceCapture {
    /// Opens a capture channel for custom [`Capture`] implementations: the
    /// frame channel for [`collect`], the id receiver for the row the
    /// capture's own store call committed.
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<Frame>, mpsc::Receiver<i64>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let (id_tx, id_rx) = mpsc::channel(1);
        (Self { tx, id_tx }, rx, id_rx)
    }

    /// Feeds one decoded uplink chunk. Non-blocking by contract: the caller
    /// is the agent driver, which must never wait on storage.
    pub fn push(&self, samples: &[f32]) {
        let _ = self.tx.send(Frame::Chunk(samples.to_vec()));
    }

    /// Attaches the capture to the message row its own store call
    /// committed: the id comes straight from the awaited `store_utterance`
    /// return, so a capture can never receive a row that is not its own —
    /// captures overlap in time, a shared routing slot could not tell them
    /// apart. A second attach is ignored.
    pub fn attach(&self, message_id: i64) {
        let _ = self.id_tx.try_send(message_id);
    }

    /// Ends the capture: the audio attaches to the user message row the
    /// memory commits (bound through [`Self::attach`]). Dropping the
    /// capture without finishing discards it.
    pub fn finish(self) {
        let _ = self.tx.send(Frame::Finish);
    }
}

/// Handle over one reply's audio capture: the synthesized reply in the
/// server's downlink format. Finishing attaches the audio to the last
/// message row the memory commits for the turn; dropping it discards the
/// capture.
#[derive(Clone)]
pub struct ReplyCapture {
    tx: mpsc::UnboundedSender<Frame>,
    id_tx: mpsc::Sender<i64>,
}

impl ReplyCapture {
    /// Opens a capture channel for custom [`Capture`] implementations: the
    /// frame channel for [`collect`], the id receiver for the row the
    /// turn's store call committed.
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<Frame>, mpsc::Receiver<i64>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let (id_tx, id_rx) = mpsc::channel(1);
        (Self { tx, id_tx }, rx, id_rx)
    }

    /// Feeds one synthesized chunk. Non-blocking by contract: the caller is
    /// the agent driver, which must never wait on storage.
    pub fn push(&self, samples: &[f32]) {
        let _ = self.tx.send(Frame::Chunk(samples.to_vec()));
    }

    /// Attaches the capture to the message row its own store call
    /// committed — the last row of the turn, full or truncated partial
    /// alike. A second attach is ignored.
    pub fn attach(&self, message_id: i64) {
        let _ = self.id_tx.try_send(message_id);
    }

    /// The attach sink for the turn's store call (`log_items` /
    /// `store_partial_reply`): a [`Clone`] of this handle behind the
    /// callback, so the id lands even though the driver's own handle was
    /// finished by then — the store runs in a background task.
    pub fn attach_sink(&self) -> OnCommit {
        let capture = self.clone();
        Box::new(move |message_id| capture.attach(message_id))
    }

    /// Ends the capture: the audio attaches to the last message row the
    /// memory commits for the turn — the final reply row, the last tool row
    /// of a reply-less turn, or the truncated partial row for a reply cut
    /// before the model finished. Dropping the capture without finishing
    /// discards it.
    pub fn finish(self) {
        let _ = self.tx.send(Frame::Finish);
    }
}

/// Drains a capture channel into its chunks. `None` when the channel closed
/// without a finish (barge-in, empty utterance, session teardown) — the
/// capture is discarded. The extension pair for custom [`Capture`]
/// implementations, together with the handles' `channel()` constructors.
pub async fn collect(rx: &mut mpsc::UnboundedReceiver<Frame>) -> Option<Vec<Vec<f32>>> {
    let mut chunks = Vec::new();
    while let Some(frame) = rx.recv().await {
        match frame {
            Frame::Chunk(chunk) => chunks.push(chunk),
            Frame::Finish => return Some(chunks),
        }
    }
    None
}

/// Stores the captured audio against the committed message row: the FLAC
/// bytes go to the storage under a fresh object key, the row records the
/// bucket and key. No storage (unconfigured), no audio chunks (a reply the
/// user never heard), or a failed put store no row at all — the message
/// row is the primary record and the audio is best-effort.
async fn store_audio(
    pool: &PgPool,
    audio_storage: Option<&Arc<dyn AudioStorage>>,
    client_id: Uuid,
    message_id: i64,
    sample_rate: u32,
    channels: u16,
    chunks: &[Vec<f32>],
) -> Result<(), String> {
    let Some(audio_storage) = audio_storage else {
        return Ok(());
    };
    let total_samples: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    if total_samples == 0 {
        return Ok(());
    }
    let duration_ms = (total_samples as i64 / i64::from(channels)) * 1000 / i64::from(sample_rate);
    let audio = encode_flac(channels, sample_rate, chunks)?;
    let key = object_key(client_id);
    let bucket = audio_storage.put(&key, audio).await?;
    insert_audio(
        pool,
        message_id,
        &bucket,
        &key,
        sample_rate,
        channels,
        duration_ms,
    )
    .await
}

/// The object key of one capture: `audios/{YYYYMMDD}/{client_id}/{uuid}.flac`
/// — a per-day prefix, the owning device's client id, and a fresh uuid, so
/// keys never collide and never leak message ids into the storage layout.
fn object_key(client_id: Uuid) -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "audios/{:04}{:02}{:02}/{client_id}/{}.flac",
        now.year(),
        u8::from(now.month()),
        now.day(),
        Uuid::new_v4().simple()
    )
}

async fn insert_audio(
    pool: &PgPool,
    message_id: i64,
    bucket: &str,
    object_key: &str,
    sample_rate: u32,
    channels: u16,
    duration_ms: i64,
) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO message_audios (message_id, bucket, object_key, sample_rate, channels, duration_ms) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(message_id)
    .bind(bucket)
    .bind(object_key)
    .bind(i32::try_from(sample_rate).unwrap_or_default())
    .bind(i32::from(channels))
    .bind(i32::try_from(duration_ms).unwrap_or_default())
    .execute(pool)
    .await
    .map_err(|err| err.to_string())?;
    Ok(())
}

/// Encodes the capture as FLAC: 16-bit PCM, the capture's sample rate and
/// channels. The audio arrives as lossy Opus already, so this is the
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

/// The audio was decoded from 16-bit Opus, so rounding back to PCM16 keeps
/// the received fidelity.
fn pcm16_sample(sample: f32) -> i32 {
    (sample * 32767.0).round().clamp(-32768.0, 32767.0) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

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
    async fn dropped_capture_collects_nothing() {
        let (capture, mut rx, _id_rx) = UtteranceCapture::channel();
        capture.push(&[0.5]);
        drop(capture);
        assert!(collect(&mut rx).await.is_none());
    }

    #[tokio::test]
    async fn finished_capture_collects_chunks_in_order() {
        let (capture, mut rx, _id_rx) = ReplyCapture::channel();
        capture.push(&[0.25, 0.5]);
        capture.push(&[-0.25]);
        capture.finish();
        assert_eq!(
            collect(&mut rx).await,
            Some(vec![vec![0.25, 0.5], vec![-0.25]])
        );
    }

    #[tokio::test]
    async fn the_first_attach_wins_and_the_id_survives_the_finish() {
        let (capture, mut rx, mut id_rx) = UtteranceCapture::channel();
        capture.attach(7);
        capture.attach(9);
        capture.finish();
        assert!(collect(&mut rx).await.is_some());
        assert_eq!(id_rx.recv().await, Some(7));
    }

    #[tokio::test]
    async fn a_finished_capture_without_an_attach_yields_no_id() {
        let (capture, _rx, mut id_rx) = ReplyCapture::channel();
        capture.finish();
        assert_eq!(id_rx.recv().await, None);
    }
}
