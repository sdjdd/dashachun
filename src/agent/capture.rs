//! Audio storage: a byproduct feature that attaches captured audio to the
//! message rows the memory commits. The driver drives the session lifecycle
//! (speech and tts boundaries) through the [`Capture`] handles; the memory's
//! [`MemoryHook`]s deliver the message ids, so audio code never writes a
//! message row and never sees one being planned.

use std::sync::Mutex;

use sqlx::PgPool;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::debug;
use tracing::warn;

use crate::agent::AgentSession;
use crate::agent::memory::MemoryHook;
use crate::audio::DOWNLINK;

const BITS_PER_SAMPLE: usize = 16;

/// The audio-storage feature: captures the device's uplink and the
/// synthesized reply, and stores them as `message_audios` rows linked by
/// `message_id` to the rows the memory commits. One instance lives per
/// connection; register it as a hook on the memory so captures learn their
/// message ids, and hand it to the driver as the [`Capture`] implementation.
pub struct AudioCapture {
    pool: PgPool,
    open_utterance: Mutex<Option<oneshot::Sender<i64>>>,
    open_reply: Mutex<Option<oneshot::Sender<i64>>>,
}

impl AudioCapture {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            open_utterance: Mutex::new(None),
            open_reply: Mutex::new(None),
        }
    }
}

impl MemoryHook for AudioCapture {
    fn utterance_stored(&self, _session: &AgentSession, message_id: i64) {
        if let Some(tx) = self.open_utterance.lock().unwrap().take() {
            let _ = tx.send(message_id);
        }
    }

    fn turn_stored(&self, _session: &AgentSession, message_id: i64) {
        if let Some(tx) = self.open_reply.lock().unwrap().take() {
            let _ = tx.send(message_id);
        }
    }
}

impl Capture for AudioCapture {
    fn start_utterance(&self, session: &AgentSession) -> UtteranceCapture {
        let (capture, mut rx) = UtteranceCapture::channel();
        let (tx_id, rx_id) = oneshot::channel();
        *self.open_utterance.lock().unwrap() = Some(tx_id);
        let pool = self.pool.clone();
        let session_id = session.id.clone();
        let sample_rate = session.sample_rate;
        let channels = session.channels;
        tokio::spawn(async move {
            let Some(chunks) = collect(&mut rx).await else {
                return;
            };
            let Ok(message_id) = rx_id.await else {
                debug!(session_id, "utterance audio discarded: no message row");
                return;
            };
            match store_audio(&pool, message_id, sample_rate, channels, &chunks).await {
                Ok(()) => debug!(session_id, message_id, "utterance audio stored"),
                Err(err) => {
                    warn!(session_id, message_id, %err, "failed to store utterance audio")
                }
            }
        });
        capture
    }

    fn start_reply(&self, session: &AgentSession) -> ReplyCapture {
        let (capture, mut rx) = ReplyCapture::channel();
        let (tx_id, rx_id) = oneshot::channel();
        *self.open_reply.lock().unwrap() = Some(tx_id);
        let pool = self.pool.clone();
        let session_id = session.id.clone();
        tokio::spawn(async move {
            let Some(chunks) = collect(&mut rx).await else {
                return;
            };
            let Ok(message_id) = rx_id.await else {
                debug!(session_id, "reply audio discarded: no message row");
                return;
            };
            match store_audio(
                &pool,
                message_id,
                DOWNLINK.sample_rate,
                DOWNLINK.channels,
                &chunks,
            )
            .await
            {
                Ok(()) => debug!(session_id, message_id, "reply audio stored"),
                Err(err) => warn!(session_id, message_id, %err, "failed to store reply audio"),
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
}

impl UtteranceCapture {
    /// Opens a capture channel for custom [`Capture`] implementations. The
    /// consumer receives chunks in order and at most one finish, then the
    /// channel closes.
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<Frame>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    /// Feeds one decoded uplink chunk. Non-blocking by contract: the caller
    /// is the agent driver, which must never wait on storage.
    pub fn push(&self, samples: &[f32]) {
        let _ = self.tx.send(Frame::Chunk(samples.to_vec()));
    }

    /// Ends the capture: the audio attaches to the user message row the
    /// memory commits (delivered through the hook). Dropping the capture
    /// without finishing discards it.
    pub fn finish(self) {
        let _ = self.tx.send(Frame::Finish);
    }
}

/// Handle over one reply's audio capture: the synthesized reply in the
/// server's downlink format. Finishing attaches the audio to the last
/// message row the memory commits for the turn; dropping it discards the
/// capture.
pub struct ReplyCapture {
    tx: mpsc::UnboundedSender<Frame>,
}

impl ReplyCapture {
    /// Opens a capture channel for custom [`Capture`] implementations. The
    /// consumer receives chunks in order and at most one finish, then the
    /// channel closes.
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<Frame>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    /// Feeds one synthesized chunk. Non-blocking by contract: the caller is
    /// the agent driver, which must never wait on storage.
    pub fn push(&self, samples: &[f32]) {
        let _ = self.tx.send(Frame::Chunk(samples.to_vec()));
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

/// Stores the captured audio against the committed message row. No audio
/// chunks (a reply the user never heard) store nothing.
async fn store_audio(
    pool: &PgPool,
    message_id: i64,
    sample_rate: u32,
    channels: u16,
    chunks: &[Vec<f32>],
) -> Result<(), String> {
    let total_samples: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    if total_samples == 0 {
        return Ok(());
    }
    let duration_ms = (total_samples as i64 / i64::from(channels)) * 1000 / i64::from(sample_rate);
    let audio = encode_flac(channels, sample_rate, chunks)?;
    insert_audio(pool, message_id, sample_rate, channels, duration_ms, &audio).await
}

async fn insert_audio(
    pool: &PgPool,
    message_id: i64,
    sample_rate: u32,
    channels: u16,
    duration_ms: i64,
    audio: &[u8],
) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO message_audios (message_id, audio, sample_rate, channels, duration_ms) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(message_id)
    .bind(audio)
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
        let (capture, mut rx) = UtteranceCapture::channel();
        capture.push(&[0.5]);
        drop(capture);
        assert!(collect(&mut rx).await.is_none());
    }

    #[tokio::test]
    async fn finished_capture_collects_chunks_in_order() {
        let (capture, mut rx) = ReplyCapture::channel();
        capture.push(&[0.25, 0.5]);
        capture.push(&[-0.25]);
        capture.finish();
        assert_eq!(
            collect(&mut rx).await,
            Some(vec![vec![0.25, 0.5], vec![-0.25]])
        );
    }
}
