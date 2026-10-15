//! Enrollment: the flow from selected user-message audio to a per-name
//! speaker. The caller names captured messages; the flow verifies each
//! one belongs to the caller's agent and is a user message with a capture
//! row, fetches the FLAC objects through the storage port (the row's
//! bucket, never the current one), decodes and gates them (16 kHz mono
//! 16-bit, 1–60 s per segment), embeds every segment with the boot-loaded
//! CAM++ model, and appends the L2-normalized vectors element-wise into the
//! name's sum. Request-side problems escalate before capability problems
//! — ownership, disabled audio capture, and rejected messages (with
//! per-message reasons) all answer before the model slot — and enrollment
//! is all-or-nothing: any rejected message fails the whole request, nothing
//! is appended.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;

use sqlx::PgPool;
use tokio::sync::Mutex;

use crate::agent::AudioStorage;
use crate::error::AppError;
use crate::voiceprint::{EMBEDDING_DIM, SAMPLE_RATE, SpeakerRecord, SpeakerStore, Voiceprint};

const MIN_SEGMENT_SAMPLES: usize = SAMPLE_RATE as usize;
const MAX_SEGMENT_SAMPLES: usize = 60 * SAMPLE_RATE as usize;

#[derive(sqlx::FromRow)]
struct MessageRow {
    id: i64,
    role: Option<String>,
    bucket: Option<String>,
    object_key: Option<String>,
}

/// Runs the enrollment for `name` under `agent_id`. `storage` and `model`
/// come from the [`crate::state::ServerState`] slots; a missing storage is
/// 400 (no audio can ever be read), a missing model 503 (the feature is
/// off). The agent's ownership is the caller's responsibility — it is
/// checked before this runs.
#[allow(clippy::too_many_arguments)]
pub async fn enroll(
    pool: &PgPool,
    storage: Option<Arc<dyn AudioStorage>>,
    model: Option<Arc<Mutex<Voiceprint>>>,
    agent_id: i64,
    user_id: i64,
    name: &str,
    description: Option<&str>,
    message_ids: &[i64],
) -> Result<SpeakerRecord, AppError> {
    let storage =
        storage.ok_or_else(|| AppError::BadRequest("audio capture is not enabled".into()))?;

    let mut ids = Vec::with_capacity(message_ids.len());
    let mut seen = HashSet::new();
    for id in message_ids {
        if seen.insert(*id) {
            ids.push(*id);
        }
    }

    let rows = sqlx::query_as::<_, MessageRow>(
        "SELECT m.id, m.message->>'role' AS role, ma.bucket, ma.object_key \
         FROM messages m \
         JOIN device_sessions ds ON ds.id = m.session_id \
         LEFT JOIN message_audios ma ON ma.message_id = m.id \
         WHERE ds.agent_id = $1 AND m.id = ANY($2) \
         ORDER BY m.id, ma.id",
    )
    .bind(agent_id)
    .bind(&ids)
    .fetch_all(pool)
    .await?;
    let mut found: HashMap<i64, MessageRow> = HashMap::with_capacity(rows.len());
    for row in rows {
        found.entry(row.id).or_insert(row);
    }

    let mut candidates = Vec::with_capacity(ids.len());
    let mut problems = Vec::new();
    for id in ids {
        match found.get(&id) {
            None => problems.push(format!("{id}: not in scope")),
            Some(row) => {
                if row.role.as_deref() != Some("user") {
                    problems.push(format!("{id}: not a user message"));
                } else {
                    match (&row.bucket, &row.object_key) {
                        (Some(bucket), Some(object_key)) => {
                            candidates.push((id, bucket, object_key))
                        }
                        _ => problems.push(format!("{id}: no audio captured")),
                    }
                }
            }
        }
    }

    let mut segments = Vec::with_capacity(candidates.len());
    for (id, bucket, object_key) in candidates {
        match storage.get(bucket, object_key).await {
            Err(err) => problems.push(format!("{id}: audio storage: {err}")),
            Ok(bytes) => match decode_segment(&bytes) {
                Err(err) => problems.push(format!("{id}: {err}")),
                Ok(pcm) => segments.push(pcm),
            },
        }
    }
    if !problems.is_empty() {
        return Err(AppError::BadRequest(format!(
            "rejected message ids: {}",
            problems.join(", ")
        )));
    }

    let model = model
        .ok_or_else(|| AppError::ServiceUnavailable("voiceprint model is not loaded".into()))?;

    let sample_count = segments.len() as i32;
    let sum = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, AppError> {
        let mut guard = model.blocking_lock();
        let mut sum = vec![0.0; EMBEDDING_DIM];
        for pcm in &segments {
            let embedding = guard
                .embed(pcm)
                .map_err(|err| AppError::Internal(format!("voiceprint embedding failed: {err}")))?;
            add_normalized(&mut sum, &embedding)
                .map_err(|err| AppError::Internal(format!("voiceprint {err}")))?;
        }
        Ok(sum)
    })
    .await
    .map_err(|err| AppError::Internal(format!("embedding task failed: {err}")))??;

    SpeakerStore::new(pool.clone())
        .create_or_append(agent_id, user_id, name, description, sum, sample_count)
        .await
}

/// Decodes one FLAC capture into the PCM the frontend expects: 16 kHz mono
/// 16-bit, between one second and one minute. The repository has no
/// resampler, so anything else is a rejected segment rather than converted.
fn decode_segment(bytes: &[u8]) -> Result<Vec<i16>, String> {
    let mut reader = claxon::FlacReader::new(Cursor::new(bytes))
        .map_err(|err| format!("audio decode failed: {err}"))?;
    let info = reader.streaminfo();
    if info.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "audio is {} Hz, expected {SAMPLE_RATE} Hz",
            info.sample_rate
        ));
    }
    if info.channels != 1 {
        return Err(format!(
            "audio has {} channels, expected mono",
            info.channels
        ));
    }
    if info.bits_per_sample != 16 {
        return Err(format!(
            "audio is {}-bit, expected 16-bit",
            info.bits_per_sample
        ));
    }
    let samples: Vec<i16> = reader
        .samples()
        .collect::<Result<Vec<i32>, _>>()
        .map_err(|err| format!("audio decode failed: {err}"))?
        .into_iter()
        .map(|sample| sample as i16)
        .collect();
    let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
    if samples.len() < MIN_SEGMENT_SAMPLES {
        return Err(format!(
            "audio too short ({seconds:.1} s, need at least 1 s)"
        ));
    }
    if samples.len() > MAX_SEGMENT_SAMPLES {
        return Err(format!("audio too long ({seconds:.1} s, at most 60 s)"));
    }
    Ok(samples)
}

/// Adds one segment's L2-normalized embedding into the running sum. The
/// stored vector is the sum of normalized segment vectors — cosine is
/// scale-invariant, so scoring the sum equals scoring the mean, and the
/// original audio never needs to be kept.
fn add_normalized(sum: &mut [f32], embedding: &[f32]) -> Result<(), String> {
    if embedding.len() != sum.len() {
        return Err(format!(
            "embedding length {} != {}",
            embedding.len(),
            sum.len()
        ));
    }
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err("produced a degenerate embedding".into());
    }
    for (acc, value) in sum.iter_mut().zip(embedding) {
        *acc += value / norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes 16-bit PCM as FLAC, mirroring the capture path's tail
    /// padding: the encoder would otherwise emit an illegal short frame.
    fn flac_bytes(sample_rate: u32, channels: usize, samples: &[i16]) -> Vec<u8> {
        use flacenc::component::BitRepr;
        use flacenc::error::Verify;

        let config = flacenc::config::Encoder::default().into_verified().unwrap();
        let mut padded: Vec<i32> = samples.iter().map(|&s| i32::from(s)).collect();
        let remainder = padded.len() % config.block_size;
        if remainder != 0 && remainder < 16 {
            padded.extend(std::iter::repeat_n(0, 16 - remainder));
        }
        let source =
            flacenc::source::MemSource::from_samples(&padded, channels, 16, sample_rate as usize);
        let stream =
            flacenc::encode_with_fixed_block_size(&config, source, config.block_size).unwrap();
        let mut sink = flacenc::bitsink::ByteSink::new();
        stream.write(&mut sink).unwrap();
        sink.as_slice().to_vec()
    }

    #[test]
    fn decodes_a_valid_segment_roundtrip() {
        let samples: Vec<i16> = (0..SAMPLE_RATE as usize)
            .map(|i| ((i % 97) as i16 * 200).saturating_sub(8000))
            .collect();
        let decoded = decode_segment(&flac_bytes(SAMPLE_RATE, 1, &samples)).unwrap();
        assert_eq!(&decoded[..samples.len()], &samples[..]);
    }

    #[test]
    fn rejects_other_sample_rates_and_channel_layouts() {
        let samples = vec![0i16; SAMPLE_RATE as usize];
        let err = decode_segment(&flac_bytes(48_000, 1, &samples)).unwrap_err();
        assert!(err.contains("Hz"), "{err}");
        let stereo = vec![0i16; 2 * SAMPLE_RATE as usize];
        let err = decode_segment(&flac_bytes(SAMPLE_RATE, 2, &stereo)).unwrap_err();
        assert!(err.contains("channels"), "{err}");
    }

    #[test]
    fn rejects_segments_outside_the_duration_window() {
        let err = decode_segment(&flac_bytes(SAMPLE_RATE, 1, &[100i16; 100])).unwrap_err();
        assert!(err.contains("too short"), "{err}");
        let long = vec![0i16; 61 * SAMPLE_RATE as usize];
        let err = decode_segment(&flac_bytes(SAMPLE_RATE, 1, &long)).unwrap_err();
        assert!(err.contains("too long"), "{err}");
    }

    #[test]
    fn rejects_bytes_that_are_not_flac() {
        let err = decode_segment(&[0u8; 64]).unwrap_err();
        assert!(err.contains("decode failed"), "{err}");
    }

    #[test]
    fn folds_l2_normalized_vectors_elementwise() {
        let mut sum = vec![0.0f32; 2];
        add_normalized(&mut sum, &[3.0, 4.0]).unwrap();
        assert_eq!(sum, [0.6, 0.8]);
        add_normalized(&mut sum, &[1.0, 0.0]).unwrap();
        assert_eq!(sum, [1.6, 0.8]);
    }

    #[test]
    fn folds_reject_degenerate_and_mismatched_embeddings() {
        let mut sum = vec![0.0f32; 2];
        assert!(add_normalized(&mut sum, &[0.0, 0.0]).is_err());
        assert!(add_normalized(&mut sum, &[1.0, 0.0, 0.0]).is_err());
    }
}
