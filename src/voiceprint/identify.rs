//! Session-side identification: the per-connection matcher over one
//! agent's enrolled voiceprints. `AgentFactory::build` constructs it once
//! per device session from the boot-loaded model and a single read of the
//! agent's voiceprints — an agent without enrollments (or without the
//! model) gets no matcher and runs no identification at all. The listening
//! stage opens a tap per utterance ([`SpeakerMatcher::tap`], driven exactly
//! like an audio capture); the collected 16 kHz speech is embedded off the
//! async runtime and scored against the enrolled labels. The outcome is a
//! debug log line plus the [`Identification`] itself, which the listening
//! stage resolves into the reply turn's speaker attribution; a threshold
//! setting comes later.

use std::sync::Arc;

use tokio::sync::{Mutex, oneshot};
use tracing::{debug, warn};

use crate::agent::{AgentSession, UtteranceCapture, collect};
use crate::voiceprint::{EnrolledEmbedding, MATCH_THRESHOLD, Voiceprint, VoiceprintError, cosine};

/// The per-connection speaker matcher: the shared CAM++ model plus the
/// agent's enrolled voiceprints as they were when the device connected —
/// enrollments made later join at the next session, never mid-session.
#[derive(Clone)]
pub struct SpeakerMatcher {
    model: Arc<Mutex<Voiceprint>>,
    candidates: Vec<EnrolledEmbedding>,
}

impl SpeakerMatcher {
    /// Builds the matcher from the model slot and the agent's enrolled
    /// voiceprints; `None` for an empty candidate set — the caller skips
    /// the whole pipeline instead of running an empty matcher.
    pub fn new(model: Arc<Mutex<Voiceprint>>, candidates: Vec<EnrolledEmbedding>) -> Option<Self> {
        if candidates.is_empty() {
            return None;
        }
        Some(Self { model, candidates })
    }

    /// Opens the identification tap for one utterance and spawns its
    /// identification task. The tap collects the speech samples exactly
    /// like an audio capture (dropping it discards the utterance — barge-in,
    /// session teardown); on finish the collected 16 kHz PCM is quantized,
    /// embedded off the async runtime and scored. The outcome is a debug
    /// log line plus the identification on the tap's result channel.
    pub fn tap(&self, session: &AgentSession) -> SpeakerTap {
        let (tap, mut rx) = UtteranceCapture::channel();
        let (result_tx, result_rx) = oneshot::channel();
        let matcher = self.clone();
        let session_id = session.id.clone();
        tokio::spawn(async move {
            let outcome = async {
                let chunks = collect(&mut rx).await?;
                let pcm: Vec<i16> = chunks
                    .iter()
                    .flatten()
                    .map(|&sample| quantize(sample))
                    .collect();
                if pcm.is_empty() {
                    return None;
                }
                Some(matcher.identify(&pcm).await)
            }
            .await;
            match &outcome {
                Some(Ok(identification)) if identification.matched => {
                    debug!(
                        session_id,
                        label = %identification.label,
                        similarity = format_args!("{:.3}", identification.similarity),
                        "voiceprint identified"
                    )
                }
                Some(Ok(identification)) => {
                    debug!(
                        session_id,
                        label = %identification.label,
                        similarity = format_args!("{:.3}", identification.similarity),
                        "voiceprint below threshold"
                    )
                }
                Some(Err(err)) => warn!(session_id, %err, "voiceprint identification failed"),
                None => {}
            }
            let _ = result_tx.send(outcome.and_then(|result| result.ok()));
        });
        SpeakerTap {
            capture: tap,
            result: result_rx,
        }
    }

    /// Embeds one utterance of 16 kHz mono s16le PCM and scores it against
    /// the enrolled voiceprints: the best cosine wins.
    pub async fn identify(&self, pcm: &[i16]) -> Result<Identification, VoiceprintError> {
        let embedding = self.embed(pcm).await?;
        Ok(score(&self.candidates, &embedding))
    }

    async fn embed(&self, pcm: &[i16]) -> Result<Vec<f32>, VoiceprintError> {
        let model = self.model.clone();
        let pcm = pcm.to_vec();
        tokio::task::spawn_blocking(move || model.blocking_lock().embed(&pcm))
            .await
            .map_err(|err| VoiceprintError::Task(err.to_string()))?
    }
}

/// The outcome of one identification: the best-scoring label, its
/// description, its cosine similarity and whether it cleared the match
/// threshold.
#[derive(Debug)]
pub struct Identification {
    pub label: String,
    pub description: String,
    pub similarity: f32,
    pub matched: bool,
}

/// The driver-side handle of one identification tap: the capture side
/// (pushed and finished exactly like an audio capture) plus the result
/// channel the listening stage resolves when the utterance's ASR final
/// turns into a turn. Dropping the tap without finishing discards the
/// utterance — the result channel then never carries a value.
pub struct SpeakerTap {
    capture: UtteranceCapture,
    result: oneshot::Receiver<Option<Identification>>,
}

impl SpeakerTap {
    /// Feeds one decoded uplink chunk, like an audio capture.
    pub fn push(&self, samples: &[f32]) {
        self.capture.push(samples);
    }

    /// Ends the tap — the speech segment is complete — and returns the
    /// result channel; the identification task sends at most one
    /// `Some(identification)` (a matched or unmatched outcome), `None`
    /// standing for a failed identification.
    pub fn finish(self) -> oneshot::Receiver<Option<Identification>> {
        self.capture.finish();
        self.result
    }
}

/// Scores one embedding against the candidates: the best cosine wins and
/// [`MATCH_THRESHOLD`] decides whether it counts as identified.
fn score(candidates: &[EnrolledEmbedding], embedding: &[f32]) -> Identification {
    let best = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.label.clone(),
                candidate.description.clone(),
                cosine(embedding, &candidate.embedding),
            )
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .expect("a matcher is only built with at least one candidate");
    Identification {
        label: best.0,
        description: best.1,
        similarity: best.2,
        matched: best.2 >= MATCH_THRESHOLD,
    }
}

/// Quantizes the uplink f32 samples to the s16le the frontend expects —
/// the same PCM16 rounding the capture path stores, so identification and
/// enrollment quantize identically. The audio arrived as lossy Opus;
/// nothing is lost beyond the decode.
fn quantize(sample: f32) -> i16 {
    (sample * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voiceprint::{DEFAULT_MODEL_PATH, SAMPLE_RATE};

    fn candidates() -> Vec<EnrolledEmbedding> {
        vec![
            EnrolledEmbedding {
                label: "a".into(),
                description: "desc a".into(),
                embedding: vec![1.0, 0.0],
            },
            EnrolledEmbedding {
                label: "b".into(),
                description: String::new(),
                embedding: vec![0.0, 1.0],
            },
        ]
    }

    #[test]
    fn scores_the_best_candidate_over_the_threshold() {
        let identification = score(&candidates(), &[0.9, 0.1]);
        assert_eq!(identification.label, "a");
        assert_eq!(identification.description, "desc a");
        assert!(identification.matched);
        assert!(identification.similarity > 0.99);
    }

    #[test]
    fn below_threshold_keeps_the_best_label_but_not_matched() {
        // -1.0 against "a", 0.0 against "b": the best score exists but is
        // nowhere near the threshold — the attribution stays undecided.
        let identification = score(&candidates(), &[-1.0, 0.0]);
        assert_eq!(identification.label, "b");
        assert_eq!(identification.similarity, 0.0);
        assert!(!identification.matched);
    }

    /// A synthetic utterance: summed sine waves at the given frequencies,
    /// one second long.
    fn tone(freqs: &[f32]) -> Vec<i16> {
        let rate = 16000.0f32;
        (0..rate as usize)
            .map(|i| {
                let t = i as f32 / rate;
                let mixed: f32 = freqs
                    .iter()
                    .map(|f| (t * f * std::f32::consts::TAU).sin())
                    .sum();
                (mixed / freqs.len() as f32 * 12000.0) as i16
            })
            .collect()
    }

    #[tokio::test]
    #[ignore] // needs the model on disk first: `just model`
    async fn identifies_the_enrolled_speaker() {
        let path =
            std::env::var("VOICEPRINT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
        let model = Arc::new(Mutex::new(Voiceprint::load(&path).expect("model loads")));

        let speech = tone(&[220.0, 440.0, 660.0]);
        let noise = tone(&[300.0, 900.0]);
        // Each embed is its own statement: a MutexGuard temporary lives to
        // the end of the statement, so two `lock().await`s in one `let`
        // would deadlock on the task's own guard.
        let speech_embedding = model.lock().await.embed(&speech).unwrap();
        let noise_embedding = model.lock().await.embed(&noise).unwrap();
        // Candidates hold the raw single-segment embeddings — cosine is
        // scale-invariant, so this scores exactly like the stored
        // normalized sums.
        let candidates = vec![
            EnrolledEmbedding {
                label: "张三".into(),
                description: "同事".into(),
                embedding: speech_embedding,
            },
            EnrolledEmbedding {
                label: "李四".into(),
                description: String::new(),
                embedding: noise_embedding,
            },
        ];
        let matcher = SpeakerMatcher::new(model, candidates).unwrap();

        let identified = matcher.identify(&speech).await.unwrap();
        assert_eq!(identified.label, "张三");
        assert!(identified.matched);
        assert!(identified.similarity > 0.9);

        let other = matcher.identify(&noise).await.unwrap();
        assert_eq!(other.label, "李四");
        assert!(other.matched);
    }

    #[tokio::test]
    #[ignore] // needs the model on disk first: `just model`
    async fn the_tap_delivers_the_identification_to_its_result_channel() {
        let path =
            std::env::var("VOICEPRINT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
        let model = Arc::new(Mutex::new(Voiceprint::load(&path).expect("model loads")));

        let speech = tone(&[220.0, 440.0, 660.0]);
        // Each embed is its own statement: a MutexGuard temporary lives to
        // the end of the statement, so two `lock().await`s in one `let`
        // would deadlock on the task's own guard.
        let speech_embedding = model.lock().await.embed(&speech).unwrap();
        let matcher = SpeakerMatcher::new(
            model,
            vec![EnrolledEmbedding {
                label: "张三".into(),
                description: "同事".into(),
                embedding: speech_embedding,
            }],
        )
        .unwrap();

        let session = AgentSession {
            id: "test".into(),
            sample_rate: SAMPLE_RATE,
            channels: 1,
            frame_duration_ms: 60,
        };
        let tap = matcher.tap(&session);
        let samples: Vec<f32> = speech.iter().map(|&s| f32::from(s) / 32768.0).collect();
        tap.push(&samples);
        let result = tap.finish().await.expect("the identification task sends");
        let identification = result.expect("the utterance embedded and scored");
        assert_eq!(identification.label, "张三");
        assert!(identification.matched);
    }
}
