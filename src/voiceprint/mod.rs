//! Speaker embeddings (voiceprint) — a standalone capability module, not an
//! agent-stage provider (no `Asr`-style trait). 16 kHz mono s16le PCM in,
//! CAM++ embedding out. The ONNX graph carries only the network; the Kaldi
//! fbank frontend (80 mel bins, 25 ms frames / 10 ms shift, dither 0) and
//! the per-utterance mel-mean normalization live here, matching the
//! sherpa-onnx `3d-speaker` export contract. Two consumers are live:
//! enrollment over captured conversation audio (`store.rs`, `enroll.rs`,
//! the `/api/agents/{agent_id}/voiceprints` REST) and session-side
//! identification (`identify.rs` — a per-utterance attribution whose matched
//! label the listening stage resolves into the reply turn's system prompt,
//! alongside the debug log line); a threshold setting comes later — the
//! integration notes live in this directory's README.md.

mod store;

pub mod dto;
mod enroll;
mod identify;

pub use enroll::enroll;
pub use identify::{Identification, SpeakerMatcher, SpeakerTap};
pub use store::{EnrolledEmbedding, VoiceprintRecord, VoiceprintStore};

use std::path::Path;

use kaldi_fbank::FbankOptions;
use ndarray::Array3;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

/// Where the pinned model is fetched from (`just model` writes it to
/// `models/`): the sherpa-onnx export of ModelScope's
/// `iic/speech_campplus_sv_zh-cn_16k-common`.
pub const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";

/// sha256 of the pinned model — the `just model` recipe verifies it, so a
/// silently rewritten file on the hosting repo cannot reach the server.
pub const MODEL_SHA256: &str = "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11";

/// The default location relative to the working directory.
pub const DEFAULT_MODEL_PATH: &str = "models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";

/// The sample rate the frontend is built for (the fbank stage computes at
/// 16 kHz); anything else must be resampled before [`Voiceprint::embed`].
pub const SAMPLE_RATE: u32 = 16_000;

/// Embedding dimension of the pinned model. A different 3d-speaker export
/// may differ — the ignored model test asserts this constant.
pub const EMBEDDING_DIM: usize = 192;

/// Cosine similarity above which an utterance is attributed to a voiceprint
/// (same speaker ≈ 0.65–0.8, different speakers ≈ 0). The voiceprint-api
/// convention; a settings surface can come later.
pub const MATCH_THRESHOLD: f32 = 0.4;

#[derive(Debug, thiserror::Error)]
pub enum VoiceprintError {
    #[error("onnx: {0}")]
    Onnx(String),
    #[error("embedding task failed: {0}")]
    Task(String),
    #[error("empty pcm input")]
    EmptyInput,
    #[error("model has no `embedding` output")]
    MissingOutput,
}

fn onnx<E: std::fmt::Display>(error: E) -> VoiceprintError {
    VoiceprintError::Onnx(error.to_string())
}

pub struct Voiceprint {
    session: Session,
}

impl Voiceprint {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, VoiceprintError> {
        let session = Session::builder()
            .map_err(onnx)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(onnx)?
            .with_intra_threads(2)
            .map_err(onnx)?
            .commit_from_file(path.as_ref())
            .map_err(onnx)?;
        Ok(Self { session })
    }

    /// Embed one utterance of 16 kHz mono s16le PCM. CPU inference is
    /// deterministic — the same samples always produce the same vector.
    pub fn embed(&mut self, pcm: &[i16]) -> Result<Vec<f32>, VoiceprintError> {
        if pcm.is_empty() {
            return Err(VoiceprintError::EmptyInput);
        }
        let samples: Vec<f32> = pcm.iter().map(|s| f32::from(*s) / 32768.0).collect();
        let mut opts = FbankOptions::default();
        opts.frame_opts.dither = 0.0;
        opts.mel_opts.num_bins = 80;
        let frames = kaldi_fbank::compute_fbank(&opts, &samples);
        let num_frames = frames.len();
        let mut mean = [0.0f32; 80];
        for frame in &frames {
            for (m, v) in mean.iter_mut().zip(frame) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= num_frames as f32;
        }
        let flat: Vec<f32> = frames
            .into_iter()
            .flat_map(|f| f.into_iter().zip(mean.iter()).map(|(v, m)| v - m))
            .collect();
        let features = Array3::from_shape_vec((1, num_frames, 80), flat)
            .expect("flat fbank data matches the frame count");

        let outputs = self
            .session
            .run(ort::inputs!["x" => Tensor::from_array(features).map_err(onnx)?])
            .map_err(onnx)?;
        let out = outputs
            .get("embedding")
            .ok_or(VoiceprintError::MissingOutput)?
            .try_extract_tensor::<f32>()
            .map_err(onnx)?;

        Ok(out.1.to_vec())
    }
}

/// Cosine similarity — the identity score between two embeddings. Two
/// utterances of the same speaker land around 0.65–0.8, different speakers
/// around 0; the voiceprint threshold is 0.4.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    let denom = na * nb;
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    #[ignore] // needs the model on disk first: `just model`
    fn embeds_deterministically_at_the_pinned_dimension() {
        let path =
            std::env::var("VOICEPRINT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
        let mut voiceprint = Voiceprint::load(&path).expect("model loads");

        let pcm: Vec<i16> = (0..SAMPLE_RATE as usize)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                ((t * 220.0).sin() * 8000.0
                    + (t * 440.0).sin() * 4000.0
                    + (t * 660.0).sin() * 2000.0) as i16
            })
            .collect();

        let first = voiceprint.embed(&pcm).expect("embeds");
        let second = voiceprint.embed(&pcm).expect("embeds again");

        assert_eq!(first.len(), EMBEDDING_DIM);
        assert!(first.iter().all(|v| v.is_finite()));
        assert!(cosine(&first, &second) > 0.999);
    }
}
