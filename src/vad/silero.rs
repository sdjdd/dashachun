//! Self-hosted Silero VAD v6 driver: an LSTM `[2, 1, 128]` state plus a
//! rolling 64-sample context (32 at 8 kHz) prepended to each 512-sample
//! window (256 at 8 kHz) form the `[1, 576]`/`[1, 288]` model input; the
//! model answers with the speech probability (`output`) and the next state
//! (`stateN`).
//!
//! The model file is fetched by `just model` (sha256-pinned; the URL/sha
//! constants live in `crate::vad` and double as the boot-error guidance).

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use super::VadError;

fn build<E: std::fmt::Display>(error: E) -> VadError {
    VadError::Build(error.to_string())
}

pub(crate) struct Silero {
    session: Session,
    sample_rate: Array1<i64>,
    state: ArrayD<f32>,
    context: Vec<f32>,
    context_size: usize,
}

impl Silero {
    pub(crate) fn new(model: Arc<[u8]>, sample_rate: u32) -> Result<Self, VadError> {
        match sample_rate {
            16000 | 8000 => {}
            other => return Err(VadError::UnsupportedSampleRate(other)),
        }
        let session = Session::builder()
            .map_err(build)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(build)?
            .commit_from_memory(&model)
            .map_err(build)?;
        let context_size = if sample_rate == 16000 { 64 } else { 32 };
        Ok(Self {
            session,
            sample_rate: Array1::from_vec(vec![i64::from(sample_rate)]),
            state: ArrayD::zeros(IxDyn(&[2, 1, 128])),
            context: vec![0.0; context_size],
            context_size,
        })
    }

    /// One window of 512 samples (256 at 8 kHz); the rolling context is
    /// prepended internally and state/context advance in place. Inference
    /// failures panic — the tensor shapes are fixed by construction, so a
    /// failure means a broken model, which matches the previous wrapper's
    /// behavior.
    pub(crate) fn predict(&mut self, window: &[f32]) -> f32 {
        let mut input = Vec::with_capacity(self.context_size + window.len());
        input.extend_from_slice(&self.context);
        input.extend_from_slice(window);
        let frame = Array2::from_shape_vec((1, input.len()), input).expect("flat input vec");
        let state = std::mem::take(&mut self.state);

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => Tensor::from_array(frame).expect("input tensor"),
                "state" => Tensor::from_array(state).expect("state tensor"),
                "sr" => Tensor::from_array(self.sample_rate.clone()).expect("sr tensor"),
            ])
            .expect("silero vad inference");

        let (shape, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .expect("state output");
        let dims: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        self.state = ArrayD::from_shape_vec(dims, state_data.to_vec()).expect("state shape");
        self.context = window[window.len() - self.context_size..].to_vec();

        outputs["output"]
            .try_extract_tensor::<f32>()
            .expect("probability output")
            .1[0]
    }
}
