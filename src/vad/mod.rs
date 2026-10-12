use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};

mod silero;

use self::silero::Silero;

use crate::audio::AudioStream;

pub const SAMPLE_RATE: u32 = 16000;

pub const MODEL_URL: &str = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx";
pub const MODEL_SHA256: &str = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3";
pub const DEFAULT_MODEL_PATH: &str = "models/silero_vad.onnx";

pub type VadEvents = Pin<Box<dyn Stream<Item = VadEvent> + Send>>;

#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    SpeechStart { at_ms: u64 },
    Speech { samples: Vec<f32> },
    SpeechEnd { at_ms: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentEvent {
    Start { at_ms: u64 },
    End { at_ms: u64 },
}

#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    pub speech_threshold: f32,
    pub silence_threshold: f32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub pre_padding_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence_threshold: 0.4,
            min_speech_ms: 200,
            min_silence_ms: 500,
            pre_padding_ms: 400,
        }
    }
}

impl VadConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(value) = env_parse("VAD_SPEECH_THRESHOLD") {
            config.speech_threshold = value;
        }
        if let Some(value) = env_parse("VAD_SILENCE_THRESHOLD") {
            config.silence_threshold = value;
        }
        if let Some(value) = env_parse("VAD_MIN_SPEECH_MS") {
            config.min_speech_ms = value;
        }
        if let Some(value) = env_parse("VAD_MIN_SILENCE_MS") {
            config.min_silence_ms = value;
        }
        if let Some(value) = env_parse("VAD_PRE_PADDING_MS") {
            config.pre_padding_ms = value;
        }
        config
    }
}

#[derive(Debug)]
struct PrePadding {
    capacity: usize,
    samples: VecDeque<f32>,
}

impl PrePadding {
    fn new(pre_padding_ms: u32, sample_rate: u32) -> Self {
        let capacity = (sample_rate as u64 * pre_padding_ms as u64 / 1000) as usize;
        Self {
            capacity,
            samples: VecDeque::with_capacity(capacity),
        }
    }

    fn push(&mut self, frame: &[f32]) {
        if self.capacity == 0 {
            return;
        }
        self.samples.extend(frame.iter().copied());
        while self.samples.len() > self.capacity {
            self.samples.pop_front();
        }
    }

    fn take(&mut self) -> Vec<f32> {
        self.samples.drain(..).collect()
    }
}

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|value| value.parse().ok())
}

#[derive(Debug, thiserror::Error)]
pub enum VadError {
    #[error("unsupported vad sample rate: {0}")]
    UnsupportedSampleRate(u32),
    #[error("vad model error: {0}")]
    Build(#[from] ort::Error),
    #[error(
        "silero vad model file {path}: {source}; download {MODEL_URL} \
         (sha256 {MODEL_SHA256}) there, or point SILERO_VAD_MODEL at the model file"
    )]
    Model {
        path: String,
        source: std::io::Error,
    },
}

pub trait Vad: Send {
    fn push(&mut self, samples: &[f32]) -> Vec<VadEvent>;
    fn flush(&mut self) -> Vec<VadEvent>;
}

pub trait VadFactory: Send + Sync {
    fn build(&self, sample_rate: u32) -> Result<Box<dyn Vad>, VadError>;
}

struct SpeechSegmenter {
    config: VadConfig,
    chunk_ms: u32,
    clock_ms: u64,
    in_speech: bool,
    speech_run_ms: u32,
    silence_run_ms: u32,
}

impl SpeechSegmenter {
    fn new(config: VadConfig, chunk_ms: u32) -> Self {
        Self {
            config,
            chunk_ms,
            clock_ms: 0,
            in_speech: false,
            speech_run_ms: 0,
            silence_run_ms: 0,
        }
    }

    fn push(&mut self, probability: f32) -> Option<SegmentEvent> {
        let at_ms = self.clock_ms;
        self.clock_ms += self.chunk_ms as u64;

        if self.in_speech {
            if probability < self.config.silence_threshold {
                self.silence_run_ms += self.chunk_ms;
                if self.silence_run_ms >= self.config.min_silence_ms {
                    self.in_speech = false;
                    self.speech_run_ms = 0;
                    self.silence_run_ms = 0;
                    return Some(SegmentEvent::End { at_ms });
                }
            } else {
                self.silence_run_ms = 0;
            }
        } else if probability >= self.config.speech_threshold {
            self.speech_run_ms += self.chunk_ms;
            if self.speech_run_ms >= self.config.min_speech_ms {
                self.in_speech = true;
                self.silence_run_ms = 0;
                self.speech_run_ms = 0;
                return Some(SegmentEvent::Start { at_ms });
            }
        } else {
            self.speech_run_ms = 0;
        }

        None
    }

    fn flush(&mut self) -> Option<SegmentEvent> {
        if !self.in_speech {
            return None;
        }
        self.in_speech = false;
        self.speech_run_ms = 0;
        self.silence_run_ms = 0;
        Some(SegmentEvent::End {
            at_ms: self.clock_ms,
        })
    }
}

pub struct SileroVad {
    model: Silero,
    chunk_size: usize,
    buffer: VecDeque<f32>,
    chunk: Vec<f32>,
    segmenter: SpeechSegmenter,
    pre_padding: PrePadding,
    active: bool,
}

impl SileroVad {
    pub fn new(sample_rate: u32, config: VadConfig, model: Arc<[u8]>) -> Result<Self, VadError> {
        let model = Silero::new(model, sample_rate)?;
        let chunk_size = if sample_rate == 16000 { 512 } else { 256 };
        let chunk_ms = (chunk_size as u64 * 1000 / sample_rate as u64) as u32;
        Ok(Self {
            model,
            chunk_size,
            buffer: VecDeque::with_capacity(chunk_size * 2),
            chunk: vec![0.0; chunk_size],
            segmenter: SpeechSegmenter::new(config, chunk_ms),
            pre_padding: PrePadding::new(config.pre_padding_ms, sample_rate),
            active: false,
        })
    }

    fn predict_push(&mut self, samples: &[f32], events: &mut Vec<SegmentEvent>) {
        self.buffer.extend(samples.iter().copied());
        while self.buffer.len() >= self.chunk_size {
            for slot in self.chunk.iter_mut() {
                *slot = self.buffer.pop_front().unwrap();
            }
            let probability = self.model.predict(&self.chunk);
            if let Some(event) = self.segmenter.push(probability) {
                events.push(event);
            }
        }
    }

    fn predict_flush(&mut self, events: &mut Vec<SegmentEvent>) {
        while !self.buffer.is_empty() {
            for slot in self.chunk.iter_mut() {
                *slot = self.buffer.pop_front().unwrap_or(0.0);
            }
            let probability = self.model.predict(&self.chunk);
            if let Some(event) = self.segmenter.push(probability) {
                events.push(event);
            }
        }
        if let Some(event) = self.segmenter.flush() {
            events.push(event);
        }
    }

    fn boundaries_to_events(&mut self, boundaries: Vec<SegmentEvent>) -> Vec<VadEvent> {
        let mut events = Vec::new();
        for event in boundaries {
            match event {
                SegmentEvent::Start { at_ms } => {
                    self.active = true;
                    let pre_padding = self.pre_padding.take();
                    events.push(VadEvent::SpeechStart { at_ms });
                    if !pre_padding.is_empty() {
                        events.push(VadEvent::Speech {
                            samples: pre_padding,
                        });
                    }
                }
                SegmentEvent::End { at_ms } => {
                    self.active = false;
                    events.push(VadEvent::SpeechEnd { at_ms });
                }
            }
        }
        events
    }
}

impl Vad for SileroVad {
    fn push(&mut self, samples: &[f32]) -> Vec<VadEvent> {
        let mut boundaries = Vec::new();
        self.predict_push(samples, &mut boundaries);
        let events = self.boundaries_to_events(boundaries);
        if self.active {
            let mut events = events;
            events.push(VadEvent::Speech {
                samples: samples.to_vec(),
            });
            events
        } else {
            self.pre_padding.push(samples);
            events
        }
    }

    fn flush(&mut self) -> Vec<VadEvent> {
        let mut boundaries = Vec::new();
        self.predict_flush(&mut boundaries);
        self.boundaries_to_events(boundaries)
    }
}

#[derive(Clone)]
pub struct SileroVadFactory {
    config: VadConfig,
    model: Arc<[u8]>,
}

impl SileroVadFactory {
    pub fn new(config: VadConfig, model: Arc<[u8]>) -> Self {
        Self { config, model }
    }

    pub fn from_env() -> Result<Self, VadError> {
        let path = std::env::var("SILERO_VAD_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_PATH.into());
        let model = std::fs::read(&path).map_err(|source| VadError::Model {
            path: path.clone(),
            source,
        })?;
        Ok(Self::new(VadConfig::from_env(), Arc::from(model)))
    }
}

impl VadFactory for SileroVadFactory {
    fn build(&self, sample_rate: u32) -> Result<Box<dyn Vad>, VadError> {
        SileroVad::new(sample_rate, self.config, self.model.clone())
            .map(|vad| Box::new(vad) as Box<dyn Vad>)
    }
}

pub fn segment(vad: Box<dyn Vad>, audio: AudioStream) -> VadEvents {
    let state = (vad, audio, VecDeque::new(), false);
    Box::pin(futures_util::stream::unfold(
        state,
        |(mut vad, mut audio, mut pending, mut finished)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((event, (vad, audio, pending, finished)));
                }
                if finished {
                    return None;
                }
                match audio.next().await {
                    Some(frame) => pending.extend(vad.push(&frame)),
                    None => {
                        pending.extend(vad.flush());
                        finished = true;
                    }
                }
            }
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segmenter() -> SpeechSegmenter {
        SpeechSegmenter::new(
            VadConfig {
                speech_threshold: 0.5,
                silence_threshold: 0.35,
                min_speech_ms: 250,
                min_silence_ms: 500,
                pre_padding_ms: 300,
            },
            32,
        )
    }

    #[test]
    fn pre_padding_keeps_only_recent_audio() {
        let mut pre_padding = PrePadding::new(100, 16000);
        pre_padding.push(&vec![1.0; 1600]);
        pre_padding.push(&[2.0; 100]);
        let kept = pre_padding.take();
        assert_eq!(kept.len(), 1600);
        assert_eq!(kept[0], 1.0);
        assert_eq!(kept[1599], 2.0);
        assert_eq!(kept.iter().filter(|&&s| s == 2.0).count(), 100);
        assert!(pre_padding.take().is_empty());
    }

    #[test]
    fn pre_padding_disabled_when_zero() {
        let mut pre_padding = PrePadding::new(0, 16000);
        pre_padding.push(&[1.0; 512]);
        assert!(pre_padding.take().is_empty());
    }

    #[test]
    fn pre_padding_take_drains() {
        let mut pre_padding = PrePadding::new(300, 16000);
        pre_padding.push(&[0.5; 4800]);
        assert_eq!(pre_padding.take().len(), 4800);
        assert!(pre_padding.take().is_empty());
        pre_padding.push(&[0.1; 100]);
        assert_eq!(pre_padding.take(), vec![0.1; 100]);
    }

    #[test]
    fn silence_emits_nothing() {
        let mut seg = segmenter();
        for _ in 0..200 {
            assert_eq!(seg.push(0.05), None);
        }
        assert_eq!(seg.flush(), None);
    }

    #[test]
    fn short_speech_is_ignored() {
        let mut seg = segmenter();
        assert_eq!(seg.push(0.9), None);
        assert_eq!(seg.push(0.9), None);
        for _ in 0..20 {
            assert_eq!(seg.push(0.05), None);
        }
    }

    #[test]
    fn speech_then_silence_emits_segment() {
        let mut seg = segmenter();
        let mut events = Vec::new();
        for _ in 0..10 {
            events.extend(seg.push(0.9));
        }
        assert_eq!(events, vec![SegmentEvent::Start { at_ms: 224 }]);

        events.clear();
        for _ in 0..20 {
            events.extend(seg.push(0.05));
        }
        assert_eq!(events, vec![SegmentEvent::End { at_ms: 800 }]);
    }

    #[test]
    fn flush_ends_open_segment() {
        let mut seg = segmenter();
        for _ in 0..10 {
            seg.push(0.9);
        }
        assert_eq!(seg.flush(), Some(SegmentEvent::End { at_ms: 320 }),);
    }

    #[test]
    fn unsupported_sample_rate() {
        assert!(matches!(
            SileroVad::new(24000, VadConfig::default(), Arc::from(Vec::new())),
            Err(VadError::UnsupportedSampleRate(24000))
        ));
    }

    #[tokio::test]
    #[ignore = "needs the silero model; run `just model`"]
    async fn real_model_silence_produces_no_events() {
        let factory = SileroVadFactory::from_env().unwrap();
        let vad = factory.build(SAMPLE_RATE).unwrap();
        let frames = (0..20).map(|_| vec![0.0f32; 960]).collect();
        let events: Vec<_> = segment(vad, audio_stream(frames)).collect().await;
        assert!(events.is_empty(), "unexpected events: {events:?}");
    }

    #[tokio::test]
    #[ignore = "needs the silero model (`just model`) and a 16 kHz mono speech wav at models/vad-test.wav (VAD_TEST_WAV overrides)"]
    async fn real_model_speech_emits_segment() {
        let wav = std::env::var("VAD_TEST_WAV").unwrap_or_else(|_| "models/vad-test.wav".into());
        let mut reader = hound::WavReader::open(&wav).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16000, "fixture must be 16 kHz");
        assert_eq!(spec.channels, 1, "fixture must be mono");
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / i16::MAX as f32)
            .collect();

        let factory = SileroVadFactory::from_env().unwrap();
        let vad = factory.build(SAMPLE_RATE).unwrap();
        let events: Vec<_> = segment(vad, audio_stream(vec![samples])).collect().await;
        assert!(
            events
                .iter()
                .any(|e| matches!(e, VadEvent::SpeechStart { .. })),
            "no SpeechStart: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, VadEvent::SpeechEnd { .. })),
            "no SpeechEnd: {events:?}"
        );
    }

    fn audio_stream(frames: Vec<Vec<f32>>) -> AudioStream {
        Box::pin(futures_util::stream::iter(frames))
    }
}
