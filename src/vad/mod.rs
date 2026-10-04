use std::collections::VecDeque;
use std::fmt;
use std::pin::Pin;

use futures_util::{Stream, StreamExt};
use voice_activity_detector::VoiceActivityDetector;

use crate::audio::AudioStream;

pub const SAMPLE_RATE: u32 = 16000;

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
            speech_threshold: 0.4,
            silence_threshold: 0.3,
            min_speech_ms: 120,
            min_silence_ms: 500,
            pre_padding_ms: 200,
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

#[derive(Debug)]
pub enum VadError {
    UnsupportedSampleRate(u32),
    Build(voice_activity_detector::Error),
}

impl fmt::Display for VadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VadError::UnsupportedSampleRate(rate) => {
                write!(f, "unsupported vad sample rate: {rate}")
            }
            VadError::Build(err) => write!(f, "vad model error: {err}"),
        }
    }
}

impl std::error::Error for VadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VadError::Build(err) => Some(err),
            VadError::UnsupportedSampleRate(_) => None,
        }
    }
}

impl From<voice_activity_detector::Error> for VadError {
    fn from(err: voice_activity_detector::Error) -> Self {
        VadError::Build(err)
    }
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

pub struct Vad {
    model: VoiceActivityDetector,
    chunk_size: usize,
    buffer: VecDeque<f32>,
    chunk: Vec<f32>,
    segmenter: SpeechSegmenter,
    pre_padding: PrePadding,
}

impl Vad {
    pub fn new(sample_rate: u32, config: VadConfig) -> Result<Self, VadError> {
        match sample_rate {
            16000 | 8000 => {}
            other => return Err(VadError::UnsupportedSampleRate(other)),
        }
        let chunk_size = if sample_rate == 8000 { 256 } else { 512 };
        let model = VoiceActivityDetector::builder()
            .sample_rate(sample_rate as i64)
            .chunk_size(chunk_size)
            .build()?;
        let chunk_ms = (chunk_size as u64 * 1000 / sample_rate as u64) as u32;
        Ok(Self {
            model,
            chunk_size,
            buffer: VecDeque::with_capacity(chunk_size * 2),
            chunk: vec![0.0; chunk_size],
            segmenter: SpeechSegmenter::new(config, chunk_ms),
            pre_padding: PrePadding::new(config.pre_padding_ms, sample_rate),
        })
    }

    pub fn segment(self, audio: AudioStream) -> VadEvents {
        let state = SegmentState {
            vad: self,
            audio,
            pending: VecDeque::new(),
            active: false,
            finished: false,
        };
        Box::pin(futures_util::stream::unfold(
            state,
            |mut state| async move {
                loop {
                    if let Some(event) = state.pending.pop_front() {
                        return Some((event, state));
                    }
                    if state.finished {
                        return None;
                    }
                    match state.audio.next().await {
                        Some(frame) => state.process(frame),
                        None => state.finish(),
                    }
                }
            },
        ))
    }

    fn push(&mut self, samples: &[f32], events: &mut Vec<SegmentEvent>) {
        self.buffer.extend(samples.iter().copied());
        while self.buffer.len() >= self.chunk_size {
            for slot in self.chunk.iter_mut() {
                *slot = self.buffer.pop_front().unwrap();
            }
            let probability = self.model.predict(self.chunk.iter().copied());
            if let Some(event) = self.segmenter.push(probability) {
                events.push(event);
            }
        }
    }

    fn flush(&mut self, events: &mut Vec<SegmentEvent>) {
        while !self.buffer.is_empty() {
            for slot in self.chunk.iter_mut() {
                *slot = self.buffer.pop_front().unwrap_or(0.0);
            }
            let probability = self.model.predict(self.chunk.iter().copied());
            if let Some(event) = self.segmenter.push(probability) {
                events.push(event);
            }
        }
        if let Some(event) = self.segmenter.flush() {
            events.push(event);
        }
    }
}

struct SegmentState {
    vad: Vad,
    audio: AudioStream,
    pending: VecDeque<VadEvent>,
    active: bool,
    finished: bool,
}

impl SegmentState {
    fn process(&mut self, frame: Vec<f32>) {
        let mut boundaries = Vec::new();
        self.vad.push(&frame, &mut boundaries);
        for event in boundaries {
            match event {
                SegmentEvent::Start { at_ms } => {
                    self.active = true;
                    let pre_padding = self.vad.pre_padding.take();
                    self.pending.push_back(VadEvent::SpeechStart { at_ms });
                    if !pre_padding.is_empty() {
                        self.pending.push_back(VadEvent::Speech {
                            samples: pre_padding,
                        });
                    }
                }
                SegmentEvent::End { at_ms } => {
                    self.active = false;
                    self.pending.push_back(VadEvent::SpeechEnd { at_ms });
                }
            }
        }
        if self.active {
            self.pending.push_back(VadEvent::Speech { samples: frame });
        } else {
            self.vad.pre_padding.push(&frame);
        }
    }

    fn finish(&mut self) {
        let mut boundaries = Vec::new();
        self.vad.flush(&mut boundaries);
        for event in boundaries {
            if let SegmentEvent::End { at_ms } = event {
                self.active = false;
                self.pending.push_back(VadEvent::SpeechEnd { at_ms });
            }
        }
        self.finished = true;
    }
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
            Vad::new(24000, VadConfig::default()),
            Err(VadError::UnsupportedSampleRate(24000))
        ));
    }

    #[tokio::test]
    async fn real_model_silence_produces_no_events() {
        let vad = Vad::new(SAMPLE_RATE, VadConfig::default()).unwrap();
        let frames = (0..20).map(|_| vec![0.0f32; 960]).collect();
        let events: Vec<_> = vad.segment(audio_stream(frames)).collect().await;
        assert!(events.is_empty(), "unexpected events: {events:?}");
    }

    fn audio_stream(frames: Vec<Vec<f32>>) -> AudioStream {
        Box::pin(futures_util::stream::iter(frames))
    }
}
