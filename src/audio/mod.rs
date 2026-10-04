use std::fmt;
use std::pin::Pin;

use futures_util::Stream;
use opus::{Application, Channels, Decoder, Encoder};

pub type AudioStream = Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>;

pub const SAMPLE_RATE: u32 = 16000;
pub const CHANNELS: u16 = 1;
pub const FRAME_DURATION_MS: u32 = 60;

const MAX_FRAME_SAMPLES: usize = 5760;

#[derive(Debug)]
pub enum DecodeError {
    UnsupportedChannels(u16),
    Opus(opus::Error),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnsupportedChannels(n) => write!(f, "unsupported channel count: {n}"),
            DecodeError::Opus(err) => write!(f, "opus decode failed: {err}"),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::Opus(err) => Some(err),
            DecodeError::UnsupportedChannels(_) => None,
        }
    }
}

impl From<opus::Error> for DecodeError {
    fn from(err: opus::Error) -> Self {
        DecodeError::Opus(err)
    }
}

#[derive(Debug)]
pub enum EncodeError {
    UnsupportedChannels(u16),
    Opus(opus::Error),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::UnsupportedChannels(n) => write!(f, "unsupported channel count: {n}"),
            EncodeError::Opus(err) => write!(f, "opus encode failed: {err}"),
        }
    }
}

impl std::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EncodeError::Opus(err) => Some(err),
            EncodeError::UnsupportedChannels(_) => None,
        }
    }
}

impl From<opus::Error> for EncodeError {
    fn from(err: opus::Error) -> Self {
        EncodeError::Opus(err)
    }
}

pub struct OpusDecoder {
    decoder: Decoder,
    channels: usize,
    buffer: Vec<f32>,
}

impl OpusDecoder {
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self, DecodeError> {
        let channel_layout = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            other => return Err(DecodeError::UnsupportedChannels(other)),
        };
        let decoder = Decoder::new(sample_rate, channel_layout)?;
        Ok(Self {
            decoder,
            channels: channels as usize,
            buffer: vec![0.0; MAX_FRAME_SAMPLES * channels as usize],
        })
    }

    pub fn decode(&mut self, packet: &[u8]) -> Result<&[f32], DecodeError> {
        let samples = self.decoder.decode_float(packet, &mut self.buffer, false)?;
        Ok(&self.buffer[..samples * self.channels])
    }
}

pub struct OpusEncoder {
    encoder: Encoder,
    frame_samples: usize,
    pending: Vec<f32>,
    packet: Vec<u8>,
}

impl OpusEncoder {
    pub fn new(
        sample_rate: u32,
        channels: u16,
        frame_duration_ms: u32,
    ) -> Result<Self, EncodeError> {
        let channel_layout = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            other => return Err(EncodeError::UnsupportedChannels(other)),
        };
        let encoder = Encoder::new(sample_rate, channel_layout, Application::Voip)?;
        let frame_samples =
            (sample_rate as usize * frame_duration_ms as usize / 1000) * channels as usize;
        Ok(Self {
            encoder,
            frame_samples,
            pending: Vec::new(),
            packet: vec![0u8; MAX_FRAME_SAMPLES * channels as usize],
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<Vec<u8>> {
        self.pending.extend_from_slice(samples);
        let mut packets = Vec::new();
        while self.pending.len() >= self.frame_samples {
            let frame: Vec<f32> = self.pending.drain(..self.frame_samples).collect();
            packets.push(self.encode_frame(&frame));
        }
        packets
    }

    pub fn flush(&mut self) -> Option<Vec<u8>> {
        if self.pending.is_empty() {
            return None;
        }
        let mut frame = std::mem::take(&mut self.pending);
        frame.resize(self.frame_samples, 0.0);
        Some(self.encode_frame(&frame))
    }

    pub fn reset(&mut self) {
        self.pending.clear();
    }

    fn encode_frame(&mut self, frame: &[f32]) -> Vec<u8> {
        let pcm: Vec<i16> = frame
            .iter()
            .map(|&sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        match self.encoder.encode(&pcm, &mut self.packet) {
            Ok(len) => self.packet[..len].to_vec(),
            Err(err) => {
                tracing::warn!(%err, "failed to encode opus frame");
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opus::{Application, Encoder};

    #[test]
    fn decodes_encoded_silence() {
        let mut encoder = Encoder::new(SAMPLE_RATE, Channels::Mono, Application::Voip).unwrap();
        let mut packet = [0u8; 4000];
        let pcm = vec![0i16; (SAMPLE_RATE / 1000 * FRAME_DURATION_MS) as usize];
        let len = encoder.encode(&pcm, &mut packet).unwrap();

        let mut decoder = OpusDecoder::new(SAMPLE_RATE, CHANNELS).unwrap();
        let decoded = decoder.decode(&packet[..len]).unwrap();
        assert_eq!(decoded.len(), pcm.len());
    }

    #[test]
    fn rejects_unsupported_channels() {
        assert!(matches!(
            OpusDecoder::new(SAMPLE_RATE, 3),
            Err(DecodeError::UnsupportedChannels(3))
        ));
    }

    #[test]
    fn encoder_buffers_to_frame_boundaries() {
        let mut encoder = OpusEncoder::new(SAMPLE_RATE, CHANNELS, FRAME_DURATION_MS).unwrap();
        let frame_samples = (SAMPLE_RATE / 1000 * FRAME_DURATION_MS) as usize;

        assert!(encoder.push(&vec![0.0; frame_samples - 1]).is_empty());
        let packets = encoder.push(&[0.0]);
        assert_eq!(packets.len(), 1);
        assert!(!packets[0].is_empty());
    }

    #[test]
    fn encoder_flush_emits_partial_frame() {
        let mut encoder = OpusEncoder::new(SAMPLE_RATE, CHANNELS, FRAME_DURATION_MS).unwrap();
        assert!(encoder.push(&vec![0.1; 100]).is_empty());
        let packet = encoder.flush().unwrap();
        assert!(!packet.is_empty());
        assert!(encoder.flush().is_none());
    }

    #[test]
    fn rejects_unsupported_encode_channels() {
        assert!(matches!(
            OpusEncoder::new(SAMPLE_RATE, 3, FRAME_DURATION_MS),
            Err(EncodeError::UnsupportedChannels(3))
        ));
    }

    #[test]
    fn every_encoded_packet_is_one_full_frame() {
        let mut encoder = OpusEncoder::new(SAMPLE_RATE, CHANNELS, FRAME_DURATION_MS).unwrap();
        let mut decoder = OpusDecoder::new(SAMPLE_RATE, CHANNELS).unwrap();
        let expected = (SAMPLE_RATE / 1000 * FRAME_DURATION_MS) as usize;

        let mut packets = Vec::new();
        for chunk in [7usize, 100, 300, 960, 50, 1, 1919] {
            let samples = vec![0.3f32; chunk];
            packets.extend(encoder.push(&samples));
        }
        packets.extend(encoder.flush());

        assert!(packets.len() >= 3);
        for packet in &packets {
            let decoded = decoder.decode(packet).unwrap();
            assert_eq!(decoded.len(), expected, "packet was not exactly one frame");
        }
    }
}
