use std::fmt;

use opus::{Channels, Decoder};

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
}
