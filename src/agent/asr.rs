use std::fmt;
use std::pin::Pin;

use futures_util::Stream;
use tokio_util::sync::CancellationToken;

pub use crate::audio::AudioStream;

pub type AsrEvents<'a> = Pin<Box<dyn Stream<Item = Result<AsrEvent, AsrError>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsrEvent {
    Partial { text: String },
    Final { text: String },
}

#[derive(Debug)]
pub enum AsrError {
    Failed(String),
}

impl fmt::Display for AsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsrError::Failed(message) => write!(f, "asr failed: {message}"),
        }
    }
}

impl std::error::Error for AsrError {}

impl From<String> for AsrError {
    fn from(message: String) -> Self {
        AsrError::Failed(message)
    }
}

impl From<&str> for AsrError {
    fn from(message: &str) -> Self {
        AsrError::Failed(message.to_string())
    }
}

/// Streaming speech recognition. Implementations must keep consuming the
/// audio stream even while nobody polls the event stream, since the agent
/// applies backpressure on the audio channel while draining events.
pub trait Asr: Send + Sync {
    fn transcribe(&self, audio: AudioStream, cancel: CancellationToken) -> AsrEvents<'_>;
}
