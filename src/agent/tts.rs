use std::fmt;
use std::pin::Pin;

use futures_util::Stream;
use tokio_util::sync::CancellationToken;

pub type TextStream = Pin<Box<dyn Stream<Item = String> + Send>>;
pub type TtsEvents<'a> = Pin<Box<dyn Stream<Item = Result<TtsEvent, TtsError>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq)]
pub struct Subtitle {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TtsEvent {
    SentenceStart { text: String },
    Subtitle(Subtitle),
    Audio(Vec<f32>),
    Done,
}

#[derive(Debug)]
pub enum TtsError {
    Failed(String),
}

impl fmt::Display for TtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TtsError::Failed(message) => write!(f, "tts failed: {message}"),
        }
    }
}

impl std::error::Error for TtsError {}

impl From<String> for TtsError {
    fn from(message: String) -> Self {
        TtsError::Failed(message)
    }
}

impl From<&str> for TtsError {
    fn from(message: &str) -> Self {
        TtsError::Failed(message.to_string())
    }
}

/// Streaming speech synthesis. Implementations must keep consuming the text
/// stream even while nobody polls the event stream, since the agent applies
/// backpressure on the text channel while draining events.
pub trait Tts: Send + Sync {
    fn synthesize(&self, text: TextStream, cancel: CancellationToken) -> TtsEvents<'_>;
}
