use std::fmt;
use std::pin::Pin;

use futures_util::{Stream, StreamExt};

pub mod volc;

pub type AudioStream = Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>;
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

pub trait Asr: Send + Sync {
    fn transcribe(&self, audio: AudioStream) -> AsrEvents<'_>;
}

pub struct StubAsr {
    text: String,
}

impl StubAsr {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl Default for StubAsr {
    fn default() -> Self {
        Self::new("你好")
    }
}

impl Asr for StubAsr {
    fn transcribe(&self, audio: AudioStream) -> AsrEvents<'_> {
        let text = self.text.clone();
        Box::pin(futures_util::stream::unfold(
            (audio, Some(text)),
            |(mut audio, text)| async move {
                let text = text?;
                while audio.next().await.is_some() {}
                Some((Ok(AsrEvent::Final { text }), (audio, None)))
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn audio_stream(mut rx: mpsc::Receiver<Vec<f32>>) -> AudioStream {
        Box::pin(futures_util::stream::poll_fn(move |cx| rx.poll_recv(cx)))
    }

    #[tokio::test]
    async fn stub_emits_final_after_stream_ends() {
        let asr = StubAsr::new("hello");
        let (tx, rx) = mpsc::channel(4);
        let mut events = asr.transcribe(audio_stream(rx));

        tx.send(vec![0.0; 160]).await.unwrap();
        assert!(tx.send(vec![0.5; 160]).await.is_ok());
        drop(tx);

        let mut finals = Vec::new();
        while let Some(event) = events.next().await {
            if let Ok(AsrEvent::Final { text }) = event {
                finals.push(text);
            }
        }
        assert_eq!(finals, vec!["hello".to_string()]);
    }
}
