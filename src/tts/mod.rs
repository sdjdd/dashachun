use std::fmt;
use std::pin::Pin;

use futures_util::{Stream, StreamExt};

pub mod volc;

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

pub trait Tts: Send + Sync {
    fn synthesize(&self, text: TextStream) -> TtsEvents<'_>;
}

pub struct StubTts;

impl Tts for StubTts {
    fn synthesize(&self, text: TextStream) -> TtsEvents<'_> {
        Box::pin(futures_util::stream::unfold(
            (text, Vec::<TtsEvent>::new(), false),
            |(mut text, mut pending, mut drained)| async move {
                if pending.is_empty() {
                    if drained {
                        return None;
                    }
                    drained = true;
                    let mut acc = String::new();
                    while let Some(chunk) = text.next().await {
                        acc.push_str(&chunk);
                    }
                    if !acc.is_empty() {
                        pending.push(TtsEvent::SentenceStart { text: acc });
                    }
                    pending.push(TtsEvent::Done);
                }
                let event = pending.remove(0);
                Some((Ok(event), (text, pending, drained)))
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_stream(chunks: Vec<&str>) -> TextStream {
        Box::pin(futures_util::stream::iter(
            chunks
                .into_iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>(),
        ))
    }

    #[tokio::test]
    async fn stub_emits_sentence_then_done() {
        let mut events = StubTts.synthesize(text_stream(vec!["hello", " world"]));
        assert_eq!(
            events.next().await.unwrap().unwrap(),
            TtsEvent::SentenceStart {
                text: "hello world".into()
            }
        );
        assert_eq!(events.next().await.unwrap().unwrap(), TtsEvent::Done);
        assert!(events.next().await.is_none());
    }

    #[tokio::test]
    async fn stub_emits_done_for_empty_text() {
        let mut events = StubTts.synthesize(text_stream(vec![]));
        assert_eq!(events.next().await.unwrap().unwrap(), TtsEvent::Done);
        assert!(events.next().await.is_none());
    }
}
