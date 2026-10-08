mod volc;

pub use volc::VolcTts;

use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::agent::{TextStream, Tts, TtsEvent, TtsEvents};

pub struct StubTts;

impl Tts for StubTts {
    fn synthesize(&self, text: TextStream, _cancel: CancellationToken) -> TtsEvents<'_> {
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
        let mut events = StubTts.synthesize(
            text_stream(vec!["hello", " world"]),
            CancellationToken::new(),
        );
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
        let mut events = StubTts.synthesize(text_stream(vec![]), CancellationToken::new());
        assert_eq!(events.next().await.unwrap().unwrap(), TtsEvent::Done);
        assert!(events.next().await.is_none());
    }
}
