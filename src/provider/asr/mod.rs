pub mod volcengine;

pub use volcengine::VolcAsr;

use std::sync::Arc;

use serde_json::Value;

use crate::provider::ProviderRegistry;

pub const STUB_KIND: &str = "stub";

pub fn register_stub(registry: &mut ProviderRegistry) {
    registry.register_asr(STUB_KIND, build_stub);
}

fn build_stub(config: &Value) -> Result<Arc<dyn Asr>, String> {
    let text = config.get("text").and_then(Value::as_str).unwrap_or("你好");
    Ok(Arc::new(StubAsr::new(text)))
}

use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::agent::{Asr, AsrEvent, AsrEvents, AudioStream};

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
    fn transcribe(&self, audio: AudioStream, _cancel: CancellationToken) -> AsrEvents<'_> {
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
        let mut events = asr.transcribe(audio_stream(rx), CancellationToken::new());

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
