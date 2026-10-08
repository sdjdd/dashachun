mod openai;

pub use openai::{OpenAiConfig, OpenAiLlm};

use tokio_util::sync::CancellationToken;

use crate::agent::{ChatItem, Llm, LlmError, LlmEvent, LlmEvents, ToolSpec};

#[derive(Default)]
pub struct StubLlm {
    reply: String,
}

impl StubLlm {
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
        }
    }
}

impl Llm for StubLlm {
    fn chat(
        &self,
        _history: Vec<ChatItem>,
        _tools: Vec<ToolSpec>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        let mut events: Vec<Result<LlmEvent, LlmError>> = Vec::new();
        if !self.reply.is_empty() {
            events.push(Ok(LlmEvent::Delta {
                text: self.reply.clone(),
            }));
        }
        events.push(Ok(LlmEvent::Done));
        Box::pin(futures_util::stream::iter(events))
    }
}
