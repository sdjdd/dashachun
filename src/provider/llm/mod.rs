pub mod openai;

pub use openai::{OpenAiConfig, OpenAiLlm};

use std::sync::Arc;

use serde_json::Value;

use crate::provider::ProviderRegistry;

pub const STUB_KIND: &str = "stub";

pub fn register_stub(registry: &mut ProviderRegistry) {
    registry.register_llm(STUB_KIND, build_stub);
}

fn build_stub(config: &Value) -> Result<Arc<dyn Llm>, String> {
    let reply = config.get("reply").and_then(Value::as_str).unwrap_or("");
    Ok(Arc::new(StubLlm::new(reply)))
}

use tokio_util::sync::CancellationToken;

use crate::agent::{
    ChatItem, Llm, LlmError, LlmEvent, LlmEvents, ReasoningEffort, ToolChoice, ToolSpec,
};

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
        _tool_choice: Option<ToolChoice>,
        _reasoning: Option<ReasoningEffort>,
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
