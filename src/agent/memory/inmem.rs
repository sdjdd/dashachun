use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::llm::ChatItem;

use super::{Memory, MemoryFactory};

/// Caps the stored window so a long session cannot grow the per-turn request
/// (token cost and latency) without bound. The window is widened backwards
/// when it would start inside a tool exchange: a `Tool` item without its
/// owning assistant tool-call message is an invalid request for
/// OpenAI-compatible APIs, so one oversized tool turn may exceed the cap but
/// an exchange is never split.
const HISTORY_LIMIT: usize = 10;

pub struct InMemMemoryFactory;

impl MemoryFactory for InMemMemoryFactory {
    fn build(&self) -> Arc<dyn Memory> {
        Arc::new(InMemMemory::default())
    }
}

/// Keeps the conversation in process memory; its state dies with the session.
#[derive(Default)]
pub struct InMemMemory {
    messages: Mutex<Vec<ChatItem>>,
}

#[async_trait]
impl Memory for InMemMemory {
    async fn history(&self) -> Vec<ChatItem> {
        self.messages.lock().unwrap().clone()
    }

    async fn append(&self, items: Vec<ChatItem>) {
        let mut messages = self.messages.lock().unwrap();
        messages.extend(items);
        trim(&mut messages);
    }
}

fn trim(history: &mut Vec<ChatItem>) {
    let overflow = history.len().saturating_sub(HISTORY_LIMIT);
    if overflow == 0 {
        return;
    }
    let mut start = overflow;
    while start > 0 && matches!(history[start], ChatItem::Tool { .. }) {
        start -= 1;
    }
    history.drain(..start);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::llm::ToolCall;

    #[tokio::test]
    async fn keeps_only_the_last_messages() {
        let memory = InMemMemory::default();
        for turn in 1..=15 {
            memory
                .append(vec![ChatItem::user(format!("u{turn}"))])
                .await;
            memory
                .append(vec![ChatItem::assistant(format!("a{turn}"))])
                .await;
        }

        let history = memory.history().await;
        assert_eq!(history.len(), 10);
        assert_eq!(history.first(), Some(&ChatItem::user("u11")));
        assert_eq!(history.last(), Some(&ChatItem::assistant("a15")));
    }

    #[tokio::test]
    async fn never_orphans_a_tool_result() {
        let memory = InMemMemory::default();
        memory.append(vec![ChatItem::user("u1")]).await;
        memory
            .append(vec![
                ChatItem::assistant_tool_calls(vec![ToolCall {
                    id: "1".into(),
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                }]),
                ChatItem::tool("1", "sunny"),
            ])
            .await;
        for turn in 2..=10 {
            memory
                .append(vec![ChatItem::user(format!("u{turn}"))])
                .await;
        }

        let history = memory.history().await;
        // 12 stored items would overflow by 2, but the window cannot start on
        // the tool result, so the owning assistant item stays and the cap is
        // exceeded by one.
        assert_eq!(history.len(), 11);
        assert!(matches!(
            history.first(),
            Some(ChatItem::Assistant { tool_calls, .. }) if !tool_calls.is_empty()
        ));
        assert_eq!(history[1], ChatItem::tool("1", "sunny"));
    }
}
