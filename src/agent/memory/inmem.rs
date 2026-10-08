use std::sync::Mutex;

use async_trait::async_trait;

use crate::agent::AgentSession;
use crate::llm::ChatItem;

use super::{Memory, trim};

/// Keeps the conversation in process memory, starting empty; its state dies
/// with the session. Test helper — production sessions preload a recent
/// window from the durable log ([`super::DbMemory`]). Mirrors the durable
/// semantics: empty utterances, empty turns and empty partials store
/// nothing; the store calls are recorded for assertions.
#[derive(Default)]
pub struct InMemMemory {
    messages: Mutex<Vec<ChatItem>>,
    utterances: Mutex<Vec<String>>,
    turns: Mutex<Vec<Vec<ChatItem>>>,
    partials: Mutex<Vec<String>>,
}

impl InMemMemory {
    /// The utterance texts passed to [`Memory::store_utterance`], in order.
    pub fn stored_utterances(&self) -> Vec<String> {
        self.utterances.lock().unwrap().clone()
    }

    /// The item batches passed to [`Memory::log_items`], in order.
    pub fn logged_turns(&self) -> Vec<Vec<ChatItem>> {
        self.turns.lock().unwrap().clone()
    }

    /// The partial texts passed to [`Memory::store_partial_reply`], in
    /// order.
    pub fn stored_partials(&self) -> Vec<String> {
        self.partials.lock().unwrap().clone()
    }
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

    async fn store_utterance(&self, _session: &AgentSession, text: &str) -> Result<(), String> {
        if !text.is_empty() {
            self.utterances.lock().unwrap().push(text.to_string());
        }
        Ok(())
    }

    fn log_items(&self, _session: &AgentSession, items: Vec<ChatItem>) {
        if !items.is_empty() {
            self.turns.lock().unwrap().push(items);
        }
    }

    fn store_partial_reply(&self, _session: &AgentSession, text: &str) {
        if !text.is_empty() {
            self.partials.lock().unwrap().push(text.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::agent::memory::HISTORY_LIMIT;
    use crate::llm::ToolCall;

    #[tokio::test]
    async fn keeps_only_the_last_messages() {
        let memory = InMemMemory::default();
        let turns = HISTORY_LIMIT / 2 + 10;
        for turn in 1..=turns {
            memory
                .append(vec![ChatItem::user(format!("u{turn}"))])
                .await;
            memory
                .append(vec![ChatItem::assistant(format!("a{turn}"))])
                .await;
        }

        let history = memory.history().await;
        assert_eq!(history.len(), HISTORY_LIMIT);
        assert_eq!(
            history.first(),
            Some(&ChatItem::user(format!(
                "u{}",
                turns - HISTORY_LIMIT / 2 + 1
            )))
        );
        assert_eq!(
            history.last(),
            Some(&ChatItem::assistant(format!("a{turns}")))
        );
    }

    #[tokio::test]
    async fn never_orphans_a_tool_result() {
        let memory = InMemMemory::default();
        for turn in 1..=(HISTORY_LIMIT - 2) {
            memory
                .append(vec![ChatItem::user(format!("u{turn}"))])
                .await;
        }
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
        for turn in 1..=(HISTORY_LIMIT - 1) {
            memory
                .append(vec![ChatItem::user(format!("late{turn}"))])
                .await;
        }

        let history = memory.history().await;
        // The overflow lands on the tool result, but the window cannot start
        // there, so the owning assistant item stays and the cap is exceeded
        // by one.
        assert_eq!(history.len(), HISTORY_LIMIT + 1);
        assert!(matches!(
            history.first(),
            Some(ChatItem::Assistant { tool_calls, .. }) if !tool_calls.is_empty()
        ));
        assert_eq!(history[1], ChatItem::tool("1", "sunny"));
        assert_eq!(
            history.last(),
            Some(&ChatItem::user(format!("late{}", HISTORY_LIMIT - 1)))
        );
    }

    #[tokio::test]
    async fn empty_store_calls_record_nothing() {
        let memory = InMemMemory::default();
        memory.store_utterance(&session(), "").await.unwrap();
        memory.log_items(&session(), vec![]);
        memory.store_partial_reply(&session(), "");

        assert!(memory.stored_utterances().is_empty());
        assert!(memory.logged_turns().is_empty());
        assert!(memory.stored_partials().is_empty());
    }

    fn session() -> AgentSession {
        AgentSession {
            id: "sess".into(),
            sample_rate: 16000,
            channels: 1,
            frame_duration_ms: 60,
        }
    }
}
