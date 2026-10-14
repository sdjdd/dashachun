use std::sync::Mutex;

use async_trait::async_trait;

use crate::agent::AgentSession;
use crate::agent::ChatItem;

use super::{Memory, Turn, trim};

/// Keeps the conversation in process memory, starting empty; its state dies
/// with the session. Test helper — mirrors the durable semantics (the
/// production [`super::DbMemory`] adds the Postgres writes and the hook
/// fan-out): empty utterances, empty turns and empty partials store
/// nothing; the store calls are recorded for assertions.
#[derive(Default)]
pub struct InMemMemory {
    turns: Mutex<Vec<Turn>>,
    utterances: Mutex<Vec<String>>,
    logged_turns: Mutex<Vec<Vec<ChatItem>>>,
    partials: Mutex<Vec<String>>,
}

impl InMemMemory {
    /// The utterance texts passed to [`Memory::store_utterance`], in order.
    pub fn stored_utterances(&self) -> Vec<String> {
        self.utterances.lock().unwrap().clone()
    }

    /// The item batches passed to [`Memory::log_items`], in order.
    pub fn logged_turns(&self) -> Vec<Vec<ChatItem>> {
        self.logged_turns.lock().unwrap().clone()
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
        self.turns
            .lock()
            .unwrap()
            .iter()
            .flat_map(|turn| turn.0.iter().cloned())
            .collect()
    }

    async fn append(&self, _session: &AgentSession, items: Vec<ChatItem>) {
        let mut turns = self.turns.lock().unwrap();
        match items.first() {
            Some(ChatItem::User { .. }) => turns.push(Turn(items)),
            Some(_) => {
                if let Some(turn) = turns.last_mut() {
                    turn.0.extend(items);
                }
            }
            None => {}
        }
        trim(&mut turns);
    }

    async fn store_utterance(&self, _session: &AgentSession, text: &str) -> Result<(), String> {
        if !text.is_empty() {
            self.utterances.lock().unwrap().push(text.to_string());
        }
        Ok(())
    }

    fn log_items(&self, _session: &AgentSession, items: Vec<ChatItem>) {
        if !items.is_empty() {
            self.logged_turns.lock().unwrap().push(items);
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

    use crate::agent::ToolCall;
    use crate::agent::memory::HISTORY_TURNS;

    #[tokio::test]
    async fn keeps_only_the_last_turns() {
        let memory = InMemMemory::default();
        let total = HISTORY_TURNS + 5;
        for turn in 1..=total {
            memory
                .append(&session(), vec![ChatItem::user(format!("u{turn}"))])
                .await;
            memory
                .append(&session(), vec![ChatItem::assistant(format!("a{turn}"))])
                .await;
        }

        let history = memory.history().await;
        assert_eq!(history.len(), HISTORY_TURNS * 2);
        assert_eq!(
            history.first(),
            Some(&ChatItem::user(format!("u{}", total - HISTORY_TURNS + 1)))
        );
        assert_eq!(
            history.last(),
            Some(&ChatItem::assistant(format!("a{total}")))
        );
    }

    #[tokio::test]
    async fn a_tool_turn_is_kept_whole() {
        let memory = InMemMemory::default();
        for turn in 1..HISTORY_TURNS {
            memory
                .append(&session(), vec![ChatItem::user(format!("u{turn}"))])
                .await;
            memory
                .append(&session(), vec![ChatItem::assistant(format!("a{turn}"))])
                .await;
        }
        memory
            .append(&session(), vec![ChatItem::user("tool turn")])
            .await;
        memory
            .append(
                &session(),
                vec![
                    ChatItem::assistant_tool_calls(vec![ToolCall {
                        id: "1".into(),
                        name: "get_weather".into(),
                        arguments: "{}".into(),
                    }]),
                    ChatItem::tool("1", "sunny"),
                ],
            )
            .await;
        for turn in 1..HISTORY_TURNS {
            memory
                .append(&session(), vec![ChatItem::user(format!("late{turn}"))])
                .await;
            memory
                .append(
                    &session(),
                    vec![ChatItem::assistant(format!("late a{turn}"))],
                )
                .await;
        }

        let history = memory.history().await;
        // The tool turn is the oldest of the ten kept turns and its items
        // exceed what the simple turns carry — the window starts at its
        // user message, never inside the exchange.
        assert_eq!(history.first(), Some(&ChatItem::user("tool turn")));
        assert!(matches!(
            history.get(1),
            Some(ChatItem::Assistant { tool_calls, .. }) if !tool_calls.is_empty()
        ));
        assert_eq!(history[2], ChatItem::tool("1", "sunny"));
        assert_eq!(history.len(), 3 + (HISTORY_TURNS - 1) * 2);
        assert_eq!(
            history.last(),
            Some(&ChatItem::assistant(format!("late a{}", HISTORY_TURNS - 1)))
        );
    }

    #[tokio::test]
    async fn interim_replies_and_tools_belong_to_one_turn() {
        let memory = InMemMemory::default();
        for turn in 1..HISTORY_TURNS {
            memory
                .append(&session(), vec![ChatItem::user(format!("u{turn}"))])
                .await;
            memory
                .append(&session(), vec![ChatItem::assistant(format!("a{turn}"))])
                .await;
        }
        memory.append(&session(), vec![ChatItem::user("u")]).await;
        memory
            .append(&session(), vec![ChatItem::assistant("thinking out loud")])
            .await;
        memory
            .append(
                &session(),
                vec![ChatItem::assistant_tool_calls(vec![ToolCall {
                    id: "1".into(),
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                }])],
            )
            .await;
        memory
            .append(&session(), vec![ChatItem::tool("1", "sunny")])
            .await;
        memory
            .append(&session(), vec![ChatItem::assistant("sunny today")])
            .await;

        let history = memory.history().await;
        // The complex turn counts once: the nine simple turns and its five
        // items all fit inside the cap.
        assert_eq!(history.len(), (HISTORY_TURNS - 1) * 2 + 5);
        assert_eq!(history[HISTORY_TURNS * 2 - 2], ChatItem::user("u"));
        assert_eq!(history.last(), Some(&ChatItem::assistant("sunny today")));
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
