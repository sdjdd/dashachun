//! The conversation memory: a warm in-process window over the durable
//! `messages` log. Text only — audio storage is a separate feature
//! ([`AudioCapture`]) that attaches to committed rows through
//! [`MemoryHook`].

mod db;
mod inmem;

use async_trait::async_trait;
use uuid::Uuid;

use crate::agent::AgentSession;
use crate::llm::ChatItem;

pub use db::DbMemory;
pub use inmem::InMemMemory;

/// Device-side identity stamped onto every message row, taken from the
/// verified device record at gateway upgrade.
#[derive(Debug, Clone)]
pub struct MemoryOwner {
    pub user_id: i64,
    pub agent_id: i64,
    pub client_id: Uuid,
    pub device_id: Option<String>,
}

/// Observes committed message rows. Called synchronously on the store's
/// task right after the insert commits: implementations must be cheap and
/// never block — spawn your own task for anything slow. Hooks never fire
/// for a row that was not committed.
pub trait MemoryHook: Send + Sync {
    /// The user message row for a final utterance was committed.
    fn utterance_stored(&self, session: &AgentSession, message_id: i64);

    /// A turn row — the tool exchange plus the final reply, or a truncated
    /// partial — was committed.
    fn turn_stored(&self, session: &AgentSession, message_id: i64);
}

/// The text side of one session's conversation store. A fresh instance is
/// built per connection; how messages are retained is the implementation's
/// policy — the durable one preloads a recent window, the in-memory one
/// starts empty. The system prompt is not part of the stored history: the
/// agent prepends its own prompt every turn.
#[async_trait]
pub trait Memory: Send + Sync {
    /// The conversation so far, oldest first.
    async fn history(&self) -> Vec<ChatItem>;

    /// Records the new items of one turn in the warm window — the user
    /// message when the utterance is final, then the assistant reply and
    /// any tool exchange when the turn completes. Batched because a tool
    /// turn is several items. The durable copy is written by the store
    /// methods below, never here.
    async fn append(&self, items: Vec<ChatItem>);

    /// Writes the user message row for a final utterance and notifies the
    /// hooks once it commits. Empty text stores nothing.
    async fn store_utterance(&self, session: &AgentSession, text: &str) -> Result<(), String>;

    /// Writes the turn row — the tool exchange plus the final reply — in a
    /// background task and notifies the hooks once it commits. A turn that
    /// produced nothing stores nothing. Fire-and-forget: failures are
    /// logged, never surfaced.
    fn log_items(&self, session: &AgentSession, items: Vec<ChatItem>);

    /// Writes a truncated assistant row for a reply cut mid-flight before
    /// the turn completed, and notifies the hooks once it commits. Empty
    /// text stores nothing. Fire-and-forget: failures are logged, never
    /// surfaced.
    fn store_partial_reply(&self, session: &AgentSession, text: &str);
}

/// Caps the stored window so a long session cannot grow the per-turn request
/// (token cost and latency) without bound. The window is widened backwards
/// when it would start inside a tool exchange: a `Tool` item without its
/// owning assistant tool-call message is an invalid request for
/// OpenAI-compatible APIs, so one oversized tool turn may exceed the cap but
/// an exchange is never split. The preload query uses the same limit in
/// rows — every turn row unfolds to at least one item, so the loaded window
/// fills the cap before trimming.
pub(crate) const HISTORY_LIMIT: usize = 20;

/// Drops the oldest items past the cap, widening the window backwards over
/// `Tool` items so an exchange is never split.
pub(crate) fn trim(history: &mut Vec<ChatItem>) {
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
