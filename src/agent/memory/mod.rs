//! The conversation memory: a warm in-process window, grouped into whole
//! conversation turns, over the durable `messages` log, plus the
//! entry-based user memory the prompt reads and the memory tools maintain.
//! Text only — audio storage is a separate feature ([`AudioCapture`]) that
//! attaches to committed rows through [`MemoryHook`].

mod db;
mod entries;
mod extract;
mod inmem;

use async_trait::async_trait;

use crate::agent::AgentSession;
use crate::agent::ChatItem;

pub use db::DbMemory;
pub use entries::{
    AddError, DbEntryMemory, EntryMemory, EntryOp, EntryOpResult, InMemEntryMemory, MAX_ENTRIES,
    MemoryEntry, entry_id,
};
pub use extract::Extractor;
pub use inmem::InMemMemory;

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
/// built per connection and starts empty — the context lives only in this
/// window for the session's lifetime; the durable `messages` log is a
/// record, never reloaded. The system prompt is not part of the stored
/// history: the agent prepends its own prompt every turn.
#[async_trait]
pub trait Memory: Send + Sync {
    /// The conversation so far, oldest first.
    async fn history(&self) -> Vec<ChatItem>;

    /// Records the new items of one turn in the warm window — the user
    /// message when the utterance is final, then the assistant reply and
    /// any tool exchange when the turn completes. A batch opening with a
    /// `User` item starts a turn; anything else extends the open one.
    /// Batched because a tool turn is several items. The durable copy is
    /// written by the store methods below, never here.
    async fn append(&self, items: Vec<ChatItem>);

    /// Writes the user message row for a final utterance and notifies the
    /// hooks once it commits. Empty text stores nothing.
    async fn store_utterance(&self, session: &AgentSession, text: &str) -> Result<(), String>;

    /// Writes the turn's assistant and tool messages in one background
    /// transaction and notifies the hooks once it commits — the hook
    /// carries the last written row's id, the capture attach point. A turn
    /// that produced nothing stores nothing. Fire-and-forget: failures are
    /// logged, never surfaced.
    fn log_items(&self, session: &AgentSession, items: Vec<ChatItem>);

    /// Writes a truncated assistant message row for a reply cut mid-flight
    /// before the turn completed, and notifies the hooks once it commits.
    /// Empty text stores nothing. Fire-and-forget: failures are logged,
    /// never surfaced.
    fn store_partial_reply(&self, session: &AgentSession, text: &str);
}

/// One conversation turn in the window: the user utterance that opens it,
/// then everything the reply stage produced for it — interim assistant
/// texts, tool calls and results, the final reply. The unit the window
/// counts and never splits.
pub(crate) struct Turn(pub(crate) Vec<ChatItem>);

/// Caps the stored window at [`HISTORY_TURNS`] turns so a long session
/// cannot grow the per-turn request (token cost and latency) without bound.
/// Turns are atomic — the window always starts at a user utterance and a
/// tool exchange is never split — so one oversized turn may exceed the cap
/// but a turn is never cut.
pub(crate) const HISTORY_TURNS: usize = 10;

/// Drops the oldest turns past the cap.
pub(crate) fn trim(turns: &mut Vec<Turn>) {
    turns.drain(..turns.len().saturating_sub(HISTORY_TURNS));
}
