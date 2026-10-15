//! The conversation memory: a warm in-process window, grouped into whole
//! conversation turns, over the durable `messages` log, plus the
//! entry-based user memory the prompt reads and the memory tools maintain.
//! Text only — audio storage is a separate feature ([`AudioCapture`]) that
//! binds to committed rows through the ids the store calls return or hand
//! to their attach sinks.

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

/// The store call's attach sink: runs once the rows the call wrote have
/// committed, with the last committed row's id — the audio attach point.
/// The caller binds the sink at the store call site, so the id always
/// lands on the capture that produced the audio, never on a newer one.
pub type OnCommit = Box<dyn FnOnce(i64) + Send>;

/// Observes the memory's warm window: every append batch the window
/// receives is announced here, verbatim. Called synchronously on the
/// caller's task right after the append lands: implementations must be
/// cheap and never block — spawn your own task for anything slow.
/// Implement what you observe; every method is a no-op by default.
pub trait MemoryHook: Send + Sync {
    /// The warm window received one append batch: the exact items the
    /// agent's context carries, speaker labels included. Announced right
    /// after the append lands — the batch is handed over verbatim, with no
    /// ids and no durable-row guarantee here; row consumers bind the ids
    /// at the store call sites instead.
    fn appended(&self, _session: &AgentSession, _items: &[ChatItem]) {}
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
    /// Batched because a tool turn is several items. The batch is announced
    /// to the hooks ([`MemoryHook::appended`]); the durable copy is written
    /// by the store methods below, never here.
    async fn append(&self, session: &AgentSession, items: Vec<ChatItem>);

    /// Writes the user message row for a final utterance and returns its id
    /// once it commits — `None` for empty text, nothing stored. `speaker`
    /// is the utterance's attribution — the matched speakers row id, `None`
    /// when the utterance identified no one or the agent runs no
    /// identification — persisted on the row for the transcript's readers.
    async fn store_utterance(
        &self,
        session: &AgentSession,
        text: &str,
        speaker: Option<i64>,
    ) -> Result<Option<i64>, String>;

    /// Writes the turn's assistant and tool messages in one background
    /// transaction and runs `attach` with the last written row's id once it
    /// commits — the capture attach point. A turn that produced nothing
    /// stores nothing and calls no sink. Fire-and-forget: failures are
    /// logged, never surfaced.
    fn log_items(&self, session: &AgentSession, items: Vec<ChatItem>, attach: Option<OnCommit>);

    /// Writes a truncated assistant message row for a reply cut mid-flight
    /// before the turn completed, and runs `attach` with the row's id once
    /// it commits. Empty text stores nothing. Fire-and-forget: failures are
    /// logged, never surfaced.
    fn store_partial_reply(&self, session: &AgentSession, text: &str, attach: Option<OnCommit>);
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
