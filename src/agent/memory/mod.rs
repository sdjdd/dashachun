//! Per-session conversation storage behind the agent.

mod inmem;

use std::sync::Arc;

use async_trait::async_trait;

use crate::llm::ChatItem;

pub use inmem::{InMemMemory, InMemMemoryFactory};

/// The conversation store behind one session's replies. A fresh instance is
/// built per connection ([`MemoryFactory`]); how messages are retained and
/// when they leave memory is the implementation's policy — the in-memory one
/// keeps a bounded recent window, a database-backed one would turn them into
/// rows.
///
/// The system prompt is not part of the stored history: the agent prepends
/// its own prompt every turn.
#[async_trait]
pub trait Memory: Send + Sync {
    /// The conversation so far, oldest first.
    async fn history(&self) -> Vec<ChatItem>;

    /// Records the new items of one turn — the user message when the
    /// utterance is final, then the assistant reply and any tool exchange
    /// when the turn completes. Batched because a tool turn is several items.
    async fn append(&self, items: Vec<ChatItem>);
}

/// Builds a fresh [`Memory`] per connection, mirroring
/// [`crate::vad::VadFactory`].
pub trait MemoryFactory: Send + Sync {
    fn build(&self) -> Arc<dyn Memory>;
}
