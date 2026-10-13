use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use sqlx::PgPool;
use tracing::debug;
use tracing::warn;

use crate::agent::AgentSession;
use crate::agent::ChatItem;

use super::{Memory, MemoryHook, trim};

/// The conversation memory backed by the durable `messages` log. The window
/// starts empty — the context lives only here for the session's lifetime,
/// the durable log is a record, never reloaded — and every turn is appended
/// by the driver while, in parallel, written durably by the store methods,
/// so per-turn requests never wait on Postgres. Each row holds one message
/// in the JSON form of a [`ChatItem`] (AI SDK ModelMessage shape, tagged by
/// `role`), so storing is a serde write. Committed rows are announced to
/// the [`MemoryHook`]s.
pub struct DbMemory {
    pool: PgPool,
    hooks: Vec<Arc<dyn MemoryHook>>,
    messages: Mutex<Vec<ChatItem>>,
}

impl DbMemory {
    /// A fresh window for one connection, starting empty.
    pub fn new(pool: PgPool, hooks: Vec<Arc<dyn MemoryHook>>) -> Self {
        Self {
            pool,
            hooks,
            messages: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Memory for DbMemory {
    async fn history(&self) -> Vec<ChatItem> {
        self.messages.lock().unwrap().clone()
    }

    async fn append(&self, items: Vec<ChatItem>) {
        let mut messages = self.messages.lock().unwrap();
        messages.extend(items);
        trim(&mut messages);
    }

    async fn store_utterance(&self, session: &AgentSession, text: &str) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        let message_id = insert_message_row(&self.pool, &session.id, &ChatItem::user(text)).await?;
        for hook in &self.hooks {
            hook.utterance_stored(session, message_id);
        }
        debug!(session_id = %session.id, message_id, "utterance stored");
        Ok(())
    }

    fn log_items(&self, session: &AgentSession, items: Vec<ChatItem>) {
        if items.is_empty() {
            return;
        }
        let pool = self.pool.clone();
        let hooks = self.hooks.clone();
        let session = session.clone();
        tokio::spawn(async move {
            match store_log_rows(&pool, &session.id, &items).await {
                Ok(Some(message_id)) => {
                    for hook in &hooks {
                        hook.turn_stored(&session, message_id);
                    }
                    debug!(session_id = %session.id, message_id, "turn stored");
                }
                Ok(None) => {}
                Err(err) => warn!(session_id = %session.id, %err, "failed to store turn items"),
            }
        });
    }

    fn store_partial_reply(&self, session: &AgentSession, text: &str) {
        if text.is_empty() {
            return;
        }
        let pool = self.pool.clone();
        let hooks = self.hooks.clone();
        let session = session.clone();
        let text = text.to_string();
        tokio::spawn(async move {
            match insert_message_row(&pool, &session.id, &ChatItem::assistant(text)).await {
                Ok(message_id) => {
                    for hook in &hooks {
                        hook.turn_stored(&session, message_id);
                    }
                    debug!(session_id = %session.id, message_id, "partial reply stored");
                }
                Err(err) => warn!(session_id = %session.id, %err, "failed to store partial reply"),
            }
        });
    }
}

/// Inserts one message row — the JSON form of `item` — and returns its id.
/// The executor is generic so the same helper serves the awaited single
/// inserts and the transactional turn batch.
async fn insert_message_row(
    executor: impl sqlx::PgExecutor<'_>,
    session_id: &str,
    item: &ChatItem,
) -> Result<i64, String> {
    let message = serde_json::to_value(item).map_err(|err| err.to_string())?;
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO messages (session_id, message) VALUES ($1, $2) RETURNING id",
    )
    .bind(session_id)
    .bind(message)
    .fetch_one(executor)
    .await
    .map_err(|err| err.to_string())?;
    Ok(id)
}

/// Writes the turn's messages in one transaction — every
/// [`ChatItem::Assistant`] and [`ChatItem::Tool`] becomes its own row, in
/// order. `System`/`User` items are never stored here: the user message is
/// written by [`Memory::store_utterance`], the system prompt is prepended
/// every turn. Returns the id of the last written row — the attachment
/// point for the turn's audio capture.
async fn store_log_rows(
    pool: &PgPool,
    session_id: &str,
    items: &[ChatItem],
) -> Result<Option<i64>, String> {
    let mut tx = pool.begin().await.map_err(|err| err.to_string())?;
    let mut last_id = None;
    for item in items {
        if matches!(item, ChatItem::System { .. } | ChatItem::User { .. }) {
            continue;
        }
        last_id = Some(insert_message_row(&mut *tx, session_id, item).await?);
    }
    tx.commit().await.map_err(|err| err.to_string())?;
    Ok(last_id)
}
