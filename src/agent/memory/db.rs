use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use sqlx::{FromRow, PgPool};
use time::OffsetDateTime;
use tracing::debug;
use tracing::warn;

use crate::agent::AgentSession;
use crate::agent::ChatItem;

use super::{Memory, MemoryHook, OnCommit, Turn, trim};

/// The conversation memory backed by the durable `messages` log. The window
/// starts empty — the context lives only here for the session's lifetime,
/// the durable log is a record, never reloaded — and every turn is appended
/// by the driver while, in parallel, written durably by the store methods,
/// so per-turn requests never wait on Postgres. The window is a list of
/// whole turns capped at [`super::HISTORY_TURNS`]. Each row of the durable log
/// holds one message in the JSON form of a [`ChatItem`] (AI SDK
/// ModelMessage shape, tagged by `role`), so storing is a serde write.
/// Window appends are announced to the [`MemoryHook`]s; the store methods
/// hand their committed row ids to the attach sinks their callers passed.
pub struct DbMemory {
    pool: PgPool,
    hooks: Vec<Arc<dyn MemoryHook>>,
    turns: Mutex<Vec<Turn>>,
}

impl DbMemory {
    /// A fresh window for one connection, starting empty.
    pub fn new(pool: PgPool, hooks: Vec<Arc<dyn MemoryHook>>) -> Self {
        Self {
            pool,
            hooks,
            turns: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Memory for DbMemory {
    async fn history(&self) -> Vec<ChatItem> {
        self.turns
            .lock()
            .unwrap()
            .iter()
            .flat_map(|turn| turn.0.iter().cloned())
            .collect()
    }

    async fn append(&self, session: &AgentSession, items: Vec<ChatItem>) {
        if items.is_empty() {
            return;
        }
        {
            let mut turns = self.turns.lock().unwrap();
            match items.first() {
                Some(ChatItem::User { .. }) => turns.push(Turn(items.clone())),
                Some(_) => {
                    if let Some(turn) = turns.last_mut() {
                        turn.0.extend(items.iter().cloned());
                    }
                }
                None => {}
            }
            trim(&mut turns);
        }
        // Outside the window lock and after trim — the hook receives the
        // batch verbatim, and everything announced is in the window: items
        // land at the tail, trim only drops from the front.
        for hook in &self.hooks {
            hook.appended(session, &items);
        }
    }

    async fn store_utterance(
        &self,
        session: &AgentSession,
        text: &str,
        speaker: Option<i64>,
    ) -> Result<Option<i64>, String> {
        if text.is_empty() {
            return Ok(None);
        }
        let message_id = insert_message_row(
            &self.pool,
            session.conversation_id,
            &ChatItem::user(text),
            speaker,
        )
        .await?;
        debug!(
            conversation_id = session.conversation_id,
            message_id, "utterance stored"
        );
        Ok(Some(message_id))
    }

    fn log_items(&self, session: &AgentSession, items: Vec<ChatItem>, attach: Option<OnCommit>) {
        if items.is_empty() {
            return;
        }
        let pool = self.pool.clone();
        let session = session.clone();
        tokio::spawn(async move {
            match store_log_rows(&pool, session.conversation_id, &items).await {
                Ok(Some(message_id)) => {
                    if let Some(attach) = attach {
                        attach(message_id);
                    }
                    debug!(
                        conversation_id = session.conversation_id,
                        message_id, "turn stored"
                    );
                }
                Ok(None) => {}
                Err(err) => {
                    warn!(conversation_id = session.conversation_id, %err, "failed to store turn items")
                }
            }
        });
    }

    fn store_partial_reply(&self, session: &AgentSession, text: &str, attach: Option<OnCommit>) {
        if text.is_empty() {
            return;
        }
        let pool = self.pool.clone();
        let session = session.clone();
        let text = text.to_string();
        tokio::spawn(async move {
            match insert_message_row(
                &pool,
                session.conversation_id,
                &ChatItem::assistant(text),
                None,
            )
            .await
            {
                Ok(message_id) => {
                    if let Some(attach) = attach {
                        attach(message_id);
                    }
                    debug!(
                        conversation_id = session.conversation_id,
                        message_id, "partial reply stored"
                    );
                }
                Err(err) => {
                    warn!(conversation_id = session.conversation_id, %err, "failed to store partial reply")
                }
            }
        });
    }
}

/// Inserts one message row — the JSON form of `item` — and returns its id.
/// The executor is generic so the same helper serves the awaited single
/// inserts and the transactional turn batch. `speaker` only ever arrives
/// for user rows (the assistant/tool writers pass `None`).
async fn insert_message_row(
    executor: impl sqlx::PgExecutor<'_>,
    conversation_id: i64,
    item: &ChatItem,
    speaker: Option<i64>,
) -> Result<i64, String> {
    let message = serde_json::to_value(item).map_err(|err| err.to_string())?;
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO messages (conversation_id, message, speaker_id) VALUES ($1, $2, $3) RETURNING id",
    )
    .bind(conversation_id)
    .bind(message)
    .bind(speaker)
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
    conversation_id: i64,
    items: &[ChatItem],
) -> Result<Option<i64>, String> {
    let mut tx = pool.begin().await.map_err(|err| err.to_string())?;
    let mut last_id = None;
    for item in items {
        if matches!(item, ChatItem::System { .. } | ChatItem::User { .. }) {
            continue;
        }
        last_id = Some(insert_message_row(&mut *tx, conversation_id, item, None).await?);
    }
    tx.commit().await.map_err(|err| err.to_string())?;
    Ok(last_id)
}

/// One durable transcript row as the read side hands it over: the stored
/// AI SDK ModelMessage JSON verbatim plus the columns a reader hangs off.
#[derive(Debug, FromRow)]
pub struct TranscriptMessage {
    pub id: i64,
    pub message: serde_json::Value,
    pub speaker_id: Option<i64>,
    pub created_at: OffsetDateTime,
}

/// Reads the durable log of one conversation, newest first, keyset-paginated
/// on the row id: `before_id` (the previous page's last id) is an exclusive
/// upper bound, so the next page continues where the previous one ended.
/// Returns at most `limit` rows.
pub async fn read_transcript(
    pool: &PgPool,
    conversation_id: i64,
    before_id: Option<i64>,
    limit: i64,
) -> Result<Vec<TranscriptMessage>, sqlx::Error> {
    sqlx::query_as(
        "SELECT id, message, speaker_id, created_at \
         FROM messages \
         WHERE conversation_id = $1 AND ($2::BIGINT IS NULL OR id < $2) \
         ORDER BY id DESC \
         LIMIT $3",
    )
    .bind(conversation_id)
    .bind(before_id)
    .bind(limit)
    .fetch_all(pool)
    .await
}
