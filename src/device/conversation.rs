//! Conversations: one row per accepted gateway connection, created by the
//! upgrade route and closed when the connection ends. The auto-increment
//! row id is the durable anchor everything the connection writes hangs
//! off; the wire `session_id` the device sees is recorded on the row so
//! device-side logs correlate with it. `device_id` and `agent_id` are the
//! binding snapshot at hello, so a later re-bind never rewrites the past.

use sqlx::{FromRow, PgPool};
use time::OffsetDateTime;

/// A recorded conversation as the read side hands it over: the binding
/// snapshot columns plus the wire `session_id` and the lifecycle stamps.
#[derive(Debug, FromRow)]
pub struct Conversation {
    pub id: i64,
    pub session_id: String,
    pub device_id: i64,
    pub created_at: OffsetDateTime,
    pub ended_at: Option<OffsetDateTime>,
}

/// Lists the agent's conversations, newest first, keyset-paginated on the
/// row id: `before_id` (the previous page's last id) is an exclusive upper
/// bound, so the next page continues where the previous one ended. Returns
/// at most `limit` rows.
pub async fn list(
    pool: &PgPool,
    agent_id: i64,
    before_id: Option<i64>,
    limit: i64,
) -> Result<Vec<Conversation>, sqlx::Error> {
    sqlx::query_as(
        "SELECT id, session_id, device_id, created_at, ended_at \
         FROM conversations \
         WHERE agent_id = $1 AND ($2::BIGINT IS NULL OR id < $2) \
         ORDER BY id DESC \
         LIMIT $3",
    )
    .bind(agent_id)
    .bind(before_id)
    .bind(limit)
    .fetch_all(pool)
    .await
}

/// Whether the conversation was opened under this agent. The agent is the
/// binding snapshot at hello — ownership never drifts when the device is
/// re-bound afterwards.
pub async fn exists_for_agent(
    pool: &PgPool,
    agent_id: i64,
    conversation_id: i64,
) -> Result<bool, sqlx::Error> {
    let hit: Option<i64> =
        sqlx::query_scalar("SELECT 1::BIGINT FROM conversations WHERE id = $1 AND agent_id = $2")
            .bind(conversation_id)
            .bind(agent_id)
            .fetch_optional(pool)
            .await?;
    Ok(hit.is_some())
}

/// Opens a conversation row for the accepted connection and returns its
/// id. A failure rejects the upgrade — a conversation whose rows could
/// not be recorded should not run.
pub async fn create(
    pool: &PgPool,
    session_id: &str,
    device_id: i64,
    agent_id: i64,
) -> Result<i64, sqlx::Error> {
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO conversations (session_id, device_id, agent_id) \
         VALUES ($1, $2, $3) RETURNING id",
    )
    .bind(session_id)
    .bind(device_id)
    .bind(agent_id)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

/// Stamps the conversation's end when the connection is over. Best-effort:
/// a crash leaves `ended_at` NULL and the row simply shows an unterminated
/// conversation.
pub async fn end(pool: &PgPool, conversation_id: i64) {
    let result = sqlx::query(
        "UPDATE conversations SET ended_at = now() \
         WHERE id = $1 AND ended_at IS NULL",
    )
    .bind(conversation_id)
    .execute(pool)
    .await;
    if let Err(err) = result {
        tracing::warn!(conversation_id, %err, "failed to record conversation end");
    }
}
