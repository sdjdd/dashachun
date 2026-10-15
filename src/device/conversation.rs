//! Conversations: one row per accepted gateway connection, created by the
//! upgrade route and closed when the connection ends. The auto-increment
//! row id is the durable anchor everything the connection writes hangs
//! off; the wire `session_id` the device sees is recorded on the row so
//! device-side logs correlate with it. `device_id` and `agent_id` are the
//! binding snapshot at hello, so a later re-bind never rewrites the past.

use sqlx::PgPool;

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
