//! Device conversation sessions: one row per accepted gateway connection,
//! created by the upgrade route with the server-assigned `session_id` and
//! closed when the connection ends. The row is the ownership anchor for the
//! session's message rows — `device_id` and `agent_id` are the binding
//! snapshot at hello, so a later re-bind never rewrites the past.

use sqlx::PgPool;

/// Opens a session row for the accepted connection. A failure rejects the
/// upgrade — a session whose rows could not be recorded should not run.
pub async fn create(
    pool: &PgPool,
    id: &str,
    device_id: i64,
    agent_id: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query("INSERT INTO device_sessions (id, device_id, agent_id) VALUES ($1, $2, $3)")
        .bind(id)
        .bind(device_id)
        .bind(agent_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Stamps the session's end when the connection is over. Best-effort: a
/// crash leaves `ended_at` NULL and the row simply shows an unterminated
/// session.
pub async fn end(pool: &PgPool, id: &str) {
    let result = sqlx::query(
        "UPDATE device_sessions SET ended_at = now() \
         WHERE id = $1 AND ended_at IS NULL",
    )
    .bind(id)
    .execute(pool)
    .await;
    if let Err(err) = result {
        tracing::warn!(session_id = %id, %err, "failed to record session end");
    }
}
