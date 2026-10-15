use std::net::IpAddr;

use sqlx::{FromRow, PgPool};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::error::AppError;

/// A session's activity snapshot (`last_active_at`, `ip`, `user_agent`) is
/// written at most once per this interval. The throttle state is the row's
/// own `last_active_at` — no in-process buffer — so the refresh is safe
/// across instances and the common request performs no write at all.
pub const TOUCH_INTERVAL_SECS: u64 = 60;

pub async fn create(
    pool: &PgPool,
    user_id: i64,
    ttl_secs: i64,
    ip: IpAddr,
    user_agent: &str,
) -> Result<Uuid, AppError> {
    let id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO auth_sessions (id, user_id, expires_at, ip, user_agent) \
         VALUES ($1, $2, now() + make_interval(secs => $3), $4, $5)",
    )
    .bind(id)
    .bind(user_id)
    .bind(ttl_secs as f64)
    .bind(ip.to_string())
    .bind(user_agent)
    .execute(pool)
    .await?;
    Ok(id)
}

pub struct LiveSession {
    pub user_id: i64,
    pub touch_due: bool,
}

/// The per-request session check: the usual expiry-scoped lookup plus the
/// staleness verdict (`last_active_at` older than [`TOUCH_INTERVAL_SECS`])
/// decided in SQL against the database clock.
pub async fn verify(pool: &PgPool, id: Uuid) -> Result<Option<LiveSession>, AppError> {
    let row = sqlx::query_as::<_, (i64, bool)>(
        "SELECT user_id, last_active_at < now() - make_interval(secs => $2) \
         FROM auth_sessions WHERE id = $1 AND expires_at > now()",
    )
    .bind(id)
    .bind(TOUCH_INTERVAL_SECS as f64)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|(user_id, touch_due)| LiveSession { user_id, touch_due }))
}

/// Refresh the activity snapshot. The predicate is the authoritative gate:
/// when concurrent touches race (same or different instance) the loser's
/// WHERE no longer holds against the committed row, so one interval yields
/// at most one write per session.
pub async fn touch(pool: &PgPool, id: Uuid, ip: IpAddr, user_agent: &str) -> Result<(), AppError> {
    sqlx::query(
        "UPDATE auth_sessions \
         SET last_active_at = now(), ip = $2, user_agent = $3 \
         WHERE id = $1 AND last_active_at < now() - make_interval(secs => $4)",
    )
    .bind(id)
    .bind(ip.to_string())
    .bind(user_agent)
    .bind(TOUCH_INTERVAL_SECS as f64)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete(pool: &PgPool, id: Uuid) -> Result<(), AppError> {
    sqlx::query("DELETE FROM auth_sessions WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn delete_others(pool: &PgPool, user_id: i64, keep: Uuid) -> Result<(), AppError> {
    sqlx::query("DELETE FROM auth_sessions WHERE user_id = $1 AND id <> $2")
        .bind(user_id)
        .bind(keep)
        .execute(pool)
        .await?;
    Ok(())
}

/// A live login session as the read side hands it over: the contact
/// metadata recorded at create (refreshed by the touch path) plus the
/// lifecycle stamps.
#[derive(Debug, FromRow)]
pub struct SessionRecord {
    pub id: Uuid,
    pub ip: String,
    pub user_agent: String,
    pub created_at: OffsetDateTime,
    pub last_active_at: OffsetDateTime,
    pub expires_at: OffsetDateTime,
}

/// Lists the account's unexpired sessions, most recently active first. The
/// per-account count is small, so there is no pagination.
pub async fn list(pool: &PgPool, user_id: i64) -> Result<Vec<SessionRecord>, AppError> {
    let rows = sqlx::query_as(
        "SELECT id, ip, user_agent, created_at, last_active_at, expires_at \
         FROM auth_sessions \
         WHERE user_id = $1 AND expires_at > now() \
         ORDER BY last_active_at DESC",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Deletes one of the account's own sessions — the single-session form of
/// [`delete_others`]. Revoking the current session is the caller's own
/// logout: the row dies and the next request fails the lookup.
pub async fn delete_owned(pool: &PgPool, user_id: i64, id: Uuid) -> Result<bool, AppError> {
    let deleted = sqlx::query("DELETE FROM auth_sessions WHERE id = $1 AND user_id = $2")
        .bind(id)
        .bind(user_id)
        .execute(pool)
        .await?
        .rows_affected();
    Ok(deleted > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_ids_are_unique() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        assert_ne!(a, b);
    }
}
