use std::sync::Arc;

use redis::aio::MultiplexedConnection;
use redis::{Client, RedisError};
use sqlx::PgPool;
use tokio::sync::OnceCell;
use uuid::Uuid;

use crate::agent::AgentStore;
use crate::error::AppError;

pub mod dto;

const MAX_CODE_ATTEMPTS: u32 = 8;

#[derive(Clone)]
pub struct DeviceStore {
    pool: PgPool,
    redis: Client,
    conn: Arc<OnceCell<MultiplexedConnection>>,
    prefix: Arc<str>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DeviceRecord {
    pub id: i64,
    pub client_id: Uuid,
    pub device_id: Option<String>,
    pub board_type: Option<String>,
    pub user_id: Option<i64>,
    pub agent_id: Option<i64>,
    pub activated_at: Option<time::OffsetDateTime>,
    pub created_at: time::OffsetDateTime,
    pub last_seen_at: time::OffsetDateTime,
}

pub enum BindOutcome {
    Bound(DeviceRecord),
    NotFound,
    Conflict,
    UnknownAgent,
}

const DEVICE_COLUMNS: &str = "id, client_id, device_id, board_type, user_id, agent_id, activated_at, created_at, last_seen_at";

impl DeviceStore {
    pub fn new(pool: PgPool, redis: Client, prefix: &str) -> Self {
        Self {
            pool,
            redis,
            conn: Arc::new(OnceCell::new()),
            prefix: Arc::from(prefix),
        }
    }

    /// Activation codes are correctness state, unlike the rate limiter:
    /// a Redis error here must surface as a 500, not fail open, so a
    /// limiter outage cannot silently hand out unusable codes.
    async fn redis_conn(&self) -> Result<MultiplexedConnection, RedisError> {
        Ok(self
            .conn
            .get_or_try_init(|| self.redis.get_multiplexed_async_connection())
            .await?
            .clone())
    }

    fn device_key(&self, client_id: Uuid) -> String {
        format!("{}:device:{client_id}", self.prefix)
    }

    fn code_key(&self, code: &str) -> String {
        format!("{}:code:{code}", self.prefix)
    }

    pub async fn upsert(
        &self,
        client_id: Uuid,
        device_id: &str,
        board_type: &str,
    ) -> Result<DeviceRecord, AppError> {
        let record = sqlx::query_as::<_, DeviceRecord>(&format!(
            "INSERT INTO devices (client_id, device_id, board_type) \
             VALUES ($1, $2, $3) \
             ON CONFLICT (client_id) DO UPDATE \
                SET device_id = EXCLUDED.device_id, \
                    board_type = EXCLUDED.board_type, \
                    last_seen_at = now() \
             RETURNING {DEVICE_COLUMNS}"
        ))
        .bind(client_id)
        .bind(device_id)
        .bind(board_type)
        .fetch_one(&self.pool)
        .await?;
        Ok(record)
    }

    /// Returns the device's current unbound activation code, rotating it
    /// only when absent (expired codes are simply gone — the TTL is the
    /// expiry). Two keys per code, written in one atomic transaction:
    /// `{prefix}:device:{cid}` is the authoritative pointer to the current
    /// code, `{prefix}:code:{code}` is the by-code lookup index consumed
    /// on bind; the `NX` on the index doubles as the collision check.
    pub async fn ensure_code(&self, client_id: Uuid, ttl_secs: i64) -> Result<String, AppError> {
        let mut conn = self.redis_conn().await?;
        let device_key = self.device_key(client_id);
        if let Some(code) = redis::cmd("GET")
            .arg(&device_key)
            .query_async::<Option<String>>(&mut conn)
            .await?
        {
            return Ok(code);
        }

        for _ in 0..MAX_CODE_ATTEMPTS {
            let code = generate_code();
            let placed = redis::pipe()
                .atomic()
                .cmd("SET")
                .arg(&device_key)
                .arg(&code)
                .arg("EX")
                .arg(ttl_secs)
                .ignore()
                .cmd("SET")
                .arg(self.code_key(&code))
                .arg(client_id.to_string())
                .arg("EX")
                .arg(ttl_secs)
                .arg("NX")
                .query_async::<Option<()>>(&mut conn)
                .await?
                .is_some();
            if placed {
                return Ok(code);
            }
        }
        Err(AppError::Internal(
            "failed to allocate activation code".into(),
        ))
    }

    pub async fn is_bound(&self, client_id: Uuid) -> Result<bool, AppError> {
        let bound = sqlx::query_scalar::<_, bool>(
            "SELECT user_id IS NOT NULL FROM devices WHERE client_id = $1",
        )
        .bind(client_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(bound.unwrap_or(false))
    }

    pub async fn rotate_token(&self, client_id: Uuid) -> Result<Option<String>, AppError> {
        let token = Uuid::new_v4().simple().to_string();
        let hash = hash_token(&token);
        let affected = sqlx::query(
            "UPDATE devices SET token_hash = $2, last_seen_at = now() \
             WHERE client_id = $1 AND user_id IS NOT NULL",
        )
        .bind(client_id)
        .bind(&hash)
        .execute(&self.pool)
        .await?
        .rows_affected();
        Ok((affected > 0).then_some(token))
    }

    pub async fn verify(
        &self,
        client_id: Uuid,
        token: &str,
    ) -> Result<Option<DeviceRecord>, AppError> {
        let hash = hash_token(token);
        let record = sqlx::query_as::<_, DeviceRecord>(&format!(
            "SELECT {DEVICE_COLUMNS} FROM devices \
             WHERE client_id = $1 AND token_hash = $2 AND user_id IS NOT NULL"
        ))
        .bind(client_id)
        .bind(&hash)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
    }

    pub async fn bind_by_code(
        &self,
        user_id: i64,
        code: &str,
        agent_id: i64,
    ) -> Result<BindOutcome, AppError> {
        let agents = AgentStore::new(self.pool.clone());
        if !agents.owns(agent_id, user_id).await? {
            return Ok(BindOutcome::UnknownAgent);
        }

        let mut conn = self.redis_conn().await?;
        let consumed: Option<String> = redis::cmd("GETDEL")
            .arg(self.code_key(code))
            .query_async(&mut conn)
            .await?;
        let Some(client_id) = consumed.and_then(|v| Uuid::parse_str(&v).ok()) else {
            return Ok(BindOutcome::NotFound);
        };
        // The lookup index can outlive the device pointer by the moments
        // around expiry; a stale index must not bind.
        let pointer: Option<String> = redis::cmd("GET")
            .arg(self.device_key(client_id))
            .query_async(&mut conn)
            .await?;
        if pointer.as_deref() != Some(code) {
            return Ok(BindOutcome::NotFound);
        }
        redis::cmd("DEL")
            .arg(self.device_key(client_id))
            .query_async::<()>(&mut conn)
            .await?;

        let updated = sqlx::query_as::<_, DeviceRecord>(&format!(
            "UPDATE devices \
                SET user_id = $1, agent_id = $2, activated_at = now() \
              WHERE client_id = $3 AND user_id IS NULL \
                AND EXISTS (\
                    SELECT 1 FROM agents \
                     WHERE agents.id = $2 AND agents.user_id = $1\
                ) \
              RETURNING {DEVICE_COLUMNS}"
        ))
        .bind(user_id)
        .bind(agent_id)
        .bind(client_id)
        .fetch_optional(&self.pool)
        .await?;
        if let Some(record) = updated {
            return Ok(BindOutcome::Bound(record));
        }
        if !agents.owns(agent_id, user_id).await? {
            return Ok(BindOutcome::UnknownAgent);
        }
        Ok(BindOutcome::Conflict)
    }

    pub async fn list_for_user(&self, user_id: i64) -> Result<Vec<DeviceRecord>, AppError> {
        let records = sqlx::query_as::<_, DeviceRecord>(&format!(
            "SELECT {DEVICE_COLUMNS} FROM devices WHERE user_id = $1 ORDER BY created_at DESC"
        ))
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(records)
    }

    pub async fn unbind(&self, user_id: i64, client_id: Uuid) -> Result<bool, AppError> {
        let affected = sqlx::query(
            "UPDATE devices \
                SET user_id = NULL, token_hash = NULL, activated_at = NULL, \
                    agent_id = NULL \
              WHERE client_id = $1 AND user_id = $2",
        )
        .bind(client_id)
        .bind(user_id)
        .execute(&self.pool)
        .await?
        .rows_affected();
        Ok(affected > 0)
    }
}

fn generate_code() -> String {
    let random = Uuid::new_v4().as_u128();
    format!("{:06}", random % 1_000_000)
}

fn hash_token(token: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_six_digits() {
        for _ in 0..1000 {
            let code = generate_code();
            assert_eq!(code.len(), 6);
            assert!(code.chars().all(|c| c.is_ascii_digit()));
        }
    }

    #[test]
    fn token_hash_is_stable_and_hex() {
        let a = hash_token("hello");
        let b = hash_token("hello");
        assert_eq!(a, b);
        assert_ne!(a, hash_token("world"));
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
