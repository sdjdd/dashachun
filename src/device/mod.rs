use sqlx::PgPool;
use uuid::Uuid;

use crate::error::AppError;

pub mod dto;

const MAX_CODE_ATTEMPTS: u32 = 8;

#[derive(Clone)]
pub struct DeviceStore {
    pool: PgPool,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DeviceRecord {
    pub id: i64,
    pub client_id: Uuid,
    pub device_id: Option<String>,
    pub board_type: Option<String>,
    pub user_id: Option<i64>,
    pub activated_at: Option<time::OffsetDateTime>,
    pub created_at: time::OffsetDateTime,
    pub last_seen_at: time::OffsetDateTime,
}

pub enum BindOutcome {
    Bound(DeviceRecord),
    NotFound,
    Conflict,
}

const DEVICE_COLUMNS: &str =
    "id, client_id, device_id, board_type, user_id, activated_at, created_at, last_seen_at";

impl DeviceStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
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

    pub async fn ensure_code(&self, client_id: Uuid, ttl_secs: i64) -> Result<String, AppError> {
        let existing = sqlx::query_scalar::<_, String>(
            "SELECT activation_code FROM devices \
             WHERE client_id = $1 AND user_id IS NULL \
               AND activation_code IS NOT NULL \
               AND activation_code_expires_at > now()",
        )
        .bind(client_id)
        .fetch_optional(&self.pool)
        .await?;
        if let Some(code) = existing {
            return Ok(code);
        }

        for _ in 0..MAX_CODE_ATTEMPTS {
            let code = generate_code();
            let result = sqlx::query(
                "UPDATE devices \
                    SET activation_code = $2, \
                        activation_code_expires_at = now() + make_interval(secs => $3) \
                  WHERE client_id = $1 AND user_id IS NULL",
            )
            .bind(client_id)
            .bind(&code)
            .bind(ttl_secs as f64)
            .execute(&self.pool)
            .await;
            match result {
                Ok(_) => return Ok(code),
                Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => {
                    continue;
                }
                Err(err) => return Err(err.into()),
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

    pub async fn verify(&self, client_id: Uuid, token: &str) -> Result<bool, AppError> {
        let hash = hash_token(token);
        let ok = sqlx::query_scalar::<_, bool>(
            "SELECT EXISTS(SELECT 1 FROM devices \
             WHERE client_id = $1 AND token_hash = $2 AND user_id IS NOT NULL)",
        )
        .bind(client_id)
        .bind(&hash)
        .fetch_one(&self.pool)
        .await?;
        Ok(ok)
    }

    pub async fn bind_by_code(&self, user_id: i64, code: &str) -> Result<BindOutcome, AppError> {
        let updated = sqlx::query_as::<_, DeviceRecord>(&format!(
            "UPDATE devices \
                SET user_id = $1, activated_at = now(), \
                    activation_code = NULL, activation_code_expires_at = NULL \
              WHERE activation_code = $2 AND user_id IS NULL \
                AND activation_code_expires_at > now() \
              RETURNING {DEVICE_COLUMNS}"
        ))
        .bind(user_id)
        .bind(code)
        .fetch_optional(&self.pool)
        .await?;
        if let Some(record) = updated {
            return Ok(BindOutcome::Bound(record));
        }

        let taken = sqlx::query_scalar::<_, bool>(
            "SELECT user_id IS NOT NULL FROM devices WHERE activation_code = $1",
        )
        .bind(code)
        .fetch_optional(&self.pool)
        .await?;
        match taken {
            Some(true) => Ok(BindOutcome::Conflict),
            _ => Ok(BindOutcome::NotFound),
        }
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
                    activation_code = NULL, activation_code_expires_at = NULL \
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
