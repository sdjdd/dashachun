pub mod dto;
pub mod extract;
pub mod password;
pub mod session;
pub mod state;

use sqlx::PgPool;

use crate::error::AppError;

pub struct UserRecord {
    pub id: i64,
    pub username: String,
    pub password_hash: String,
}

pub async fn create_user(pool: &PgPool, username: &str, password: &str) -> Result<i64, AppError> {
    let password_hash = password::hash(password)?;
    let result = sqlx::query_scalar::<_, i64>(
        "INSERT INTO users (username, password_hash) VALUES ($1, $2) RETURNING id",
    )
    .bind(username)
    .bind(&password_hash)
    .fetch_one(pool)
    .await;

    match result {
        Ok(id) => Ok(id),
        Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => {
            Err(AppError::Conflict("username already taken".into()))
        }
        Err(err) => Err(err.into()),
    }
}

pub async fn find_user(pool: &PgPool, username: &str) -> Result<Option<UserRecord>, AppError> {
    let record = sqlx::query_as::<_, (i64, String, String)>(
        "SELECT id, username, password_hash FROM users WHERE username = $1",
    )
    .bind(username)
    .fetch_optional(pool)
    .await?;

    Ok(record.map(|(id, username, password_hash)| UserRecord {
        id,
        username,
        password_hash,
    }))
}

pub async fn find_user_by_id(pool: &PgPool, user_id: i64) -> Result<Option<UserRecord>, AppError> {
    let record = sqlx::query_as::<_, (i64, String, String)>(
        "SELECT id, username, password_hash FROM users WHERE id = $1",
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await?;

    Ok(record.map(|(id, username, password_hash)| UserRecord {
        id,
        username,
        password_hash,
    }))
}

pub async fn update_password(pool: &PgPool, user_id: i64, password: &str) -> Result<(), AppError> {
    let password_hash = password::hash(password)?;
    sqlx::query("UPDATE users SET password_hash = $1, updated_at = now() WHERE id = $2")
        .bind(&password_hash)
        .bind(user_id)
        .execute(pool)
        .await?;
    Ok(())
}
