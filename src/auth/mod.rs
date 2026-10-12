pub mod dto;
pub mod extract;
pub mod password;
pub mod session;
pub mod state;

use serde::Serialize;
use sqlx::PgPool;

use crate::error::AppError;

pub struct UserRecord {
    pub id: i64,
    pub username: String,
    pub password_hash: String,
    pub role: Role,
}

/// The account's standing: `admin` runs the operator surface (`/api/providers`,
/// `/api/settings/*`), `user` is everyone else. The first registered account
/// becomes admin, so a fresh server is configurable without SQL access;
/// later promotions and demotions are SQL-only — there is no role REST.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, sqlx::Type)]
#[serde(rename_all = "lowercase")]
#[sqlx(type_name = "text", rename_all = "lowercase")]
pub enum Role {
    User,
    Admin,
}

pub async fn create_user(
    pool: &PgPool,
    username: &str,
    password: &str,
) -> Result<(i64, Role), AppError> {
    let password_hash = password::hash(password)?;
    let mut tx = pool.begin().await?;
    // The bootstrap check and the insert must not race: two concurrent first
    // registrations could both see an empty table and both land as admin.
    // One advisory lock serializes the pair; the Argon2 work above stays
    // outside it.
    sqlx::query("SELECT pg_advisory_xact_lock(hashtext('dashachun:bootstrap-admin'))")
        .execute(&mut *tx)
        .await?;
    let result = sqlx::query_as::<_, (i64, Role)>(
        "INSERT INTO users (username, password_hash, role) \
         VALUES ($1, $2, CASE WHEN EXISTS (SELECT 1 FROM users) THEN 'user' ELSE 'admin' END) \
         RETURNING id, role",
    )
    .bind(username)
    .bind(&password_hash)
    .fetch_one(&mut *tx)
    .await;

    match result {
        Ok((id, role)) => {
            tx.commit().await?;
            Ok((id, role))
        }
        Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => {
            Err(AppError::Conflict("username already taken".into()))
        }
        Err(err) => Err(err.into()),
    }
}

pub async fn find_user(pool: &PgPool, username: &str) -> Result<Option<UserRecord>, AppError> {
    let record = sqlx::query_as::<_, (i64, String, String, Role)>(
        "SELECT id, username, password_hash, role FROM users WHERE username = $1",
    )
    .bind(username)
    .fetch_optional(pool)
    .await?;

    Ok(
        record.map(|(id, username, password_hash, role)| UserRecord {
            id,
            username,
            password_hash,
            role,
        }),
    )
}

pub async fn find_user_by_id(pool: &PgPool, user_id: i64) -> Result<Option<UserRecord>, AppError> {
    let record = sqlx::query_as::<_, (i64, String, String, Role)>(
        "SELECT id, username, password_hash, role FROM users WHERE id = $1",
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await?;

    Ok(
        record.map(|(id, username, password_hash, role)| UserRecord {
            id,
            username,
            password_hash,
            role,
        }),
    )
}

/// The role behind a session's user id — `AdminUser`'s second lookup. A
/// session whose user row is gone is `Unauthorized` (no user deletion exists
/// today; this is belt-and-braces).
pub async fn user_role(pool: &PgPool, user_id: i64) -> Result<Role, AppError> {
    let role: Option<Role> = sqlx::query_scalar("SELECT role FROM users WHERE id = $1")
        .bind(user_id)
        .fetch_optional(pool)
        .await?;
    role.ok_or(AppError::Unauthorized)
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
