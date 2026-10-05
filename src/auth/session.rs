use sqlx::PgPool;
use uuid::Uuid;

use crate::error::AppError;

pub async fn create(pool: &PgPool, user_id: i64, ttl_secs: i64) -> Result<Uuid, AppError> {
    let id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO sessions (id, user_id, expires_at) \
         VALUES ($1, $2, now() + make_interval(secs => $3))",
    )
    .bind(id)
    .bind(user_id)
    .bind(ttl_secs as f64)
    .execute(pool)
    .await?;
    Ok(id)
}

pub async fn user_id(pool: &PgPool, id: Uuid) -> Result<Option<i64>, AppError> {
    let row = sqlx::query_scalar::<_, i64>(
        "SELECT user_id FROM sessions WHERE id = $1 AND expires_at > now()",
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

pub async fn delete(pool: &PgPool, id: Uuid) -> Result<(), AppError> {
    sqlx::query("DELETE FROM sessions WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn delete_others(pool: &PgPool, user_id: i64, keep: Uuid) -> Result<(), AppError> {
    sqlx::query("DELETE FROM sessions WHERE user_id = $1 AND id <> $2")
        .bind(user_id)
        .bind(keep)
        .execute(pool)
        .await?;
    Ok(())
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
