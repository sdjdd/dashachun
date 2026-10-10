use serde_json::Value;
use sqlx::PgPool;

use crate::error::AppError;

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ProviderRecord {
    pub id: i64,
    #[sqlx(rename = "type")]
    pub r#type: String,
    pub name: String,
    pub kind: String,
    pub config: Value,
    pub created_at: time::OffsetDateTime,
    pub updated_at: time::OffsetDateTime,
}

/// The provider catalog: which provider instances exist, identified by
/// `(type, name)` — e.g. type=tts, name=volcengine. Which one each type
/// uses lives in the `settings` table (`{"provider": <id>}`), so this
/// store is pure CRUD — activation is orchestrated by the admin routes.
#[derive(Clone)]
pub struct ProviderStore {
    pool: PgPool,
}

impl ProviderStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn list(&self) -> Result<Vec<ProviderRecord>, AppError> {
        let rows = sqlx::query_as::<_, ProviderRecord>(
            "SELECT id, type, name, kind, config, created_at, updated_at \
             FROM providers ORDER BY type, id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    pub async fn find(&self, id: i64) -> Result<Option<ProviderRecord>, AppError> {
        let record = sqlx::query_as::<_, ProviderRecord>(
            "SELECT id, type, name, kind, config, created_at, updated_at \
             FROM providers WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
    }

    pub async fn create(
        &self,
        r#type: &str,
        name: &str,
        kind: &str,
        config: &Value,
    ) -> Result<ProviderRecord, AppError> {
        let record = sqlx::query_as::<_, ProviderRecord>(
            "INSERT INTO providers (type, name, kind, config) VALUES ($1, $2, $3, $4) \
             RETURNING id, type, name, kind, config, created_at, updated_at",
        )
        .bind(r#type)
        .bind(name)
        .bind(kind)
        .bind(config)
        .fetch_one(&self.pool)
        .await;

        match record {
            Ok(record) => Ok(record),
            Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => Err(
                AppError::Conflict("provider name already taken for this type".into()),
            ),
            Err(err) => Err(err.into()),
        }
    }

    pub async fn update(
        &self,
        id: i64,
        name: Option<&str>,
        kind: Option<&str>,
        config: Option<&Value>,
    ) -> Result<Option<ProviderRecord>, AppError> {
        let record = sqlx::query_as::<_, ProviderRecord>(
            "UPDATE providers SET \
                name = COALESCE($2, name), \
                kind = COALESCE($3, kind), \
                config = COALESCE($4, config), \
                updated_at = now() \
             WHERE id = $1 \
             RETURNING id, type, name, kind, config, created_at, updated_at",
        )
        .bind(id)
        .bind(name)
        .bind(kind)
        .bind(config)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
    }

    pub async fn delete(&self, id: i64) -> Result<bool, AppError> {
        let result = sqlx::query("DELETE FROM providers WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }
}
