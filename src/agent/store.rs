use sqlx::PgPool;

use crate::error::AppError;

/// DB-backed agent definitions. Distinct from the runtime [`crate::agent::Agent`]:
/// this is the user-owned configuration (currently just the persona prompt) that
/// a device is bound to.
#[derive(Clone)]
pub struct AgentStore {
    pool: PgPool,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct AgentRecord {
    pub id: i64,
    pub user_id: i64,
    pub name: String,
    pub persona_prompt: String,
    pub created_at: time::OffsetDateTime,
}

impl AgentStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn find(&self, agent_id: i64) -> Result<Option<AgentRecord>, AppError> {
        let record = sqlx::query_as::<_, AgentRecord>(
            "SELECT id, user_id, name, persona_prompt, created_at FROM agents WHERE id = $1",
        )
        .bind(agent_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
    }

    pub async fn list_for_user(&self, user_id: i64) -> Result<Vec<AgentRecord>, AppError> {
        let records = sqlx::query_as::<_, AgentRecord>(
            "SELECT id, user_id, name, persona_prompt, created_at FROM agents \
             WHERE user_id = $1 ORDER BY id",
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(records)
    }

    pub async fn create(
        &self,
        user_id: i64,
        name: &str,
        persona_prompt: &str,
    ) -> Result<AgentRecord, AppError> {
        let record = sqlx::query_as::<_, AgentRecord>(
            "INSERT INTO agents (user_id, name, persona_prompt) VALUES ($1, $2, $3) \
             RETURNING id, user_id, name, persona_prompt, created_at",
        )
        .bind(user_id)
        .bind(name)
        .bind(persona_prompt)
        .fetch_one(&self.pool)
        .await;

        match record {
            Ok(record) => Ok(record),
            Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => {
                Err(AppError::Conflict("agent name already taken".into()))
            }
            Err(err) => Err(err.into()),
        }
    }

    pub async fn update(
        &self,
        agent_id: i64,
        user_id: i64,
        name: Option<&str>,
        persona_prompt: Option<&str>,
    ) -> Result<Option<AgentRecord>, AppError> {
        let record = sqlx::query_as::<_, AgentRecord>(
            "UPDATE agents SET name = COALESCE($1, name), \
                 persona_prompt = COALESCE($2, persona_prompt), updated_at = now() \
             WHERE id = $3 AND user_id = $4 \
             RETURNING id, user_id, name, persona_prompt, created_at",
        )
        .bind(name)
        .bind(persona_prompt)
        .bind(agent_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await;
        match record {
            Ok(record) => Ok(record),
            Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => {
                Err(AppError::Conflict("agent name already taken".into()))
            }
            Err(err) => Err(err.into()),
        }
    }

    /// Deletes the agent and everything scoped to it in one transaction:
    /// its devices' bindings (the device rows survive, unbound and
    /// token-cleared, ready to re-bind), its speakers, its memory entries
    /// (hard-deleted — the (user_id, agent_id) scope dies with the agent and
    /// ids are never reused), and its conversations with their message and
    /// capture rows. The capture objects are deliberately left in their
    /// bucket — audio retention is the storage's periodic-reaping concern,
    /// not an app-side delete; with the rows gone the app cannot reach the
    /// objects anyway. Returns `false` when the agent does not exist or is
    /// not the caller's.
    pub async fn delete(&self, agent_id: i64, user_id: i64) -> Result<bool, AppError> {
        let mut tx = self.pool.begin().await?;
        let deleted = sqlx::query("DELETE FROM agents WHERE id = $1 AND user_id = $2")
            .bind(agent_id)
            .bind(user_id)
            .execute(&mut *tx)
            .await?
            .rows_affected();
        if deleted == 0 {
            return Ok(false);
        }
        sqlx::query(
            "UPDATE devices SET user_id = NULL, token_hash = NULL, activated_at = NULL, \
                    agent_id = NULL \
              WHERE agent_id = $1 AND user_id IS NOT NULL",
        )
        .bind(agent_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "DELETE FROM message_audios ma USING messages m \
              WHERE ma.message_id = m.id \
                AND m.conversation_id IN (SELECT id FROM conversations WHERE agent_id = $1)",
        )
        .bind(agent_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "DELETE FROM messages \
              WHERE conversation_id IN (SELECT id FROM conversations WHERE agent_id = $1)",
        )
        .bind(agent_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query("DELETE FROM conversations WHERE agent_id = $1")
            .bind(agent_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM speakers WHERE agent_id = $1")
            .bind(agent_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM memory_entries WHERE agent_id = $1")
            .bind(agent_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(true)
    }

    pub async fn owns(&self, agent_id: i64, user_id: i64) -> Result<bool, AppError> {
        let owned = sqlx::query_scalar::<_, bool>(
            "SELECT EXISTS(SELECT 1 FROM agents WHERE id = $1 AND user_id = $2)",
        )
        .bind(agent_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(owned)
    }
}
