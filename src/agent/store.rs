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
}

impl AgentStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn find(&self, agent_id: i64) -> Result<Option<AgentRecord>, AppError> {
        let record = sqlx::query_as::<_, AgentRecord>(
            "SELECT id, user_id, name, persona_prompt FROM agents WHERE id = $1",
        )
        .bind(agent_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
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
