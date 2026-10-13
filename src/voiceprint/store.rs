//! DB-backed per-agent speaker voiceprints: the durable store for the
//! embeddings [`crate::voiceprint`] produces. One row per (agent, label);
//! `embedding` holds the element-wise sum of the L2-normalized per-segment
//! vectors, so enrolling more audio is a vector addition — one transaction
//! that row-locks the existing vector (`SELECT … FOR UPDATE`) before adding;
//! a concurrent insert of the same brand-new label loses its insert race and
//! retries as an append. Every method takes the caller's `user_id` and
//! resolves ownership through `agents` inline: a row behind an agent the
//! caller does not own simply does not exist (`None` / `false` /
//! [`AppError::NotFound`]).

use sqlx::PgPool;

use crate::error::AppError;

pub struct VoiceprintStore {
    pool: PgPool,
}

/// The row as the REST surface returns it — the 192-float embedding is
/// deliberately left out: lists and edits never carry the vector, and the
/// identification flow reads it through its own query
/// ([`VoiceprintStore::embeddings_for_agent`]).
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct VoiceprintRecord {
    pub id: i64,
    pub agent_id: i64,
    pub label: String,
    pub description: String,
    pub sample_count: i32,
    pub created_at: time::OffsetDateTime,
    pub updated_at: time::OffsetDateTime,
}

/// The identification read shape: one enrolled label with its description
/// and embedding vector — the vector the REST surface deliberately never
/// carries.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct EnrolledEmbedding {
    pub label: String,
    pub description: String,
    pub embedding: Vec<f32>,
}

impl VoiceprintStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Inserts a new voiceprint for `label`, or — when the agent already has
    /// one — adds `addend` element-wise to its embedding and accumulates
    /// `sample_count`, inside one transaction that row-locks the existing
    /// vector first. `description` overwrites on `Some` and stays empty /
    /// untouched on `None`. An agent the caller does not own is
    /// [`AppError::NotFound`], so a cross-agent label can never be touched.
    pub async fn create_or_append(
        &self,
        agent_id: i64,
        user_id: i64,
        label: &str,
        description: Option<&str>,
        addend: Vec<f32>,
        sample_count: i32,
    ) -> Result<VoiceprintRecord, AppError> {
        loop {
            let mut tx = self.pool.begin().await?;
            let current = sqlx::query_as::<_, (i64, Vec<f32>)>(
                "SELECT v.id, v.embedding \
                 FROM voiceprints v JOIN agents a ON a.id = v.agent_id \
                 WHERE v.agent_id = $1 AND v.label = $2 AND a.user_id = $3 \
                 FOR UPDATE OF v",
            )
            .bind(agent_id)
            .bind(label)
            .bind(user_id)
            .fetch_optional(&mut *tx)
            .await?;
            let record = if let Some((id, current)) = current {
                if current.len() != addend.len() {
                    return Err(AppError::Internal(format!(
                        "voiceprint embedding length mismatch: {} != {}",
                        current.len(),
                        addend.len()
                    )));
                }
                let mut sum = current;
                for (acc, value) in sum.iter_mut().zip(addend.iter()) {
                    *acc += value;
                }
                sqlx::query_as::<_, VoiceprintRecord>(
                    "UPDATE voiceprints \
                     SET embedding = $2, sample_count = sample_count + $3, \
                         description = COALESCE($4, description), updated_at = now() \
                     WHERE id = $1 \
                     RETURNING id, agent_id, label, description, sample_count, \
                               created_at, updated_at",
                )
                .bind(id)
                .bind(sum)
                .bind(sample_count)
                .bind(description)
                .fetch_one(&mut *tx)
                .await
            } else {
                match sqlx::query_as::<_, VoiceprintRecord>(
                    "INSERT INTO voiceprints (agent_id, label, description, embedding, \
                     sample_count) \
                     SELECT $1, $3, COALESCE($4, ''), $5, $6 \
                     WHERE EXISTS (SELECT 1 FROM agents WHERE id = $1 AND user_id = $2) \
                     RETURNING id, agent_id, label, description, sample_count, \
                               created_at, updated_at",
                )
                .bind(user_id)
                .bind(agent_id)
                .bind(label)
                .bind(description)
                .bind(&addend)
                .bind(sample_count)
                .fetch_optional(&mut *tx)
                .await
                {
                    Ok(Some(record)) => Ok(record),
                    Ok(None) => return Err(AppError::NotFound),
                    Err(err) => Err(err),
                }
            };
            match record {
                Ok(record) => {
                    tx.commit().await?;
                    return Ok(record);
                }
                Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => {
                    continue;
                }
                Err(err) => return Err(err.into()),
            }
        }
    }

    pub async fn list_for_agent(
        &self,
        agent_id: i64,
        user_id: i64,
    ) -> Result<Vec<VoiceprintRecord>, AppError> {
        let rows = sqlx::query_as::<_, VoiceprintRecord>(
            "SELECT v.id, v.agent_id, v.label, v.description, v.sample_count, \
                    v.created_at, v.updated_at \
             FROM voiceprints v JOIN agents a ON a.id = v.agent_id \
             WHERE v.agent_id = $1 AND a.user_id = $2 \
             ORDER BY v.id",
        )
        .bind(agent_id)
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// The agent's enrolled voiceprints with their embedding vectors — the
    /// session-side identification read (`AgentFactory::build`). Server-internal:
    /// `agent_id` comes from the verified device binding, so unlike the REST
    /// queries there is no caller scope to resolve. An agent without
    /// voiceprints yields an empty vec — no identification pipeline at all.
    pub async fn embeddings_for_agent(
        &self,
        agent_id: i64,
    ) -> Result<Vec<EnrolledEmbedding>, AppError> {
        let rows = sqlx::query_as::<_, EnrolledEmbedding>(
            "SELECT label, description, embedding FROM voiceprints \
             WHERE agent_id = $1 ORDER BY id",
        )
        .bind(agent_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    pub async fn get(
        &self,
        agent_id: i64,
        voiceprint_id: i64,
        user_id: i64,
    ) -> Result<Option<VoiceprintRecord>, AppError> {
        let record = sqlx::query_as::<_, VoiceprintRecord>(
            "SELECT v.id, v.agent_id, v.label, v.description, v.sample_count, \
                    v.created_at, v.updated_at \
             FROM voiceprints v JOIN agents a ON a.id = v.agent_id \
             WHERE v.id = $2 AND v.agent_id = $1 AND a.user_id = $3",
        )
        .bind(agent_id)
        .bind(voiceprint_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
    }

    /// Applies `label` / `description` when given, keeping the other as-is;
    /// renaming onto a label the agent already uses is
    /// [`AppError::Conflict`]. `None` when the voiceprint does not exist
    /// for this caller.
    pub async fn update(
        &self,
        agent_id: i64,
        voiceprint_id: i64,
        user_id: i64,
        label: Option<&str>,
        description: Option<&str>,
    ) -> Result<Option<VoiceprintRecord>, AppError> {
        let record = sqlx::query_as::<_, VoiceprintRecord>(
            "UPDATE voiceprints v \
             SET label = COALESCE($1, v.label), \
                 description = COALESCE($2, v.description), \
                 updated_at = now() \
             FROM agents a \
             WHERE v.id = $3 AND v.agent_id = $4 AND a.id = v.agent_id AND a.user_id = $5 \
             RETURNING v.id, v.agent_id, v.label, v.description, v.sample_count, \
                       v.created_at, v.updated_at",
        )
        .bind(label)
        .bind(description)
        .bind(voiceprint_id)
        .bind(agent_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await;
        match record {
            Ok(record) => Ok(record),
            Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => {
                Err(AppError::Conflict("voiceprint label already taken".into()))
            }
            Err(err) => Err(err.into()),
        }
    }

    pub async fn delete(
        &self,
        agent_id: i64,
        voiceprint_id: i64,
        user_id: i64,
    ) -> Result<bool, AppError> {
        let result = sqlx::query(
            "DELETE FROM voiceprints v USING agents a \
             WHERE v.id = $1 AND v.agent_id = $2 AND a.id = v.agent_id AND a.user_id = $3",
        )
        .bind(voiceprint_id)
        .bind(agent_id)
        .bind(user_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }
}
