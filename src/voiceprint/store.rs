//! DB-backed per-agent speakers: the durable store for the identities the
//! embeddings [`crate::voiceprint`] produce. One row per (agent, name);
//! `embedding` holds the element-wise sum of the L2-normalized per-segment
//! vectors, so enrolling more audio is a vector addition — one transaction
//! that row-locks the existing vector (`SELECT … FOR UPDATE`) before adding;
//! a concurrent insert of the same brand-new name loses its insert race and
//! retries as an append. Every method takes the caller's `user_id` and
//! resolves ownership through `agents` inline: a row behind an agent the
//! caller does not own simply does not exist (`None` / `false` /
//! [`AppError::NotFound`]).

use sqlx::PgPool;

use crate::error::AppError;

pub struct SpeakerStore {
    pool: PgPool,
}

/// The row as the REST surface returns it — the 192-float embedding is
/// deliberately left out: lists and edits never carry the vector, and the
/// identification flow reads it through its own query
/// ([`SpeakerStore::embeddings_for_agent`]).
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SpeakerRecord {
    pub id: i64,
    pub agent_id: i64,
    pub name: String,
    pub description: String,
    pub sample_count: i32,
    pub created_at: time::OffsetDateTime,
    pub updated_at: time::OffsetDateTime,
}

/// The identification read shape: one enrolled speaker with its row id (the
/// stable speaker id the prompt renders, in the roster and in the
/// context's message labels), its name as the label, its description and
/// embedding vector — the vector the REST surface deliberately never
/// carries.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct EnrolledEmbedding {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub embedding: Vec<f32>,
}

impl SpeakerStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Inserts a new speaker under `name`, or — when the agent already has
    /// one — adds `addend` element-wise to its embedding and accumulates
    /// `sample_count`, inside one transaction that row-locks the existing
    /// vector first. `description` overwrites on `Some` and stays empty /
    /// untouched on `None`. An agent the caller does not own is
    /// [`AppError::NotFound`], so a cross-agent name can never be touched.
    pub async fn create_or_append(
        &self,
        agent_id: i64,
        user_id: i64,
        name: &str,
        description: Option<&str>,
        addend: Vec<f32>,
        sample_count: i32,
    ) -> Result<SpeakerRecord, AppError> {
        loop {
            let mut tx = self.pool.begin().await?;
            let current = sqlx::query_as::<_, (i64, Vec<f32>)>(
                "SELECT s.id, s.embedding \
                 FROM speakers s JOIN agents a ON a.id = s.agent_id \
                 WHERE s.agent_id = $1 AND s.name = $2 AND a.user_id = $3 \
                 FOR UPDATE OF s",
            )
            .bind(agent_id)
            .bind(name)
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
                sqlx::query_as::<_, SpeakerRecord>(
                    "UPDATE speakers \
                     SET embedding = $2, sample_count = sample_count + $3, \
                         description = COALESCE($4, description), updated_at = now() \
                     WHERE id = $1 \
                     RETURNING id, agent_id, name, description, sample_count, \
                               created_at, updated_at",
                )
                .bind(id)
                .bind(sum)
                .bind(sample_count)
                .bind(description)
                .fetch_one(&mut *tx)
                .await
            } else {
                match sqlx::query_as::<_, SpeakerRecord>(
                    "INSERT INTO speakers (agent_id, name, description, embedding, \
                     sample_count) \
                     SELECT $1, $3, COALESCE($4, ''), $5, $6 \
                     WHERE EXISTS (SELECT 1 FROM agents WHERE id = $1 AND user_id = $2) \
                     RETURNING id, agent_id, name, description, sample_count, \
                               created_at, updated_at",
                )
                .bind(user_id)
                .bind(agent_id)
                .bind(name)
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
    ) -> Result<Vec<SpeakerRecord>, AppError> {
        let rows = sqlx::query_as::<_, SpeakerRecord>(
            "SELECT s.id, s.agent_id, s.name, s.description, s.sample_count, \
                    s.created_at, s.updated_at \
             FROM speakers s JOIN agents a ON a.id = s.agent_id \
             WHERE s.agent_id = $1 AND a.user_id = $2 \
             ORDER BY s.id",
        )
        .bind(agent_id)
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// The agent's enrolled speakers with their embedding vectors — the
    /// session-side identification read (`AgentFactory::build`). Server-internal:
    /// `agent_id` comes from the verified device binding, so unlike the REST
    /// queries there is no caller scope to resolve. An agent without
    /// enrollments yields an empty vec — no identification pipeline at all.
    pub async fn embeddings_for_agent(
        &self,
        agent_id: i64,
    ) -> Result<Vec<EnrolledEmbedding>, AppError> {
        let rows = sqlx::query_as::<_, EnrolledEmbedding>(
            "SELECT id, name, description, embedding FROM speakers \
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
        speaker_id: i64,
        user_id: i64,
    ) -> Result<Option<SpeakerRecord>, AppError> {
        let record = sqlx::query_as::<_, SpeakerRecord>(
            "SELECT s.id, s.agent_id, s.name, s.description, s.sample_count, \
                    s.created_at, s.updated_at \
             FROM speakers s JOIN agents a ON a.id = s.agent_id \
             WHERE s.id = $2 AND s.agent_id = $1 AND a.user_id = $3",
        )
        .bind(agent_id)
        .bind(speaker_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
    }

    /// Applies `name` / `description` when given, keeping the other as-is;
    /// renaming onto a name the agent already uses is
    /// [`AppError::Conflict`]. `None` when the speaker does not exist
    /// for this caller.
    pub async fn update(
        &self,
        agent_id: i64,
        speaker_id: i64,
        user_id: i64,
        name: Option<&str>,
        description: Option<&str>,
    ) -> Result<Option<SpeakerRecord>, AppError> {
        let record = sqlx::query_as::<_, SpeakerRecord>(
            "UPDATE speakers s \
             SET name = COALESCE($1, s.name), \
                 description = COALESCE($2, s.description), \
                 updated_at = now() \
             FROM agents a \
             WHERE s.id = $3 AND s.agent_id = $4 AND a.id = s.agent_id AND a.user_id = $5 \
             RETURNING s.id, s.agent_id, s.name, s.description, s.sample_count, \
                       s.created_at, s.updated_at",
        )
        .bind(name)
        .bind(description)
        .bind(speaker_id)
        .bind(agent_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await;
        match record {
            Ok(record) => Ok(record),
            Err(sqlx::Error::Database(err)) if err.code().as_deref() == Some("23505") => {
                Err(AppError::Conflict("speaker name already taken".into()))
            }
            Err(err) => Err(err.into()),
        }
    }

    pub async fn delete(
        &self,
        agent_id: i64,
        speaker_id: i64,
        user_id: i64,
    ) -> Result<bool, AppError> {
        let result = sqlx::query(
            "DELETE FROM speakers s USING agents a \
             WHERE s.id = $1 AND s.agent_id = $2 AND a.id = s.agent_id AND a.user_id = $3",
        )
        .bind(speaker_id)
        .bind(agent_id)
        .bind(user_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }
}
