use std::sync::Mutex;

use async_trait::async_trait;
use sqlx::PgPool;
use tracing::warn;

/// The per-scope cap: at most this many active entries per
/// (user_id, agent_id). The prompt states it and [`EntryMemory::add`]
/// refuses beyond it — the model must update or delete instead.
pub const MAX_ENTRIES: usize = 20;

/// The display form of an entry id, the `id` attribute of a `<memory>`
/// element in the system prompt and echoed back by the memory tools.
pub fn entry_id(mem_no: i32) -> String {
    mem_no.to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEntry {
    pub mem_no: i32,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddError {
    Full,
    Failed(String),
}

/// One proposed change to the scope's entries, already validated — the
/// payload the extraction tool submits, applied as one batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryOp {
    Add(String),
    Update(i32, String),
    Delete(i32),
}

/// The per-op outcome of [`EntryMemory::apply`], in the submitted order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryOpResult {
    Added(i32),
    Updated(i32),
    Deleted(i32),
    Skipped(String),
}

/// The entry-based user memory for one (user_id, agent_id) scope: short
/// durable facts the LLM maintains through the memory tools and the prompt
/// reads every turn. A fresh instance is built per connection, bound to the
/// device's owner like [`super::DbMemory`].
#[async_trait]
pub trait EntryMemory: Send + Sync {
    /// The active entries, oldest first.
    async fn list(&self) -> Vec<MemoryEntry>;

    /// Stores a new entry with the next monotonic id. Fails with
    /// [`AddError::Full`] when the scope already holds [`MAX_ENTRIES`]
    /// active entries.
    async fn add(&self, content: &str) -> Result<MemoryEntry, AddError>;

    /// Replaces an entry's content; `false` when the id is unknown.
    async fn update(&self, mem_no: i32, content: &str) -> Result<bool, String>;

    /// Removes an entry; `false` when the id is unknown. The id is never
    /// reused within the scope.
    async fn delete(&self, mem_no: i32) -> Result<bool, String>;

    /// Applies a batch of changes in one transaction, in order, and reports
    /// one outcome per submitted op. Best-effort: a full memory skips the
    /// remaining adds, an unknown id skips that op, a store failure skips
    /// them all — nothing about one op blocks the others.
    async fn apply(&self, ops: Vec<EntryOp>) -> Vec<EntryOpResult>;
}

pub struct DbEntryMemory {
    pool: PgPool,
    user_id: i64,
    agent_id: i64,
}

impl DbEntryMemory {
    pub fn new(pool: PgPool, user_id: i64, agent_id: i64) -> Self {
        Self {
            pool,
            user_id,
            agent_id,
        }
    }
}

#[async_trait]
impl EntryMemory for DbEntryMemory {
    async fn list(&self) -> Vec<MemoryEntry> {
        match sqlx::query_as::<_, (i32, String)>(
            "SELECT mem_no, content FROM memory_entries \
             WHERE user_id = $1 AND agent_id = $2 AND deleted_at IS NULL \
             ORDER BY mem_no",
        )
        .bind(self.user_id)
        .bind(self.agent_id)
        .fetch_all(&self.pool)
        .await
        {
            Ok(rows) => rows
                .into_iter()
                .map(|(mem_no, content)| MemoryEntry { mem_no, content })
                .collect(),
            Err(err) => {
                warn!(%err, "failed to load memory entries");
                Vec::new()
            }
        }
    }

    async fn add(&self, content: &str) -> Result<MemoryEntry, AddError> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|err| AddError::Failed(err.to_string()))?;
        // Serialize the scope so the cap check and the id assignment stay
        // atomic even with the same agent talking on two devices.
        sqlx::query("SELECT pg_advisory_xact_lock(hashtext($1))")
            .bind(format!("{}:{}", self.user_id, self.agent_id))
            .execute(&mut *tx)
            .await
            .map_err(|err| AddError::Failed(err.to_string()))?;
        let count: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM memory_entries \
             WHERE user_id = $1 AND agent_id = $2 AND deleted_at IS NULL",
        )
        .bind(self.user_id)
        .bind(self.agent_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|err| AddError::Failed(err.to_string()))?;
        if count as usize >= MAX_ENTRIES {
            return Err(AddError::Full);
        }
        // max() runs over every row including soft-deleted ones, so freed
        // ids are never handed out again.
        let (mem_no, content): (i32, String) = sqlx::query_as(
            "INSERT INTO memory_entries (user_id, agent_id, mem_no, content) \
             VALUES ($1, $2, (SELECT coalesce(max(mem_no), 0) + 1 FROM memory_entries \
             WHERE user_id = $1 AND agent_id = $2), $3) \
             RETURNING mem_no, content",
        )
        .bind(self.user_id)
        .bind(self.agent_id)
        .bind(content)
        .fetch_one(&mut *tx)
        .await
        .map_err(|err| AddError::Failed(err.to_string()))?;
        tx.commit()
            .await
            .map_err(|err| AddError::Failed(err.to_string()))?;
        Ok(MemoryEntry { mem_no, content })
    }

    async fn update(&self, mem_no: i32, content: &str) -> Result<bool, String> {
        let result = sqlx::query(
            "UPDATE memory_entries SET content = $3, updated_at = now() \
             WHERE user_id = $1 AND agent_id = $2 AND mem_no = $4 AND deleted_at IS NULL",
        )
        .bind(self.user_id)
        .bind(self.agent_id)
        .bind(content)
        .bind(mem_no)
        .execute(&self.pool)
        .await
        .map_err(|err| err.to_string())?;
        Ok(result.rows_affected() > 0)
    }

    async fn delete(&self, mem_no: i32) -> Result<bool, String> {
        let result = sqlx::query(
            "UPDATE memory_entries SET deleted_at = now() \
             WHERE user_id = $1 AND agent_id = $2 AND mem_no = $3 AND deleted_at IS NULL",
        )
        .bind(self.user_id)
        .bind(self.agent_id)
        .bind(mem_no)
        .execute(&self.pool)
        .await
        .map_err(|err| err.to_string())?;
        Ok(result.rows_affected() > 0)
    }

    async fn apply(&self, ops: Vec<EntryOp>) -> Vec<EntryOpResult> {
        match self.apply_tx(&ops).await {
            Ok(results) => results,
            Err(err) => {
                warn!(%err, "failed to apply a memory entry batch");
                ops.iter()
                    .map(|_| EntryOpResult::Skipped("memory store failed".into()))
                    .collect()
            }
        }
    }
}

impl DbEntryMemory {
    /// One transaction for the whole batch — the advisory lock serializes
    /// the scope, so the running active count and the monotonic id
    /// assignment stay exact even across devices.
    async fn apply_tx(&self, ops: &[EntryOp]) -> Result<Vec<EntryOpResult>, String> {
        let mut tx = self.pool.begin().await.map_err(|err| err.to_string())?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtext($1))")
            .bind(format!("{}:{}", self.user_id, self.agent_id))
            .execute(&mut *tx)
            .await
            .map_err(|err| err.to_string())?;
        let mut active: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM memory_entries \
             WHERE user_id = $1 AND agent_id = $2 AND deleted_at IS NULL",
        )
        .bind(self.user_id)
        .bind(self.agent_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|err| err.to_string())?;
        let mut results = Vec::with_capacity(ops.len());
        for op in ops {
            results.push(match op {
                EntryOp::Add(content) => {
                    if active >= MAX_ENTRIES as i64 {
                        EntryOpResult::Skipped(format!(
                            "memory is full ({MAX_ENTRIES} entries)"
                        ))
                    } else {
                        let (mem_no, _): (i32, String) = sqlx::query_as(
                            "INSERT INTO memory_entries (user_id, agent_id, mem_no, content) \
                             VALUES ($1, $2, (SELECT coalesce(max(mem_no), 0) + 1 FROM memory_entries \
                             WHERE user_id = $1 AND agent_id = $2), $3) \
                             RETURNING mem_no, content",
                        )
                        .bind(self.user_id)
                        .bind(self.agent_id)
                        .bind(content)
                        .fetch_one(&mut *tx)
                        .await
                        .map_err(|err| err.to_string())?;
                        active += 1;
                        EntryOpResult::Added(mem_no)
                    }
                }
                EntryOp::Update(mem_no, content) => {
                    let updated = sqlx::query(
                        "UPDATE memory_entries SET content = $3, updated_at = now() \
                         WHERE user_id = $1 AND agent_id = $2 AND mem_no = $4 AND deleted_at IS NULL",
                    )
                    .bind(self.user_id)
                    .bind(self.agent_id)
                    .bind(content)
                    .bind(mem_no)
                    .execute(&mut *tx)
                    .await
                    .map_err(|err| err.to_string())?;
                    if updated.rows_affected() > 0 {
                        EntryOpResult::Updated(*mem_no)
                    } else {
                        EntryOpResult::Skipped(format!("unknown {}", entry_id(*mem_no)))
                    }
                }
                EntryOp::Delete(mem_no) => {
                    let deleted = sqlx::query(
                        "UPDATE memory_entries SET deleted_at = now() \
                         WHERE user_id = $1 AND agent_id = $2 AND mem_no = $3 AND deleted_at IS NULL",
                    )
                    .bind(self.user_id)
                    .bind(self.agent_id)
                    .bind(mem_no)
                    .execute(&mut *tx)
                    .await
                    .map_err(|err| err.to_string())?;
                    if deleted.rows_affected() > 0 {
                        active -= 1;
                        EntryOpResult::Deleted(*mem_no)
                    } else {
                        EntryOpResult::Skipped(format!("unknown {}", entry_id(*mem_no)))
                    }
                }
            });
        }
        tx.commit().await.map_err(|err| err.to_string())?;
        Ok(results)
    }
}

#[derive(Default)]
struct InMemState {
    entries: Vec<MemoryEntry>,
    next_no: i32,
}

#[derive(Default)]
pub struct InMemEntryMemory {
    state: Mutex<InMemState>,
}

impl InMemEntryMemory {
    pub fn stored(&self) -> Vec<MemoryEntry> {
        self.state.lock().unwrap().entries.clone()
    }
}

#[async_trait]
impl EntryMemory for InMemEntryMemory {
    async fn list(&self) -> Vec<MemoryEntry> {
        self.state.lock().unwrap().entries.clone()
    }

    async fn add(&self, content: &str) -> Result<MemoryEntry, AddError> {
        let mut state = self.state.lock().unwrap();
        if state.entries.len() >= MAX_ENTRIES {
            return Err(AddError::Full);
        }
        // Monotonic like the durable store: freed ids are never reused.
        state.next_no += 1;
        let entry = MemoryEntry {
            mem_no: state.next_no,
            content: content.to_string(),
        };
        state.entries.push(entry.clone());
        Ok(entry)
    }

    async fn update(&self, mem_no: i32, content: &str) -> Result<bool, String> {
        let mut state = self.state.lock().unwrap();
        match state
            .entries
            .iter_mut()
            .find(|entry| entry.mem_no == mem_no)
        {
            Some(entry) => {
                entry.content = content.to_string();
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete(&self, mem_no: i32) -> Result<bool, String> {
        let mut state = self.state.lock().unwrap();
        let before = state.entries.len();
        state.entries.retain(|entry| entry.mem_no != mem_no);
        Ok(state.entries.len() < before)
    }

    async fn apply(&self, ops: Vec<EntryOp>) -> Vec<EntryOpResult> {
        let mut state = self.state.lock().unwrap();
        let mut results = Vec::with_capacity(ops.len());
        for op in ops {
            results.push(match op {
                EntryOp::Add(content) => {
                    if state.entries.len() >= MAX_ENTRIES {
                        EntryOpResult::Skipped(format!("memory is full ({MAX_ENTRIES} entries)"))
                    } else {
                        state.next_no += 1;
                        let mem_no = state.next_no;
                        state.entries.push(MemoryEntry { mem_no, content });
                        EntryOpResult::Added(mem_no)
                    }
                }
                EntryOp::Update(mem_no, content) => {
                    match state
                        .entries
                        .iter_mut()
                        .find(|entry| entry.mem_no == mem_no)
                    {
                        Some(entry) => {
                            entry.content = content;
                            EntryOpResult::Updated(mem_no)
                        }
                        None => EntryOpResult::Skipped(format!("unknown {}", entry_id(mem_no))),
                    }
                }
                EntryOp::Delete(mem_no) => {
                    let before = state.entries.len();
                    state.entries.retain(|entry| entry.mem_no != mem_no);
                    if state.entries.len() < before {
                        EntryOpResult::Deleted(mem_no)
                    } else {
                        EntryOpResult::Skipped(format!("unknown {}", entry_id(mem_no)))
                    }
                }
            });
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inmem_add_lists_and_keeps_ids_monotonic() {
        let memory = InMemEntryMemory::default();
        let first = memory.add("one").await.unwrap();
        let second = memory.add("two").await.unwrap();
        assert_eq!(first.mem_no, 1);
        assert_eq!(second.mem_no, 2);
        assert_eq!(memory.list().await, vec![first, second.clone()]);
        assert!(memory.delete(2).await.unwrap());
        let third = memory.add("three").await.unwrap();
        assert_eq!(third.mem_no, 3);
        assert_eq!(
            memory.list().await,
            vec![
                MemoryEntry {
                    mem_no: 1,
                    content: "one".into()
                },
                third,
            ]
        );
    }

    #[tokio::test]
    async fn inmem_add_hits_the_cap() {
        let memory = InMemEntryMemory::default();
        for i in 0..MAX_ENTRIES {
            memory.add(&format!("fact {i}")).await.unwrap();
        }
        assert_eq!(memory.add("one more").await, Err(AddError::Full));
        assert_eq!(memory.list().await.len(), MAX_ENTRIES);
    }

    #[tokio::test]
    async fn inmem_update_and_delete_report_unknown_ids() {
        let memory = InMemEntryMemory::default();
        memory.add("one").await.unwrap();
        assert!(memory.update(1, "renamed").await.unwrap());
        assert!(!memory.update(9, "x").await.unwrap());
        assert!(memory.delete(1).await.unwrap());
        assert!(!memory.delete(1).await.unwrap());
        assert!(memory.list().await.is_empty());
    }

    #[tokio::test]
    async fn inmem_apply_runs_a_mixed_batch_best_effort() {
        let memory = InMemEntryMemory::default();
        memory.add("old").await.unwrap();
        let results = memory
            .apply(vec![
                EntryOp::Update(1, "renamed".into()),
                EntryOp::Add("new fact".into()),
                EntryOp::Delete(9),
            ])
            .await;
        assert_eq!(
            results,
            vec![
                EntryOpResult::Updated(1),
                EntryOpResult::Added(2),
                EntryOpResult::Skipped("unknown 9".into()),
            ]
        );
        assert_eq!(
            memory.list().await,
            vec![
                MemoryEntry {
                    mem_no: 1,
                    content: "renamed".into()
                },
                MemoryEntry {
                    mem_no: 2,
                    content: "new fact".into()
                },
            ]
        );
    }

    #[tokio::test]
    async fn inmem_apply_fills_to_the_cap_and_skips_the_rest() {
        let memory = InMemEntryMemory::default();
        for i in 0..MAX_ENTRIES - 1 {
            memory.add(&format!("fact {i}")).await.unwrap();
        }
        let results = memory
            .apply(vec![
                EntryOp::Add("fits".into()),
                EntryOp::Add("overflows".into()),
            ])
            .await;
        assert_eq!(results[0], EntryOpResult::Added(MAX_ENTRIES as i32));
        assert!(matches!(&results[1], EntryOpResult::Skipped(reason) if reason.contains("full")));
        assert_eq!(memory.list().await.len(), MAX_ENTRIES);
    }

    #[tokio::test]
    async fn inmem_apply_frees_a_slot_within_one_batch() {
        let memory = InMemEntryMemory::default();
        memory.add("stale").await.unwrap();
        let results = memory
            .apply(vec![EntryOp::Delete(1), EntryOp::Add("fresh".into())])
            .await;
        assert_eq!(
            results,
            vec![EntryOpResult::Deleted(1), EntryOpResult::Added(2)]
        );
        assert_eq!(
            memory.list().await,
            vec![MemoryEntry {
                mem_no: 2,
                content: "fresh".into()
            }]
        );
    }
}
