use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

mod common;

use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

use common::{RecordingHook, owner, session};
use dashachun::agent::{
    AddError, DbEntryMemory, DbMemory, EntryMemory, EntryOp, EntryOpResult, Extractor, Llm,
    LlmError, LlmEvent, LlmEvents, MAX_ENTRIES, Memory, MemoryEntry, MemoryHook, ToolChoice,
    ToolSpec,
};
use dashachun::agent::{ChatItem, ToolCall};

#[sqlx::test]
#[ignore]
async fn entry_memory_roundtrips_through_the_db(pool: PgPool) {
    let memory = DbEntryMemory::new(pool.clone(), 7, 3);
    let first = memory.add("likes Rust").await.unwrap();
    let second = memory.add("building an ESP32 assistant").await.unwrap();
    assert_eq!(first.mem_no, 1);
    assert_eq!(first.content, "likes Rust");
    assert_eq!(second.mem_no, 2);

    assert_eq!(memory.list().await, vec![first.clone(), second.clone()]);

    assert!(memory.update(1, "loves Rust").await.unwrap());
    let listed = memory.list().await;
    assert_eq!(listed[0].content, "loves Rust");

    // Deleting frees nothing: the next id is monotonic, so a stale prompt
    // reference can never point at a different entry.
    assert!(memory.delete(2).await.unwrap());
    let third = memory.add("third fact").await.unwrap();
    assert_eq!(third.mem_no, 3);
    assert_eq!(
        memory.list().await,
        vec![
            MemoryEntry {
                mem_no: 1,
                content: "loves Rust".into()
            },
            third.clone(),
        ]
    );

    // Unknown ids report false instead of failing.
    assert!(!memory.update(9, "x").await.unwrap());
    assert!(!memory.delete(9).await.unwrap());

    // Another scope sees nothing of it.
    let other = DbEntryMemory::new(pool, 8, 4);
    assert!(other.list().await.is_empty());
    let other_entry = other.add("bob's fact").await.unwrap();
    assert_eq!(other_entry.mem_no, 1);
}

#[sqlx::test]
#[ignore]
async fn entry_memory_enforces_the_cap(pool: PgPool) {
    let memory = DbEntryMemory::new(pool, 7, 3);
    for i in 0..MAX_ENTRIES {
        memory.add(&format!("fact {i}")).await.unwrap();
    }
    match memory.add("one more").await {
        Err(AddError::Full) => {}
        other => panic!("expected Full, got {other:?}"),
    }
    assert_eq!(memory.list().await.len(), MAX_ENTRIES);

    // Deleting frees a slot but not an id.
    assert!(memory.delete(20).await.unwrap());
    let entry = memory.add("after delete").await.unwrap();
    assert_eq!(entry.mem_no, 21);
    assert_eq!(memory.list().await.len(), MAX_ENTRIES);
}

#[sqlx::test]
#[ignore]
async fn history_preloads_the_full_tool_exchange_across_sessions(pool: PgPool) {
    let owner = owner(7, 3);
    let memory = DbMemory::load(pool.clone(), owner.clone(), vec![])
        .await
        .unwrap();
    memory
        .store_utterance(&session("s1"), "hello")
        .await
        .unwrap();
    memory.log_items(
        &session("s1"),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_a".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"hangzhou\"}".into(),
            }]),
            ChatItem::tool("call_a", "sunny 26 degrees"),
            ChatItem::assistant("sunny today"),
        ],
    );
    common::wait_for_count(&pool, "messages", 4).await;

    // A brand-new session (s2) of the same user + agent picks the rows up:
    // `load` has no session filter, `session_id` is provenance only.
    let reloaded = DbMemory::load(pool.clone(), owner, vec![]).await.unwrap();
    assert_eq!(
        reloaded.history().await,
        vec![
            ChatItem::user("hello"),
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_a".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"hangzhou\"}".into(),
            }]),
            ChatItem::tool("call_a", "sunny 26 degrees"),
            ChatItem::assistant("sunny today"),
        ]
    );
}

#[sqlx::test]
#[ignore]
async fn history_is_scoped_to_the_owner(pool: PgPool) {
    let alice = owner(7, 3);
    let bob = owner(8, 4);
    let alice_memory = DbMemory::load(pool.clone(), alice.clone(), vec![])
        .await
        .unwrap();
    let bob_memory = DbMemory::load(pool.clone(), bob.clone(), vec![])
        .await
        .unwrap();
    alice_memory
        .store_utterance(&session("s1"), "from alice")
        .await
        .unwrap();
    bob_memory
        .store_utterance(&session("s1"), "from bob")
        .await
        .unwrap();

    // The warm caches only see their own appends; a fresh load reads the
    // durable rows, where each owner sees exactly its own utterance.
    let reloaded_alice = DbMemory::load(pool.clone(), alice, vec![]).await.unwrap();
    assert_eq!(
        reloaded_alice.history().await,
        vec![ChatItem::user("from alice")]
    );
    let reloaded_bob = DbMemory::load(pool, bob, vec![]).await.unwrap();
    assert_eq!(
        reloaded_bob.history().await,
        vec![ChatItem::user("from bob")]
    );
}

#[sqlx::test]
#[ignore]
async fn history_caps_at_the_window_limit(pool: PgPool) {
    // Mirrors the agent's HISTORY_LIMIT (pub(crate) there).
    const HISTORY_LIMIT: usize = 20;

    let owner = owner(7, 3);
    let memory = DbMemory::load(pool.clone(), owner.clone(), vec![])
        .await
        .unwrap();
    for turn in 1..=(HISTORY_LIMIT + 5) {
        memory
            .store_utterance(&session("s1"), &format!("u{turn}"))
            .await
            .unwrap();
    }

    let reloaded = DbMemory::load(pool, owner, vec![]).await.unwrap();
    let history = reloaded.history().await;
    assert_eq!(history.len(), HISTORY_LIMIT);
    assert_eq!(history.first(), Some(&ChatItem::user("u6")));
    assert_eq!(
        history.last(),
        Some(&ChatItem::user(format!("u{}", HISTORY_LIMIT + 5)))
    );
}

#[sqlx::test]
#[ignore]
async fn loaded_history_keeps_appending_under_the_cap(pool: PgPool) {
    // Mirrors the agent's HISTORY_LIMIT (pub(crate) there).
    const HISTORY_LIMIT: usize = 20;

    let owner = owner(7, 3);
    let memory = DbMemory::load(pool, owner, vec![]).await.unwrap();
    for turn in 1..=25 {
        memory
            .append(vec![
                ChatItem::user(format!("u{turn}")),
                ChatItem::assistant(format!("a{turn}")),
            ])
            .await;
    }

    let history = memory.history().await;
    assert_eq!(history.len(), HISTORY_LIMIT);
    assert_eq!(history.first(), Some(&ChatItem::user("u16")));
    assert_eq!(history.last(), Some(&ChatItem::assistant("a25")));
}

#[sqlx::test]
#[ignore]
async fn utterance_row_is_written_and_announced(pool: PgPool) {
    let owner = owner(7, 3);
    let hook = std::sync::Arc::new(RecordingHook::default());
    let memory = DbMemory::load(
        pool.clone(),
        owner.clone(),
        vec![hook.clone() as std::sync::Arc<dyn MemoryHook>],
    )
    .await
    .unwrap();

    memory
        .store_utterance(&session("sess-1"), "hello there")
        .await
        .unwrap();

    common::wait_for_count(&pool, "messages", 1).await;
    let row = sqlx::query_as::<_, (String, Option<String>)>(
        "SELECT message->>'role', message->>'content' FROM messages",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0, "user");
    assert_eq!(row.1.as_deref(), Some("hello there"));

    let id: i64 = sqlx::query_scalar("SELECT id FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(hook.utterance_ids(), vec![id]);
    assert!(hook.turn_ids().is_empty());

    // An empty utterance writes nothing and announces nothing.
    memory
        .store_utterance(&session("sess-1"), "")
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(hook.utterance_ids().len(), 1);
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 1);
}

#[sqlx::test]
#[ignore]
async fn turn_messages_are_stored_one_row_per_message(pool: PgPool) {
    let owner = owner(7, 3);
    let hook = std::sync::Arc::new(RecordingHook::default());
    let memory = DbMemory::load(
        pool.clone(),
        owner,
        vec![hook.clone() as std::sync::Arc<dyn MemoryHook>],
    )
    .await
    .unwrap();

    memory.log_items(
        &session("sess-2"),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"hangzhou\"}".into(),
            }]),
            ChatItem::tool("call_x", "sunny 26 degrees"),
            ChatItem::assistant("🙂sunny today"),
        ],
    );

    common::wait_for_count(&pool, "messages", 3).await;
    let rows: Vec<serde_json::Value> =
        sqlx::query_scalar("SELECT message FROM messages ORDER BY id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows[0],
        serde_json::json!({
            "role": "assistant",
            "tool_calls": [{"id": "call_x", "name": "get_weather", "arguments": "{\"city\":\"hangzhou\"}"}]
        })
    );
    assert_eq!(
        rows[1],
        serde_json::json!({"role": "tool", "tool_call_id": "call_x", "content": "sunny 26 degrees"})
    );
    assert_eq!(
        rows[2],
        serde_json::json!({"role": "assistant", "content": "🙂sunny today"})
    );

    // The hook carries the last written row's id — the audio attach point.
    let id: i64 = sqlx::query_scalar("SELECT id FROM messages ORDER BY id DESC LIMIT 1")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(hook.turn_ids(), vec![id]);
}

#[sqlx::test]
#[ignore]
async fn multi_round_tool_exchange_stores_one_row_per_message(pool: PgPool) {
    let owner = owner(7, 3);
    let memory = DbMemory::load(pool.clone(), owner, vec![]).await.unwrap();
    memory.log_items(
        &session("sess-9"),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_a".into(),
                name: "lookup".into(),
                arguments: "{\"q\":\"a\"}".into(),
            }]),
            ChatItem::tool("call_a", "result a"),
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_b".into(),
                name: "lookup".into(),
                arguments: "{\"q\":\"b\"}".into(),
            }]),
            ChatItem::tool("call_b", "result b"),
            ChatItem::assistant("done"),
        ],
    );

    common::wait_for_count(&pool, "messages", 5).await;
    let rows: Vec<serde_json::Value> =
        sqlx::query_scalar("SELECT message FROM messages ORDER BY id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0]["role"], "assistant");
    assert_eq!(rows[0]["tool_calls"][0]["id"], "call_a");
    assert_eq!(rows[1]["role"], "tool");
    assert_eq!(rows[1]["tool_call_id"], "call_a");
    assert_eq!(rows[2]["tool_calls"][0]["id"], "call_b");
    assert_eq!(rows[3]["tool_call_id"], "call_b");
    assert_eq!(
        rows[4],
        serde_json::json!({"role": "assistant", "content": "done"})
    );
}

#[sqlx::test]
#[ignore]
async fn tool_only_turn_stores_rows_without_content(pool: PgPool) {
    let owner = owner(7, 3);
    let hook = std::sync::Arc::new(RecordingHook::default());
    let memory = DbMemory::load(
        pool.clone(),
        owner,
        vec![hook.clone() as std::sync::Arc<dyn MemoryHook>],
    )
    .await
    .unwrap();

    memory.log_items(
        &session("sess-10"),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_x", "result"),
        ],
    );

    common::wait_for_count(&pool, "messages", 2).await;
    let rows: Vec<serde_json::Value> =
        sqlx::query_scalar("SELECT message FROM messages ORDER BY id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(rows[0]["role"], "assistant");
    assert!(
        rows[0].get("content").is_none(),
        "a call-only round carries no content"
    );
    assert_eq!(
        rows[1],
        serde_json::json!({"role": "tool", "tool_call_id": "call_x", "content": "result"})
    );
    let id: i64 = sqlx::query_scalar("SELECT id FROM messages ORDER BY id DESC LIMIT 1")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(hook.turn_ids(), vec![id]);
}

#[sqlx::test]
#[ignore]
async fn turns_without_storable_content_write_nothing(pool: PgPool) {
    let owner = owner(7, 3);
    let hook = std::sync::Arc::new(RecordingHook::default());
    let memory = DbMemory::load(
        pool.clone(),
        owner,
        vec![hook.clone() as std::sync::Arc<dyn MemoryHook>],
    )
    .await
    .unwrap();

    memory.log_items(&session("sess-3"), vec![ChatItem::system("system prompt")]);
    memory.log_items(&session("sess-3"), vec![]);

    tokio::time::sleep(Duration::from_millis(100)).await;
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0);
    assert!(hook.turn_ids().is_empty());
}

#[sqlx::test]
#[ignore]
async fn partial_reply_row_is_written_and_announced(pool: PgPool) {
    let owner = owner(7, 3);
    let hook = std::sync::Arc::new(RecordingHook::default());
    let memory = DbMemory::load(
        pool.clone(),
        owner,
        vec![hook.clone() as std::sync::Arc<dyn MemoryHook>],
    )
    .await
    .unwrap();

    memory.store_partial_reply(&session("sess-4"), "partially spoke");
    memory.log_items(&session("sess-4"), vec![ChatItem::assistant("ok")]);
    common::wait_for_count(&pool, "messages", 2).await;

    // The two writes run in independent background tasks, so the commit
    // order — and with it the id order — is not fixed; assert the set.
    let mut rows = sqlx::query_as::<_, (String, Option<String>)>(
        "SELECT message->>'role', message->>'content' FROM messages",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    rows.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, "assistant");
    assert_eq!(rows[0].1.as_deref(), Some("ok"));
    assert_eq!(rows[1].0, "assistant");
    assert_eq!(rows[1].1.as_deref(), Some("partially spoke"));
    assert_eq!(hook.turn_ids().len(), 2);

    // An empty partial stores nothing.
    memory.store_partial_reply(&session("sess-4"), "");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 2);
}

// extractor

/// A one-shot LLM: pops one scripted round per `chat` call and records the
/// received history, so tests can assert on the prompts it was handed.
struct ProposeLlm {
    rounds: Mutex<VecDeque<Vec<Result<LlmEvent, LlmError>>>>,
    calls: Mutex<Vec<Vec<ChatItem>>>,
}

impl ProposeLlm {
    fn new(rounds: Vec<Vec<Result<LlmEvent, LlmError>>>) -> Self {
        Self {
            rounds: Mutex::new(rounds.into()),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    fn last_prompt(&self) -> Vec<ChatItem> {
        self.calls.lock().unwrap().last().unwrap().clone()
    }
}

impl Llm for ProposeLlm {
    fn chat(
        &self,
        history: Vec<ChatItem>,
        _tools: Vec<ToolSpec>,
        _tool_choice: Option<ToolChoice>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        self.calls.lock().unwrap().push(history);
        let round = self.rounds.lock().unwrap().pop_front().unwrap_or_default();
        Box::pin(futures_util::stream::iter(round))
    }
}

fn propose_call(arguments: &str) -> Result<LlmEvent, LlmError> {
    Ok(LlmEvent::ToolCall(ToolCall {
        id: "call_p".into(),
        name: "memory_propose".into(),
        arguments: arguments.into(),
    }))
}

#[sqlx::test]
#[ignore]
async fn entry_memory_applies_a_batch_in_one_transaction(pool: PgPool) {
    let memory = DbEntryMemory::new(pool.clone(), 7, 3);
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
            EntryOpResult::Skipped("unknown mem_09".into()),
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

    // Deleting frees a slot within the same batch, ids stay monotonic, and
    // another scope sees none of it.
    let results = memory
        .apply(vec![EntryOp::Delete(1), EntryOp::Add("third".into())])
        .await;
    assert_eq!(
        results,
        vec![EntryOpResult::Deleted(1), EntryOpResult::Added(3)]
    );
    let other = DbEntryMemory::new(pool, 8, 4);
    assert!(other.list().await.is_empty());
}

#[sqlx::test]
#[ignore]
async fn extractor_flushes_a_full_batch_into_entries(pool: PgPool) {
    // Mirrors the extractor's BATCH_ROWS.
    const BATCH_ROWS: usize = 12;

    let owner = owner(7, 3);
    let entries: Arc<dyn EntryMemory> = Arc::new(DbEntryMemory::new(pool.clone(), 7, 3));
    let llm = Arc::new(ProposeLlm::new(vec![vec![
        propose_call(
            r#"{"memories":[
                {"action":"add","content":"用户每天用 Go 写后端"},
                {"action":"add","content":"用户在杭州工作"}]}"#,
        ),
        Ok(LlmEvent::Done),
    ]]));
    let extractor = Arc::new(Extractor::new(
        pool.clone(),
        owner.clone(),
        llm.clone(),
        entries.clone(),
    ));
    let memory = DbMemory::load(
        pool.clone(),
        owner,
        vec![extractor.clone() as Arc<dyn MemoryHook>],
    )
    .await
    .unwrap();

    for turn in 1..=(BATCH_ROWS - 1) {
        memory
            .store_utterance(&session("s1"), &format!("turn {turn}"))
            .await
            .unwrap();
    }
    // One tool turn: its tool rows are plumbing — only the final reply row
    // is announced to the hook and reaches the dialogue.
    memory.log_items(
        &session("s1"),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_w".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_w", "sunny 26 degrees"),
            ChatItem::assistant("今天晴"),
        ],
    );

    // The 12th announced row triggers the flush; wait for the entries.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while entries.list().await.len() < 2 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "extraction did not run"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let listed = entries.list().await;
    assert_eq!(listed[0].content, "用户每天用 Go 写后端");
    assert_eq!(listed[1].content, "用户在杭州工作");

    // The dialogue held the utterances and the spoken reply, not the tool
    // exchange.
    let prompt = llm.last_prompt();
    assert_eq!(prompt.len(), 2);
    assert!(matches!(&prompt[0], ChatItem::System { .. }));
    let ChatItem::User { content } = &prompt[1] else {
        panic!("expected the dialogue batch")
    };
    assert!(content.contains("用户：turn 1\n"));
    assert!(content.contains("用户：turn 11"));
    assert!(content.contains("助手：今天晴"));
    assert!(!content.contains("sunny 26 degrees"));
}

#[sqlx::test]
#[ignore]
async fn a_failed_flush_stays_pending_and_retries_on_the_next_batch(pool: PgPool) {
    const BATCH_ROWS: usize = 12;

    let owner = owner(7, 3);
    let entries: Arc<dyn EntryMemory> = Arc::new(DbEntryMemory::new(pool.clone(), 7, 3));
    let llm = Arc::new(ProposeLlm::new(vec![
        vec![Err(LlmError::Failed("provider down".into()))],
        vec![
            propose_call(r#"{"memories":[{"action":"add","content":"用户喜欢深夜编码"}]}"#),
            Ok(LlmEvent::Done),
        ],
    ]));
    let extractor = Arc::new(Extractor::new(
        pool.clone(),
        owner.clone(),
        llm.clone(),
        entries.clone(),
    ));
    let memory = DbMemory::load(pool.clone(), owner, vec![extractor as Arc<dyn MemoryHook>])
        .await
        .unwrap();

    for turn in 1..=BATCH_ROWS {
        memory
            .store_utterance(&session("s1"), &format!("turn {turn}"))
            .await
            .unwrap();
    }
    // Round 1 fails; wait for it to have happened, nothing is recorded.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while llm.call_count() < 1 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the first flush never ran"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(entries.list().await.is_empty());

    // The failed rows stayed pending: one more utterance completes a new
    // full batch over all 13 rows and the retry lands the entry.
    memory
        .store_utterance(&session("s1"), "turn 13")
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while entries.list().await.is_empty() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the retry never ran"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(llm.call_count(), 2);
    let listed = entries.list().await;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].content, "用户喜欢深夜编码");
    // The retry batch covered every row, including the failed round's.
    let ChatItem::User { content } = &llm.last_prompt()[1] else {
        panic!("expected the dialogue batch")
    };
    assert!(content.contains("用户：turn 1\n"));
    assert!(content.contains("用户：turn 13"));
}

#[sqlx::test]
#[ignore]
async fn teardown_flushes_the_remaining_rows(pool: PgPool) {
    let owner = owner(7, 3);
    let entries: Arc<dyn EntryMemory> = Arc::new(DbEntryMemory::new(pool.clone(), 7, 3));
    let llm = Arc::new(ProposeLlm::new(vec![vec![
        propose_call(r#"{"memories":[{"action":"add","content":"用户养了一只猫"}]}"#),
        Ok(LlmEvent::Done),
    ]]));
    let extractor = Arc::new(Extractor::new(
        pool.clone(),
        owner.clone(),
        llm.clone(),
        entries.clone(),
    ));
    {
        let memory = DbMemory::load(
            pool.clone(),
            owner,
            vec![extractor.clone() as Arc<dyn MemoryHook>],
        )
        .await
        .unwrap();
        for turn in 1..=3 {
            memory
                .store_utterance(&session("s1"), &format!("turn {turn}"))
                .await
                .unwrap();
        }
    }

    // Below the batch threshold nothing ran while connected.
    assert!(entries.list().await.is_empty());
    assert_eq!(llm.call_count(), 0);

    // The connection is gone: the teardown flush runs the remainder.
    drop(extractor);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while entries.list().await.is_empty() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the teardown flush never ran"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(entries.list().await[0].content, "用户养了一只猫");
}
