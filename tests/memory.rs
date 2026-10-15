use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

mod common;

use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

use axum::http::StatusCode;

use common::{IdSink, RecordingHook, app, expect_json, register, request, send, session};
use dashachun::agent::{
    AddError, DbEntryMemory, DbMemory, EntryMemory, EntryOp, EntryOpResult, Extractor, Llm,
    LlmError, LlmEvent, LlmEvents, MAX_ENTRIES, Memory, MemoryEntry, MemoryHook, ReasoningEffort,
    ToolChoice, ToolSpec,
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
async fn a_new_session_starts_with_an_empty_window(pool: PgPool) {
    let memory = DbMemory::new(pool.clone(), vec![]);
    memory
        .store_utterance(&session(1), "hello", None)
        .await
        .unwrap();
    memory.log_items(
        &session(1),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_a".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"hangzhou\"}".into(),
            }]),
            ChatItem::tool("call_a", "sunny 26 degrees"),
            ChatItem::assistant("sunny today"),
        ],
        None,
    );
    common::wait_for_count(&pool, "messages", 4).await;

    // The rows are durable, but the context lives only in the session's
    // window: a brand-new session starts empty and never reloads them.
    let fresh = DbMemory::new(pool, vec![]);
    assert!(fresh.history().await.is_empty());
}

#[sqlx::test]
#[ignore]
async fn the_window_caps_at_the_turns(pool: PgPool) {
    // Mirrors the agent's HISTORY_TURNS (pub(crate) there).
    const HISTORY_TURNS: usize = 10;

    let memory = DbMemory::new(pool, vec![]);
    for turn in 1..=(HISTORY_TURNS + 5) {
        memory
            .append(&session(1), vec![ChatItem::user(format!("u{turn}"))])
            .await;
        memory
            .append(&session(1), vec![ChatItem::assistant(format!("a{turn}"))])
            .await;
    }

    let history = memory.history().await;
    assert_eq!(history.len(), HISTORY_TURNS * 2);
    assert_eq!(history.first(), Some(&ChatItem::user("u6")));
    assert_eq!(
        history.last(),
        Some(&ChatItem::assistant(format!("a{}", HISTORY_TURNS + 5)))
    );
}

#[sqlx::test]
#[ignore]
async fn appended_batches_are_announced_to_the_hooks(pool: PgPool) {
    let hook = Arc::new(RecordingHook::default());
    let memory = DbMemory::new(pool, vec![hook.clone() as Arc<dyn MemoryHook>]);

    memory
        .append(&session(1), vec![ChatItem::user("hello")])
        .await;
    memory
        .append(&session(1), vec![ChatItem::assistant("hi there")])
        .await;
    // An empty batch mutates nothing and announces nothing.
    memory.append(&session(1), vec![]).await;

    assert_eq!(
        hook.appends(),
        vec![
            vec![ChatItem::user("hello")],
            vec![ChatItem::assistant("hi there")],
        ]
    );
}

#[sqlx::test]
#[ignore]
async fn utterance_row_is_written_and_announced(pool: PgPool) {
    let memory = DbMemory::new(pool.clone(), vec![]);

    let message_id = memory
        .store_utterance(&session(11), "hello there", None)
        .await
        .unwrap()
        .expect("a non-empty utterance stores a row");

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
    assert_eq!(message_id, id);

    // An empty utterance stores nothing and returns no id.
    let empty = memory
        .store_utterance(&session(11), "", None)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(empty, None);
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 1);
}

#[sqlx::test]
#[ignore]
async fn the_utterance_row_records_its_speaker(pool: PgPool) {
    let memory = DbMemory::new(pool.clone(), vec![]);

    memory
        .store_utterance(&session(1), "identified", Some(7))
        .await
        .unwrap();
    memory
        .store_utterance(&session(1), "unidentified", None)
        .await
        .unwrap();
    common::wait_for_count(&pool, "messages", 2).await;

    // The speaker rides the row: the matched speakers id on identified
    // utterances, NULL — not a sentinel — when nothing was identified.
    // Turn rows never carry one; only store_utterance writes a speaker.
    let speakers: Vec<(String, Option<i64>)> =
        sqlx::query_as("SELECT message->>'content', speaker_id FROM messages ORDER BY id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(
        speakers,
        vec![
            ("identified".to_owned(), Some(7)),
            ("unidentified".to_owned(), None),
        ]
    );

    memory.log_items(&session(1), vec![ChatItem::assistant("hi")], None);
    common::wait_for_count(&pool, "messages", 3).await;
    let assistant_speaker: Option<i64> =
        sqlx::query_scalar("SELECT speaker_id FROM messages ORDER BY id DESC LIMIT 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(assistant_speaker, None);
}

#[sqlx::test]
#[ignore]
async fn turn_messages_are_stored_one_row_per_message(pool: PgPool) {
    let ids = Arc::new(IdSink::default());
    let memory = DbMemory::new(pool.clone(), vec![]);

    memory.log_items(
        &session(2),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"hangzhou\"}".into(),
            }]),
            ChatItem::tool("call_x", "sunny 26 degrees"),
            ChatItem::assistant("🙂sunny today"),
        ],
        Some(ids.sink()),
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

    // The sink carries the last written row's id — the audio attach point.
    let id: i64 = sqlx::query_scalar("SELECT id FROM messages ORDER BY id DESC LIMIT 1")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(ids.ids(), vec![id]);
}

#[sqlx::test]
#[ignore]
async fn multi_round_tool_exchange_stores_one_row_per_message(pool: PgPool) {
    let memory = DbMemory::new(pool.clone(), vec![]);
    memory.log_items(
        &session(9),
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
        None,
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
    let ids = Arc::new(IdSink::default());
    let memory = DbMemory::new(pool.clone(), vec![]);

    memory.log_items(
        &session(10),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_x", "result"),
        ],
        Some(ids.sink()),
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
    assert_eq!(ids.ids(), vec![id]);
}

#[sqlx::test]
#[ignore]
async fn turns_without_storable_content_write_nothing(pool: PgPool) {
    let ids = Arc::new(IdSink::default());
    let memory = DbMemory::new(pool.clone(), vec![]);

    memory.log_items(
        &session(3),
        vec![ChatItem::system("system prompt")],
        Some(ids.sink()),
    );
    memory.log_items(&session(3), vec![], Some(ids.sink()));

    tokio::time::sleep(Duration::from_millis(100)).await;
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0);
    assert!(ids.ids().is_empty());
}

#[sqlx::test]
#[ignore]
async fn partial_reply_row_is_written_and_announced(pool: PgPool) {
    let ids = Arc::new(IdSink::default());
    let memory = DbMemory::new(pool.clone(), vec![]);

    memory.store_partial_reply(&session(4), "partially spoke", Some(ids.sink()));
    memory.log_items(
        &session(4),
        vec![ChatItem::assistant("ok")],
        Some(ids.sink()),
    );
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

    // Each write handed its own row's id to the sink.
    let mut stored: Vec<i64> = sqlx::query_scalar("SELECT id FROM messages ORDER BY id")
        .fetch_all(&pool)
        .await
        .unwrap();
    stored.sort();
    let mut announced = ids.ids();
    announced.sort();
    assert_eq!(announced, stored);

    // An empty partial stores nothing.
    memory.store_partial_reply(&session(4), "", None);
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
    efforts: Mutex<Vec<Option<ReasoningEffort>>>,
}

impl ProposeLlm {
    fn new(rounds: Vec<Vec<Result<LlmEvent, LlmError>>>) -> Self {
        Self {
            rounds: Mutex::new(rounds.into()),
            calls: Mutex::new(Vec::new()),
            efforts: Mutex::new(Vec::new()),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    fn last_prompt(&self) -> Vec<ChatItem> {
        self.calls.lock().unwrap().last().unwrap().clone()
    }

    fn efforts(&self) -> Vec<Option<ReasoningEffort>> {
        self.efforts.lock().unwrap().clone()
    }
}

impl Llm for ProposeLlm {
    fn chat(
        &self,
        history: Vec<ChatItem>,
        _tools: Vec<ToolSpec>,
        _tool_choice: Option<ToolChoice>,
        reasoning: Option<ReasoningEffort>,
        _cancel: CancellationToken,
    ) -> LlmEvents<'_> {
        self.calls.lock().unwrap().push(history);
        self.efforts.lock().unwrap().push(reasoning);
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
    // Mirrors the extractor's BATCH_ITEMS.
    const BATCH_ITEMS: usize = 12;

    let entries: Arc<dyn EntryMemory> = Arc::new(DbEntryMemory::new(pool.clone(), 7, 3));
    let llm = Arc::new(ProposeLlm::new(vec![vec![
        propose_call(
            r#"{"memories":[
                {"action":"add","content":"用户每天用 Go 写后端"},
                {"action":"add","content":"用户在杭州工作"}]}"#,
        ),
        Ok(LlmEvent::Done),
    ]]));
    let extractor = Arc::new(Extractor::new(llm.clone(), entries.clone(), vec![]));
    let memory = DbMemory::new(pool.clone(), vec![extractor.clone() as Arc<dyn MemoryHook>]);

    for turn in 1..=(BATCH_ITEMS - 1) {
        memory
            .append(&session(1), vec![ChatItem::user(format!("turn {turn}"))])
            .await;
    }
    // One tool turn: its exchange is plumbing — the utterance and the
    // spoken replies are what reach the dialogue.
    memory
        .append(
            &session(1),
            vec![
                ChatItem::assistant_tool_calls(vec![ToolCall {
                    id: "call_w".into(),
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                }]),
                ChatItem::tool("call_w", "sunny 26 degrees"),
                ChatItem::assistant("今天晴"),
            ],
        )
        .await;

    // The 12th buffered item triggers the flush; wait for the entries.
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
    // The flush runs at low effort — a background batch nobody waits on.
    assert_eq!(llm.efforts(), vec![Some(ReasoningEffort::Low)]);

    // The dialogue is the agent's message list: the utterances and the
    // spoken reply, never the tool exchange.
    let prompt = llm.last_prompt();
    assert_eq!(prompt.len(), BATCH_ITEMS + 1);
    assert!(
        matches!(&prompt[0], ChatItem::System { content } if content.contains("本批对话发生在"))
    );
    assert_eq!(prompt[1], ChatItem::user("turn 1"));
    assert_eq!(prompt[11], ChatItem::user("turn 11"));
    assert_eq!(prompt[12], ChatItem::assistant("今天晴"));
    assert!(
        !prompt
            .iter()
            .any(|item| matches!(item, ChatItem::Tool { .. }))
    );
}

#[sqlx::test]
#[ignore]
async fn a_failed_flush_stays_pending_and_retries_on_the_next_batch(pool: PgPool) {
    const BATCH_ITEMS: usize = 12;

    let entries: Arc<dyn EntryMemory> = Arc::new(DbEntryMemory::new(pool.clone(), 7, 3));
    let llm = Arc::new(ProposeLlm::new(vec![
        vec![Err(LlmError::Failed("provider down".into()))],
        vec![
            propose_call(r#"{"memories":[{"action":"add","content":"用户喜欢深夜编码"}]}"#),
            Ok(LlmEvent::Done),
        ],
    ]));
    let extractor = Arc::new(Extractor::new(llm.clone(), entries.clone(), vec![]));
    let memory = DbMemory::new(pool.clone(), vec![extractor as Arc<dyn MemoryHook>]);

    for turn in 1..=BATCH_ITEMS {
        memory
            .append(&session(1), vec![ChatItem::user(format!("turn {turn}"))])
            .await;
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

    // The failed items stayed pending: one more item completes a new full
    // batch over all 13 and the retry lands the entry.
    memory
        .append(&session(1), vec![ChatItem::user("turn 13")])
        .await;
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
    // The retry batch covered every item, including the failed round's.
    let prompt = llm.last_prompt();
    assert_eq!(prompt[1], ChatItem::user("turn 1"));
    assert_eq!(prompt[13], ChatItem::user("turn 13"));
}

#[sqlx::test]
#[ignore]
async fn teardown_flushes_the_remaining_items(pool: PgPool) {
    let entries: Arc<dyn EntryMemory> = Arc::new(DbEntryMemory::new(pool.clone(), 7, 3));
    let llm = Arc::new(ProposeLlm::new(vec![vec![
        propose_call(r#"{"memories":[{"action":"add","content":"用户养了一只猫"}]}"#),
        Ok(LlmEvent::Done),
    ]]));
    let extractor = Arc::new(Extractor::new(llm.clone(), entries.clone(), vec![]));
    {
        let memory = DbMemory::new(pool.clone(), vec![extractor.clone() as Arc<dyn MemoryHook>]);
        for turn in 1..=3 {
            memory
                .append(&session(1), vec![ChatItem::user(format!("turn {turn}"))])
                .await;
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

#[sqlx::test]
#[ignore]
async fn memories_rest_lists_and_deletes_within_the_agent_scope(pool: PgPool) {
    let app = app(pool.clone());
    let (cookie, user_id) = register(&app, &pool).await;
    let created = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    let agent = created["id"].as_i64().unwrap();

    let memory = DbEntryMemory::new(pool.clone(), user_id, agent);
    memory.add("第一条").await.unwrap();
    memory.add("第二条").await.unwrap();

    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/memories"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    let items = body.as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["mem_no"], 1);
    assert_eq!(items[0]["content"], "第一条");
    assert_eq!(items[1]["mem_no"], 2);
    assert_eq!(items[1]["content"], "第二条");

    let (other, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/memories"),
            None,
            Some(&other),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{agent}/memories/2"),
            None,
            Some(&other),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{agent}/memories/1"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let body = expect_json(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/memories"),
            None,
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    let items = body.as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["mem_no"], 2);

    let res = send(
        &app,
        request(
            "DELETE",
            &format!("/api/agents/{agent}/memories/1"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let third = memory.add("第三条").await.unwrap();
    assert_eq!(third.mem_no, 3);
}
