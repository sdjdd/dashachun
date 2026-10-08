use std::time::Duration;

mod common;

use sqlx::PgPool;

use common::{RecordingHook, owner, session};
use dashachun::agent::{ChatItem, ToolCall};
use dashachun::agent::{DbMemory, Memory, MemoryHook};

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
    common::wait_for_count(&pool, "messages", 2).await;

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
    let row = sqlx::query_as::<_, (String, Option<String>)>("SELECT role, content FROM messages")
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
async fn turn_row_persists_the_tool_exchange_and_is_announced(pool: PgPool) {
    let owner = owner(7, 3);
    let hook = std::sync::Arc::new(RecordingHook::default());
    let memory = DbMemory::load(
        pool.clone(),
        owner.clone(),
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

    common::wait_for_count(&pool, "messages", 1).await;
    let row = sqlx::query_as::<_, (String, Option<String>, serde_json::Value)>(
        "SELECT role, content, parts FROM messages",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0, "assistant");
    assert_eq!(row.1.as_deref(), Some("🙂sunny today"));
    let parts = row.2.as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "tool_call");
    assert_eq!(parts[0]["id"], "call_x");
    assert_eq!(parts[0]["name"], "get_weather");
    assert_eq!(parts[0]["arguments"], "{\"city\":\"hangzhou\"}");
    assert_eq!(parts[1]["type"], "tool_result");
    assert_eq!(parts[1]["tool_call_id"], "call_x");
    assert_eq!(parts[1]["content"], "sunny 26 degrees");

    let id: i64 = sqlx::query_scalar("SELECT id FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(hook.turn_ids(), vec![id]);
}

#[sqlx::test]
#[ignore]
async fn multi_round_tool_exchange_flattens_into_parts_in_order(pool: PgPool) {
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

    common::wait_for_count(&pool, "messages", 1).await;
    let parts: serde_json::Value = sqlx::query_scalar("SELECT parts FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    let parts = parts.as_array().unwrap();
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0]["type"], "tool_call");
    assert_eq!(parts[0]["id"], "call_a");
    assert_eq!(parts[1]["type"], "tool_result");
    assert_eq!(parts[1]["tool_call_id"], "call_a");
    assert_eq!(parts[2]["type"], "tool_call");
    assert_eq!(parts[2]["id"], "call_b");
    assert_eq!(parts[3]["type"], "tool_result");
    assert_eq!(parts[3]["tool_call_id"], "call_b");
}

#[sqlx::test]
#[ignore]
async fn tool_only_turn_stores_null_content_and_is_announced(pool: PgPool) {
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

    common::wait_for_count(&pool, "messages", 1).await;
    let row = sqlx::query_as::<_, (Option<String>, serde_json::Value)>(
        "SELECT content, parts FROM messages",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0, None);
    assert_eq!(row.1.as_array().unwrap().len(), 2);
    let id: i64 = sqlx::query_scalar("SELECT id FROM messages")
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
    let mut rows =
        sqlx::query_as::<_, (String, Option<String>)>("SELECT role, content FROM messages")
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
