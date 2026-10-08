use std::sync::Arc;
use std::time::Duration;

use sqlx::PgPool;
use uuid::Uuid;

use dashachun::agent::{
    AgentSession, AudioCapture, Capture, DbMemory, Memory, MemoryHook, MemoryOwner,
};
use dashachun::llm::{ChatItem, ToolCall};

fn session(id: &str) -> AgentSession {
    AgentSession {
        id: id.into(),
        sample_rate: 16000,
        channels: 1,
        frame_duration_ms: 60,
    }
}

fn owner() -> MemoryOwner {
    MemoryOwner {
        user_id: 7,
        agent_id: 3,
        client_id: Uuid::new_v4(),
        device_id: Some("aa:bb:cc:dd:ee:ff".into()),
    }
}

/// A memory wired with the audio capture as its hook — the production
/// assembly under test.
async fn wired_memory(pool: &PgPool) -> (DbMemory, Arc<AudioCapture>) {
    let audio = Arc::new(AudioCapture::new(pool.clone()));
    let memory = DbMemory::load(
        pool.clone(),
        owner(),
        vec![audio.clone() as Arc<dyn MemoryHook>],
    )
    .await
    .unwrap();
    (memory, audio)
}

async fn wait_for_audio_count(pool: &PgPool, count: i64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let current: i64 = sqlx::query_scalar("SELECT count(*) FROM message_audios")
            .fetch_one(pool)
            .await
            .unwrap();
        if current == count {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "expected {count} audios, got {current}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[sqlx::test]
#[ignore]
async fn utterance_audio_attaches_to_the_user_row_through_the_hook(pool: PgPool) {
    let (memory, audio) = wired_memory(&pool).await;

    let recording = audio.start_utterance(&session("sess-1"));
    recording.push(&[0.5; 960]);
    memory
        .store_utterance(&session("sess-1"), "hello")
        .await
        .unwrap();
    // The hook fired inside store_utterance, so by the time the capture is
    // finished its message id is already in hand.
    recording.finish();

    wait_for_audio_count(&pool, 1).await;
    let (message_id, sample_rate, channels): (i64, i32, i32) =
        sqlx::query_as("SELECT message_id, sample_rate, channels FROM message_audios")
            .fetch_one(&pool)
            .await
            .unwrap();
    let (row_id, role): (i64, String) = sqlx::query_as("SELECT id, role FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(role, "user");
    assert_eq!(message_id, row_id);
    assert_eq!(sample_rate, 16000);
    assert_eq!(channels, 1);
}

#[sqlx::test]
#[ignore]
async fn reply_audio_attaches_to_the_turn_row_through_the_hook(pool: PgPool) {
    let (memory, audio) = wired_memory(&pool).await;

    let recording = audio.start_reply(&session("sess-2"));
    recording.push(&[0.25; 960]);
    // The turn row is written in the background: finishing before the commit
    // lands exercises the capture waiting for the hook.
    memory.log_items(
        &session("sess-2"),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_x", "sunny"),
            ChatItem::assistant("sunny today"),
        ],
    );
    recording.finish();

    wait_for_audio_count(&pool, 1).await;
    let message_id: i64 = sqlx::query_scalar("SELECT message_id FROM message_audios")
        .fetch_one(&pool)
        .await
        .unwrap();
    let (row_id, role): (i64, String) = sqlx::query_as("SELECT id, role FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(role, "assistant");
    assert_eq!(message_id, row_id);
}

#[sqlx::test]
#[ignore]
async fn partial_reply_audio_attaches_to_the_truncated_row(pool: PgPool) {
    let (memory, audio) = wired_memory(&pool).await;

    let recording = audio.start_reply(&session("sess-3"));
    recording.push(&[0.25; 960]);
    memory.store_partial_reply(&session("sess-3"), "partially spoke");
    recording.finish();

    wait_for_audio_count(&pool, 1).await;
    let message_id: i64 = sqlx::query_scalar("SELECT message_id FROM message_audios")
        .fetch_one(&pool)
        .await
        .unwrap();
    let (row_id, content): (i64, Option<String>) =
        sqlx::query_as("SELECT id, content FROM messages")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(content.as_deref(), Some("partially spoke"));
    assert_eq!(message_id, row_id);
}

#[sqlx::test]
#[ignore]
async fn dropped_captures_store_no_audio(pool: PgPool) {
    let (memory, audio) = wired_memory(&pool).await;

    let utterance = audio.start_utterance(&session("sess-4"));
    utterance.push(&[0.5; 960]);
    drop(utterance);
    memory
        .store_utterance(&session("sess-4"), "hello")
        .await
        .unwrap();

    let reply = audio.start_reply(&session("sess-4"));
    reply.push(&[0.25; 960]);
    drop(reply);
    memory.log_items(&session("sess-4"), vec![ChatItem::assistant("ok")]);

    tokio::time::sleep(Duration::from_millis(200)).await;
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM message_audios")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0);
    // The text rows exist regardless — audio is the byproduct.
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 2);
}

#[sqlx::test]
#[ignore]
async fn utterance_without_text_stores_no_row_and_no_audio(pool: PgPool) {
    let (memory, audio) = wired_memory(&pool).await;

    let recording = audio.start_utterance(&session("sess-5"));
    recording.push(&[0.5; 960]);
    memory
        .store_utterance(&session("sess-5"), "")
        .await
        .unwrap();
    // No row, no hook, so finishing the capture stores nothing either.
    recording.finish();

    tokio::time::sleep(Duration::from_millis(200)).await;
    let messages: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(messages, 0);
    let audios: i64 = sqlx::query_scalar("SELECT count(*) FROM message_audios")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(audios, 0);
}
