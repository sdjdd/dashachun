mod common;

use std::sync::Arc;
use std::time::Duration;

use sqlx::PgPool;

use dashachun::agent::{AudioCapture, AudioStorage, Capture, DbMemory, Memory, MemoryHook};
use dashachun::agent::{ChatItem, ToolCall};

use common::MemoryStorage;

/// A memory wired with the audio capture as its hook — the production
/// assembly under test. The client id is generated here and handed to the
/// capture exactly as `AgentFactory::build` does; the storage doubles as
/// the port, and `None` disables storage entirely.
async fn wired_memory(
    pool: &PgPool,
    audio_storage: Option<Arc<dyn AudioStorage>>,
) -> (DbMemory, Arc<AudioCapture>, uuid::Uuid) {
    let client_id = uuid::Uuid::new_v4();
    let audio = Arc::new(AudioCapture::new(pool.clone(), client_id, audio_storage));
    let memory = DbMemory::new(pool.clone(), vec![audio.clone() as Arc<dyn MemoryHook>]);
    (memory, audio, client_id)
}

/// The capture key's day prefix, `audios/{YYYYMMDD}/`.
fn date_prefix() -> String {
    let now = time::OffsetDateTime::now_utc();
    format!(
        "audios/{:04}{:02}{:02}/",
        now.year(),
        u8::from(now.month()),
        now.day()
    )
}

#[sqlx::test]
#[ignore]
async fn utterance_audio_attaches_to_the_user_row_through_the_hook(pool: PgPool) {
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let recording = audio.start_utterance(&common::session("sess-1"));
    recording.push(&[0.5; 960]);
    memory
        .store_utterance(&common::session("sess-1"), "hello")
        .await
        .unwrap();
    // The hook fired inside store_utterance, so by the time the capture is
    // finished its message id is already in hand.
    recording.finish();

    common::wait_for_count(&pool, "message_audios", 1).await;
    let (message_id, bucket, object_key, sample_rate, channels): (i64, String, String, i32, i32) =
        sqlx::query_as(
            "SELECT message_id, bucket, object_key, sample_rate, channels FROM message_audios",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
    let (row_id, role): (i64, String) = sqlx::query_as("SELECT id, message->>'role' FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(role, "user");
    assert_eq!(message_id, row_id);
    assert_eq!(sample_rate, 16000);
    assert_eq!(channels, 1);
    // The FLAC bytes live in the storage under the row's object key.
    assert_eq!(audio_storage.keys(), vec![object_key.clone()]);
    assert_eq!(bucket, audio_storage.bucket());
    assert!(object_key.starts_with(&date_prefix()));
    assert!(object_key.contains(&client_id.to_string()));
    assert!(object_key.ends_with(".flac"));
    assert!(audio_storage.get(&object_key).unwrap().starts_with(b"fLaC"));
}

#[sqlx::test]
#[ignore]
async fn reply_audio_attaches_to_the_final_reply_row_through_the_hook(pool: PgPool) {
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, _client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let recording = audio.start_reply(&common::session("sess-2"));
    recording.push(&[0.25; 960]);
    // The turn's rows are written in the background: finishing before the
    // commit lands exercises the capture waiting for the hook.
    memory.log_items(
        &common::session("sess-2"),
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

    common::wait_for_count(&pool, "message_audios", 1).await;
    let (message_id, bucket, object_key): (i64, String, String) =
        sqlx::query_as("SELECT message_id, bucket, object_key FROM message_audios")
            .fetch_one(&pool)
            .await
            .unwrap();
    let (row_id, role, content): (i64, String, Option<String>) = sqlx::query_as(
        "SELECT id, message->>'role', message->>'content' FROM messages ORDER BY id DESC LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(role, "assistant");
    assert_eq!(content.as_deref(), Some("sunny today"));
    assert_eq!(message_id, row_id);
    assert_eq!(bucket, audio_storage.bucket());
    assert_eq!(audio_storage.keys(), vec![object_key]);
}

#[sqlx::test]
#[ignore]
async fn partial_reply_audio_attaches_to_the_truncated_row(pool: PgPool) {
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, _client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let recording = audio.start_reply(&common::session("sess-3"));
    recording.push(&[0.25; 960]);
    memory.store_partial_reply(&common::session("sess-3"), "partially spoke");
    recording.finish();

    common::wait_for_count(&pool, "message_audios", 1).await;
    let (message_id, bucket, object_key): (i64, String, String) =
        sqlx::query_as("SELECT message_id, bucket, object_key FROM message_audios")
            .fetch_one(&pool)
            .await
            .unwrap();
    let (row_id, content): (i64, Option<String>) =
        sqlx::query_as("SELECT id, message->>'content' FROM messages")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(content.as_deref(), Some("partially spoke"));
    assert_eq!(message_id, row_id);
    assert_eq!(bucket, audio_storage.bucket());
    assert_eq!(audio_storage.keys(), vec![object_key]);
}

#[sqlx::test]
#[ignore]
async fn dropped_captures_store_no_audio(pool: PgPool) {
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, _client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let utterance = audio.start_utterance(&common::session("sess-4"));
    utterance.push(&[0.5; 960]);
    drop(utterance);
    memory
        .store_utterance(&common::session("sess-4"), "hello")
        .await
        .unwrap();

    let reply = audio.start_reply(&common::session("sess-4"));
    reply.push(&[0.25; 960]);
    drop(reply);
    memory.log_items(&common::session("sess-4"), vec![ChatItem::assistant("ok")]);

    tokio::time::sleep(Duration::from_millis(200)).await;
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM message_audios")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0);
    assert!(audio_storage.is_empty());
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
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, _client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let recording = audio.start_utterance(&common::session("sess-5"));
    recording.push(&[0.5; 960]);
    memory
        .store_utterance(&common::session("sess-5"), "")
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
    assert!(audio_storage.is_empty());
}

#[sqlx::test]
#[ignore]
async fn no_sink_stores_no_audio_rows(pool: PgPool) {
    let (memory, audio, _client_id) = wired_memory(&pool, None).await;

    let recording = audio.start_utterance(&common::session("sess-6"));
    recording.push(&[0.5; 960]);
    memory
        .store_utterance(&common::session("sess-6"), "hello")
        .await
        .unwrap();
    recording.finish();

    // The message row is the primary record; the unconfigured storage
    // leaves no audio rows behind — no BYTEA fallback.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let messages: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(messages, 1);
    let audios: i64 = sqlx::query_scalar("SELECT count(*) FROM message_audios")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(audios, 0);
}
