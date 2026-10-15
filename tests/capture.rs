mod common;

use std::sync::Arc;
use std::time::Duration;

use sqlx::PgPool;

use dashachun::agent::{AudioCapture, AudioStorage, Capture, DbMemory, Memory};
use dashachun::agent::{ChatItem, ToolCall};

use common::MemoryStorage;

/// A memory and the audio capture, built exactly as `AgentFactory::build`
/// wires them: the capture is the driver's `Capture`, the memory carries no
/// hook — each capture binds its row at the store call site. The client id
/// is generated here and handed to the capture; the storage doubles as the
/// port, and `None` disables storage entirely.
async fn wired_memory(
    pool: &PgPool,
    audio_storage: Option<Arc<dyn AudioStorage>>,
) -> (DbMemory, Arc<AudioCapture>, uuid::Uuid) {
    let client_id = uuid::Uuid::new_v4();
    let audio = Arc::new(AudioCapture::new(pool.clone(), client_id, audio_storage));
    let memory = DbMemory::new(pool.clone(), vec![]);
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
async fn utterance_audio_attaches_to_the_user_row(pool: PgPool) {
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let recording = audio.start_utterance(&common::session(1));
    recording.push(&[0.5; 960]);
    let message_id = memory
        .store_utterance(&common::session(1), "hello", None)
        .await
        .unwrap()
        .unwrap();
    // The id this store call returned is the capture's own row — attaching
    // before or after `finish` works; production attaches first.
    recording.attach(message_id);
    recording.finish();

    common::wait_for_count(&pool, "message_audios", 1).await;
    let (stored_id, bucket, object_key, sample_rate, channels): (i64, String, String, i32, i32) =
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
    assert_eq!(stored_id, row_id);
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
async fn reply_audio_attaches_to_the_final_reply_row(pool: PgPool) {
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, _client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let recording = audio.start_reply(&common::session(2));
    recording.push(&[0.25; 960]);
    // The turn's rows are written in the background: the sink rides the
    // store call, so finishing before the commit lands still binds the id.
    memory.log_items(
        &common::session(2),
        vec![
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_x", "sunny"),
            ChatItem::assistant("sunny today"),
        ],
        Some(recording.attach_sink()),
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

    let recording = audio.start_reply(&common::session(3));
    recording.push(&[0.25; 960]);
    memory.store_partial_reply(
        &common::session(3),
        "partially spoke",
        Some(recording.attach_sink()),
    );
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

    let utterance = audio.start_utterance(&common::session(4));
    utterance.push(&[0.5; 960]);
    drop(utterance);
    memory
        .store_utterance(&common::session(4), "hello", None)
        .await
        .unwrap();

    let reply = audio.start_reply(&common::session(4));
    reply.push(&[0.25; 960]);
    drop(reply);
    memory.log_items(&common::session(4), vec![ChatItem::assistant("ok")], None);

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

    let recording = audio.start_utterance(&common::session(5));
    recording.push(&[0.5; 960]);
    let message_id = memory
        .store_utterance(&common::session(5), "", None)
        .await
        .unwrap();
    // No row, no id — there is nothing to attach, so finishing the capture
    // stores nothing either.
    assert_eq!(message_id, None);
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

    let recording = audio.start_utterance(&common::session(6));
    recording.push(&[0.5; 960]);
    let message_id = memory
        .store_utterance(&common::session(6), "hello", None)
        .await
        .unwrap()
        .unwrap();
    recording.attach(message_id);
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

/// The misrouting that motivated the attach redesign: utterance B's capture
/// opens before utterance A's row commits. A shared id slot attached B's
/// audio to A's row and dropped A's; the per-capture id channel keeps every
/// capture on its own row.
#[sqlx::test]
#[ignore]
async fn overlapping_utterance_captures_each_attach_to_their_own_row(pool: PgPool) {
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, _client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let first = audio.start_utterance(&common::session(7));
    first.push(&[0.5; 960]); // 60 ms
    let second = audio.start_utterance(&common::session(7));
    second.push(&[0.25; 480]); // 30 ms

    let first_id = memory
        .store_utterance(&common::session(7), "first", None)
        .await
        .unwrap()
        .unwrap();
    first.attach(first_id);
    first.finish();
    let second_id = memory
        .store_utterance(&common::session(7), "second", None)
        .await
        .unwrap()
        .unwrap();
    second.attach(second_id);
    second.finish();

    common::wait_for_count(&pool, "message_audios", 2).await;
    let rows: Vec<(String, i32)> = sqlx::query_as(
        "SELECT m.message->>'content', a.duration_ms \
         FROM message_audios a JOIN messages m ON m.id = a.message_id \
         ORDER BY a.duration_ms",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    // The utterance text identifies the row: each capture's audio carries
    // its own utterance's duration on its own row.
    assert_eq!(
        rows,
        vec![("second".to_owned(), 30), ("first".to_owned(), 60)]
    );
    assert_eq!(audio_storage.keys().len(), 2);
}

/// The reply-side twin: turn 2's capture opens before turn 1's store call
/// commits. The sink rides the call, so the late commit cannot land in
/// turn 2's capture.
#[sqlx::test]
#[ignore]
async fn overlapping_reply_captures_each_attach_to_their_own_row(pool: PgPool) {
    let audio_storage = Arc::new(MemoryStorage::new("memory-bucket"));
    let (memory, audio, _client_id) = wired_memory(&pool, Some(audio_storage.clone())).await;

    let first = audio.start_reply(&common::session(8));
    first.push(&[0.5; 960]); // 60 ms
    let second = audio.start_reply(&common::session(8));
    second.push(&[0.25; 480]); // 30 ms

    memory.log_items(
        &common::session(8),
        vec![ChatItem::assistant("one")],
        Some(first.attach_sink()),
    );
    memory.log_items(
        &common::session(8),
        vec![ChatItem::assistant("two")],
        Some(second.attach_sink()),
    );
    first.finish();
    second.finish();

    common::wait_for_count(&pool, "message_audios", 2).await;
    // The two commits race, so order the result by duration: the audio of
    // reply "one" must sit on the "one" row and "two"'s on the "two" row.
    let rows: Vec<(String, i32)> = sqlx::query_as(
        "SELECT m.message->>'content', a.duration_ms \
         FROM message_audios a JOIN messages m ON m.id = a.message_id \
         ORDER BY a.duration_ms",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(rows, vec![("two".to_owned(), 30), ("one".to_owned(), 60)]);
    assert_eq!(audio_storage.keys().len(), 2);
}
