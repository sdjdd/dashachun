use std::time::Duration;

use sqlx::PgPool;
use uuid::Uuid;

use dashachun::agent::transcript::DbTranscriptSink;
use dashachun::agent::{AgentSession, TranscriptOwner, TranscriptSink};

fn session(id: &str) -> AgentSession {
    AgentSession {
        id: id.into(),
        sample_rate: 16000,
        channels: 1,
        frame_duration_ms: 60,
    }
}

fn owner() -> TranscriptOwner {
    TranscriptOwner {
        user_id: 7,
        agent_id: 3,
        client_id: Uuid::new_v4(),
        device_id: Some("aa:bb:cc:dd:ee:ff".into()),
    }
}

fn sink(pool: PgPool) -> DbTranscriptSink {
    DbTranscriptSink::new(pool, owner())
}

async fn wait_for_message_count(pool: &PgPool, count: i64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let current: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
            .fetch_one(pool)
            .await
            .unwrap();
        if current == count {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "expected {count} messages, got {current}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
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

async fn message_count(pool: &PgPool) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn audio_count(pool: &PgPool) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM message_audios")
        .fetch_one(pool)
        .await
        .unwrap()
}

#[sqlx::test]
#[ignore]
async fn finished_utterance_writes_message_and_audio(pool: PgPool) {
    let own = owner();
    let sink = DbTranscriptSink::new(pool.clone(), own.clone());
    let recording = sink.start_utterance(&session("sess-1"));
    recording.push(&vec![0.25; 16000]);
    recording.finish("hello".into());

    wait_for_message_count(&pool, 1).await;
    let row = sqlx::query_as::<_, (String, Option<String>, i64, i64, Uuid, Option<String>)>(
        "SELECT session_id, content, user_id, agent_id, client_id, device_id \
         FROM messages WHERE role = 'user'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0, "sess-1");
    assert_eq!(row.1.as_deref(), Some("hello"));
    assert_eq!(row.2, 7);
    assert_eq!(row.3, 3);
    assert_eq!(row.4, own.client_id);
    assert_eq!(row.5.as_deref(), Some("aa:bb:cc:dd:ee:ff"));

    let audio = sqlx::query_as::<_, (Vec<u8>, i32, i32, i32)>(
        "SELECT a.audio, a.sample_rate, a.channels, a.duration_ms \
         FROM message_audios a JOIN messages m ON m.id = a.message_id \
         WHERE m.role = 'user'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(audio.0.starts_with(b"fLaC"));
    assert_eq!(audio.1, 16000);
    assert_eq!(audio.2, 1);
    assert_eq!(audio.3, 1000);
}

#[sqlx::test]
#[ignore]
async fn dropped_recording_writes_nothing(pool: PgPool) {
    let sink = sink(pool.clone());
    let recording = sink.start_utterance(&session("sess-1"));
    recording.push(&[0.5, 0.5]);
    drop(recording);
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(message_count(&pool).await, 0);
    assert_eq!(audio_count(&pool).await, 0);
}

#[sqlx::test]
#[ignore]
async fn empty_transcript_writes_nothing(pool: PgPool) {
    let sink = sink(pool.clone());
    let recording = sink.start_utterance(&session("sess-1"));
    recording.push(&[0.5, 0.5]);
    recording.finish(String::new());
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(message_count(&pool).await, 0);
}

#[sqlx::test]
#[ignore]
async fn turn_items_persist_assistant_tool_and_reply_rows(pool: PgPool) {
    use dashachun::llm::{ChatItem, ToolCall};

    let receipt = sink(pool.clone()).log_items(
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

    wait_for_message_count(&pool, 3).await;
    let rows = sqlx::query_as::<
        _,
        (
            i64,
            String,
            Option<String>,
            Option<String>,
            Option<serde_json::Value>,
        ),
    >("SELECT id, role, content, tool_call_id, tool_calls FROM messages ORDER BY id")
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(rows[0].1, "assistant");
    assert_eq!(rows[0].2, None);
    let calls = rows[0].4.as_ref().unwrap();
    assert_eq!(calls[0]["id"], "call_x");
    assert_eq!(calls[0]["name"], "get_weather");
    assert_eq!(calls[0]["arguments"], "{\"city\":\"hangzhou\"}");
    assert_eq!(rows[1].1, "tool");
    assert_eq!(rows[1].2.as_deref(), Some("sunny 26 degrees"));
    assert_eq!(rows[1].3.as_deref(), Some("call_x"));
    assert_eq!(rows[2].1, "assistant");
    assert_eq!(rows[2].2.as_deref(), Some("🙂sunny today"));
    assert!(rows[2].4.is_none());
    // The receipt resolves to the reply's message row.
    assert_eq!(receipt.resolve().await, Some(rows[2].0));
}

#[sqlx::test]
#[ignore]
async fn system_items_are_never_stored(pool: PgPool) {
    use dashachun::llm::ChatItem;

    let _ =
        sink(pool.clone()).log_items(&session("sess-3"), vec![ChatItem::system("system prompt")]);
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(message_count(&pool).await, 0);
}

#[sqlx::test]
#[ignore]
async fn reply_audio_attaches_to_the_assistant_row_via_receipt(pool: PgPool) {
    use dashachun::llm::ChatItem;

    let sink = sink(pool.clone());
    let receipt = sink.log_items(&session("sess-4"), vec![ChatItem::assistant("hi there")]);
    let recording = sink.start_reply(&session("sess-4"));
    recording.push(&vec![0.25; 16000]);
    recording.finish(receipt);

    wait_for_message_count(&pool, 1).await;
    wait_for_audio_count(&pool, 1).await;
    let row = sqlx::query_as::<_, (i64, String, Option<String>)>(
        "SELECT id, role, content FROM messages",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.1, "assistant");
    assert_eq!(row.2.as_deref(), Some("hi there"));

    let audio = sqlx::query_as::<_, (Vec<u8>, i32, i32, i32)>(
        "SELECT audio, sample_rate, channels, duration_ms FROM message_audios",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(audio.0.starts_with(b"fLaC"));
    assert_eq!(audio.1, dashachun::audio::DOWNLINK.sample_rate as i32);
    assert_eq!(audio.2, dashachun::audio::DOWNLINK.channels as i32);
    // 16000 samples at the downlink rate is exactly one second.
    assert_eq!(audio.3, 1000);
}

#[sqlx::test]
#[ignore]
async fn partial_reply_writes_the_truncated_row_and_audio(pool: PgPool) {
    let recording = sink(pool.clone()).start_reply(&session("sess-5"));
    recording.push(&vec![0.25; 8000]);
    recording.finish_partial("partial reply".into());

    wait_for_message_count(&pool, 1).await;
    let row = sqlx::query_as::<_, (String, Option<String>)>("SELECT role, content FROM messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(row.0, "assistant");
    assert_eq!(row.1.as_deref(), Some("partial reply"));

    let audio =
        sqlx::query_as::<_, (i32, i32)>("SELECT sample_rate, duration_ms FROM message_audios")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(audio.0, dashachun::audio::DOWNLINK.sample_rate as i32);
    assert_eq!(audio.1, 500);
}

#[sqlx::test]
#[ignore]
async fn partial_reply_without_audio_writes_the_row_only(pool: PgPool) {
    let recording = sink(pool.clone()).start_reply(&session("sess-6"));
    recording.finish_partial("text only".into());

    wait_for_message_count(&pool, 1).await;
    assert_eq!(audio_count(&pool).await, 0);
}

#[sqlx::test]
#[ignore]
async fn partial_reply_with_empty_text_writes_nothing(pool: PgPool) {
    let recording = sink(pool.clone()).start_reply(&session("sess-7"));
    recording.push(&[0.5]);
    recording.finish_partial(String::new());
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(message_count(&pool).await, 0);
    assert_eq!(audio_count(&pool).await, 0);
}

#[sqlx::test]
#[ignore]
async fn unlinked_receipt_discards_the_reply_audio(pool: PgPool) {
    use dashachun::llm::ChatItem;

    let sink = sink(pool.clone());
    // System-only items store no rows, so the receipt resolves to None and
    // the audio has nothing to attach to.
    let receipt = sink.log_items(&session("sess-8"), vec![ChatItem::system("system prompt")]);
    let recording = sink.start_reply(&session("sess-8"));
    recording.push(&[0.5]);
    recording.finish(receipt);
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(message_count(&pool).await, 0);
    assert_eq!(audio_count(&pool).await, 0);
}
