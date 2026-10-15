//! The audio-capture surfaces: the admin settings API (auth, validation,
//! the hot-swap apply path) and the per-message playback read route. The
//! storage-verification tests need the compose RustFS — export
//! `S3_ENDPOINT`/`S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY` (same variables
//! as the `oss` suite) alongside `DATABASE_URL`/`REDIS_URL`.

mod common;

use std::sync::Arc;

use axum::body::to_bytes;
use axum::http::StatusCode;
use sqlx::PgPool;

use common::{MemoryStorage, expect_json, register, request, send};

use dashachun::agent::{AgentStore, AudioStorage};

async fn owned_agent(pool: &PgPool, user_id: i64, name: &str) -> i64 {
    AgentStore::new(pool.clone())
        .create(user_id, name, "")
        .await
        .unwrap()
        .id
}

async fn message_row(pool: &PgPool, agent_id: i64, role: &str) -> i64 {
    let conversation_id = common::conversation(pool, 1, agent_id).await;
    sqlx::query_scalar(
        "INSERT INTO messages (conversation_id, message) VALUES ($1, $2) RETURNING id",
    )
    .bind(conversation_id)
    .bind(serde_json::json!({ "role": role, "content": "你好" }))
    .fetch_one(pool)
    .await
    .unwrap()
}

async fn audio_row(pool: &PgPool, message_id: i64, bucket: &str, key: &str) {
    sqlx::query(
        "INSERT INTO message_audios \
         (message_id, bucket, object_key, sample_rate, channels, duration_ms) \
         VALUES ($1, $2, $3, 16000, 1, 1000)",
    )
    .bind(message_id)
    .bind(bucket)
    .bind(key)
    .execute(pool)
    .await
    .unwrap();
}

#[sqlx::test]
#[ignore]
async fn audio_capture_settings_authenticate_and_roundtrip(pool: PgPool) {
    let (app, state) = common::app_with_state(pool.clone());
    let (cookie, _) = register(&app, &pool).await;

    // Unauthenticated reads and writes are rejected.
    let res = send(
        &app,
        request("GET", "/api/settings/audio-capture", None, None),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let res = send(
        &app,
        request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":false,"bucket":""}"#),
            None,
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // A fresh deployment reads the default section (disabled, no bucket).
    let body = expect_json(
        &app,
        request("GET", "/api/settings/audio-capture", None, Some(&cookie)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["enabled"], false);
    assert_eq!(body["bucket"], "");
    assert!(state.audio_storage().is_none());

    // Disabling stores the section as-is (no storage round trip) and keeps
    // the slot empty.
    let body = expect_json(
        &app,
        request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":false,"bucket":""}"#),
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["enabled"], false);
    assert!(state.audio_storage().is_none());
    let (stored,): (serde_json::Value,) =
        sqlx::query_as("SELECT value FROM settings WHERE key = 'audio_capture'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(stored["enabled"], false);

    // Enabling without a bucket is a validation error — nothing stored,
    // the previous section stands.
    let res = send(
        &app,
        request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":true,"bucket":"   "}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let (stored,): (serde_json::Value,) =
        sqlx::query_as("SELECT value FROM settings WHERE key = 'audio_capture'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(stored["enabled"], false);
    assert!(state.audio_storage().is_none());
}

#[sqlx::test]
#[ignore]
async fn audio_capture_settings_are_admin_only(pool: PgPool) {
    let (app, _) = common::app_with_state(pool.clone());
    let (admin_cookie, _) = register(&app, &pool).await;
    let (user_cookie, _) = register(&app, &pool).await;

    for (method, body) in [
        ("GET", None),
        ("PUT", Some(r#"{"enabled":false,"bucket":""}"#)),
    ] {
        let res = send(
            &app,
            request(
                method,
                "/api/settings/audio-capture",
                body,
                Some(&user_cookie),
            ),
        )
        .await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN, "{method}");
    }

    let res = send(
        &app,
        request(
            "GET",
            "/api/settings/audio-capture",
            None,
            Some(&admin_cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
}

#[sqlx::test]
#[ignore]
async fn enabling_validates_the_storage_then_swaps_the_slot(pool: PgPool) {
    std::env::var("S3_ENDPOINT").expect("S3_ENDPOINT must be set (compose rustfs)");
    std::env::var("S3_ACCESS_KEY_ID").expect("S3_ACCESS_KEY_ID must be set (compose rustfs)");
    std::env::var("S3_SECRET_ACCESS_KEY").expect("S3_SECRET_ACCESS_KEY must be set");
    let (app, state) = common::app_with_state(pool.clone());
    let (cookie, _) = register(&app, &pool).await;

    // Enabling eagerly ensures the bucket, stores the section and swaps
    // the slot in — all before the response returns.
    expect_json(
        &app,
        request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":true,"bucket":"dsc-audio-routetest"}"#),
            Some(&cookie),
        ),
        StatusCode::OK,
    )
    .await;
    assert!(state.audio_storage().is_some());
    let (stored,): (serde_json::Value,) =
        sqlx::query_as("SELECT value FROM settings WHERE key = 'audio_capture'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(stored["enabled"], true);
    assert_eq!(stored["bucket"], "dsc-audio-routetest");

    // A storage-side rejection (RustFS rejects this bucket name) keeps the
    // previous settings and slot standing — the eager check failed before
    // anything was written.
    let res = send(
        &app,
        request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":true,"bucket":"INVALID/NAME"}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let (stored,): (serde_json::Value,) =
        sqlx::query_as("SELECT value FROM settings WHERE key = 'audio_capture'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(stored["bucket"], "dsc-audio-routetest");
    assert!(state.audio_storage().is_some());

    // Disabling empties the slot again.
    let res = send(
        &app,
        request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":false,"bucket":"dsc-audio-routetest"}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(state.audio_storage().is_none());
}

#[sqlx::test]
#[ignore]
async fn playback_serves_the_captured_bytes(pool: PgPool) {
    let (app, state) = common::app_with_state(pool.clone());
    let storage = Arc::new(MemoryStorage::new("bucket"));
    state.set_audio_storage(Some(storage.clone()));

    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let uplink = message_row(&pool, agent, "user").await;
    let downlink = message_row(&pool, agent, "assistant").await;
    storage
        .put("audios/test/up.flac", b"fLaC-uplink".to_vec())
        .await
        .unwrap();
    storage
        .put("audios/test/down.flac", b"fLaC-downlink".to_vec())
        .await
        .unwrap();
    audio_row(&pool, uplink, "bucket", "audios/test/up.flac").await;
    audio_row(&pool, downlink, "bucket", "audios/test/down.flac").await;

    let path = format!("/api/agents/{agent}/messages/{uplink}/audio");
    let res = send(&app, request("GET", &path, None, Some(&cookie))).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "audio/flac");
    assert_eq!(res.headers()["cache-control"], "private");
    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    assert_eq!(&bytes[..], b"fLaC-uplink");

    // The downlink capture (assistant row) plays back the same way — the
    // route is not role-gated.
    let path = format!("/api/agents/{agent}/messages/{downlink}/audio");
    let res = send(&app, request("GET", &path, None, Some(&cookie))).await;
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    assert_eq!(&bytes[..], b"fLaC-downlink");
}

#[sqlx::test]
#[ignore]
async fn playback_scopes_by_ownership_and_reports_missing_audio(pool: PgPool) {
    let (app, state) = common::app_with_state(pool.clone());
    let storage = Arc::new(MemoryStorage::new("bucket"));
    state.set_audio_storage(Some(storage.clone()));

    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let other_agent = owned_agent(&pool, user_id, "b").await;
    let (other_cookie, other_user) = register(&app, &pool).await;
    let foreign_agent = owned_agent(&pool, other_user, "c").await;

    let uncaptured = message_row(&pool, agent, "user").await;
    let cross_agent = message_row(&pool, other_agent, "user").await;
    let foreign = message_row(&pool, foreign_agent, "user").await;
    let vanished = message_row(&pool, agent, "user").await;
    audio_row(&pool, vanished, "bucket", "never-stored.flac").await;

    let path = |agent: i64, message: i64| format!("/api/agents/{agent}/messages/{message}/audio");

    let res = send(&app, request("GET", &path(agent, uncaptured), None, None)).await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // Foreign agents and their messages stay invisible.
    let res = send(
        &app,
        request("GET", &path(foreign_agent, foreign), None, Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = send(
        &app,
        request("GET", &path(agent, foreign), None, Some(&other_cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // The agent is owned but the message belongs to the caller's other
    // agent — out of scope.
    let res = send(
        &app,
        request("GET", &path(agent, cross_agent), None, Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // Unknown message, and a message whose capture never happened.
    let res = send(
        &app,
        request("GET", &path(agent, 999_999), None, Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = send(
        &app,
        request("GET", &path(agent, uncaptured), None, Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // A row whose object is gone reads as an internal error — the row
    // stands, the storage read failed.
    let res = send(
        &app,
        request("GET", &path(agent, vanished), None, Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[sqlx::test]
#[ignore]
async fn playback_503s_while_audio_storage_is_disabled(pool: PgPool) {
    let (app, _state) = common::app_with_state(pool.clone());

    let (cookie, user_id) = register(&app, &pool).await;
    let agent = owned_agent(&pool, user_id, "a").await;
    let message = message_row(&pool, agent, "user").await;
    audio_row(&pool, message, "bucket", "audios/test/up.flac").await;

    // The capture row stands but the storage slot is empty (the feature
    // is off or degraded) — the read is retryable, not missing.
    let res = send(
        &app,
        request(
            "GET",
            &format!("/api/agents/{agent}/messages/{message}/audio"),
            None,
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
}
