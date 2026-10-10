//! The audio-capture settings API: auth, validation and the hot-swap apply
//! path. The storage-verification tests need the compose RustFS — export
//! `S3_ENDPOINT`/`S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY` (same variables
//! as the `oss` suite) alongside `DATABASE_URL`/`REDIS_URL`.

mod common;

use axum::body::Body;
use axum::http::Request;
use sqlx::PgPool;
use tower::ServiceExt;

fn request(method: &str, path: &str, body: Option<&str>, cookie: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(path);
    if body.is_some() {
        builder = builder.header("content-type", "application/json");
    }
    if let Some(cookie) = cookie {
        builder = builder.header("cookie", cookie);
    }
    builder
        .body(Body::from(body.unwrap_or_default().to_owned()))
        .unwrap()
}

#[sqlx::test]
#[ignore]
async fn audio_capture_settings_authenticate_and_roundtrip(pool: PgPool) {
    let (app, state) = common::app_with_state(pool.clone());
    let (cookie, _) = common::register(&app, &pool).await;

    // Unauthenticated reads and writes are rejected.
    let res = app
        .clone()
        .oneshot(request("GET", "/api/settings/audio-capture", None, None))
        .await
        .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::UNAUTHORIZED);
    let res = app
        .clone()
        .oneshot(request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":false,"bucket":""}"#),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::UNAUTHORIZED);

    // A fresh deployment reads the default section (disabled, no bucket).
    let res = app
        .clone()
        .oneshot(request(
            "GET",
            "/api/settings/audio-capture",
            None,
            Some(&cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);
    let body = common::body_json(res).await;
    assert_eq!(body["enabled"], false);
    assert_eq!(body["bucket"], "");
    assert!(state.audio_storage().is_none());

    // Disabling stores the section as-is (no storage round trip) and keeps
    // the slot empty.
    let res = app
        .clone()
        .oneshot(request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":false,"bucket":""}"#),
            Some(&cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);
    let body = common::body_json(res).await;
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
    let res = app
        .clone()
        .oneshot(request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":true,"bucket":"   "}"#),
            Some(&cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::BAD_REQUEST);
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
async fn enabling_validates_the_storage_then_swaps_the_slot(pool: PgPool) {
    std::env::var("S3_ENDPOINT").expect("S3_ENDPOINT must be set (compose rustfs)");
    std::env::var("S3_ACCESS_KEY_ID").expect("S3_ACCESS_KEY_ID must be set (compose rustfs)");
    std::env::var("S3_SECRET_ACCESS_KEY").expect("S3_SECRET_ACCESS_KEY must be set");
    let (app, state) = common::app_with_state(pool.clone());
    let (cookie, _) = common::register(&app, &pool).await;

    // Enabling eagerly ensures the bucket, stores the section and swaps
    // the slot in — all before the response returns.
    let res = app
        .clone()
        .oneshot(request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":true,"bucket":"dsc-audio-routetest"}"#),
            Some(&cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);
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
    let res = app
        .clone()
        .oneshot(request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":true,"bucket":"INVALID/NAME"}"#),
            Some(&cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::BAD_REQUEST);
    let (stored,): (serde_json::Value,) =
        sqlx::query_as("SELECT value FROM settings WHERE key = 'audio_capture'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(stored["bucket"], "dsc-audio-routetest");
    assert!(state.audio_storage().is_some());

    // Disabling empties the slot again.
    let res = app
        .clone()
        .oneshot(request(
            "PUT",
            "/api/settings/audio-capture",
            Some(r#"{"enabled":false,"bucket":"dsc-audio-routetest"}"#),
            Some(&cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);
    assert!(state.audio_storage().is_none());
}
