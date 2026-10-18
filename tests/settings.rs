mod common;

use axum::http::StatusCode;
use sqlx::PgPool;

use common::{app, expect_json, register, request, send};

use dashachun::settings::Settings;

async fn insert(pool: &PgPool, key: &str, value: serde_json::Value) {
    sqlx::query("INSERT INTO settings (key, value) VALUES ($1, $2)")
        .bind(key)
        .bind(value)
        .execute(pool)
        .await
        .unwrap();
}

#[sqlx::test]
#[ignore]
async fn load_reads_stored_selections(pool: PgPool) {
    insert(&pool, "llm_provider_id", serde_json::json!(7)).await;
    insert(
        &pool,
        "ota",
        serde_json::json!({"websocket_url": "wss://example.com/gateway", "timezone_offset": 0}),
    )
    .await;
    insert(
        &pool,
        "voiceprint",
        serde_json::json!({"match_threshold": 0.55}),
    )
    .await;

    let settings = Settings::load(&pool).await.unwrap();
    assert_eq!(settings.llm, Some(7));
    let ota = settings.ota.unwrap();
    assert_eq!(
        ota.websocket_url.as_deref(),
        Some("wss://example.com/gateway")
    );
    assert_eq!(ota.timezone_offset, 0);
    assert_eq!(settings.voiceprint.unwrap().match_threshold, 0.55);
}

#[sqlx::test]
#[ignore]
async fn load_rejects_a_malformed_selection_row(pool: PgPool) {
    insert(&pool, "asr_provider_id", serde_json::json!("not-an-id")).await;

    let err = Settings::load(&pool).await.unwrap_err();
    assert!(err.contains("asr_provider_id"), "unexpected error: {err}");
}

#[sqlx::test]
#[ignore]
async fn load_skips_unknown_keys(pool: PgPool) {
    insert(&pool, "wizard", serde_json::json!({"step": 1})).await;

    let settings = Settings::load(&pool).await.unwrap();
    assert!(settings.llm.is_none());
    assert!(settings.asr.is_none());
}

#[sqlx::test]
#[ignore]
async fn ensure_session_secret_generates_stores_and_reuses(pool: PgPool) {
    let mut settings = Settings::default();
    let secret = settings.ensure_session_secret(&pool).await.unwrap();
    assert_eq!(secret.len(), 128);
    assert!(secret.chars().all(|c| c.is_ascii_hexdigit()));

    let mut reloaded = Settings::load(&pool).await.unwrap();
    assert_eq!(reloaded.ensure_session_secret(&pool).await.unwrap(), secret);

    let stored: i64 = sqlx::query_scalar("SELECT count(*) FROM settings WHERE key = 'security'")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(stored, 1);
}

#[sqlx::test]
#[ignore]
async fn ota_settings_rest_reads_defaults_and_roundtrips(pool: PgPool) {
    let app = app(pool.clone());
    let (admin, _) = register(&app, &pool).await;

    let body = expect_json(
        &app,
        request("GET", "/api/settings/ota", None, Some(&admin)),
        StatusCode::OK,
    )
    .await;
    assert!(body["websocket_url"].is_null());
    assert_eq!(body["timezone_offset"], 480);

    let body = expect_json(
        &app,
        request(
            "PUT",
            "/api/settings/ota",
            Some(r#"{"websocket_url":"wss://example.com/gateway","timezone_offset":-300}"#),
            Some(&admin),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["websocket_url"], "wss://example.com/gateway");
    assert_eq!(body["timezone_offset"], -300);

    let body = expect_json(
        &app,
        request("GET", "/api/settings/ota", None, Some(&admin)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["websocket_url"], "wss://example.com/gateway");
    assert_eq!(body["timezone_offset"], -300);

    for payload in [
        r#"{"websocket_url":"https://example.com","timezone_offset":0}"#,
        r#"{"websocket_url":null,"timezone_offset":99999}"#,
        r#"{"websocket_url":null,"timezone_offset":-1000}"#,
    ] {
        let res = send(
            &app,
            request("PUT", "/api/settings/ota", Some(payload), Some(&admin)),
        )
        .await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    let (non_admin, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request("GET", "/api/settings/ota", None, Some(&non_admin)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
    let res = send(
        &app,
        request(
            "PUT",
            "/api/settings/ota",
            Some(r#"{"websocket_url":null,"timezone_offset":0}"#),
            Some(&non_admin),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::FORBIDDEN);

    let settings = Settings::load(&pool).await.unwrap();
    let ota = settings.ota.unwrap();
    assert_eq!(
        ota.websocket_url.as_deref(),
        Some("wss://example.com/gateway")
    );
    assert_eq!(ota.timezone_offset, -300);
}

#[sqlx::test]
#[ignore]
async fn voiceprint_settings_rest_reads_defaults_and_roundtrips(pool: PgPool) {
    let app = app(pool.clone());
    let (admin, _) = register(&app, &pool).await;

    let body = expect_json(
        &app,
        request("GET", "/api/settings/voiceprint", None, Some(&admin)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["match_threshold"], 0.4);

    let body = expect_json(
        &app,
        request(
            "PUT",
            "/api/settings/voiceprint",
            Some(r#"{"match_threshold":0.55}"#),
            Some(&admin),
        ),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["match_threshold"], 0.55);

    let body = expect_json(
        &app,
        request("GET", "/api/settings/voiceprint", None, Some(&admin)),
        StatusCode::OK,
    )
    .await;
    assert_eq!(body["match_threshold"], 0.55);

    for payload in [r#"{"match_threshold":-0.1}"#, r#"{"match_threshold":1.5}"#] {
        let res = send(
            &app,
            request(
                "PUT",
                "/api/settings/voiceprint",
                Some(payload),
                Some(&admin),
            ),
        )
        .await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    let (non_admin, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request("GET", "/api/settings/voiceprint", None, Some(&non_admin)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
    let res = send(
        &app,
        request(
            "PUT",
            "/api/settings/voiceprint",
            Some(r#"{"match_threshold":0.4}"#),
            Some(&non_admin),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::FORBIDDEN);

    let settings = Settings::load(&pool).await.unwrap();
    assert_eq!(settings.voiceprint.unwrap().match_threshold, 0.55);
}
