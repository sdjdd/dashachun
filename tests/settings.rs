use sqlx::PgPool;

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
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn load_reads_stored_sections(pool: PgPool) {
    insert(
        &pool,
        "llm",
        serde_json::json!({
            "base_url": "http://localhost/v1",
            "api_key": "k",
            "model": "m",
            "max_tokens": 512
        }),
    )
    .await;
    insert(
        &pool,
        "ota",
        serde_json::json!({"websocket_url": "wss://example.com/gateway", "timezone_offset": 0}),
    )
    .await;

    let settings = Settings::load(&pool).await.unwrap();
    let llm = settings.llm.unwrap();
    assert_eq!(llm.base_url, "http://localhost/v1");
    assert_eq!(llm.model, "m");
    assert_eq!(llm.max_tokens, Some(512));
    let ota = settings.ota.unwrap();
    assert_eq!(
        ota.websocket_url.as_deref(),
        Some("wss://example.com/gateway")
    );
    assert_eq!(ota.timezone_offset, 0);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn load_fails_on_an_invalid_row(pool: PgPool) {
    insert(
        &pool,
        "llm",
        serde_json::json!({"base_url": "http://localhost/v1"}),
    )
    .await;

    let err = Settings::load(&pool).await.unwrap_err();
    assert!(err.contains("llm"), "unexpected error: {err}");
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn load_skips_unknown_keys(pool: PgPool) {
    insert(&pool, "wizard", serde_json::json!({"step": 1})).await;

    let settings = Settings::load(&pool).await.unwrap();
    assert!(settings.llm.is_none());
    assert!(settings.asr.is_none());
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
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
