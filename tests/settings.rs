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
#[ignore]
async fn load_reads_stored_selections(pool: PgPool) {
    insert(&pool, "llm_provider_id", serde_json::json!(7)).await;
    insert(
        &pool,
        "ota",
        serde_json::json!({"websocket_url": "wss://example.com/gateway", "timezone_offset": 0}),
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
