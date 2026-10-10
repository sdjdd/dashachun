mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use sqlx::PgPool;
use tower::ServiceExt;

use common::{app_with_registry, app_with_state, body_json, register};

use dashachun::provider::ProviderRegistry;

fn combined_registry() -> ProviderRegistry {
    let mut registry = ProviderRegistry::with_builtins();
    dashachun::provider::asr::register_stub(&mut registry);
    dashachun::provider::llm::register_stub(&mut registry);
    dashachun::provider::tts::register_stub(&mut registry);
    registry
}

fn request(
    method: Method,
    path: &str,
    body: Option<&str>,
    cookie: Option<String>,
) -> Request<Body> {
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

async fn send(
    app: &tower_http::normalize_path::NormalizePath<axum::Router>,
    request: Request<Body>,
) -> axum::response::Response {
    app.clone().oneshot(request).await.unwrap()
}

async fn create(
    app: &tower_http::normalize_path::NormalizePath<axum::Router>,
    cookie: &str,
    body: &str,
) -> axum::response::Response {
    send(
        app,
        request(
            Method::POST,
            "/api/providers",
            Some(body),
            Some(cookie.into()),
        ),
    )
    .await
}

const STUB_LLM: &str = r#"{"type":"llm","name":"chat","kind":"stub","config":{"reply":"hi"}}"#;

#[sqlx::test]
#[ignore]
async fn create_without_kind_resolves_the_sole_registered_kind(pool: PgPool) {
    let (app, _) = app_with_state(pool.clone());
    let (cookie, _) = register(&app, &pool).await;

    let res = create(
        &app,
        &cookie,
        r#"{"type":"llm","name":"chat","config":{"reply":"hi"}}"#,
    )
    .await;
    assert_eq!(res.status(), StatusCode::CREATED);
    let body = body_json(res).await;
    assert_eq!(body["kind"], "stub");
    assert_eq!(body["config"]["reply"], "hi");
}

#[sqlx::test]
#[ignore]
async fn create_without_kind_is_ambiguous_when_a_type_has_several(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    let res = create(
        &app,
        &cookie,
        r#"{"type":"llm","name":"chat","config":{"reply":"hi"}}"#,
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let text = String::from_utf8(
        axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(
        text.contains("several provider kinds") && text.contains("stub") && text.contains("openai"),
        "unexpected: {text}"
    );
}

#[sqlx::test]
#[ignore]
async fn create_requires_session(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let res = create(&app, "", STUB_LLM).await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[sqlx::test]
#[ignore]
async fn create_list_and_get_roundtrip_the_row(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    let res = create(&app, &cookie, STUB_LLM).await;
    assert_eq!(res.status(), StatusCode::CREATED);
    let body = body_json(res).await;
    assert!(body["id"].is_i64());
    assert_eq!(body["type"], "llm");
    assert_eq!(body["name"], "chat");
    assert_eq!(body["kind"], "stub");
    assert_eq!(body["config"]["reply"], "hi");
    assert_eq!(body["selected"], false);
    assert!(body["created_at"].is_string());
    assert!(body["updated_at"].is_string());
    let id = body["id"].as_i64().unwrap();

    let res = send(
        &app,
        request(Method::GET, "/api/providers", None, Some(cookie.clone())),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_json(res).await;
    assert_eq!(body.as_array().unwrap().len(), 1);
    assert_eq!(body[0]["id"], id);

    let res = send(
        &app,
        request(
            Method::GET,
            &format!("/api/providers/{id}"),
            None,
            Some(cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await["name"], "chat");
}

#[sqlx::test]
#[ignore]
async fn create_rejects_unknown_type_and_kind(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    let res = create(
        &app,
        &cookie,
        r#"{"type":"mic","name":"x","kind":"stub","config":{}}"#,
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let body = res.into_body();
    let text = String::from_utf8(
        axum::body::to_bytes(body, usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(text.contains("unknown provider type"), "unexpected: {text}");

    let res = create(
        &app,
        &cookie,
        r#"{"type":"llm","name":"x","kind":"bogus","config":{}}"#,
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[sqlx::test]
#[ignore]
async fn create_rejects_an_unbuildable_config(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    let res = create(
        &app,
        &cookie,
        r#"{"type":"asr","name":"x","kind":"volcengine","config":{"base_url":"wss://asr","api_key":"k"}}"#,
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let text = String::from_utf8(
        axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(text.contains("resource_id"), "unexpected: {text}");

    let rows: i64 = sqlx::query_scalar("SELECT count(*) FROM providers")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(rows, 0);
}

#[sqlx::test]
#[ignore]
async fn duplicate_name_conflicts_within_stage_but_not_across(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    let first = create(&app, &cookie, STUB_LLM).await;
    assert_eq!(first.status(), StatusCode::CREATED);
    let second = create(&app, &cookie, STUB_LLM).await;
    assert_eq!(second.status(), StatusCode::CONFLICT);
    let other_stage = create(
        &app,
        &cookie,
        r#"{"type":"tts","name":"chat","kind":"stub","config":{}}"#,
    )
    .await;
    assert_eq!(other_stage.status(), StatusCode::CREATED);
}

#[sqlx::test]
#[ignore]
async fn update_patches_the_provided_fields(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;
    let id = body_json(create(&app, &cookie, STUB_LLM).await).await["id"]
        .as_i64()
        .unwrap();

    let res = send(
        &app,
        request(
            Method::PUT,
            &format!("/api/providers/{id}"),
            Some(r#"{"name":"renamed"}"#),
            Some(cookie.clone()),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_json(res).await;
    assert_eq!(body["name"], "renamed");
    assert_eq!(body["kind"], "stub");
    assert_eq!(body["config"]["reply"], "hi");

    let res = send(
        &app,
        request(
            Method::PUT,
            &format!("/api/providers/{id}"),
            Some(r#"{"config":{"reply":"changed"}}"#),
            Some(cookie.clone()),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_json(res).await;
    assert_eq!(body["name"], "renamed");
    assert_eq!(body["config"]["reply"], "changed");
}

#[sqlx::test]
#[ignore]
async fn update_rejects_an_unbuildable_config_and_keeps_the_row(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;
    let id = body_json(create(
        &app,
        &cookie,
        r#"{"type":"asr","name":"x","kind":"volcengine","config":{"base_url":"wss://asr","api_key":"k","resource_id":"r"}}"#,
    )
    .await).await["id"]
        .as_i64()
        .unwrap();

    let res = send(
        &app,
        request(
            Method::PUT,
            &format!("/api/providers/{id}"),
            Some(r#"{"config":{"base_url":"wss://asr","api_key":"k"}}"#),
            Some(cookie.clone()),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    let body = body_json(
        send(
            &app,
            request(
                Method::GET,
                &format!("/api/providers/{id}"),
                None,
                Some(cookie),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(body["config"]["resource_id"], "r");
}

#[sqlx::test]
#[ignore]
async fn update_missing_provider_is_404(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            Method::PUT,
            "/api/providers/999",
            Some(r#"{"name":"y"}"#),
            Some(cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[sqlx::test]
#[ignore]
async fn activate_applies_only_when_every_stage_is_selected(pool: PgPool) {
    let (app, state) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    let mut ids = Vec::new();
    for provider_type in ["asr", "llm", "tts"] {
        let body = format!(
            r#"{{"type":"{provider_type}","name":"{provider_type}","kind":"stub","config":{{}}}}"#
        );
        let id = body_json(create(&app, &cookie, &body).await).await["id"]
            .as_i64()
            .unwrap();
        ids.push(id);
    }

    let initial = state.agent_factory().unwrap();

    // The first two activations store their selection but cannot assemble a
    // full trio yet — the generation is untouched.
    for id in &ids[..2] {
        let res = send(
            &app,
            request(
                Method::POST,
                &format!("/api/providers/{id}/activate"),
                None,
                Some(cookie.clone()),
            ),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body_json(res).await["selected"], true);
        assert!(
            Arc::ptr_eq(&state.agent_factory().unwrap(), &initial),
            "an incomplete selection must not swap the generation"
        );
    }

    // Completing the trio swaps the generation.
    let res = send(
        &app,
        request(
            Method::POST,
            &format!("/api/providers/{}/activate", ids[2]),
            None,
            Some(cookie.clone()),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await["selected"], true);
    let after = state.agent_factory().unwrap();
    assert!(!Arc::ptr_eq(&after, &initial));
    drop(initial);

    for (key, id) in [("asr", ids[0]), ("llm", ids[1]), ("tts", ids[2])] {
        let stored: serde_json::Value = sqlx::query_scalar(&format!(
            "SELECT value FROM settings WHERE key = '{key}_provider_id'"
        ))
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(stored, serde_json::json!(id), "unexpected {key} selection");
    }
}

#[sqlx::test]
#[ignore]
async fn activate_missing_provider_is_404(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            Method::POST,
            "/api/providers/999/activate",
            None,
            Some(cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[sqlx::test]
#[ignore]
async fn activate_with_a_broken_selection_keeps_the_generation(pool: PgPool) {
    let (app, state) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    for provider_type in ["asr", "llm", "tts"] {
        let body = format!(
            r#"{{"type":"{provider_type}","name":"{provider_type}","kind":"stub","config":{{}}}}"#
        );
        let id = body_json(create(&app, &cookie, &body).await).await["id"]
            .as_i64()
            .unwrap();
        let res = send(
            &app,
            request(
                Method::POST,
                &format!("/api/providers/{id}/activate"),
                None,
                Some(cookie.clone()),
            ),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
    }
    let before = state.agent_factory().unwrap();

    // A hand-inserted row the registry cannot build (REST creation would
    // have validated it).
    let bad_id: i64 = sqlx::query_scalar(
        "INSERT INTO providers (type, name, kind, config) \
         VALUES ('llm', 'broken', 'volcengine', '{\"base_url\": \"wss://x\"}') RETURNING id",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    let res = send(
        &app,
        request(
            Method::POST,
            &format!("/api/providers/{bad_id}/activate"),
            None,
            Some(cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    // The selection stands, the generation is untouched.
    let stored: serde_json::Value =
        sqlx::query_scalar("SELECT value FROM settings WHERE key = 'llm_provider_id'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(stored, serde_json::json!(bad_id));
    let after = state.agent_factory().unwrap();
    assert!(Arc::ptr_eq(&after, &before));
}

#[sqlx::test]
#[ignore]
async fn updating_the_selected_provider_hot_applies(pool: PgPool) {
    let (app, state) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    let mut llm_id = None;
    for provider_type in ["asr", "llm", "tts"] {
        let body = format!(
            r#"{{"type":"{provider_type}","name":"{provider_type}","kind":"stub","config":{{}}}}"#
        );
        let id = body_json(create(&app, &cookie, &body).await).await["id"]
            .as_i64()
            .unwrap();
        if provider_type == "llm" {
            llm_id = Some(id);
        }
        let res = send(
            &app,
            request(
                Method::POST,
                &format!("/api/providers/{id}/activate"),
                None,
                Some(cookie.clone()),
            ),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
    }
    let before = state.agent_factory().unwrap();

    // A rename does not touch the wiring.
    let res = send(
        &app,
        request(
            Method::PUT,
            &format!("/api/providers/{}", llm_id.unwrap()),
            Some(r#"{"name":"renamed"}"#),
            Some(cookie.clone()),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(Arc::ptr_eq(&state.agent_factory().unwrap(), &before));

    // A config change on the selected row swaps the generation.
    let res = send(
        &app,
        request(
            Method::PUT,
            &format!("/api/providers/{}", llm_id.unwrap()),
            Some(r#"{"config":{"reply":"changed"}}"#),
            Some(cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let after = state.agent_factory().unwrap();
    assert!(!Arc::ptr_eq(&after, &before));
}

#[sqlx::test]
#[ignore]
async fn delete_blocks_the_selected_provider(pool: PgPool) {
    let (app, _) = app_with_registry(pool.clone(), combined_registry());
    let (cookie, _) = register(&app, &pool).await;

    let mut selected = None;
    for provider_type in ["asr", "llm", "tts"] {
        let body = format!(
            r#"{{"type":"{provider_type}","name":"{provider_type}","kind":"stub","config":{{}}}}"#
        );
        let id = body_json(create(&app, &cookie, &body).await).await["id"]
            .as_i64()
            .unwrap();
        if provider_type == "llm" {
            selected = Some(id);
        }
        let res = send(
            &app,
            request(
                Method::POST,
                &format!("/api/providers/{id}/activate"),
                None,
                Some(cookie.clone()),
            ),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
    }
    let spare = body_json(create(&app, &cookie, STUB_LLM).await).await["id"]
        .as_i64()
        .unwrap();

    let res = send(
        &app,
        request(
            Method::DELETE,
            &format!("/api/providers/{}", selected.unwrap()),
            None,
            Some(cookie.clone()),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CONFLICT);

    let res = send(
        &app,
        request(
            Method::DELETE,
            &format!("/api/providers/{spare}"),
            None,
            Some(cookie.clone()),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = send(
        &app,
        request(Method::DELETE, "/api/providers/999", None, Some(cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}
