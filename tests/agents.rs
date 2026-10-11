mod common;

use axum::http::StatusCode;
use sqlx::PgPool;

use common::{expect_json, register, request, send};

#[sqlx::test]
#[ignore]
async fn create_agent_returns_created_agent(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let body = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default","persona_prompt":"你是一个测试助手"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    assert_eq!(body["name"], "default");
    assert_eq!(body["persona_prompt"], "你是一个测试助手");
    assert!(body["id"].is_i64());
    assert!(body["created_at"].is_string());
}

#[sqlx::test]
#[ignore]
async fn create_agent_defaults_persona_prompt_to_empty(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let body = expect_json(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&cookie),
        ),
        StatusCode::CREATED,
    )
    .await;
    assert_eq!(body["persona_prompt"], "");
}

#[sqlx::test]
#[ignore]
async fn duplicate_agent_name_conflicts_within_user_but_not_across_users(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CREATED);
    let res = send(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&cookie),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CONFLICT);
    let (other, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request(
            "POST",
            "/api/agents",
            Some(r#"{"name":"default"}"#),
            Some(&other),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::CREATED);
}

#[sqlx::test]
#[ignore]
async fn create_agent_rejects_blank_name(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let res = send(
        &app,
        request("POST", "/api/agents", Some(r#"{"name":""}"#), Some(&cookie)),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[sqlx::test]
#[ignore]
async fn create_agent_requires_session(pool: PgPool) {
    let app = common::app(pool.clone());
    let res = send(
        &app,
        request("POST", "/api/agents", Some(r#"{"name":"default"}"#), None),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}
