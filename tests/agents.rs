mod common;

use axum::http::StatusCode;
use sqlx::PgPool;
use tower::ServiceExt;

use common::{body_json, post_json, register};

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn create_agent_returns_created_agent(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/agents",
            r#"{"name":"default","persona_prompt":"你是一个测试助手"}"#,
            Some(cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let body = body_json(res).await;
    assert_eq!(body["name"], "default");
    assert_eq!(body["persona_prompt"], "你是一个测试助手");
    assert!(body["id"].is_i64());
    assert!(body["created_at"].is_string());
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn create_agent_defaults_persona_prompt_to_empty(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/agents",
            r#"{"name":"default"}"#,
            Some(cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let body = body_json(res).await;
    assert_eq!(body["persona_prompt"], "");
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn duplicate_agent_name_conflicts_within_user_but_not_across_users(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let first = app
        .clone()
        .oneshot(post_json(
            "/api/agents",
            r#"{"name":"default"}"#,
            Some(cookie.clone()),
        ))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::CREATED);
    let second = app
        .clone()
        .oneshot(post_json(
            "/api/agents",
            r#"{"name":"default"}"#,
            Some(cookie.clone()),
        ))
        .await
        .unwrap();
    assert_eq!(second.status(), StatusCode::CONFLICT);
    let (other, _) = register(&app, &pool).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/agents",
            r#"{"name":"default"}"#,
            Some(other),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn create_agent_rejects_blank_name(pool: PgPool) {
    let app = common::app(pool.clone());
    let (cookie, _) = register(&app, &pool).await;
    let res = app
        .clone()
        .oneshot(post_json("/api/agents", r#"{"name":""}"#, Some(cookie)))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn create_agent_requires_session(pool: PgPool) {
    let app = common::app(pool.clone());
    let res = app
        .clone()
        .oneshot(post_json("/api/agents", r#"{"name":"default"}"#, None))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}
