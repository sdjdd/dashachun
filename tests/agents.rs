use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use axum_extra::extract::cookie::Key;
use sqlx::PgPool;
use tower::ServiceExt;

use dashachun::auth::state::AuthState;
use dashachun::config::{AppConfig, DeviceConfig, OtaConfig, ServerConfig};
use dashachun::device::DeviceStore;
use dashachun::state::ServerState;

fn test_name() -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!("u{}", &id[..16])
}

fn auth_state(pool: PgPool) -> AuthState {
    AuthState {
        pool,
        key: Key::from(&[7u8; 64]),
        ttl_secs: 3600,
        cookie_name: "xz_session".into(),
        cookie_secure: false,
    }
}

fn server_state(pool: PgPool) -> ServerState {
    ServerState::new(
        AppConfig {
            server: ServerConfig {
                bind_addr: "127.0.0.1:0".into(),
                playback_prebuffer_ms: 180,
                shutdown_grace_ms: 5000,
            },
            ota: OtaConfig {
                websocket_url: None,
                timezone_offset: 480,
            },
            device: DeviceConfig {
                activation_ttl_secs: 600,
            },
        },
        Arc::new(dashachun::agent::AgentFactory::new(
            Arc::new(dashachun::asr::StubAsr::default()),
            Arc::new(dashachun::llm::StubLlm::default()),
            Arc::new(dashachun::tts::StubTts),
            Arc::new(dashachun::vad::SileroVadFactory::new(
                dashachun::vad::VadConfig::default(),
            )),
            Arc::new(dashachun::agent::InMemMemoryFactory),
            Arc::new(dashachun::agent::ToolRegistry::new(Vec::new())),
            dashachun::agent::AgentStore::new(pool.clone()),
        )),
        DeviceStore::new(pool),
    )
}

fn app(pool: PgPool) -> tower_http::normalize_path::NormalizePath<axum::Router> {
    let auth = auth_state(pool.clone());
    dashachun::app(server_state(pool), auth)
}

fn post_json(path: &str, body: &str, cookie: Option<String>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    if let Some(cookie) = cookie {
        builder = builder.header("cookie", cookie);
    }
    builder.body(Body::from(body.to_owned())).unwrap()
}

fn session_cookie(res: &axum::response::Response) -> String {
    let value = res.headers().get("set-cookie").unwrap().to_str().unwrap();
    value.split(';').next().unwrap().to_owned()
}

async fn body_json(res: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(&to_bytes(res.into_body(), usize::MAX).await.unwrap()).unwrap()
}

async fn register(app: &tower_http::normalize_path::NormalizePath<axum::Router>) -> String {
    let username = test_name();
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/auth/register",
            &format!(r#"{{"username":"{username}","password":"supersecret1"}}"#),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    session_cookie(&res)
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn create_agent_returns_created_agent(pool: PgPool) {
    let app = app(pool);
    let cookie = register(&app).await;
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
    let app = app(pool);
    let cookie = register(&app).await;
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
    let app = app(pool);
    let cookie = register(&app).await;
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
    let other = register(&app).await;
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
    let app = app(pool);
    let cookie = register(&app).await;
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
    let app = app(pool);
    let res = app
        .clone()
        .oneshot(post_json("/api/agents", r#"{"name":"default"}"#, None))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}
