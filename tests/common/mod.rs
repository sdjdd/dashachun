//! Shared scaffolding for the integration tests: the assembled REST app,
//! small HTTP request/response helpers, and the DB-side memory fixtures.
//! Compiled once per test target, so not every helper is used by every
//! binary.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use axum_extra::extract::cookie::Key;
use sqlx::PgPool;
use tower::ServiceExt;

use dashachun::agent::{
    AgentFactory, AgentSession, AgentStore, MemoryHook, MemoryOwner, ToolRegistry,
};
use dashachun::auth::state::AuthState;
use dashachun::config::{AppConfig, DeviceConfig, OtaConfig, ServerConfig};
use dashachun::device::DeviceStore;
use dashachun::provider::asr::StubAsr;
use dashachun::provider::llm::StubLlm;
use dashachun::provider::tts::StubTts;
use dashachun::state::ServerState;
use dashachun::vad::{SileroVadFactory, VadConfig};

// ---------------------------------------------------------------- REST app

/// A random unique username so re-runs never collide.
pub fn test_name() -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!("u{}", &id[..16])
}

pub fn auth_state(pool: PgPool) -> AuthState {
    AuthState {
        pool,
        key: Key::from(&[7u8; 64]),
        ttl_secs: 3600,
        cookie_name: "xz_session".into(),
        cookie_secure: false,
    }
}

pub fn test_state(pool: PgPool, websocket_url: Option<String>) -> ServerState {
    ServerState::new(
        AppConfig {
            server: ServerConfig {
                bind_addr: "127.0.0.1:0".into(),
                playback_prebuffer_ms: 180,
                shutdown_grace_ms: 5000,
            },
            ota: OtaConfig {
                websocket_url,
                timezone_offset: 480,
            },
            device: DeviceConfig {
                activation_ttl_secs: 600,
            },
        },
        Arc::new(AgentFactory::new(
            Arc::new(StubAsr::default()),
            Arc::new(StubLlm::default()),
            Arc::new(StubTts),
            Arc::new(SileroVadFactory::new(VadConfig::default())),
            Arc::new(ToolRegistry::new(Vec::new())),
            AgentStore::new(pool.clone()),
            pool.clone(),
        )),
        DeviceStore::new(pool),
    )
}

/// The full app with a Host-derived gateway URL.
pub fn app(pool: PgPool) -> tower_http::normalize_path::NormalizePath<axum::Router> {
    let auth = auth_state(pool.clone());
    dashachun::app(test_state(pool, None), auth)
}

pub fn post_json(path: &str, body: &str, cookie: Option<String>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    if let Some(cookie) = cookie {
        builder = builder.header("cookie", cookie);
    }
    builder.body(Body::from(body.to_owned())).unwrap()
}

pub fn session_cookie(res: &axum::response::Response) -> String {
    let value = res.headers().get("set-cookie").unwrap().to_str().unwrap();
    value.split(';').next().unwrap().to_owned()
}

pub async fn body_json(res: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(&to_bytes(res.into_body(), usize::MAX).await.unwrap()).unwrap()
}

/// Registers a fresh user and returns the session cookie plus the row id.
pub async fn register(
    app: &tower_http::normalize_path::NormalizePath<axum::Router>,
    pool: &PgPool,
) -> (String, i64) {
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
    assert_eq!(res.status(), axum::http::StatusCode::CREATED);
    let cookie = session_cookie(&res);
    let user_id = sqlx::query_scalar::<_, i64>("SELECT id FROM users WHERE username = $1")
        .bind(&username)
        .fetch_one(pool)
        .await
        .unwrap();
    (cookie, user_id)
}

// -------------------------------------------------------- memory / capture

pub fn session(id: &str) -> AgentSession {
    AgentSession {
        id: id.into(),
        sample_rate: 16000,
        channels: 1,
        frame_duration_ms: 60,
    }
}

pub fn owner(user_id: i64, agent_id: i64) -> MemoryOwner {
    MemoryOwner {
        user_id,
        agent_id,
        client_id: uuid::Uuid::new_v4(),
        device_id: Some("aa:bb:cc:dd:ee:ff".into()),
    }
}

/// Records the message ids the memory announces, for hook assertions.
#[derive(Default)]
pub struct RecordingHook {
    utterances: Mutex<Vec<i64>>,
    turns: Mutex<Vec<i64>>,
}

impl MemoryHook for RecordingHook {
    fn utterance_stored(&self, _session: &AgentSession, message_id: i64) {
        self.utterances.lock().unwrap().push(message_id);
    }

    fn turn_stored(&self, _session: &AgentSession, message_id: i64) {
        self.turns.lock().unwrap().push(message_id);
    }
}

impl RecordingHook {
    pub fn utterance_ids(&self) -> Vec<i64> {
        self.utterances.lock().unwrap().clone()
    }

    pub fn turn_ids(&self) -> Vec<i64> {
        self.turns.lock().unwrap().clone()
    }
}

/// Waits until `table` holds exactly `count` rows — the writes run in
/// background tasks, so the test must poll.
pub async fn wait_for_count(pool: &PgPool, table: &str, count: i64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let current: i64 = sqlx::query_scalar(&format!("SELECT count(*) FROM {table}"))
            .fetch_one(pool)
            .await
            .unwrap();
        if current == count {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "expected {count} rows in {table}, got {current}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
