//! Shared scaffolding for the integration tests: the assembled REST app,
//! small HTTP request/response helpers, and the DB-side memory fixtures.
//! Compiled once per test target, so not every helper is used by every
//! binary.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use arc_swap::ArcSwapOption;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use axum_extra::extract::cookie::Key;
use sqlx::PgPool;
use tower::ServiceExt;

pub type TestApp = tower_http::normalize_path::NormalizePath<axum::Router>;

use dashachun::agent::{
    AgentFactory, AgentSession, AgentStore, AudioStorage, MemoryHook, MemoryOwner, ToolSet,
};
use dashachun::auth::state::AuthState;
use dashachun::config::{AppConfig, DeviceConfig, ServerConfig};
use dashachun::device::DeviceStore;
use dashachun::provider::{ProviderRegistry, ProviderSpec};
use dashachun::ratelimit::RedisLimiter;
use dashachun::settings::OtaSettings;
use dashachun::state::ServerState;
use dashachun::vad::{SileroVadFactory, VadConfig};
use dashachun::voiceprint::Voiceprint;

// REST app

/// A random unique username so re-runs never collide.
pub fn test_name() -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!("u{}", &id[..16])
}

/// A limiter with a per-call unique key prefix, so concurrently running
/// tests never share rate-limit buckets over the shared Redis.
pub fn test_limiter() -> RedisLimiter {
    let url = std::env::var("REDIS_URL")
        .expect("REDIS_URL must be set for the DB-backed suites (redis:7 on 6379)");
    RedisLimiter::new(&url, &format!("rl-test-{}", uuid::Uuid::new_v4().simple())).unwrap()
}

pub fn auth_state(pool: PgPool) -> AuthState {
    AuthState {
        pool,
        key: Key::from(&[7u8; 64]),
        ttl_secs: 3600,
        cookie_name: "xz_session".into(),
        cookie_secure: false,
        limiter: test_limiter(),
        trust_proxy: false,
    }
}

/// A device store backed by the test Redis with a per-call unique key
/// prefix, so concurrently running tests never share activation codes.
/// Returns the prefix, so a test can manipulate the keys directly.
pub fn test_device_store(pool: PgPool) -> (DeviceStore, String) {
    let url = std::env::var("REDIS_URL")
        .expect("REDIS_URL must be set for the DB-backed suites (redis:7 on 6379)");
    let prefix = format!("act-test-{}", uuid::Uuid::new_v4().simple());
    (
        DeviceStore::new(pool, redis::Client::open(url).unwrap(), &prefix),
        prefix,
    )
}

pub fn test_state(pool: PgPool, websocket_url: Option<String>) -> ServerState {
    let (devices, _) = test_device_store(pool.clone());
    test_state_with_devices(pool, websocket_url, devices)
}

pub fn test_state_with_devices(
    pool: PgPool,
    websocket_url: Option<String>,
    devices: DeviceStore,
) -> ServerState {
    test_state_with_registry_inner(
        pool,
        websocket_url,
        ProviderRegistry::with_stubs(),
        devices,
        None,
    )
}

pub fn test_state_with_registry(
    pool: PgPool,
    websocket_url: Option<String>,
    registry: ProviderRegistry,
) -> ServerState {
    let (devices, _) = test_device_store(pool.clone());
    test_state_with_registry_inner(pool, websocket_url, registry, devices, None)
}

/// Like [`test_state`], but with the voiceprint enrollment model in its
/// boot slot — the happy-path enrollment tests load the pinned model and
/// hand it in here.
pub fn test_state_with_voiceprint(
    pool: PgPool,
    websocket_url: Option<String>,
    voiceprint: Arc<tokio::sync::Mutex<Voiceprint>>,
) -> ServerState {
    let (devices, _) = test_device_store(pool.clone());
    test_state_with_registry_inner(
        pool,
        websocket_url,
        ProviderRegistry::with_stubs(),
        devices,
        Some(voiceprint),
    )
}

fn test_state_with_registry_inner(
    pool: PgPool,
    websocket_url: Option<String>,
    registry: ProviderRegistry,
    devices: DeviceStore,
    voiceprint: Option<Arc<tokio::sync::Mutex<Voiceprint>>>,
) -> ServerState {
    let vad = Arc::new(SileroVadFactory::new(
        VadConfig::default(),
        Arc::from(Vec::new()),
    ));
    let tools = ToolSet::new();
    let factory = AgentFactory::assemble(
        &registry,
        &ProviderSpec::stub(),
        &ProviderSpec::stub(),
        &ProviderSpec::stub(),
        vad.clone(),
        voiceprint.clone(),
        tools.clone(),
        AgentStore::new(pool.clone()),
        pool.clone(),
    )
    .unwrap();
    ServerState::new(
        AppConfig {
            server: ServerConfig {
                bind_addr: "127.0.0.1:0".into(),
                playback_prebuffer_ms: 180,
                shutdown_grace_ms: 5000,
            },
            ota: OtaSettings {
                websocket_url,
                timezone_offset: 480,
            },
            device: DeviceConfig {
                activation_ttl_secs: 600,
            },
        },
        Arc::new(registry),
        vad,
        tools,
        None,
        Arc::new(ArcSwapOption::from_pointee(factory)),
        devices,
        voiceprint,
    )
}

/// The full app with a Host-derived gateway URL, plus the state handle for
/// assertions on the published agent generation.
pub fn app_with_state(pool: PgPool) -> (TestApp, ServerState) {
    app_with_registry(pool, ProviderRegistry::with_stubs())
}

/// Like [`app_with_state`], but with the voiceprint enrollment model loaded
/// into the boot slot.
pub fn app_with_voiceprint(
    pool: PgPool,
    voiceprint: Arc<tokio::sync::Mutex<Voiceprint>>,
) -> (TestApp, ServerState) {
    let state = test_state_with_voiceprint(pool.clone(), None, voiceprint);
    let auth = auth_state(pool);
    (dashachun::app(state.clone(), auth), state)
}

/// Like [`app_with_state`], but with a caller-supplied provider registry
/// (e.g. builtins plus stubs) for tests that exercise real provider kinds.
pub fn app_with_registry(pool: PgPool, registry: ProviderRegistry) -> (TestApp, ServerState) {
    let state = test_state_with_registry(pool.clone(), None, registry);
    let auth = auth_state(pool);
    (dashachun::app(state.clone(), auth), state)
}

/// The full app with a Host-derived gateway URL.
pub fn app(pool: PgPool) -> TestApp {
    app_with_state(pool).0
}

pub fn request(
    method: &str,
    path: &str,
    body: Option<&str>,
    cookie: Option<&str>,
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

pub async fn send(app: &TestApp, request: Request<Body>) -> axum::response::Response {
    app.clone().oneshot(request).await.unwrap()
}

/// Sends the request and asserts the status, returning the parsed JSON body.
/// A mismatch panics with the body, so the server's error JSON is visible.
pub async fn expect_json(
    app: &TestApp,
    request: Request<Body>,
    expected: StatusCode,
) -> serde_json::Value {
    let response = send(app, request).await;
    let status = response.status();
    let value = body_json(response).await;
    assert_eq!(status, expected, "unexpected status, body: {value}");
    value
}

pub fn session_cookie(res: &axum::response::Response) -> String {
    let value = res.headers().get("set-cookie").unwrap().to_str().unwrap();
    value.split(';').next().unwrap().to_owned()
}

pub async fn body_json(res: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(&to_bytes(res.into_body(), usize::MAX).await.unwrap()).unwrap()
}

/// Registers a fresh user and returns the session cookie plus the row id.
pub async fn register(app: &TestApp, pool: &PgPool) -> (String, i64) {
    let username = test_name();
    let body = serde_json::json!({"username": username, "password": "supersecret1"}).to_string();
    let res = send(
        app,
        request("POST", "/api/auth/register", Some(&body), None),
    )
    .await;
    assert_eq!(res.status(), axum::http::StatusCode::CREATED);
    let cookie = session_cookie(&res);
    let user_id = sqlx::query_scalar::<_, i64>("SELECT id FROM users WHERE username = $1")
        .bind(&username)
        .fetch_one(pool)
        .await
        .unwrap();
    (cookie, user_id)
}

// memory / capture

pub fn session(id: &str) -> AgentSession {
    AgentSession {
        id: id.into(),
        sample_rate: 16000,
        channels: 1,
        frame_duration_ms: 60,
    }
}

pub fn owner(user_id: i64, agent_id: i64) -> MemoryOwner {
    owner_with_client(user_id, agent_id, uuid::Uuid::new_v4())
}

pub fn owner_with_client(user_id: i64, agent_id: i64, client_id: uuid::Uuid) -> MemoryOwner {
    MemoryOwner {
        user_id,
        agent_id,
        client_id,
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

/// An in-memory [`AudioStorage`] recording every put — the test double for the
/// OSS, so capture suites assert keys and bytes without a storage service.
#[derive(Default)]
pub struct MemoryStorage {
    bucket: String,
    objects: Mutex<HashMap<String, Vec<u8>>>,
}

#[async_trait::async_trait]
impl AudioStorage for MemoryStorage {
    async fn put(&self, key: &str, bytes: Vec<u8>) -> Result<String, String> {
        self.objects.lock().unwrap().insert(key.to_owned(), bytes);
        Ok(self.bucket.clone())
    }

    async fn get(&self, bucket: &str, key: &str) -> Result<Vec<u8>, String> {
        if bucket != self.bucket {
            return Err(format!("get {bucket}/{key}: no such bucket"));
        }
        self.objects
            .lock()
            .unwrap()
            .get(key)
            .cloned()
            .ok_or_else(|| format!("get {bucket}/{key}: no such object"))
    }
}

impl MemoryStorage {
    pub fn new(bucket: &str) -> Self {
        Self {
            bucket: bucket.to_owned(),
            objects: Mutex::new(HashMap::new()),
        }
    }

    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    pub fn keys(&self) -> Vec<String> {
        self.objects.lock().unwrap().keys().cloned().collect()
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.objects.lock().unwrap().get(key).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.lock().unwrap().is_empty()
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
