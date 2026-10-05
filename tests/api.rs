use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{HeaderValue, Request, StatusCode};
use axum_extra::extract::cookie::Key;
use sqlx::PgPool;
use tower::ServiceExt;

use dashachun::agent::{Agent, CompositeAgent, ToolRegistry};
use dashachun::asr::StubAsr;
use dashachun::auth::state::AuthState;
use dashachun::config::{AppConfig, DeviceConfig, OtaConfig, ServerConfig};
use dashachun::device::DeviceStore;
use dashachun::state::ServerState;
use dashachun::vad::{SileroVadFactory, VadConfig};

const CLIENT_ID: &str = "00000000-0000-0000-0000-000000000000";

fn auth_state(pool: PgPool) -> AuthState {
    AuthState {
        pool,
        key: Key::from(&[7u8; 64]),
        ttl_secs: 3600,
        cookie_name: "xz_session".into(),
        cookie_secure: false,
    }
}

fn state_with_url(pool: PgPool, websocket_url: Option<String>) -> ServerState {
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
        Arc::new(CompositeAgent {
            asr: Arc::new(StubAsr::default()),
            llm: None,
            tts: None,
            vad: Arc::new(SileroVadFactory::new(VadConfig::default())),
            tools: Arc::new(ToolRegistry::new(Vec::new())),
        }) as Arc<dyn Agent>,
        DeviceStore::new(pool),
    )
}

fn app(pool: PgPool) -> tower_http::normalize_path::NormalizePath<axum::Router> {
    let auth = auth_state(pool.clone());
    dashachun::app(
        state_with_url(pool, Some("ws://configured/gateway".into())),
        auth,
    )
}

fn ota_request(uri: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("host", "192.168.1.50:3000")
        .header("content-type", "application/json")
        .header("device-id", "aa:bb:cc:dd:ee:ff")
        .header("client-id", CLIENT_ID)
        .body(Body::from(
            r#"{"application":{"version":"1.0.0"},"board":{"type":"test-board"}}"#,
        ))
        .unwrap()
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn root_returns_ok(pool: PgPool) {
    let res = app(pool)
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn ota_uses_configured_websocket_url(pool: PgPool) {
    let res = app(pool).oneshot(ota_request("/api/ota")).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["websocket"]["url"], "ws://configured/gateway");
    assert_eq!(v["websocket"]["token"], "");
    assert_eq!(v["server_time"]["timezone_offset"], 480);
    assert_eq!(v["firmware"]["version"], "1.0.0");
    assert!(v["activation"]["code"].is_string());
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn ota_derives_websocket_url_from_host(pool: PgPool) {
    let auth = auth_state(pool.clone());
    let state = state_with_url(pool, None);
    let res = dashachun::app(state, auth)
        .oneshot(ota_request("/api/ota"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["websocket"]["url"], "ws://192.168.1.50:3000/gateway");
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn ota_trailing_slash_is_normalized(pool: PgPool) {
    let res = app(pool).oneshot(ota_request("/api/ota/")).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn ota_missing_headers_is_bad_request(pool: PgPool) {
    let req = Request::builder()
        .method("POST")
        .uri("/api/ota")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let res = app(pool).oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn unknown_route_is_not_found(pool: PgPool) {
    let res = app(pool)
        .oneshot(
            Request::builder()
                .uri("/does-not-exist")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn shutdown_closes_websocket_and_serve_returns(pool: PgPool) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let token = bound_device_token(&pool).await;

    let auth = auth_state(pool.clone());
    let state = state_with_url(pool, None);
    let shutdown_tx = state.shutdown_sender();
    let shutdown_rx = state.shutdown_signal();
    let app = dashachun::app(state, auth);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let grace = std::time::Duration::from_secs(5);
    let server = tokio::spawn(dashachun::serve(listener, app, shutdown_rx, grace));

    let mut request =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(format!(
            "ws://{addr}/gateway"
        ))
        .unwrap();
    request
        .headers_mut()
        .insert("client-id", HeaderValue::from_static(CLIENT_ID));
    request.headers_mut().insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
    );

    let (mut ws, _) = tokio_tungstenite::connect_async(request).await.unwrap();
    ws.send(Message::text(
        r#"{"type":"hello","version":1,"transport":"websocket","features":{"mcp":false,"aec":false}}"#,
    ))
    .await
    .unwrap();

    let hello = ws.next().await.unwrap().unwrap();
    assert!(matches!(hello, Message::Text(_)));

    shutdown_tx.send(true).unwrap();

    let mut closed = false;
    while let Some(message) = ws.next().await {
        match message {
            Ok(Message::Close(_)) => {
                closed = true;
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    assert!(closed, "client did not receive a close frame");

    tokio::time::timeout(grace, server)
        .await
        .expect("serve did not return within grace")
        .unwrap();
}

async fn bound_device_token(pool: &PgPool) -> String {
    let devices = DeviceStore::new(pool.clone());
    let uuid = uuid::Uuid::parse_str(CLIENT_ID).unwrap();
    devices
        .upsert(uuid, "aa:bb:cc:dd:ee:ff", "test-board")
        .await
        .unwrap();

    let user_id: i64 = sqlx::query_scalar(
        "INSERT INTO users (username, password_hash) VALUES ($1, 'x') RETURNING id",
    )
    .bind(format!("u{}", uuid::Uuid::new_v4().simple()))
    .fetch_one(pool)
    .await
    .unwrap();

    sqlx::query("UPDATE devices SET user_id = $1, activated_at = now() WHERE client_id = $2")
        .bind(user_id)
        .bind(uuid)
        .execute(pool)
        .await
        .unwrap();

    devices.rotate_token(uuid).await.unwrap().unwrap()
}
