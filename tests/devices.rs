use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{HeaderValue, Request, StatusCode};
use axum_extra::extract::cookie::Key;
use sqlx::PgPool;
use tower::ServiceExt;

use xiaozhi_server_rs::auth::state::AuthState;
use xiaozhi_server_rs::config::{AppConfig, DeviceConfig, OtaConfig, ServerConfig};
use xiaozhi_server_rs::device::DeviceStore;
use xiaozhi_server_rs::state::ServerState;

const CLIENT_ID: &str = "11111111-1111-4111-8111-111111111111";

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

fn server_state() -> ServerState {
    ServerState::new(
        AppConfig {
            server: ServerConfig {
                bind_addr: "127.0.0.1:0".into(),
                playback_prebuffer_ms: 180,
                shutdown_grace_ms: 5000,
            },
            ota: OtaConfig {
                websocket_url: None,
                token: "test-token".into(),
                timezone_offset: 480,
            },
            device: DeviceConfig {
                activation_ttl_secs: 600,
            },
        },
        Arc::new(xiaozhi_server_rs::agent::CompositeAgent {
            asr: Arc::new(xiaozhi_server_rs::asr::StubAsr::default()),
            llm: None,
            tts: None,
            vad: Arc::new(xiaozhi_server_rs::vad::SileroVadFactory::new(
                xiaozhi_server_rs::vad::VadConfig::default(),
            )),
            tools: Arc::new(xiaozhi_server_rs::agent::ToolRegistry::new(Vec::new())),
        }),
    )
}

fn app(pool: PgPool) -> tower_http::normalize_path::NormalizePath<axum::Router> {
    xiaozhi_server_rs::app(server_state(), Some(auth_state(pool)))
}

fn ota_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/api/ota")
        .header("content-type", "application/json")
        .header("host", "127.0.0.1:3000")
        .header("device-id", "aa:bb:cc:dd:ee:ff")
        .header("client-id", CLIENT_ID)
        .body(Body::from(
            r#"{"application":{"version":"1.0.0"},"board":{"type":"test-board"}}"#,
        ))
        .unwrap()
}

fn activate_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/api/ota/activate")
        .header("client-id", CLIENT_ID)
        .body(Body::from("{}"))
        .unwrap()
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

fn get_with_cookie(path: &str, cookie: String) -> Request<Body> {
    Request::builder()
        .uri(path)
        .header("cookie", cookie)
        .body(Body::empty())
        .unwrap()
}

fn delete_with_cookie(path: &str, cookie: String) -> Request<Body> {
    Request::builder()
        .method("DELETE")
        .uri(path)
        .header("cookie", cookie)
        .body(Body::empty())
        .unwrap()
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
async fn ota_creates_device_and_returns_code(pool: PgPool) {
    let app = app(pool);
    let res = app.clone().oneshot(ota_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_json(res).await;
    let code = body["activation"]["code"]
        .as_str()
        .expect("activation code");
    assert_eq!(code.len(), 6);
    assert!(code.chars().all(|c| c.is_ascii_digit()));
    assert_eq!(body["activation"]["message"], code);
    assert!(body["activation"]["challenge"].is_string());
    assert_eq!(body["websocket"]["token"], "");

    // the same device re-polls OTA: the code is reused, not rotated
    let res = app.clone().oneshot(ota_request()).await.unwrap();
    let body = body_json(res).await;
    assert_eq!(body["activation"]["code"].as_str(), Some(code));

    let res = app.clone().oneshot(activate_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::ACCEPTED);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn bind_flow_issues_token_and_gateway_enforces(pool: PgPool) {
    let app = app(pool.clone());
    let code = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let cookie = register(&app).await;

    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}"}}"#),
            Some(cookie.clone()),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let device = body_json(res).await;
    assert_eq!(device["client_id"], CLIENT_ID);
    assert!(device.get("id").is_none());
    assert!(device["created_at"].as_str().unwrap().contains('T'));

    let res = app
        .clone()
        .oneshot(get_with_cookie("/api/devices", cookie.clone()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await.as_array().unwrap().len(), 1);

    assert_eq!(
        app.clone()
            .oneshot(activate_request())
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );

    let token = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        assert!(body.get("activation").is_none());
        body["websocket"]["token"].as_str().unwrap().to_owned()
    };
    assert_eq!(token.len(), 32);

    let state = server_state();
    let shutdown_tx = state.shutdown_sender();
    let shutdown_rx = state.shutdown_signal();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = xiaozhi_server_rs::app(state, Some(auth_state(pool.clone())));
    let server = tokio::spawn(xiaozhi_server_rs::serve(
        listener,
        served,
        shutdown_rx,
        std::time::Duration::from_secs(5),
    ));

    assert!(connect_gateway(addr, Some(&token)).await.is_ok());
    assert!(connect_gateway(addr, Some("wrong-token")).await.is_err());
    assert!(connect_gateway(addr, None).await.is_err());

    let res = app
        .clone()
        .oneshot(delete_with_cookie(
            &format!("/api/devices/{CLIENT_ID}"),
            cookie,
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    assert!(connect_gateway(addr, Some(&token)).await.is_err());

    let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
    assert!(body["activation"]["code"].is_string());
    assert_eq!(body["websocket"]["token"], "");

    shutdown_tx.send(true).unwrap();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), server).await;
}

async fn connect_gateway(
    addr: std::net::SocketAddr,
    token: Option<&str>,
) -> Result<(), tokio_tungstenite::tungstenite::Error> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let mut request = format!("ws://{addr}/gateway")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("client-id", HeaderValue::from_static(CLIENT_ID));
    if let Some(token) = token {
        request.headers_mut().insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
    }
    let (_ws, _) = tokio_tungstenite::connect_async(request).await?;
    Ok(())
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn expired_code_is_not_found_and_reissued(pool: PgPool) {
    let app = app(pool.clone());
    let old_code = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let client_id = uuid::Uuid::parse_str(CLIENT_ID).unwrap();
    DeviceStore::new(pool.clone())
        .upsert(client_id, "aa:bb:cc:dd:ee:ff", "test-board")
        .await
        .unwrap();
    sqlx::query(
        "UPDATE devices SET activation_code_expires_at = now() - interval '1 minute' \
         WHERE client_id = $1",
    )
    .bind(client_id)
    .execute(&pool)
    .await
    .unwrap();

    let cookie = register(&app).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{old_code}"}}"#),
            Some(cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
    let new_code = body["activation"]["code"].as_str().unwrap();
    assert_ne!(new_code, old_code);
}

#[sqlx::test]
#[ignore = "requires a running postgres; run with cargo test -- --ignored"]
async fn binding_taken_code_is_not_found(pool: PgPool) {
    let app = app(pool.clone());
    let code = {
        let body = body_json(app.clone().oneshot(ota_request()).await.unwrap()).await;
        body["activation"]["code"].as_str().unwrap().to_owned()
    };

    let cookie = register(&app).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}"}}"#),
            Some(cookie),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let other = register(&app).await;
    let res = app
        .clone()
        .oneshot(post_json(
            "/api/devices/activate",
            &format!(r#"{{"code":"{code}"}}"#),
            Some(other),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}
