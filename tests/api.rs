use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use xiaozhi_server_rs::agent::{Agent, CompositeAgent, ToolRegistry};
use xiaozhi_server_rs::asr::StubAsr;
use xiaozhi_server_rs::config::{AppConfig, DeviceConfig, OtaConfig, ServerConfig};
use xiaozhi_server_rs::state::ServerState;
use xiaozhi_server_rs::vad::{SileroVadFactory, VadConfig};

fn state() -> ServerState {
    state_with_url(Some("ws://configured/gateway".into()))
}

fn state_with_url(websocket_url: Option<String>) -> ServerState {
    ServerState::new(
        AppConfig {
            server: ServerConfig {
                bind_addr: "127.0.0.1:0".into(),
                playback_prebuffer_ms: 180,
                shutdown_grace_ms: 5000,
            },
            ota: OtaConfig {
                websocket_url,
                token: "test-token".into(),
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
    )
}

fn ota_request(uri: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("host", "192.168.1.50:3000")
        .header("content-type", "application/json")
        .header("device-id", "aa:bb:cc:dd:ee:ff")
        .header("client-id", "00000000-0000-0000-0000-000000000000")
        .body(Body::from(
            r#"{"application":{"version":"1.0.0"},"board":{"type":"test-board"}}"#,
        ))
        .unwrap()
}

#[tokio::test]
async fn root_returns_ok() {
    let res = xiaozhi_server_rs::app(state(), None)
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn ota_uses_configured_websocket_url() {
    let res = xiaozhi_server_rs::app(state(), None)
        .oneshot(ota_request("/api/ota"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["websocket"]["url"], "ws://configured/gateway");
    assert_eq!(v["websocket"]["token"], "test-token");
    assert_eq!(v["server_time"]["timezone_offset"], 480);
    assert_eq!(v["firmware"]["version"], "1.0.0");
    assert!(v.get("activation").is_none());
}

#[tokio::test]
async fn ota_derives_websocket_url_from_host() {
    let res = xiaozhi_server_rs::app(state_with_url(None), None)
        .oneshot(ota_request("/api/ota"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["websocket"]["url"], "ws://192.168.1.50:3000/gateway");
}

#[tokio::test]
async fn ota_trailing_slash_is_normalized() {
    let res = xiaozhi_server_rs::app(state(), None)
        .oneshot(ota_request("/api/ota/"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn ota_missing_headers_is_bad_request() {
    let req = Request::builder()
        .method("POST")
        .uri("/api/ota")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let res = xiaozhi_server_rs::app(state(), None)
        .oneshot(req)
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unknown_route_is_not_found() {
    let res = xiaozhi_server_rs::app(state(), None)
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

#[tokio::test]
async fn shutdown_closes_websocket_and_serve_returns() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let state = state_with_url(None);
    let shutdown_tx = state.shutdown_sender();
    let shutdown_rx = state.shutdown_signal();
    let app = xiaozhi_server_rs::app(state, None);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let grace = std::time::Duration::from_secs(5);
    let server = tokio::spawn(xiaozhi_server_rs::serve(listener, app, shutdown_rx, grace));

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/gateway"))
        .await
        .unwrap();
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
