use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use xiaozhi_server_rs::asr::StubAsr;
use xiaozhi_server_rs::config::{AppConfig, OtaConfig, ServerConfig};
use xiaozhi_server_rs::state::AppState;

fn state() -> AppState {
    state_with_url(Some("ws://configured/gateway".into()))
}

fn state_with_url(websocket_url: Option<String>) -> AppState {
    AppState {
        config: AppConfig {
            server: ServerConfig {
                bind_addr: "127.0.0.1:0".into(),
            },
            ota: OtaConfig {
                websocket_url,
                token: "test-token".into(),
                timezone_offset: 480,
            },
        },
        asr: Arc::new(StubAsr::default()),
    }
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
    let res = xiaozhi_server_rs::app(state())
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn ota_uses_configured_websocket_url() {
    let res = xiaozhi_server_rs::app(state())
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
}

#[tokio::test]
async fn ota_derives_websocket_url_from_host() {
    let res = xiaozhi_server_rs::app(state_with_url(None))
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
    let res = xiaozhi_server_rs::app(state())
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
    let res = xiaozhi_server_rs::app(state()).oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unknown_route_is_not_found() {
    let res = xiaozhi_server_rs::app(state())
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
