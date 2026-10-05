use axum::Router;
use axum::extract::State;
use axum::extract::ws::WebSocketUpgrade;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tracing::{info, warn};
use uuid::Uuid;

use crate::gateway::Gateway;
use crate::state::ServerState;

pub fn routes() -> Router<ServerState> {
    Router::new().route("/", get(handle_device_connect))
}

async fn handle_device_connect(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
    headers: HeaderMap,
) -> Response {
    info!(
        device_id = header_str(&headers, "device-id"),
        client_id = header_str(&headers, "client-id"),
        user_agent = header_str(&headers, "user-agent"),
        "gateway upgrade"
    );

    if let Some(devices) = &state.devices {
        let Some(client_id) = header_str(&headers, "client-id") else {
            return StatusCode::UNAUTHORIZED.into_response();
        };
        let Ok(uuid) = Uuid::parse_str(client_id) else {
            return StatusCode::UNAUTHORIZED.into_response();
        };
        let Some(token) = bearer_token(&headers) else {
            return StatusCode::UNAUTHORIZED.into_response();
        };
        match devices.verify(uuid, token).await {
            Ok(true) => {}
            Ok(false) => {
                warn!(client_id, "gateway rejected unbound or invalid token");
                return StatusCode::UNAUTHORIZED.into_response();
            }
            Err(err) => {
                warn!(%err, "gateway token verification failed");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        }
    }

    let shutdown = state.shutdown_signal();
    ws.on_upgrade(move |socket| Gateway::new(state, shutdown).run(socket))
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = header_str(headers, "authorization")?;
    value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
        .map(str::trim)
        .filter(|token| !token.is_empty())
}

fn header_str<'a>(headers: &'a HeaderMap, name: &'static str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}
