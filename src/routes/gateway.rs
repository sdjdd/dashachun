use axum::Router;
use axum::extract::State;
use axum::extract::ws::WebSocketUpgrade;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::get;
use tracing::info;

use crate::gateway::Gateway;
use crate::state::ServerState;

pub fn routes() -> Router<ServerState> {
    Router::new().route("/gateway", get(handle_device_connect))
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
    let shutdown = state.shutdown_signal();
    ws.on_upgrade(move |socket| Gateway::new(state, shutdown).run(socket))
}

fn header_str<'a>(headers: &'a HeaderMap, name: &'static str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}
