use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;
use tracing::{info, warn};
use uuid::Uuid;

use crate::dto::ws::{AudioParams, ClientHello, ServerHello};
use crate::state::AppState;

pub fn routes() -> Router<AppState> {
    Router::new().route("/gateway", get(handle_device_connect))
}

async fn handle_device_connect(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_device_socket(socket, state))
}

async fn handle_device_socket(mut socket: WebSocket, _state: AppState) {
    info!("device connected");
    let mut session_id: Option<String> = None;

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(Message::Close(_)) => break,
            Ok(msg) => msg,
            Err(err) => {
                warn!(%err, "websocket error");
                break;
            }
        };

        let Message::Text(txt) = &msg else {
            info!("non-text message ignored");
            continue;
        };

        let value: serde_json::Value = match serde_json::from_str(txt.as_str()) {
            Ok(value) => value,
            Err(err) => {
                warn!(%err, "invalid json message");
                continue;
            }
        };

        match value.get("type").and_then(|t| t.as_str()) {
            Some("hello") => {
                let hello: ClientHello = match serde_json::from_value(value) {
                    Ok(hello) => hello,
                    Err(err) => {
                        warn!(%err, "invalid hello message");
                        continue;
                    }
                };
                let id = Uuid::new_v4().to_string();
                info!(
                    session_id = %id,
                    version = hello.version,
                    transport = %hello.transport,
                    mcp = hello.features.mcp,
                    aec = hello.features.aec,
                    "device hello"
                );

                let reply = ServerHello::new(id.clone(), AudioParams::default());
                let Ok(payload) = serde_json::to_string(&reply) else {
                    warn!("failed to serialize server hello");
                    continue;
                };
                if socket.send(Message::text(payload)).await.is_err() {
                    break;
                }
                session_id = Some(id);
            }
            Some(other) => {
                info!(session_id = ?session_id, %other, "unhandled message");
            }
            None => {
                warn!(%txt, "message missing type");
            }
        }
    }
    info!("device disconnected");
}
