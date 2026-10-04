use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;
use tracing::{info, warn};

use crate::state::AppState;

pub fn routes() -> Router<AppState> {
    Router::new().route("/gateway", get(handle_device_connect))
}

async fn handle_device_connect(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_device_socket(socket, state))
}

async fn handle_device_socket(mut socket: WebSocket, _state: AppState) {
    info!("device connected");
    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(Message::Close(_)) => break,
            Ok(msg) => msg,
            Err(err) => {
                warn!(%err, "websocket error");
                break;
            }
        };

        if let Message::Text(txt) = &msg {
            info!(%txt, "device message");
        }

        if socket.send(msg).await.is_err() {
            break;
        }
    }
    info!("device disconnected");
}
