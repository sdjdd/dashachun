use axum::Router;
use axum::extract::State;
use axum::extract::ws::WebSocketUpgrade;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tracing::{info, warn};
use uuid::Uuid;

use crate::device::{self};
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

    let Some(client_id) = header_str(&headers, "client-id") else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let Ok(uuid) = Uuid::parse_str(client_id) else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let Some(token) = bearer_token(&headers) else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let record = match state.devices.verify(uuid, token).await {
        Ok(Some(record)) => record,
        Ok(None) => {
            warn!(client_id, "gateway rejected unbound or invalid token");
            return StatusCode::UNAUTHORIZED.into_response();
        }
        Err(err) => {
            warn!(%err, "gateway token verification failed");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let Some(factory) = state.agent_factory() else {
        warn!(
            client_id,
            "gateway rejected: no provider generation configured"
        );
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let Some(agent_id) = record.agent_id else {
        warn!(client_id, "gateway rejected: device has no assigned agent");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let agent = match factory.build(&record, state.audio_storage()).await {
        Ok(agent) => agent,
        Err(err) => {
            warn!(client_id, %err, "gateway rejected: agent unavailable");
            return err.into_response();
        }
    };

    // The conversation row is the ownership anchor for everything the
    // connection writes, so a failure to record it rejects the upgrade.
    // The wire `session_id` is assigned here too — the value the hello ack
    // announces and the device echoes — and recorded on the row for
    // correlation; it never keys anything server-side.
    let session_id = Uuid::new_v4().to_string();
    let conversation_id =
        match device::conversation::create(state.devices.pool(), &session_id, record.id, agent_id)
            .await
        {
            Ok(id) => id,
            Err(err) => {
                warn!(client_id, %err, "gateway rejected: conversation could not be recorded");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        };

    let shutdown = state.shutdown_signal();
    let pool = state.devices.pool().clone();
    ws.on_upgrade(move |socket| async move {
        Gateway::new(state, agent, shutdown, session_id, conversation_id)
            .run(socket)
            .await;
        device::conversation::end(&pool, conversation_id).await;
    })
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
