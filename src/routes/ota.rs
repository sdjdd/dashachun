use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;
use axum::{Json, Router, extract::State, routing::post};
use tracing::{debug, info};

use crate::{
    dto::ota::{Firmware, OtaRequest, OtaResponse, ServerTime, Websocket},
    error::AppError,
    extract::{ClientId, DeviceId},
    state::AppState,
};

pub fn routes() -> Router<AppState> {
    Router::new().route("/api/ota", post(handle_ota))
}

async fn handle_ota(
    State(state): State<AppState>,
    DeviceId(device_id): DeviceId,
    ClientId(client_id): ClientId,
    headers: HeaderMap,
    Json(body): Json<OtaRequest>,
) -> Result<Json<OtaResponse>, AppError> {
    let version = body.application.version;
    let board_type = body.board.board_type;

    info!(device_id, client_id, %version, %board_type, "OTA request");

    let websocket_url = match &state.config.ota.websocket_url {
        Some(url) => url.clone(),
        None => {
            let host = headers
                .get(axum::http::header::HOST)
                .ok_or(AppError::MissingHeader("host"))?
                .to_str()
                .map_err(|_| AppError::InvalidHeader("host"))?;
            format!("ws://{host}/gateway")
        }
    };

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let firmware_version = if version.is_empty() {
        "0.0.0".to_string()
    } else {
        version
    };

    debug!(%websocket_url, firmware_version, "OTA response");

    Ok(Json(OtaResponse {
        server_time: ServerTime {
            timestamp,
            timezone_offset: state.config.ota.timezone_offset,
        },
        firmware: Firmware {
            version: firmware_version,
            url: String::new(),
        },
        websocket: Websocket {
            url: websocket_url,
            token: state.config.ota.token.clone(),
        },
    }))
}
