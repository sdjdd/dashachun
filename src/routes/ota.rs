use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderMap, StatusCode};
use axum::{Json, Router, extract::State, routing::post};
use tracing::{debug, info};
use uuid::Uuid;

use crate::{
    dto::ota::{Activation, Firmware, OtaRequest, OtaResponse, ServerTime, Websocket},
    error::AppError,
    extract::{ClientId, DeviceId},
    state::ServerState,
};

pub fn routes() -> Router<ServerState> {
    Router::new()
        .route("/", post(handle_ota))
        .route("/activate", post(handle_activate))
}

async fn handle_ota(
    State(state): State<ServerState>,
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

    let (token, activation) = match &state.devices {
        Some(devices) => {
            let uuid = parse_client_id(&client_id)?;
            devices.upsert(uuid, &device_id, &board_type).await?;
            if devices.is_bound(uuid).await? {
                (devices.rotate_token(uuid).await?.unwrap_or_default(), None)
            } else {
                let code = devices
                    .ensure_code(uuid, state.config.device.activation_ttl_secs)
                    .await?;
                (
                    String::new(),
                    Some(Activation {
                        message: code.clone(),
                        challenge: Uuid::new_v4().simple().to_string(),
                        code,
                        timeout_ms: (state.config.device.activation_ttl_secs as u64) * 1000,
                    }),
                )
            }
        }
        None => (state.config.ota.token.clone(), None),
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
            token,
        },
        activation,
    }))
}

async fn handle_activate(
    State(state): State<ServerState>,
    ClientId(client_id): ClientId,
) -> Result<StatusCode, AppError> {
    let Some(devices) = &state.devices else {
        return Ok(StatusCode::OK);
    };
    let uuid = parse_client_id(&client_id)?;
    if devices.is_bound(uuid).await? {
        Ok(StatusCode::OK)
    } else {
        Ok(StatusCode::ACCEPTED)
    }
}

fn parse_client_id(value: &str) -> Result<Uuid, AppError> {
    Uuid::parse_str(value).map_err(|_| AppError::InvalidHeader("client-id"))
}
