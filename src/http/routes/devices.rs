use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::{Json, Router, routing::delete, routing::get};
use uuid::Uuid;

use super::auth::WINDOW;
use crate::agent::AgentStore;
use crate::device::BindOutcome;
use crate::error::{AppError, ErrorResponse};
use crate::http::dto::device::{ActivateDeviceArgs, DeviceResponse};
use crate::http::extract::Validated;
use crate::http::extract::auth::AuthUser;
use crate::http::session::AuthState;
use crate::http::state::AgentsState;
use crate::ratelimit::Decision;
use crate::state::ServerState;

/// Bind attempts are a fixed window per account: the activation code is
/// six digits over a ten-minute TTL, so an unthrottled bind endpoint is a
/// brute-force surface over the pairing space. The account is the subject,
/// not the client IP — binding requires a login, and an account cap
/// survives IP rotation.
pub const BIND_LIMIT: u32 = 5;

pub fn routes() -> Router<AgentsState> {
    Router::new()
        .route(
            "/{agent_id}/devices",
            get(list_devices).post(activate_device),
        )
        .route("/{agent_id}/devices/{client_id}", delete(unbind_device))
}

#[utoipa::path(
    post,
    path = "/api/agents/{agent_id}/devices",
    tag = "devices",
    security(("cookieAuth" = [])),
    request_body = ActivateDeviceArgs,
    params(("agent_id" = i64, Path, description = "Id of the agent to bind the device under")),
    responses(
        (status = 200, body = DeviceResponse, description = "Device bound to the account"),
        (status = 400, body = ErrorResponse, description = "Invalid payload"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Unknown activation code or agent"),
        (status = 409, body = ErrorResponse, description = "Device already bound"),
        (status = 429, body = ErrorResponse, description = "Rate limited"),
    ),
)]
pub(crate) async fn activate_device(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
    Validated(args): Validated<ActivateDeviceArgs>,
) -> Result<Json<DeviceResponse>, AppError> {
    match auth
        .limiter
        .hit("bind", &user.id.to_string(), BIND_LIMIT, WINDOW)
        .await
    {
        Decision::Allowed => {}
        Decision::Limited { retry_after } => {
            return Err(AppError::TooManyRequests(retry_after.as_secs().max(1)));
        }
    }
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    match server
        .devices
        .bind_by_code(user.id, &args.code, agent_id)
        .await?
    {
        BindOutcome::Bound(record) => Ok(Json(record.into())),
        BindOutcome::NotFound | BindOutcome::UnknownAgent => Err(AppError::NotFound),
        BindOutcome::Conflict => Err(AppError::Conflict("device already bound".into())),
    }
}

#[utoipa::path(
    get,
    path = "/api/agents/{agent_id}/devices",
    tag = "devices",
    security(("cookieAuth" = [])),
    params(("agent_id" = i64, Path, description = "Id of the agent owning the devices")),
    responses(
        (status = 200, body = [DeviceResponse], description = "The agent's bound devices"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
    ),
)]
pub(crate) async fn list_devices(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
) -> Result<Json<Vec<DeviceResponse>>, AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let records = server.devices.list_for_agent(agent_id).await?;
    Ok(Json(
        records.into_iter().map(DeviceResponse::from).collect(),
    ))
}

#[utoipa::path(
    delete,
    path = "/api/agents/{agent_id}/devices/{client_id}",
    tag = "devices",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent the device is bound under"),
        ("client_id" = String, Path, description = "Client id of the device to unbind"),
    ),
    responses(
        (status = 204, description = "Device unbound"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Device not bound under this agent"),
    ),
)]
pub(crate) async fn unbind_device(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    user: AuthUser,
    Path((agent_id, client_id)): Path<(i64, String)>,
) -> Result<StatusCode, AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let client_id = Uuid::parse_str(&client_id).map_err(|_| AppError::NotFound)?;
    if server.devices.unbind(agent_id, client_id).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound)
    }
}
