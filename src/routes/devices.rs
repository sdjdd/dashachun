use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::{Json, Router, routing::delete, routing::get, routing::post};
use uuid::Uuid;

use crate::auth::extract::{AuthUser, Validated};
use crate::auth::state::AuthState;
use crate::device::BindOutcome;
use crate::device::DeviceStore;
use crate::device::dto::{ActivateDeviceArgs, DeviceResponse};
use crate::error::AppError;

pub fn routes() -> Router<AuthState> {
    Router::new()
        .route("/", get(list_devices))
        .route("/activate", post(activate_device))
        .route("/{client_id}", delete(unbind_device))
}

async fn activate_device(
    State(state): State<AuthState>,
    user: AuthUser,
    Validated(args): Validated<ActivateDeviceArgs>,
) -> Result<Json<DeviceResponse>, AppError> {
    let devices = DeviceStore::new(state.pool.clone());
    match devices
        .bind_by_code(user.id, &args.code, args.agent_id)
        .await?
    {
        BindOutcome::Bound(record) => Ok(Json(record.into())),
        BindOutcome::NotFound => Err(AppError::NotFound),
        BindOutcome::UnknownAgent => Err(AppError::NotFound),
        BindOutcome::Conflict => Err(AppError::Conflict("device already bound".into())),
    }
}

async fn list_devices(
    State(state): State<AuthState>,
    user: AuthUser,
) -> Result<Json<Vec<DeviceResponse>>, AppError> {
    let devices = DeviceStore::new(state.pool.clone());
    let records = devices.list_for_user(user.id).await?;
    Ok(Json(
        records.into_iter().map(DeviceResponse::from).collect(),
    ))
}

async fn unbind_device(
    State(state): State<AuthState>,
    user: AuthUser,
    Path(client_id): Path<String>,
) -> Result<StatusCode, AppError> {
    let client_id = Uuid::parse_str(&client_id).map_err(|_| AppError::NotFound)?;
    let devices = DeviceStore::new(state.pool.clone());
    if devices.unbind(user.id, client_id).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound)
    }
}
