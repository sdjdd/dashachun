use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};

use crate::error::{AppError, ErrorResponse};
use crate::http::extract::auth::AdminUser;
use crate::http::session::AuthState;
use crate::http::state::SettingsState;
use crate::settings::{self, OtaSettings};

pub fn routes() -> Router<SettingsState> {
    Router::new().route("/", get(get_ota).put(put_ota))
}

#[utoipa::path(
    get,
    path = "/api/settings/ota",
    tag = "settings",
    security(("cookieAuth" = [])),
    responses(
        (status = 200, body = OtaSettings, description = "The standing ota section — an absent row reads as the defaults"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
    ),
)]
pub(crate) async fn get_ota(
    State(auth): State<AuthState>,
    _user: AdminUser,
) -> Result<Json<OtaSettings>, AppError> {
    let ota = settings::load_ota(&auth.pool)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(ota))
}

// Stores the ota section. Not hot-applied: the section is resolved once at
// boot, so a change takes effect on restart — the response says stored, the
// operator restarts.
#[utoipa::path(
    put,
    path = "/api/settings/ota",
    tag = "settings",
    security(("cookieAuth" = [])),
    request_body = OtaSettings,
    responses(
        (status = 200, body = OtaSettings, description = "Section stored — takes effect on the next restart"),
        (status = 400, body = ErrorResponse, description = "Invalid section"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
    ),
)]
pub(crate) async fn put_ota(
    State(auth): State<AuthState>,
    _user: AdminUser,
    Json(ota): Json<OtaSettings>,
) -> Result<Json<OtaSettings>, AppError> {
    if !(-720..=840).contains(&ota.timezone_offset) {
        return Err(AppError::BadRequest(
            "timezone_offset must be within -720..=840".into(),
        ));
    }
    if let Some(url) = &ota.websocket_url
        && !(url.starts_with("ws://") || url.starts_with("wss://"))
    {
        return Err(AppError::BadRequest(
            "websocket_url must use the ws:// or wss:// scheme".into(),
        ));
    }
    settings::set_ota(&auth.pool, &ota)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(ota))
}
