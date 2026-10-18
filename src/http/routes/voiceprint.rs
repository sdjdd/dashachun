use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};

use crate::error::{AppError, ErrorResponse};
use crate::http::extract::auth::AdminUser;
use crate::http::session::AuthState;
use crate::http::state::SettingsState;
use crate::settings::{self, VoiceprintSettings};

pub fn routes() -> Router<SettingsState> {
    Router::new().route("/", get(get_threshold).put(put_threshold))
}

#[utoipa::path(
    get,
    path = "/api/settings/voiceprint",
    tag = "settings",
    security(("cookieAuth" = [])),
    responses(
        (status = 200, body = VoiceprintSettings, description = "The standing voiceprint section — an absent row reads as the defaults"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
    ),
)]
pub(crate) async fn get_threshold(
    State(auth): State<AuthState>,
    _user: AdminUser,
) -> Result<Json<VoiceprintSettings>, AppError> {
    let voiceprint = settings::load_voiceprint(&auth.pool)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(voiceprint))
}

// Stores the voiceprint section. Nothing to apply eagerly: the factory
// reads the row once per device session, so the new threshold reaches
// every new session as it connects and in-flight sessions keep theirs —
// the same snapshot semantics as the speaker roster.
#[utoipa::path(
    put,
    path = "/api/settings/voiceprint",
    tag = "settings",
    security(("cookieAuth" = [])),
    request_body = VoiceprintSettings,
    responses(
        (status = 200, body = VoiceprintSettings, description = "Section stored — applies to every new session"),
        (status = 400, body = ErrorResponse, description = "Invalid section"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
    ),
)]
pub(crate) async fn put_threshold(
    State(auth): State<AuthState>,
    _user: AdminUser,
    Json(voiceprint): Json<VoiceprintSettings>,
) -> Result<Json<VoiceprintSettings>, AppError> {
    // The cosine domain is [-1, 1], but a negative threshold would match
    // every utterance against everything; only the useful half is accepted.
    // `contains` also rejects NaN.
    if !(0.0..=1.0).contains(&voiceprint.match_threshold) {
        return Err(AppError::BadRequest(
            "match_threshold must be within 0.0..=1.0".into(),
        ));
    }
    settings::set_voiceprint(&auth.pool, &voiceprint)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(voiceprint))
}
