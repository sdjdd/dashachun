use std::sync::Arc;

use axum::extract::{FromRef, State};
use axum::routing::get;
use axum::{Json, Router};

use crate::agent::AudioStorage;
use crate::auth::extract::AdminUser;
use crate::auth::state::AuthState;
use crate::config::OssConfig;
use crate::error::AppError;
use crate::oss::Oss;
use crate::settings::{self, AudioCaptureSettings};
use crate::state::ServerState;

/// Router state for the audio-capture settings API: session auth plus the
/// server handle whose storage slot the apply path swaps.
#[derive(Clone)]
pub struct AudioCaptureState {
    pub auth: AuthState,
    pub server: ServerState,
}

impl FromRef<AudioCaptureState> for AuthState {
    fn from_ref(state: &AudioCaptureState) -> Self {
        state.auth.clone()
    }
}

impl FromRef<AudioCaptureState> for ServerState {
    fn from_ref(state: &AudioCaptureState) -> Self {
        state.server.clone()
    }
}

pub fn routes() -> Router<AudioCaptureState> {
    Router::new().route("/", get(get_capture).put(put_capture))
}

async fn get_capture(
    State(auth): State<AuthState>,
    _user: AdminUser,
) -> Result<Json<AudioCaptureSettings>, AppError> {
    let capture = settings::load_audio_capture(&auth.pool)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(capture))
}

/// Stores and hot-applies the audio-capture section. Enabling runs the
/// storage check eagerly (build + `ensure_bucket`) before anything is
/// stored — a failed check keeps the current settings standing, exactly
/// like the provider build path. In-flight sessions keep their storage
/// clone across the swap.
async fn put_capture(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    _user: AdminUser,
    Json(capture): Json<AudioCaptureSettings>,
) -> Result<Json<AudioCaptureSettings>, AppError> {
    let bucket = capture.bucket.trim().to_owned();
    if capture.enabled && bucket.is_empty() {
        return Err(AppError::BadRequest(
            "bucket is required when audio capture is enabled".into(),
        ));
    }
    let capture = AudioCaptureSettings {
        enabled: capture.enabled,
        bucket,
    };
    let next: Option<Arc<dyn AudioStorage>> = if capture.enabled {
        let config = OssConfig::from_env(true)
            .map_err(AppError::BadRequest)?
            .ok_or_else(|| AppError::BadRequest("S3_ENDPOINT is not configured".into()))?;
        let oss = Oss::new(&config, &capture.bucket);
        oss.ensure_bucket()
            .await
            .map_err(|err| AppError::BadRequest(format!("storage check failed: {err}")))?;
        Some(Arc::new(oss))
    } else {
        None
    };
    settings::set_audio_capture(&auth.pool, &capture)
        .await
        .map_err(AppError::Internal)?;
    server.set_audio_storage(next);
    Ok(Json(capture))
}
