use axum::Router;
use axum::extract::{Path, State};
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;

use crate::agent::AgentStore;
use crate::error::{AppError, ErrorResponse};
use crate::http::extract::auth::AuthUser;
use crate::http::session::AuthState;
use crate::http::state::AgentsState;
use crate::state::ServerState;

pub fn routes() -> Router<AgentsState> {
    Router::new().route(
        "/{agent_id}/messages/{message_id}/audio",
        get(message_audio),
    )
}

// Playback serves the captured bytes verbatim through the server: the
// files are small (≤ ~2 MB), the auth never leaves the process, and no
// storage endpoint has to be reachable by the client. The bucket comes
// from the row — it may predate a bucket switch — and the first audio
// row wins, matching how enrollment picks its capture. Guards escalate
// with specificity — ownership 404, missing capture 404, storage off
// 503 — so the response never turns a standing row into a lie.
#[utoipa::path(
    get,
    path = "/api/agents/{agent_id}/messages/{message_id}/audio",
    tag = "agents",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the message"),
        ("message_id" = i64, Path, description = "Id of the captured message"),
    ),
    responses(
        (status = 200, content_type = "audio/flac", description = "The captured FLAC audio"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller, or the message has no captured audio"),
        (status = 503, body = ErrorResponse, description = "Audio storage is not currently configured"),
    ),
)]
pub(crate) async fn message_audio(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    user: AuthUser,
    Path((agent_id, message_id)): Path<(i64, i64)>,
) -> Result<impl IntoResponse, AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let audio: Option<(String, String)> = sqlx::query_as(
        "SELECT ma.bucket, ma.object_key \
         FROM messages m \
         JOIN conversations c ON c.id = m.conversation_id \
         JOIN message_audios ma ON ma.message_id = m.id \
         WHERE c.agent_id = $1 AND m.id = $2 \
         ORDER BY ma.id \
         LIMIT 1",
    )
    .bind(agent_id)
    .bind(message_id)
    .fetch_optional(&auth.pool)
    .await?;
    let (bucket, object_key) = audio.ok_or(AppError::NotFound)?;
    let storage = server
        .audio_storage()
        .ok_or_else(|| AppError::ServiceUnavailable("audio storage is not configured".into()))?;
    let bytes = storage.get(&bucket, &object_key).await.map_err(|err| {
        tracing::warn!(bucket = %bucket, key = %object_key, %err, "audio playback read failed");
        AppError::Internal(format!("audio storage: {err}"))
    })?;
    Ok((
        [
            (header::CONTENT_TYPE, "audio/flac".to_owned()),
            (header::CACHE_CONTROL, "private".to_owned()),
        ],
        bytes,
    ))
}
