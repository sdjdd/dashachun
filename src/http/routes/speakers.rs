use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, put};
use axum::{Json, Router};

use crate::agent::AgentStore;
use crate::error::{AppError, ErrorResponse};
use crate::http::dto::speaker::{CreateSpeakerArgs, SpeakerResponse, UpdateSpeakerArgs};
use crate::http::extract::Validated;
use crate::http::extract::auth::AuthUser;
use crate::http::session::AuthState;
use crate::http::state::AgentsState;
use crate::state::ServerState;
use crate::voiceprint::SpeakerStore;
use crate::voiceprint::enroll;

pub fn routes() -> Router<AgentsState> {
    Router::new()
        .route(
            "/{agent_id}/speakers",
            get(list_speakers).post(create_speaker),
        )
        .route(
            "/{agent_id}/speakers/{speaker_id}",
            put(update_speaker).delete(delete_speaker),
        )
}

#[utoipa::path(
    get,
    path = "/api/agents/{agent_id}/speakers",
    tag = "speakers",
    security(("cookieAuth" = [])),
    params(("agent_id" = i64, Path, description = "Id of the agent owning the speakers")),
    responses(
        (status = 200, body = [SpeakerResponse], description = "The agent's speakers, without the embeddings"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
    ),
)]
pub(crate) async fn list_speakers(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
) -> Result<Json<Vec<SpeakerResponse>>, AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let speakers = SpeakerStore::new(auth.pool.clone())
        .list_for_agent(agent_id, user.id)
        .await?;
    Ok(Json(
        speakers.into_iter().map(SpeakerResponse::from).collect(),
    ))
}

// Enrollment: the audio of the named user messages (captured by the audio
// feature) is fetched from the storage, decoded, embedded, and appended
// into the name's vector sum. Guards escalate with specificity —
// ownership 404, disabled audio capture 400, rejected messages 400 with
// per-message reasons, missing model 503 — so request-side problems are
// reported even while the feature is off.
#[utoipa::path(
    post,
    path = "/api/agents/{agent_id}/speakers",
    tag = "speakers",
    security(("cookieAuth" = [])),
    request_body = CreateSpeakerArgs,
    params(("agent_id" = i64, Path, description = "Id of the agent to enroll the speaker under")),
    responses(
        (status = 201, body = SpeakerResponse, description = "Speaker created or appended"),
        (status = 400, body = ErrorResponse, description = "Invalid payload, audio capture disabled, or rejected message ids"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
        (status = 503, body = ErrorResponse, description = "Voiceprint model not loaded"),
    ),
)]
pub(crate) async fn create_speaker(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
    Validated(args): Validated<CreateSpeakerArgs>,
) -> Result<(StatusCode, Json<SpeakerResponse>), AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let record = enroll(
        &auth.pool,
        server.audio_storage(),
        server.voiceprint(),
        agent_id,
        user.id,
        &args.name,
        args.description.as_deref(),
        &args.message_ids,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(record.into())))
}

#[utoipa::path(
    put,
    path = "/api/agents/{agent_id}/speakers/{speaker_id}",
    tag = "speakers",
    security(("cookieAuth" = [])),
    request_body = UpdateSpeakerArgs,
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the speaker"),
        ("speaker_id" = i64, Path, description = "Id of the speaker"),
    ),
    responses(
        (status = 200, body = SpeakerResponse, description = "Name and/or description updated"),
        (status = 400, body = ErrorResponse, description = "Invalid payload or an empty patch"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Speaker not owned by the caller"),
        (status = 409, body = ErrorResponse, description = "New name already taken on this agent"),
    ),
)]
pub(crate) async fn update_speaker(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path((agent_id, speaker_id)): Path<(i64, i64)>,
    Validated(args): Validated<UpdateSpeakerArgs>,
) -> Result<Json<SpeakerResponse>, AppError> {
    if args.name.is_none() && args.description.is_none() {
        return Err(AppError::BadRequest(
            "at least one of name or description is required".into(),
        ));
    }
    let record = SpeakerStore::new(auth.pool.clone())
        .update(
            agent_id,
            speaker_id,
            user.id,
            args.name.as_deref(),
            args.description.as_deref(),
        )
        .await?
        .ok_or(AppError::NotFound)?;
    Ok(Json(record.into()))
}

#[utoipa::path(
    delete,
    path = "/api/agents/{agent_id}/speakers/{speaker_id}",
    tag = "speakers",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the speaker"),
        ("speaker_id" = i64, Path, description = "Id of the speaker"),
    ),
    responses(
        (status = 204, description = "Speaker deleted"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Speaker not owned by the caller"),
    ),
)]
pub(crate) async fn delete_speaker(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path((agent_id, speaker_id)): Path<(i64, i64)>,
) -> Result<StatusCode, AppError> {
    if SpeakerStore::new(auth.pool.clone())
        .delete(agent_id, speaker_id, user.id)
        .await?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound)
    }
}
