use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, put};
use axum::{Json, Router};

use crate::agent::AgentStore;
use crate::auth::extract::AuthUser;
use crate::auth::state::AuthState;
use crate::error::{AppError, ErrorResponse};
use crate::extract::Validated;
use crate::routes::agents::AgentsState;
use crate::state::ServerState;
use crate::voiceprint::VoiceprintStore;
use crate::voiceprint::dto::{CreateVoiceprintArgs, UpdateVoiceprintArgs, VoiceprintResponse};
use crate::voiceprint::enroll;

pub fn routes() -> Router<AgentsState> {
    Router::new()
        .route(
            "/{agent_id}/voiceprints",
            get(list_voiceprints).post(create_voiceprint),
        )
        .route(
            "/{agent_id}/voiceprints/{voiceprint_id}",
            put(update_voiceprint).delete(delete_voiceprint),
        )
}

#[utoipa::path(
    get,
    path = "/api/agents/{agent_id}/voiceprints",
    tag = "voiceprints",
    security(("cookieAuth" = [])),
    params(("agent_id" = i64, Path, description = "Id of the agent owning the voiceprints")),
    responses(
        (status = 200, body = [VoiceprintResponse], description = "The agent's voiceprints, without the embeddings"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
    ),
)]
pub(crate) async fn list_voiceprints(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
) -> Result<Json<Vec<VoiceprintResponse>>, AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let voiceprints = VoiceprintStore::new(auth.pool.clone())
        .list_for_agent(agent_id, user.id)
        .await?;
    Ok(Json(
        voiceprints
            .into_iter()
            .map(VoiceprintResponse::from)
            .collect(),
    ))
}

// Enrollment: the audio of the named user messages (captured by the audio
// feature) is fetched from the storage, decoded, embedded, and appended
// into the label's vector sum. Guards escalate with specificity —
// ownership 404, disabled audio capture 400, rejected messages 400 with
// per-message reasons, missing model 503 — so request-side problems are
// reported even while the feature is off.
#[utoipa::path(
    post,
    path = "/api/agents/{agent_id}/voiceprints",
    tag = "voiceprints",
    security(("cookieAuth" = [])),
    request_body = CreateVoiceprintArgs,
    params(("agent_id" = i64, Path, description = "Id of the agent to enroll the voiceprint under")),
    responses(
        (status = 201, body = VoiceprintResponse, description = "Voiceprint created or appended"),
        (status = 400, body = ErrorResponse, description = "Invalid payload, audio capture disabled, or rejected message ids"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
        (status = 503, body = ErrorResponse, description = "Voiceprint model not loaded"),
    ),
)]
pub(crate) async fn create_voiceprint(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
    Validated(args): Validated<CreateVoiceprintArgs>,
) -> Result<(StatusCode, Json<VoiceprintResponse>), AppError> {
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
        &args.label,
        args.description.as_deref(),
        &args.message_ids,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(record.into())))
}

#[utoipa::path(
    put,
    path = "/api/agents/{agent_id}/voiceprints/{voiceprint_id}",
    tag = "voiceprints",
    security(("cookieAuth" = [])),
    request_body = UpdateVoiceprintArgs,
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the voiceprint"),
        ("voiceprint_id" = i64, Path, description = "Id of the voiceprint"),
    ),
    responses(
        (status = 200, body = VoiceprintResponse, description = "Label and/or description updated"),
        (status = 400, body = ErrorResponse, description = "Invalid payload or an empty patch"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Voiceprint not owned by the caller"),
        (status = 409, body = ErrorResponse, description = "New label already taken on this agent"),
    ),
)]
pub(crate) async fn update_voiceprint(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path((agent_id, voiceprint_id)): Path<(i64, i64)>,
    Validated(args): Validated<UpdateVoiceprintArgs>,
) -> Result<Json<VoiceprintResponse>, AppError> {
    if args.label.is_none() && args.description.is_none() {
        return Err(AppError::BadRequest(
            "at least one of label or description is required".into(),
        ));
    }
    let record = VoiceprintStore::new(auth.pool.clone())
        .update(
            agent_id,
            voiceprint_id,
            user.id,
            args.label.as_deref(),
            args.description.as_deref(),
        )
        .await?
        .ok_or(AppError::NotFound)?;
    Ok(Json(record.into()))
}

#[utoipa::path(
    delete,
    path = "/api/agents/{agent_id}/voiceprints/{voiceprint_id}",
    tag = "voiceprints",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the voiceprint"),
        ("voiceprint_id" = i64, Path, description = "Id of the voiceprint"),
    ),
    responses(
        (status = 204, description = "Voiceprint deleted"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Voiceprint not owned by the caller"),
    ),
)]
pub(crate) async fn delete_voiceprint(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path((agent_id, voiceprint_id)): Path<(i64, i64)>,
) -> Result<StatusCode, AppError> {
    if VoiceprintStore::new(auth.pool.clone())
        .delete(agent_id, voiceprint_id, user.id)
        .await?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound)
    }
}
