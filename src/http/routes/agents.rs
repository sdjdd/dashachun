use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, put};
use axum::{Json, Router};

use crate::agent::AgentStore;
use crate::error::{AppError, ErrorResponse};
use crate::http::dto::agent::{AgentResponse, CreateAgentArgs, UpdateAgentArgs};
use crate::http::extract::Validated;
use crate::http::extract::auth::AuthUser;
use crate::http::session::AuthState;
use crate::http::state::AgentsState;

pub fn routes() -> Router<AgentsState> {
    Router::new()
        .route("/", get(list_agents).post(create_agent))
        .route("/{agent_id}", put(update_agent).delete(delete_agent))
        .merge(crate::http::routes::memories::routes())
        .merge(crate::http::routes::speakers::routes())
        .merge(crate::http::routes::conversations::routes())
        .merge(crate::http::routes::devices::routes())
}

#[utoipa::path(
    get,
    path = "/api/agents",
    tag = "agents",
    security(("cookieAuth" = [])),
    responses(
        (status = 200, body = [AgentResponse], description = "The caller's agents, in creation order"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
    ),
)]
pub(crate) async fn list_agents(
    State(state): State<AuthState>,
    user: AuthUser,
) -> Result<Json<Vec<AgentResponse>>, AppError> {
    let agents = AgentStore::new(state.pool.clone());
    let records = agents.list_for_user(user.id).await?;
    Ok(Json(records.into_iter().map(Into::into).collect()))
}

#[utoipa::path(
    post,
    path = "/api/agents",
    tag = "agents",
    security(("cookieAuth" = [])),
    request_body = CreateAgentArgs,
    responses(
        (status = 201, body = AgentResponse, description = "Agent created"),
        (status = 400, body = ErrorResponse, description = "Invalid payload"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
    ),
)]
pub(crate) async fn create_agent(
    State(state): State<AuthState>,
    user: AuthUser,
    Validated(args): Validated<CreateAgentArgs>,
) -> Result<(StatusCode, Json<AgentResponse>), AppError> {
    let agents = AgentStore::new(state.pool.clone());
    let record = agents
        .create(
            user.id,
            &args.name,
            args.persona_prompt.as_deref().unwrap_or_default(),
        )
        .await?;
    Ok((StatusCode::CREATED, Json(record.into())))
}

#[utoipa::path(
    put,
    path = "/api/agents/{agent_id}",
    tag = "agents",
    security(("cookieAuth" = [])),
    request_body = UpdateAgentArgs,
    params(("agent_id" = i64, Path, description = "Id of the agent to update")),
    responses(
        (status = 200, body = AgentResponse, description = "Name and/or persona prompt updated"),
        (status = 400, body = ErrorResponse, description = "Invalid payload or an empty patch"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
        (status = 409, body = ErrorResponse, description = "New name already taken by one of the caller's agents"),
    ),
)]
pub(crate) async fn update_agent(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
    Validated(args): Validated<UpdateAgentArgs>,
) -> Result<Json<AgentResponse>, AppError> {
    if args.name.is_none() && args.persona_prompt.is_none() {
        return Err(AppError::BadRequest(
            "at least one of name or persona_prompt is required".into(),
        ));
    }
    let record = AgentStore::new(auth.pool.clone())
        .update(
            agent_id,
            user.id,
            args.name.as_deref(),
            args.persona_prompt.as_deref(),
        )
        .await?
        .ok_or(AppError::NotFound)?;
    Ok(Json(record.into()))
}

// Deleting an agent removes everything scoped to it in one transaction —
// its devices' bindings, speakers, memory entries, conversations with their
// message and capture rows. The capture objects are deliberately left in
// their bucket: audio retention is the storage's periodic-reaping concern,
// not an app-side delete.
#[utoipa::path(
    delete,
    path = "/api/agents/{agent_id}",
    tag = "agents",
    security(("cookieAuth" = [])),
    params(("agent_id" = i64, Path, description = "Id of the agent to delete")),
    responses(
        (status = 204, description = "Agent and its sub-resources deleted"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
    ),
)]
pub(crate) async fn delete_agent(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
) -> Result<StatusCode, AppError> {
    if AgentStore::new(auth.pool.clone())
        .delete(agent_id, user.id)
        .await?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound)
    }
}
