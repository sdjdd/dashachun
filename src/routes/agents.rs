use axum::extract::{FromRef, State};
use axum::http::StatusCode;
use axum::{Json, Router, routing::post};

use crate::agent::AgentStore;
use crate::agent::dto::{AgentResponse, CreateAgentArgs};
use crate::auth::extract::AuthUser;
use crate::auth::state::AuthState;
use crate::error::{AppError, ErrorResponse};
use crate::extract::Validated;
use crate::state::ServerState;

/// The agent REST routes need both halves: session auth and the server
/// whose audio-storage and voiceprint-model slots the enrollment POST reads
/// (the speakers sub-routes merge in here, so one state shape carries
/// both).
#[derive(Clone)]
pub struct AgentsState {
    pub auth: AuthState,
    pub server: ServerState,
}

impl FromRef<AgentsState> for AuthState {
    fn from_ref(state: &AgentsState) -> Self {
        state.auth.clone()
    }
}

impl FromRef<AgentsState> for ServerState {
    fn from_ref(state: &AgentsState) -> Self {
        state.server.clone()
    }
}

pub fn routes() -> Router<AgentsState> {
    Router::new()
        .route("/", post(create_agent))
        .merge(crate::routes::speakers::routes())
        .merge(crate::routes::messages::routes())
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
