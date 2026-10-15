use axum::extract::State;
use axum::http::StatusCode;
use axum::{Json, Router, routing::post};

use crate::agent::AgentStore;
use crate::error::{AppError, ErrorResponse};
use crate::http::dto::agent::{AgentResponse, CreateAgentArgs};
use crate::http::extract::Validated;
use crate::http::extract::auth::AuthUser;
use crate::http::session::AuthState;
use crate::http::state::AgentsState;

pub fn routes() -> Router<AgentsState> {
    Router::new()
        .route("/", post(create_agent))
        .merge(crate::http::routes::speakers::routes())
        .merge(crate::http::routes::messages::routes())
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
