use axum::extract::State;
use axum::http::StatusCode;
use axum::{Json, Router, routing::post};

use crate::agent::AgentStore;
use crate::agent::dto::{AgentResponse, CreateAgentArgs};
use crate::auth::extract::AuthUser;
use crate::auth::state::AuthState;
use crate::error::AppError;
use crate::extract::Validated;

pub fn routes() -> Router<AuthState> {
    Router::new().route("/", post(create_agent))
}

async fn create_agent(
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
