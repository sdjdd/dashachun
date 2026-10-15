use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{delete, get};
use axum::{Json, Router};

use crate::agent::{AgentStore, DbEntryMemory, EntryMemory};
use crate::error::{AppError, ErrorResponse};
use crate::http::dto::memory::MemoryResponse;
use crate::http::extract::auth::AuthUser;
use crate::http::session::AuthState;
use crate::http::state::AgentsState;

pub fn routes() -> Router<AgentsState> {
    Router::new()
        .route("/{agent_id}/memories", get(list_memories))
        .route("/{agent_id}/memories/{mem_no}", delete(delete_memory))
}

#[utoipa::path(
    get,
    path = "/api/agents/{agent_id}/memories",
    tag = "memories",
    security(("cookieAuth" = [])),
    params(("agent_id" = i64, Path, description = "Id of the agent owning the memory entries")),
    responses(
        (status = 200, body = [MemoryResponse], description = "The agent's memory entries, in prompt-catalog order"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
    ),
)]
pub(crate) async fn list_memories(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
) -> Result<Json<Vec<MemoryResponse>>, AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let entries = DbEntryMemory::new(auth.pool.clone(), user.id, agent_id)
        .list()
        .await;
    Ok(Json(
        entries.into_iter().map(MemoryResponse::from).collect(),
    ))
}

#[utoipa::path(
    delete,
    path = "/api/agents/{agent_id}/memories/{mem_no}",
    tag = "memories",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the memory entry"),
        ("mem_no" = i32, Path, description = "The entry's mem_no, as shown in the list and the prompt catalog"),
    ),
    responses(
        (status = 204, description = "Entry deleted — the agent sees it gone on its next turn"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent or entry not owned by the caller"),
    ),
)]
pub(crate) async fn delete_memory(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path((agent_id, mem_no)): Path<(i64, i32)>,
) -> Result<StatusCode, AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let memory = DbEntryMemory::new(auth.pool.clone(), user.id, agent_id);
    match memory.delete(mem_no).await {
        Ok(true) => Ok(StatusCode::NO_CONTENT),
        Ok(false) => Err(AppError::NotFound),
        Err(err) => Err(AppError::Internal(err)),
    }
}
