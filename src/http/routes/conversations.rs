use axum::Json;
use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::{delete, get};

use crate::agent::AgentStore;
use crate::agent::read_transcript;
use crate::device::conversation;
use crate::error::{AppError, ErrorResponse};
use crate::http::dto::conversation::{
    ConversationPage, ConversationResponse, MessagePage, MessageResponse, PageArgs,
};
use crate::http::extract::auth::AuthUser;
use crate::http::session::AuthState;
use crate::http::state::AgentsState;
use crate::state::ServerState;

pub fn routes() -> Router<AgentsState> {
    Router::new()
        .route("/{agent_id}/conversations", get(list_conversations))
        .route(
            "/{agent_id}/conversations/{conversation_id}",
            delete(delete_conversation),
        )
        .route(
            "/{agent_id}/conversations/{conversation_id}/messages",
            get(list_messages),
        )
        .route(
            "/{agent_id}/conversations/{conversation_id}/messages/{message_id}/audio",
            get(message_audio),
        )
}

const DEFAULT_PAGE_LIMIT: i32 = 50;
const MAX_PAGE_LIMIT: i32 = 200;

// The paged reads share one convention: keyset pagination on the row id,
// newest first. `before_id` is the previous page's last item id (an
// exclusive upper bound), `limit` must sit in 1..=200 — a negative LIMIT
// is unbounded SQL, zero an empty window nobody asked for.
fn page(args: PageArgs) -> Result<(Option<i64>, i64), AppError> {
    let limit = args.limit.unwrap_or(DEFAULT_PAGE_LIMIT);
    if !(1..=MAX_PAGE_LIMIT).contains(&limit) {
        return Err(AppError::BadRequest(format!(
            "limit must be between 1 and {MAX_PAGE_LIMIT}"
        )));
    }
    Ok((args.before_id, limit as i64))
}

// The read side of the durable conversation record: the agent scoping
// rides the conversation's binding snapshot (`agent_id` at hello), so a
// later re-bind never rewires what the past conversations belong to.
#[utoipa::path(
    get,
    path = "/api/agents/{agent_id}/conversations",
    tag = "conversations",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the conversations"),
        ("before_id" = Option<i64>, Query, description = "Exclusive upper id bound — the previous page's last item id"),
        ("limit" = Option<i32>, Query, description = "Page size, 1–200 (default 50)"),
    ),
    responses(
        (status = 200, body = ConversationPage, description = "The agent's conversations, newest first"),
        (status = 400, body = ErrorResponse, description = "Invalid pagination"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller"),
    ),
)]
pub(crate) async fn list_conversations(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path(agent_id): Path<i64>,
    Query(args): Query<PageArgs>,
) -> Result<Json<ConversationPage>, AppError> {
    let (before_id, limit) = page(args)?;
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    let rows = conversation::list(&auth.pool, agent_id, before_id, limit + 1).await?;
    let has_more = rows.len() as i64 > limit;
    Ok(Json(ConversationPage {
        items: rows
            .into_iter()
            .take(limit as usize)
            .map(ConversationResponse::from)
            .collect(),
        has_more,
    }))
}

#[utoipa::path(
    get,
    path = "/api/agents/{agent_id}/conversations/{conversation_id}/messages",
    tag = "conversations",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the conversation"),
        ("conversation_id" = i64, Path, description = "Id of the conversation"),
        ("before_id" = Option<i64>, Query, description = "Exclusive upper id bound — the previous page's last item id"),
        ("limit" = Option<i32>, Query, description = "Page size, 1–200 (default 50)"),
    ),
    responses(
        (status = 200, body = MessagePage, description = "The conversation's messages, newest first"),
        (status = 400, body = ErrorResponse, description = "Invalid pagination"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller, or the conversation belongs to another agent"),
    ),
)]
pub(crate) async fn list_messages(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path((agent_id, conversation_id)): Path<(i64, i64)>,
    Query(args): Query<PageArgs>,
) -> Result<Json<MessagePage>, AppError> {
    let (before_id, limit) = page(args)?;
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    if !conversation::exists_for_agent(&auth.pool, agent_id, conversation_id).await? {
        return Err(AppError::NotFound);
    }
    let rows = read_transcript(&auth.pool, conversation_id, before_id, limit + 1).await?;
    let has_more = rows.len() as i64 > limit;
    Ok(Json(MessagePage {
        items: rows
            .into_iter()
            .take(limit as usize)
            .map(MessageResponse::from)
            .collect(),
        has_more,
    }))
}

// The privacy deletion of the record: one conversation with its message
// and capture rows goes in one transaction, the conversation row itself
// scoped by the binding-snapshot agent as the ownership guard. The capture
// objects are deliberately left in their bucket — audio retention is the
// storage's periodic-reaping concern, not an app-side delete.
#[utoipa::path(
    delete,
    path = "/api/agents/{agent_id}/conversations/{conversation_id}",
    tag = "conversations",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the conversation"),
        ("conversation_id" = i64, Path, description = "Id of the conversation to delete"),
    ),
    responses(
        (status = 204, description = "Conversation deleted with its message and capture rows"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller, or the conversation belongs to another agent"),
    ),
)]
pub(crate) async fn delete_conversation(
    State(auth): State<AuthState>,
    user: AuthUser,
    Path((agent_id, conversation_id)): Path<(i64, i64)>,
) -> Result<StatusCode, AppError> {
    if !AgentStore::new(auth.pool.clone())
        .owns(agent_id, user.id)
        .await?
    {
        return Err(AppError::NotFound);
    }
    if conversation::delete(&auth.pool, agent_id, conversation_id).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound)
    }
}

// Playback serves the captured bytes verbatim through the server: the
// files are small (≤ ~2 MB), the auth never leaves the process, and no
// storage endpoint has to be reachable by the client. The bucket comes
// from the row — it may predate a bucket switch — and the first audio
// row wins, matching how enrollment picks its capture. Guards escalate
// with specificity — ownership 404, conversation or message out of scope
// 404, missing capture 404, storage off 503 — so the response never
// turns a standing row into a lie.
#[utoipa::path(
    get,
    path = "/api/agents/{agent_id}/conversations/{conversation_id}/messages/{message_id}/audio",
    tag = "conversations",
    security(("cookieAuth" = [])),
    params(
        ("agent_id" = i64, Path, description = "Id of the agent owning the conversation"),
        ("conversation_id" = i64, Path, description = "Id of the conversation carrying the message"),
        ("message_id" = i64, Path, description = "Id of the captured message"),
    ),
    responses(
        (status = 200, content_type = "audio/flac", description = "The captured FLAC audio"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 404, body = ErrorResponse, description = "Agent not owned by the caller, the message is out of scope, or it has no captured audio"),
        (status = 503, body = ErrorResponse, description = "Audio storage is not currently configured"),
    ),
)]
pub(crate) async fn message_audio(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    user: AuthUser,
    Path((agent_id, conversation_id, message_id)): Path<(i64, i64, i64)>,
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
         WHERE c.agent_id = $1 AND c.id = $2 AND m.id = $3 \
         ORDER BY ma.id \
         LIMIT 1",
    )
    .bind(agent_id)
    .bind(conversation_id)
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
