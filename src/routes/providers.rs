use std::sync::Arc;

use axum::extract::{FromRef, Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use sqlx::PgPool;

use crate::agent::{AgentFactory, AgentStore};
use crate::auth::extract::AdminUser;
use crate::auth::state::AuthState;
use crate::error::{AppError, ErrorResponse};
use crate::extract::Validated;
use crate::provider::ProviderSpec;
use crate::provider::dto::{CreateProviderArgs, ProviderResponse, UpdateProviderArgs};
use crate::provider::store::{ProviderRecord, ProviderStore};
use crate::settings;
use crate::state::ServerState;

/// Router state for the provider admin API: session auth plus the server
/// wiring (registry, shared VAD/tools and the hot-swap slot) that
/// activation needs to assemble and publish a new agent generation.
#[derive(Clone)]
pub struct ProvidersState {
    pub auth: AuthState,
    pub server: ServerState,
}

impl FromRef<ProvidersState> for AuthState {
    fn from_ref(state: &ProvidersState) -> Self {
        state.auth.clone()
    }
}

impl FromRef<ProvidersState> for ServerState {
    fn from_ref(state: &ProvidersState) -> Self {
        state.server.clone()
    }
}

pub fn routes() -> Router<ProvidersState> {
    Router::new()
        .route("/", get(list_providers).post(create_provider))
        .route(
            "/{id}",
            get(get_provider)
                .put(update_provider)
                .delete(delete_provider),
        )
        .route("/{id}/activate", post(activate_provider))
}

#[utoipa::path(
    get,
    path = "/api/providers",
    tag = "providers",
    security(("cookieAuth" = [])),
    responses(
        (status = 200, body = [ProviderResponse], description = "All provider rows with selection state"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
    ),
)]
pub(crate) async fn list_providers(
    State(auth): State<AuthState>,
    _user: AdminUser,
) -> Result<Json<Vec<ProviderResponse>>, AppError> {
    let selections = provider_selections(&auth.pool).await?;
    let rows = ProviderStore::new(auth.pool.clone()).list().await?;
    Ok(Json(
        rows.into_iter()
            .map(|row| {
                let selected = is_selected(&selections, &row);
                ProviderResponse::new(row, selected)
            })
            .collect(),
    ))
}

#[utoipa::path(
    get,
    path = "/api/providers/{id}",
    tag = "providers",
    security(("cookieAuth" = [])),
    params(("id" = i64, Path, description = "Provider row id")),
    responses(
        (status = 200, body = ProviderResponse, description = "The provider row"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
        (status = 404, body = ErrorResponse, description = "Unknown provider id"),
    ),
)]
pub(crate) async fn get_provider(
    State(auth): State<AuthState>,
    _user: AdminUser,
    Path(id): Path<i64>,
) -> Result<Json<ProviderResponse>, AppError> {
    let store = ProviderStore::new(auth.pool.clone());
    let Some(row) = store.find(id).await? else {
        return Err(AppError::NotFound);
    };
    let selections = provider_selections(&auth.pool).await?;
    let selected = is_selected(&selections, &row);
    Ok(Json(ProviderResponse::new(row, selected)))
}

#[utoipa::path(
    post,
    path = "/api/providers",
    tag = "providers",
    security(("cookieAuth" = [])),
    request_body = CreateProviderArgs,
    responses(
        (status = 201, body = ProviderResponse, description = "Provider row created"),
        (status = 400, body = ErrorResponse, description = "Invalid payload or config rejected by the type"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
    ),
)]
pub(crate) async fn create_provider(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    _user: AdminUser,
    Validated(args): Validated<CreateProviderArgs>,
) -> Result<(StatusCode, Json<ProviderResponse>), AppError> {
    // An omitted kind resolves to the type's sole registered implementation
    // and is stored, so every row stays self-describing.
    let kind = match args.kind.as_deref() {
        Some(kind) => kind,
        None => server
            .registry
            .sole_kind(&args.r#type)
            .map_err(AppError::BadRequest)?,
    };
    server
        .registry
        .validate(&args.r#type, kind, &args.config)
        .map_err(AppError::BadRequest)?;
    let store = ProviderStore::new(auth.pool.clone());
    let row = store
        .create(&args.r#type, &args.name, kind, &args.config)
        .await?;
    let selections = provider_selections(&auth.pool).await?;
    let selected = is_selected(&selections, &row);
    Ok((
        StatusCode::CREATED,
        Json(ProviderResponse::new(row, selected)),
    ))
}

#[utoipa::path(
    put,
    path = "/api/providers/{id}",
    tag = "providers",
    security(("cookieAuth" = [])),
    params(("id" = i64, Path, description = "Provider row id")),
    request_body = UpdateProviderArgs,
    responses(
        (status = 200, body = ProviderResponse, description = "Provider row updated"),
        (status = 400, body = ErrorResponse, description = "Invalid payload or config rejected by the type"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
        (status = 404, body = ErrorResponse, description = "Unknown provider id"),
    ),
)]
pub(crate) async fn update_provider(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    _user: AdminUser,
    Path(id): Path<i64>,
    Validated(args): Validated<UpdateProviderArgs>,
) -> Result<Json<ProviderResponse>, AppError> {
    let store = ProviderStore::new(auth.pool.clone());
    let Some(existing) = store.find(id).await? else {
        return Err(AppError::NotFound);
    };
    let changed = args.kind.is_some() || args.config.is_some();
    if changed {
        let kind = args.kind.as_deref().unwrap_or(&existing.kind);
        let config = args
            .config
            .clone()
            .unwrap_or_else(|| existing.config.clone());
        server
            .registry
            .validate(&existing.r#type, kind, &config)
            .map_err(AppError::BadRequest)?;
    }
    let Some(row) = store
        .update(
            id,
            args.name.as_deref(),
            args.kind.as_deref(),
            args.config.as_ref(),
        )
        .await?
    else {
        return Err(AppError::NotFound);
    };
    let selections = provider_selections(&auth.pool).await?;
    let selected = is_selected(&selections, &row);
    // Editing the selected provider's kind/config hot-applies a new
    // generation; a rename does not change the wiring (selections are by id).
    if changed && selected {
        apply_generation(&server, &auth.pool).await?;
    }
    Ok(Json(ProviderResponse::new(row, selected)))
}

#[utoipa::path(
    delete,
    path = "/api/providers/{id}",
    tag = "providers",
    security(("cookieAuth" = [])),
    params(("id" = i64, Path, description = "Provider row id")),
    responses(
        (status = 204, description = "Provider row deleted"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
        (status = 404, body = ErrorResponse, description = "Unknown provider id"),
        (status = 409, body = ErrorResponse, description = "Provider is selected for its type"),
    ),
)]
pub(crate) async fn delete_provider(
    State(auth): State<AuthState>,
    _user: AdminUser,
    Path(id): Path<i64>,
) -> Result<StatusCode, AppError> {
    let store = ProviderStore::new(auth.pool.clone());
    let Some(row) = store.find(id).await? else {
        return Err(AppError::NotFound);
    };
    let selections = provider_selections(&auth.pool).await?;
    if is_selected(&selections, &row) {
        return Err(AppError::Conflict(
            "provider is selected for its type; activate another one first".into(),
        ));
    }
    if !store.delete(id).await? {
        return Err(AppError::NotFound);
    }
    Ok(StatusCode::NO_CONTENT)
}

#[utoipa::path(
    post,
    path = "/api/providers/{id}/activate",
    tag = "providers",
    security(("cookieAuth" = [])),
    params(("id" = i64, Path, description = "Provider row id")),
    responses(
        (status = 200, body = ProviderResponse, description = "Provider selected and generation hot-applied"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
        (status = 403, body = ErrorResponse, description = "Admins only"),
        (status = 404, body = ErrorResponse, description = "Unknown provider id"),
    ),
)]
pub(crate) async fn activate_provider(
    State(auth): State<AuthState>,
    State(server): State<ServerState>,
    _user: AdminUser,
    Path(id): Path<i64>,
) -> Result<Json<ProviderResponse>, AppError> {
    let store = ProviderStore::new(auth.pool.clone());
    let Some(row) = store.find(id).await? else {
        return Err(AppError::NotFound);
    };
    settings::set_provider_selection(&auth.pool, &row.r#type, row.id)
        .await
        .map_err(AppError::Internal)?;
    apply_generation(&server, &auth.pool).await?;
    // The selection was just written for this row.
    Ok(Json(ProviderResponse::new(row, true)))
}

fn is_selected(selections: &settings::ProviderSelections, row: &ProviderRecord) -> bool {
    selections.selected(&row.r#type) == Some(row.id)
}

/// Rebuilds the agent generation from the stored selections and publishes
/// it. The swap lock serializes the read-assemble-swap sequence, so the
/// published generation always matches the stored selections. An
/// incomplete selection set (a fresh server being filled in type by type)
/// leaves the current generation untouched; a failed build returns the
/// builder error and leaves the selection standing — fix the config and
/// re-activate.
async fn apply_generation(server: &ServerState, pool: &PgPool) -> Result<(), AppError> {
    let _guard = server.swap_lock.lock().await;
    let selections = provider_selections(pool).await?;
    let (Some(asr_id), Some(llm_id), Some(tts_id)) =
        (selections.asr, selections.llm, selections.tts)
    else {
        tracing::info!("provider selection incomplete; generation not applied yet");
        return Ok(());
    };
    let store = ProviderStore::new(pool.clone());
    let asr = selected_row(&store, asr_id, "asr").await?;
    let llm = selected_row(&store, llm_id, "llm").await?;
    let tts = selected_row(&store, tts_id, "tts").await?;
    let asr_spec = ProviderSpec::new(asr.kind, asr.config);
    let llm_spec = ProviderSpec::new(llm.kind, llm.config);
    let tts_spec = ProviderSpec::new(tts.kind, tts.config);
    let factory = AgentFactory::assemble(
        &server.registry,
        &asr_spec,
        &llm_spec,
        &tts_spec,
        server.vad.clone(),
        server.tools.clone(),
        AgentStore::new(pool.clone()),
        pool.clone(),
    )
    .map_err(AppError::BadRequest)?;
    server.swap_agent_factory(Arc::new(factory));
    Ok(())
}

async fn selected_row(
    store: &ProviderStore,
    id: i64,
    r#type: &str,
) -> Result<ProviderRecord, AppError> {
    store.find(id).await?.ok_or_else(|| {
        AppError::BadRequest(format!(
            "the {type} selection points at provider {id}, which does not exist"
        ))
    })
}

async fn provider_selections(pool: &PgPool) -> Result<settings::ProviderSelections, AppError> {
    settings::load_provider_selections(pool)
        .await
        .map_err(AppError::Internal)
}
