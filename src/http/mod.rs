//! The HTTP server surface: every axum route (the browser/operator REST,
//! the device-facing OTA endpoints and the gateway websocket upgrade) with
//! its extractors, wire DTOs, error rendering and the OpenAPI document.
//! Domain stores and the shared `ServerState` live outside; this module
//! depends on them, never the other way around — a future gRPC surface
//! would sit beside it as a sibling module over the same service layer.

pub mod docs;
pub mod dto;
pub mod extract;
pub mod routes;
pub mod session;
pub mod state;

use axum::Router;
use axum::routing::get;
use tower::Layer;
use tower_http::normalize_path::{NormalizePath, NormalizePathLayer};
use utoipa::OpenApi;
use utoipa_scalar::{Scalar, Servable};

use crate::state::ServerState;

use self::session::AuthState;
use self::state::{AgentsState, ProvidersState, SettingsState};

pub fn app(server: ServerState, auth: AuthState) -> NormalizePath<Router> {
    let providers = routes::providers::routes().with_state(ProvidersState {
        auth: auth.clone(),
        server: server.clone(),
    });
    let audio_capture = routes::audio_capture::routes().with_state(SettingsState {
        auth: auth.clone(),
        server: server.clone(),
    });
    let ota_settings = routes::settings::routes().with_state(SettingsState {
        auth: auth.clone(),
        server: server.clone(),
    });
    let agents = routes::agents::routes().with_state(AgentsState {
        auth: auth.clone(),
        server: server.clone(),
    });
    let router = Router::new()
        .route("/", get(async || "王老师，你看我鞋好看不"))
        .nest("/gateway", routes::gateway::routes())
        .nest("/api/ota", routes::ota::routes())
        .with_state(server)
        .nest("/api/providers", providers)
        .nest("/api/auth", routes::auth::routes().with_state(auth.clone()))
        .nest("/api/agents", agents)
        .nest("/api/settings/audio-capture", audio_capture)
        .nest("/api/settings/ota", ota_settings)
        .merge(Scalar::with_url("/scalar", docs::ApiDoc::openapi()))
        .route("/api/openapi.json", get(docs::spec));

    NormalizePathLayer::trim_trailing_slash().layer(router)
}
