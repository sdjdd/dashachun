//! The HTTP server surface: every axum route (the browser/operator REST,
//! the device-facing OTA endpoints and the gateway websocket upgrade), its
//! extractors, wire DTOs, error rendering, the OpenAPI document and the
//! optional static web console serving.
//! Domain stores and the shared `ServerState` live outside; this module
//! depends on them, never the other way around — a future gRPC surface
//! would sit beside it as a sibling module over the same service layer.

pub mod docs;
pub mod dto;
pub mod extract;
pub mod routes;
pub mod session;
pub mod state;
pub mod static_assets;

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
    let voiceprint_settings = routes::voiceprint::routes().with_state(SettingsState {
        auth: auth.clone(),
        server: server.clone(),
    });
    let agents = routes::agents::routes().with_state(AgentsState {
        auth: auth.clone(),
        server: server.clone(),
    });
    let router = Router::new()
        .nest("/gateway", routes::gateway::routes())
        .nest("/api/ota", routes::ota::routes())
        .nest("/healthz", routes::health::routes())
        .with_state(server.clone())
        .nest("/api/providers", providers)
        .nest("/api/auth", routes::auth::routes().with_state(auth.clone()))
        .nest("/api/agents", agents)
        .nest("/api/settings/audio-capture", audio_capture)
        .nest("/api/settings/ota", ota_settings)
        .nest("/api/settings/voiceprint", voiceprint_settings)
        .merge(Scalar::with_url("/scalar", docs::ApiDoc::openapi()))
        .route("/api/openapi.json", get(docs::spec));

    let router = match &server.config.server.static_dir {
        Some(dir) => router.fallback_service(static_assets::routes(dir)),
        None => router,
    };

    NormalizePathLayer::trim_trailing_slash().layer(router)
}
