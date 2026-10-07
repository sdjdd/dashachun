use std::time::Duration;

use axum::Router;
use axum::routing::get;
use tokio::sync::watch;
use tower::Layer;
use tower_http::normalize_path::{NormalizePath, NormalizePathLayer};

use crate::auth::state::AuthState;
use crate::state::ServerState;

pub mod agent;
pub mod asr;
pub mod audio;
pub mod auth;
pub mod config;
pub mod device;
pub mod dto;
pub mod error;
pub mod extract;
pub mod gateway;
pub mod llm;
pub mod routes;
pub mod state;
pub mod tts;
pub mod vad;

pub fn app(server: ServerState, auth: AuthState) -> NormalizePath<Router> {
    let router = Router::new()
        .route("/", get(async || "Hello, world!"))
        .nest("/gateway", routes::gateway::routes())
        .nest("/api/ota", routes::ota::routes())
        .with_state(server)
        .nest("/api/auth", routes::auth::routes().with_state(auth.clone()))
        .nest(
            "/api/agents",
            routes::agents::routes().with_state(auth.clone()),
        )
        .nest("/api/devices", routes::devices::routes().with_state(auth));

    NormalizePathLayer::trim_trailing_slash().layer(router)
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    app: NormalizePath<Router>,
    shutdown: watch::Receiver<bool>,
    _grace: Duration,
) {
    let signal = async move {
        let mut shutdown = shutdown;
        let _ = shutdown.wait_for(|value| *value).await;
    };
    if let Err(err) = axum::serve(
        listener,
        axum::ServiceExt::<axum::http::Request<axum::body::Body>>::into_make_service(app),
    )
    .with_graceful_shutdown(signal)
    .await
    {
        tracing::warn!(%err, "server stopped with error");
    }
}
