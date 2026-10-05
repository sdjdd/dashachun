use std::time::Duration;

use axum::Router;
use axum::routing::get;
use tokio::sync::watch;
use tower::Layer;
use tower_http::normalize_path::{NormalizePath, NormalizePathLayer};

use crate::state::AppState;

pub mod agent;
pub mod asr;
pub mod audio;
pub mod config;
pub mod dto;
pub mod error;
pub mod extract;
pub mod llm;
pub mod routes;
pub mod state;
pub mod tts;
pub mod vad;

pub fn app(state: AppState) -> NormalizePath<Router> {
    let router = Router::new()
        .route("/", get(async || "Hello, world!"))
        .merge(routes::gateway::routes())
        .merge(routes::ota::routes())
        .with_state(state);

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
