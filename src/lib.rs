use axum::Router;
use axum::routing::get;
use tower::Layer;
use tower_http::normalize_path::{NormalizePath, NormalizePathLayer};

use crate::state::AppState;

pub mod config;
pub mod dto;
pub mod error;
pub mod extract;
pub mod routes;
pub mod state;

pub fn app(state: AppState) -> NormalizePath<Router> {
    let router = Router::new()
        .route("/", get(async || "Hello, world!"))
        .merge(routes::gateway::routes())
        .merge(routes::ota::routes())
        .with_state(state);

    NormalizePathLayer::trim_trailing_slash().layer(router)
}
