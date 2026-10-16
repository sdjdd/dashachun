pub mod agent;
pub mod audio;
pub mod auth;
pub mod config;
pub mod device;
pub mod error;
pub mod gateway;
pub mod http;
pub mod oss;
pub mod provider;
pub mod ratelimit;
pub mod settings;
pub mod state;
pub mod telemetry;
pub mod util;
pub mod vad;
pub mod voiceprint;

use std::net::SocketAddr;
use std::time::Duration;

use tokio::sync::watch;
use tower_http::normalize_path::NormalizePath;

pub async fn serve(
    listener: tokio::net::TcpListener,
    app: NormalizePath<axum::Router>,
    shutdown: watch::Receiver<bool>,
    _grace: Duration,
) {
    let signal = async move {
        let mut shutdown = shutdown;
        let _ = shutdown.wait_for(|value| *value).await;
    };
    if let Err(err) = axum::serve(
        listener,
        app.into_inner()
            .into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(signal)
    .await
    {
        tracing::warn!(%err, "server stopped with error");
    }
}
