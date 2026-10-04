use axum::ServiceExt;
use axum::body::Body;
use axum::http::Request;
use tracing_subscriber::EnvFilter;

use xiaozhi_server_rs::config::AppConfig;
use xiaozhi_server_rs::state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::from_env();
    let bind_addr = config.server.bind_addr.clone();
    let app = xiaozhi_server_rs::app(AppState { config });

    let listener = tokio::net::TcpListener::bind(&bind_addr).await.unwrap();
    tracing::info!("listening on {bind_addr}");
    axum::serve(
        listener,
        ServiceExt::<Request<Body>>::into_make_service(app),
    )
    .await
    .unwrap();
}
