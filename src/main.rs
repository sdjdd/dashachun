use std::sync::Arc;

use axum::ServiceExt;
use axum::body::Body;
use axum::http::Request;
use tracing_subscriber::EnvFilter;

use xiaozhi_server_rs::agent::{Agent, CompositeAgent};
use xiaozhi_server_rs::asr::volc::VolcAsr;
use xiaozhi_server_rs::asr::{Asr, StubAsr};
use xiaozhi_server_rs::config::AppConfig;
use xiaozhi_server_rs::llm::{Llm, OpenAiConfig, OpenAiLlm};
use xiaozhi_server_rs::state::AppState;
use xiaozhi_server_rs::vad::SileroVadFactory;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("install rustls crypto provider");

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let filter = filter.add_directive("ort=off".parse().expect("valid directive"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let config = AppConfig::from_env();
    let bind_addr = config.server.bind_addr.clone();
    let asr: Arc<dyn Asr> = match VolcAsr::from_env() {
        Some(volc) => {
            tracing::info!("using volc asr provider");
            Arc::new(volc)
        }
        None => {
            tracing::info!("using stub asr provider");
            Arc::new(StubAsr::default())
        }
    };
    let vad = Arc::new(SileroVadFactory::from_env());
    let llm: Option<Arc<dyn Llm>> = match OpenAiConfig::from_env() {
        Some(config) => {
            tracing::info!(model = %config.model, "using openai responses llm provider");
            Some(Arc::new(OpenAiLlm::new(config)))
        }
        None => {
            tracing::info!("no llm provider configured");
            None
        }
    };
    let agent: Arc<dyn Agent> = Arc::new(CompositeAgent::new(asr, llm, None, vad));
    let app = xiaozhi_server_rs::app(AppState { config, agent });

    let listener = tokio::net::TcpListener::bind(&bind_addr).await.unwrap();
    tracing::info!("listening on {bind_addr}");
    axum::serve(
        listener,
        ServiceExt::<Request<Body>>::into_make_service(app),
    )
    .await
    .unwrap();
}
