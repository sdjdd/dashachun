use std::sync::Arc;

use axum::ServiceExt;
use axum::body::Body;
use axum::http::Request;

use xiaozhi_server_rs::agent::{Agent, CompositeAgent};
use xiaozhi_server_rs::asr::volc::VolcAsr;
use xiaozhi_server_rs::asr::{Asr, StubAsr};
use xiaozhi_server_rs::config::AppConfig;
use xiaozhi_server_rs::llm::{Llm, OpenAiConfig, OpenAiLlm};
use xiaozhi_server_rs::state::AppState;
use xiaozhi_server_rs::tts::Tts;
use xiaozhi_server_rs::tts::volc::VolcTts;
use xiaozhi_server_rs::vad::SileroVadFactory;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt::init();

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
            tracing::info!(model = %config.model, "using openai llm provider");
            Some(Arc::new(OpenAiLlm::new(config)))
        }
        None => {
            tracing::info!("no llm provider configured");
            None
        }
    };
    let tts: Option<Arc<dyn Tts>> = match VolcTts::from_env() {
        Some(volc) => {
            tracing::info!("using volc tts provider");
            Some(Arc::new(volc))
        }
        None => {
            tracing::info!("no tts provider configured");
            None
        }
    };
    let agent: Arc<dyn Agent> = Arc::new(CompositeAgent::new(asr, llm, tts, vad));
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
