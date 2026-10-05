use std::sync::Arc;
use std::time::Duration;

use xiaozhi_server_rs::agent::{
    Agent, CompositeAgent, ToolRegistry, tool::GetWeather, tool::SetEmotion,
};
use xiaozhi_server_rs::asr::volc::VolcAsr;
use xiaozhi_server_rs::asr::{Asr, StubAsr};
use xiaozhi_server_rs::auth::state::AuthState;
use xiaozhi_server_rs::config::{AppConfig, AuthConfig};
use xiaozhi_server_rs::llm::{Llm, OpenAiConfig, OpenAiLlm};
use xiaozhi_server_rs::state::ServerState;
use xiaozhi_server_rs::tts::Tts;
use xiaozhi_server_rs::tts::volc::VolcTts;
use xiaozhi_server_rs::vad::SileroVadFactory;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt::init();

    let config = AppConfig::from_env();
    let bind_addr = config.server.bind_addr.clone();
    let grace = Duration::from_millis(config.server.shutdown_grace_ms);
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
    let tools = Arc::new(ToolRegistry::new(vec![
        Arc::new(SetEmotion),
        Arc::new(GetWeather::new()),
    ]));
    let agent: Arc<dyn Agent> = Arc::new(CompositeAgent {
        asr,
        llm,
        tts,
        vad,
        tools,
    });
    let auth = Some(build_auth_state().await);
    let state = ServerState::new(config, agent);
    let shutdown_tx = state.shutdown_sender();
    let shutdown_rx = state.shutdown_signal();
    let app = xiaozhi_server_rs::app(state, auth);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await.unwrap();
    tracing::info!("listening on {bind_addr}");

    tokio::spawn(async move {
        shutdown_on_signal().await;
        tracing::info!("shutdown signal received, draining");
        let _ = shutdown_tx.send(true);
        tokio::time::sleep(grace).await;
        tracing::warn!("graceful shutdown timed out, forcing exit");
        std::process::exit(0);
    });

    xiaozhi_server_rs::serve(listener, app, shutdown_rx, grace).await;
    tracing::info!("server stopped");
}

async fn build_auth_state() -> AuthState {
    let config = match AuthConfig::from_env() {
        Ok(config) => config,
        Err(err) => {
            tracing::error!(%err, "invalid auth config");
            std::process::exit(1);
        }
    };
    let pool = match sqlx::PgPool::connect(&config.database_url).await {
        Ok(pool) => pool,
        Err(err) => {
            tracing::error!(%err, "failed to connect to database");
            std::process::exit(1);
        }
    };
    if let Err(err) = sqlx::migrate!("./migrations").run(&pool).await {
        tracing::error!(%err, "failed to run migrations");
        std::process::exit(1);
    }
    let key = axum_extra::extract::cookie::Key::from(config.session_secret.as_bytes());
    tracing::info!("auth routes enabled");
    AuthState {
        pool,
        key,
        ttl_secs: config.session_ttl_secs,
        cookie_name: config.cookie_name,
        cookie_secure: config.cookie_secure,
    }
}

async fn shutdown_on_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(term) => term,
            Err(err) => {
                tracing::warn!(%err, "failed to install sigterm handler");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
