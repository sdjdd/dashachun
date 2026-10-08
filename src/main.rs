use std::sync::Arc;
use std::time::Duration;

use dashachun::agent::{AgentFactory, AgentStore, ToolRegistry, tool::GetWeather};
use dashachun::asr::Asr;
use dashachun::asr::volc::VolcAsr;
use dashachun::audio::DOWNLINK;
use dashachun::auth::state::AuthState;
use dashachun::config::{AppConfig, AuthConfig};
use dashachun::device::DeviceStore;
use dashachun::llm::{Llm, OpenAiConfig, OpenAiLlm};
use dashachun::state::ServerState;
use dashachun::tts::Tts;
use dashachun::tts::volc::VolcTts;
use dashachun::vad::SileroVadFactory;

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
            tracing::error!("VOLC_ASR_API_KEY and VOLC_ASR_BASE_URL are required");
            std::process::exit(1);
        }
    };
    let vad = Arc::new(SileroVadFactory::from_env());
    let llm: Arc<dyn Llm> = match OpenAiConfig::from_env() {
        Some(config) => {
            tracing::info!(model = %config.model, "using openai llm provider");
            Arc::new(OpenAiLlm::new(config))
        }
        None => {
            tracing::error!("LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are required");
            std::process::exit(1);
        }
    };
    let tts: Arc<dyn Tts> = match VolcTts::from_env(DOWNLINK) {
        Ok(Some(volc)) => {
            tracing::info!("using volc tts provider");
            Arc::new(volc)
        }
        Ok(None) => {
            tracing::error!(
                "VOLC_TTS_API_KEY, VOLC_TTS_BASE_URL and VOLC_TTS_SPEAKER are required"
            );
            std::process::exit(1);
        }
        Err(err) => {
            tracing::error!(%err, "invalid volc tts configuration");
            std::process::exit(1);
        }
    };
    let tools = Arc::new(ToolRegistry::new(vec![Arc::new(GetWeather::new())]));
    let auth_config = match AuthConfig::from_env() {
        Ok(config) => config,
        Err(err) => {
            tracing::error!(%err, "invalid auth config");
            std::process::exit(1);
        }
    };
    let pool = build_pool(&auth_config).await;
    let auth = build_auth_state(auth_config, pool.clone());
    let agent_factory = Arc::new(AgentFactory::new(
        asr,
        llm,
        tts,
        vad,
        tools,
        AgentStore::new(pool.clone()),
        pool.clone(),
    ));
    let state = ServerState::new(config, agent_factory, DeviceStore::new(pool));
    let shutdown_tx = state.shutdown_sender();
    let shutdown_rx = state.shutdown_signal();
    let app = dashachun::app(state, auth);

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

    dashachun::serve(listener, app, shutdown_rx, grace).await;
    tracing::info!("server stopped");
}

async fn build_pool(config: &AuthConfig) -> sqlx::PgPool {
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
    pool
}

fn build_auth_state(config: AuthConfig, pool: sqlx::PgPool) -> AuthState {
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
