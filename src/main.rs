use std::env;
use std::sync::Arc;
use std::time::Duration;

use dashachun::agent::tool::{GetDateTime, GetWeather};
use dashachun::agent::{AgentFactory, AgentStore, Asr, Llm, ToolRegistry, Tts};
use dashachun::audio::DOWNLINK;
use dashachun::auth::state::AuthState;
use dashachun::config::{AppConfig, AuthConfig, DeviceConfig, ServerConfig};
use dashachun::device::DeviceStore;
use dashachun::provider::asr::VolcAsr;
use dashachun::provider::llm::{OpenAiConfig, OpenAiLlm};
use dashachun::provider::tts::VolcTts;
use dashachun::settings::Settings;
use dashachun::state::ServerState;
use dashachun::vad::SileroVadFactory;

const SHUTDOWN_BUDGET: Duration = Duration::from_secs(10);

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt::init();

    let server = ServerConfig::from_env();
    let bind_addr = server.bind_addr.clone();
    let grace = Duration::from_millis(server.shutdown_grace_ms);
    let device = DeviceConfig::from_env();

    let database_url = match env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            tracing::error!("DATABASE_URL is required");
            std::process::exit(1);
        }
    };
    let pool = build_pool(&database_url).await;
    let mut settings = match Settings::load(&pool).await {
        Ok(settings) => settings,
        Err(err) => {
            tracing::error!(%err, "invalid settings");
            std::process::exit(1);
        }
    };
    let session_secret = match settings.ensure_session_secret(&pool).await {
        Ok(secret) => secret,
        Err(err) => {
            tracing::error!(%err, "invalid session secret");
            std::process::exit(1);
        }
    };
    let ota = settings.ota.unwrap_or_default();

    let asr: Arc<dyn Asr> = match settings.asr {
        Some(config) => {
            tracing::info!("using volc asr provider");
            Arc::new(VolcAsr::new(
                config.base_url,
                config.api_key,
                config.resource_id,
            ))
        }
        None => {
            tracing::error!(
                "ASR is not configured: insert the \"asr\" settings row (base_url, api_key, resource_id)"
            );
            std::process::exit(1);
        }
    };
    let vad = Arc::new(SileroVadFactory::from_env());
    let llm: Arc<dyn Llm> = match settings.llm {
        Some(config) => {
            tracing::info!(model = %config.model, "using openai llm provider");
            Arc::new(OpenAiLlm::new(OpenAiConfig {
                base_url: config.base_url,
                api_key: config.api_key,
                model: config.model,
                max_tokens: config.max_tokens,
                reasoning_effort: config.reasoning_effort,
            }))
        }
        None => {
            tracing::error!(
                "LLM is not configured: insert the \"llm\" settings row (base_url, api_key, model)"
            );
            std::process::exit(1);
        }
    };
    let tts: Arc<dyn Tts> = match settings.tts {
        Some(config) => match VolcTts::new(
            config.base_url,
            config.api_key,
            config.resource_id,
            config.speaker,
            DOWNLINK,
        ) {
            Ok(volc) => {
                tracing::info!("using volc tts provider");
                Arc::new(volc)
            }
            Err(err) => {
                tracing::error!(%err, "invalid volc tts configuration");
                std::process::exit(1);
            }
        },
        None => {
            tracing::error!(
                "TTS is not configured: insert the \"tts\" settings row (base_url, api_key, speaker, resource_id)"
            );
            std::process::exit(1);
        }
    };
    let tools = Arc::new(ToolRegistry::new(vec![
        Arc::new(GetWeather::new()),
        Arc::new(GetDateTime::new(ota.timezone_offset)),
    ]));
    let config = AppConfig {
        server,
        ota,
        device,
    };
    let auth_config = AuthConfig::from_env(database_url, session_secret);
    let auth = build_auth_state(auth_config, pool.clone());
    let agent_factory = Arc::new(AgentFactory::new(
        asr,
        llm,
        tts.clone(),
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
        tokio::time::sleep(grace + SHUTDOWN_BUDGET).await;
        tracing::warn!("graceful shutdown timed out, forcing exit");
        std::process::exit(0);
    });

    dashachun::serve(listener, app, shutdown_rx, grace).await;
    tts.shutdown().await;
    tracing::info!("server stopped");
}

async fn build_pool(database_url: &str) -> sqlx::PgPool {
    let pool = match sqlx::PgPool::connect(database_url).await {
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
