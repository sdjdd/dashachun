use std::env;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;

use dashachun::agent::tool::{GetDateTime, GetWeather};
use dashachun::agent::{AgentFactory, AgentStore, ToolRegistry};
use dashachun::auth::state::AuthState;
use dashachun::config::{AppConfig, AuthConfig, DeviceConfig, ServerConfig};
use dashachun::device::DeviceStore;
use dashachun::provider::asr::volc::KIND as VOLC_ASR;
use dashachun::provider::llm::openai::KIND as OPENAI_LLM;
use dashachun::provider::tts::volc::KIND as VOLC_TTS;
use dashachun::provider::{ProviderRegistry, ProviderSpec};
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

    let registry = ProviderRegistry::with_builtins();
    let asr = provider_spec(
        settings.asr.take(),
        "asr",
        VOLC_ASR,
        "(base_url, api_key, resource_id)",
    );
    let llm = provider_spec(
        settings.llm.take(),
        "llm",
        OPENAI_LLM,
        "(base_url, api_key, model)",
    );
    let tts = provider_spec(
        settings.tts.take(),
        "tts",
        VOLC_TTS,
        "(base_url, api_key, speaker, resource_id)",
    );
    let vad = Arc::new(SileroVadFactory::from_env());
    let tools = Arc::new(ToolRegistry::new(vec![
        Arc::new(GetWeather::new()),
        Arc::new(GetDateTime::new(ota.timezone_offset)),
    ]));
    let agent_factory = match AgentFactory::assemble(
        &registry,
        &asr,
        &llm,
        &tts,
        vad,
        tools,
        AgentStore::new(pool.clone()),
        pool.clone(),
    ) {
        Ok(factory) => Arc::new(ArcSwap::from_pointee(factory)),
        Err(err) => {
            tracing::error!(%err, "invalid provider configuration");
            std::process::exit(1);
        }
    };
    let config = AppConfig {
        server,
        ota,
        device,
    };
    let auth_config = AuthConfig::from_env(database_url, session_secret);
    let auth = build_auth_state(auth_config, pool.clone());
    let state = ServerState::new(config, agent_factory.clone(), DeviceStore::new(pool));
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
    agent_factory.load_full().shutdown().await;
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

fn provider_spec(
    row: Option<serde_json::Value>,
    stage: &str,
    kind: &str,
    fields: &str,
) -> ProviderSpec {
    ProviderSpec::new(
        kind,
        match row {
            Some(config) => config,
            None => missing_provider(stage, fields),
        },
    )
}

fn missing_provider(stage: &str, fields: &str) -> ! {
    tracing::error!("{stage} is not configured: insert the \"{stage}\" settings row {fields}");
    std::process::exit(1);
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
