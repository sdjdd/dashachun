use std::env;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwapOption;

use dashachun::agent::tool::{GetDateTime, GetWeather};
use dashachun::agent::{AgentFactory, AgentStore, AudioStorage, ToolSet};
use dashachun::auth::state::AuthState;
use dashachun::config::{AppConfig, AuthConfig, DeviceConfig, OssConfig, ServerConfig};
use dashachun::device::DeviceStore;
use dashachun::error::AppError;
use dashachun::oss::Oss;
use dashachun::provider::store::ProviderStore;
use dashachun::provider::{ProviderRegistry, ProviderSpec};
use dashachun::ratelimit::RedisLimiter;
use dashachun::settings::{AudioCaptureSettings, Settings};
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
    let redis_url = match env::var("REDIS_URL") {
        Ok(url) => url,
        Err(_) => {
            tracing::error!("REDIS_URL is required");
            std::process::exit(1);
        }
    };
    let limiter = match RedisLimiter::new(&redis_url, "rl") {
        Ok(limiter) => limiter,
        Err(err) => {
            tracing::error!(%err, "invalid REDIS_URL");
            std::process::exit(1);
        }
    };
    if let Err(err) = limiter.ping().await {
        tracing::error!(%err, "failed to connect to redis");
        std::process::exit(1);
    }
    let redis = match redis::Client::open(redis_url.as_str()) {
        Ok(client) => client,
        Err(err) => {
            tracing::error!(%err, "invalid REDIS_URL");
            std::process::exit(1);
        }
    };
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
    let audio_capture = settings.audio_capture.clone().unwrap_or_default();
    let audio_storage = build_audio_storage(&audio_capture).await;
    let specs = match resolve_specs(&ProviderStore::new(pool.clone()), &settings).await {
        Ok(specs) => specs,
        Err(err) => {
            tracing::error!(%err, "failed to load provider selections");
            std::process::exit(1);
        }
    };
    let ota = settings.ota.unwrap_or_default();

    let registry = Arc::new(ProviderRegistry::with_builtins());
    let vad = match SileroVadFactory::from_env() {
        Ok(vad) => Arc::new(vad),
        Err(err) => {
            tracing::error!(%err, "failed to load the silero vad model");
            std::process::exit(1);
        }
    };
    let tools = ToolSet::new()
        .tool(GetWeather::new())
        .tool(GetDateTime::new(ota.timezone_offset));
    let agent_factory = match specs {
        Some((asr, llm, tts)) => match AgentFactory::assemble(
            &registry,
            &asr,
            &llm,
            &tts,
            vad.clone(),
            tools.clone(),
            AgentStore::new(pool.clone()),
            pool.clone(),
        ) {
            Ok(factory) => Arc::new(ArcSwapOption::from_pointee(factory)),
            Err(err) => {
                tracing::error!(
                    %err,
                    "provider configuration failed to build; starting without agents (fix via /api/providers)"
                );
                Arc::new(ArcSwapOption::empty())
            }
        },
        None => {
            tracing::warn!(
                "no provider selected for every type yet; the gateway stays unavailable until /api/providers is configured"
            );
            Arc::new(ArcSwapOption::empty())
        }
    };
    let config = AppConfig {
        server,
        ota,
        device,
    };
    let auth_config = AuthConfig::from_env(database_url, session_secret);
    let auth = build_auth_state(auth_config, pool.clone(), limiter);
    let state = ServerState::new(
        config,
        registry,
        vad,
        tools,
        audio_storage,
        agent_factory.clone(),
        DeviceStore::new(pool, redis, "act"),
    );
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
    if let Some(factory) = agent_factory.load_full() {
        factory.shutdown().await;
    }
    tracing::info!("server stopped");
}

/// Builds the boot-time audio storage. Storage problems never block the
/// service — the audio is a byproduct, so any misconfiguration or
/// unreachability degrades to captures-off with a warning; the settings
/// API applies and validates storage eagerly when operators change it,
/// and a later apply (or restart) picks the storage back up.
async fn build_audio_storage(capture: &AudioCaptureSettings) -> Option<Arc<dyn AudioStorage>> {
    if !capture.enabled {
        return None;
    }
    if capture.bucket.trim().is_empty() {
        tracing::warn!(
            "audio capture is enabled but audio_capture.bucket is not set; captures are not stored"
        );
        return None;
    }
    let config = match OssConfig::from_env(true) {
        Ok(Some(config)) => config,
        Ok(None) => {
            tracing::warn!(
                "audio capture is enabled but S3_ENDPOINT is not configured; captures are not stored"
            );
            return None;
        }
        Err(err) => {
            tracing::warn!(
                %err,
                "audio capture is enabled but its storage is misconfigured; captures are not stored"
            );
            return None;
        }
    };
    let oss = Oss::new(&config, &capture.bucket);
    if let Err(err) = oss.ensure_bucket().await {
        tracing::warn!(
            bucket = %capture.bucket,
            %err,
            "audio storage is unreachable at boot; captures are not stored until it is re-applied via /api/settings/audio-capture"
        );
        return None;
    }
    tracing::info!(bucket = %capture.bucket, "audio capture storage ready");
    Some(Arc::new(oss))
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

fn build_auth_state(config: AuthConfig, pool: sqlx::PgPool, limiter: RedisLimiter) -> AuthState {
    let key = axum_extra::extract::cookie::Key::from(config.session_secret.as_bytes());
    tracing::info!("auth routes enabled");
    AuthState {
        pool,
        key,
        ttl_secs: config.session_ttl_secs,
        cookie_name: config.cookie_name,
        cookie_secure: config.cookie_secure,
        limiter,
        trust_proxy: config.trust_proxy,
    }
}

/// Resolves the three type selections into provider specs; `None` means
/// some type has no selection (or its row vanished) — the server starts
/// without a generation and the setup completes via /api/providers.
async fn resolve_specs(
    store: &ProviderStore,
    settings: &Settings,
) -> Result<Option<(ProviderSpec, ProviderSpec, ProviderSpec)>, AppError> {
    let Some(asr) = type_spec(store, settings.asr).await? else {
        return Ok(None);
    };
    let Some(llm) = type_spec(store, settings.llm).await? else {
        return Ok(None);
    };
    let Some(tts) = type_spec(store, settings.tts).await? else {
        return Ok(None);
    };
    Ok(Some((asr, llm, tts)))
}

async fn type_spec(
    store: &ProviderStore,
    provider: Option<i64>,
) -> Result<Option<ProviderSpec>, AppError> {
    let Some(id) = provider else {
        return Ok(None);
    };
    match store.find(id).await? {
        Some(record) => Ok(Some(ProviderSpec::new(record.kind, record.config))),
        None => {
            tracing::error!(id, "selected provider row is missing");
            Ok(None)
        }
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
