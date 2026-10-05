use std::env;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub ota: OtaConfig,
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub playback_prebuffer_ms: u32,
    pub shutdown_grace_ms: u64,
}

#[derive(Clone, Debug)]
pub struct OtaConfig {
    pub websocket_url: Option<String>,
    pub token: String,
    pub timezone_offset: i32,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            server: ServerConfig::from_env(),
            ota: OtaConfig::from_env(),
        }
    }
}

impl ServerConfig {
    pub fn from_env() -> Self {
        Self {
            bind_addr: env_or("BIND_ADDR", "0.0.0.0:3000"),
            playback_prebuffer_ms: env_or("PLAYBACK_PREBUFFER_MS", "180")
                .parse()
                .unwrap_or(180),
            shutdown_grace_ms: env_or("SHUTDOWN_GRACE_MS", "5000").parse().unwrap_or(5000),
        }
    }
}

impl OtaConfig {
    pub fn from_env() -> Self {
        Self {
            websocket_url: env::var("OTA_WEBSOCKET_URL").ok(),
            token: env_or("OTA_TOKEN", "test-token"),
            timezone_offset: env_or("OTA_TIMEZONE_OFFSET", "480").parse().unwrap_or(480),
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}
