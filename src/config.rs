use std::env;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub ota: OtaConfig,
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub bind_addr: String,
}

#[derive(Clone, Debug)]
pub struct OtaConfig {
    pub websocket_url: String,
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
        }
    }
}

impl OtaConfig {
    pub fn from_env() -> Self {
        Self {
            websocket_url: env_or("OTA_WEBSOCKET_URL", "ws://127.0.0.1:3000/gateway"),
            token: env_or("OTA_TOKEN", "test-token"),
            timezone_offset: env_or("OTA_TIMEZONE_OFFSET", "480").parse().unwrap_or(480),
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}
