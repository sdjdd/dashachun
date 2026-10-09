use std::env;

use crate::settings::OtaSettings;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub ota: OtaSettings,
    pub device: DeviceConfig,
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub playback_prebuffer_ms: u32,
    pub shutdown_grace_ms: u64,
}

#[derive(Clone, Debug)]
pub struct DeviceConfig {
    pub activation_ttl_secs: i64,
}

#[derive(Clone)]
pub struct AuthConfig {
    pub database_url: String,
    pub session_secret: String,
    pub session_ttl_secs: i64,
    pub cookie_secure: bool,
    pub cookie_name: String,
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

impl DeviceConfig {
    pub fn from_env() -> Self {
        Self {
            activation_ttl_secs: env_or("DEVICE_ACTIVATION_TTL_SECS", "600")
                .parse()
                .unwrap_or(600),
        }
    }
}

impl AuthConfig {
    pub fn from_env(database_url: String, session_secret: String) -> Self {
        Self {
            database_url,
            session_secret,
            session_ttl_secs: env_or("SESSION_TTL_SECS", "2592000")
                .parse()
                .unwrap_or(2592000),
            cookie_secure: env_or("COOKIE_SECURE", "true").parse().unwrap_or(true),
            cookie_name: env_or("SESSION_COOKIE_NAME", "xz_session"),
        }
    }
}

impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthConfig")
            .field("database_url", &self.database_url)
            .field("session_secret", &"<redacted>")
            .field("session_ttl_secs", &self.session_ttl_secs)
            .field("cookie_secure", &self.cookie_secure)
            .field("cookie_name", &self.cookie_name)
            .finish()
    }
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}
