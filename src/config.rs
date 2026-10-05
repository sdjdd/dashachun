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

#[derive(Clone)]
pub struct AuthConfig {
    pub database_url: String,
    pub session_secret: String,
    pub session_ttl_secs: i64,
    pub cookie_secure: bool,
    pub cookie_name: String,
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

impl AuthConfig {
    pub fn from_env() -> Result<Self, String> {
        let database_url =
            env::var("DATABASE_URL").map_err(|_| "DATABASE_URL is required".to_string())?;
        let session_secret =
            env::var("SESSION_SECRET").map_err(|_| "SESSION_SECRET is required".to_string())?;
        if session_secret.len() < 64 {
            return Err("SESSION_SECRET must be at least 64 bytes".to_string());
        }
        Ok(Self {
            database_url,
            session_secret,
            session_ttl_secs: env_or("SESSION_TTL_SECS", "2592000")
                .parse()
                .unwrap_or(2592000),
            cookie_secure: env_or("COOKIE_SECURE", "true").parse().unwrap_or(true),
            cookie_name: env_or("SESSION_COOKIE_NAME", "xz_session"),
        })
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
