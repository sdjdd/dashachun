use std::env;

use crate::settings::OtaSettings;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub ota: OtaSettings,
    pub device: DeviceConfig,
}

#[derive(Clone)]
pub struct OssConfig {
    pub endpoint: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub force_path_style: bool,
}

impl OssConfig {
    /// `S3_ENDPOINT` presence means the OSS is configured; `capture_enabled`
    /// is the settings-backed audio-capture switch. Storage without the
    /// switch initializes nothing, a switch without storage is a boot error.
    /// The bucket is not env config — it lives in the `audio_capture`
    /// settings section.
    pub fn from_env(capture_enabled: bool) -> Result<Option<Self>, String> {
        let configured = env::var("S3_ENDPOINT")
            .ok()
            .filter(|v| !v.trim().is_empty());
        match (capture_enabled, configured) {
            (true, None) => Err(
                "audio capture is enabled but S3_ENDPOINT is not set: captured audio has nowhere to go"
                    .into(),
            ),
            (false, _) => Ok(None),
            (true, Some(endpoint)) => Ok(Some(Self {
                endpoint,
                region: env_or("S3_REGION", "us-east-1"),
                access_key_id: env_required("S3_ACCESS_KEY_ID")?,
                secret_access_key: env_required("S3_SECRET_ACCESS_KEY")?,
                force_path_style: env_or("S3_FORCE_PATH_STYLE", "true")
                    .parse()
                    .unwrap_or(true),
            })),
        }
    }
}

impl std::fmt::Debug for OssConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OssConfig")
            .field("endpoint", &self.endpoint)
            .field("region", &self.region)
            .field("access_key_id", &self.access_key_id)
            .field("secret_access_key", &"<redacted>")
            .field("force_path_style", &self.force_path_style)
            .finish()
    }
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
    pub trust_proxy: bool,
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
            trust_proxy: env_or("TRUST_PROXY", "false").parse().unwrap_or(false),
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
            .field("trust_proxy", &self.trust_proxy)
            .finish()
    }
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_required(key: &str) -> Result<String, String> {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{key} is required when S3_ENDPOINT is set"))
}
