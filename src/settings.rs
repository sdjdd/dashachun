use std::fmt;

use rand::TryRngCore;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

const LLM_KEY: &str = "llm";
const ASR_KEY: &str = "asr";
const TTS_KEY: &str = "tts";
const OTA_KEY: &str = "ota";
const SECURITY_KEY: &str = "security";

const SESSION_SECRET_BYTES: usize = 64;
const MIN_SESSION_SECRET_LEN: usize = 64;

#[derive(Clone, Serialize, Deserialize)]
pub struct LlmSettings {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

impl fmt::Debug for LlmSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmSettings")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("max_tokens", &self.max_tokens)
            .field("reasoning_effort", &self.reasoning_effort)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AsrSettings {
    pub base_url: String,
    pub api_key: String,
    pub resource_id: String,
}

impl fmt::Debug for AsrSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsrSettings")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("resource_id", &self.resource_id)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TtsSettings {
    pub base_url: String,
    pub api_key: String,
    pub speaker: String,
    pub resource_id: String,
}

impl fmt::Debug for TtsSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TtsSettings")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("speaker", &self.speaker)
            .field("resource_id", &self.resource_id)
            .finish()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OtaSettings {
    pub websocket_url: Option<String>,
    #[serde(default = "default_timezone_offset")]
    pub timezone_offset: i32,
}

fn default_timezone_offset() -> i32 {
    480
}

impl Default for OtaSettings {
    fn default() -> Self {
        Self {
            websocket_url: None,
            timezone_offset: default_timezone_offset(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SecuritySettings {
    pub session_secret: String,
}

impl fmt::Debug for SecuritySettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecuritySettings")
            .field("session_secret", &"<redacted>")
            .finish()
    }
}

#[derive(Default)]
pub struct Settings {
    pub llm: Option<LlmSettings>,
    pub asr: Option<AsrSettings>,
    pub tts: Option<TtsSettings>,
    pub ota: Option<OtaSettings>,
    session_secret: Option<String>,
}

impl fmt::Debug for Settings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Settings")
            .field("llm", &self.llm)
            .field("asr", &self.asr)
            .field("tts", &self.tts)
            .field("ota", &self.ota)
            .field(
                "session_secret",
                &self.session_secret.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

impl Settings {
    pub async fn load(pool: &PgPool) -> Result<Self, String> {
        let rows: Vec<(String, serde_json::Value)> =
            sqlx::query_as("SELECT key, value FROM settings")
                .fetch_all(pool)
                .await
                .map_err(|err| format!("failed to load settings: {err}"))?;
        let mut settings = Self::default();
        for (key, value) in rows {
            match key.as_str() {
                LLM_KEY => settings.llm = Some(parse_row(LLM_KEY, value)?),
                ASR_KEY => settings.asr = Some(parse_row(ASR_KEY, value)?),
                TTS_KEY => settings.tts = Some(parse_row(TTS_KEY, value)?),
                OTA_KEY => settings.ota = Some(parse_row(OTA_KEY, value)?),
                SECURITY_KEY => {
                    settings.session_secret =
                        Some(parse_row::<SecuritySettings>(SECURITY_KEY, value)?.session_secret)
                }
                unknown => tracing::warn!(key = %unknown, "unknown settings key, ignoring"),
            }
        }
        Ok(settings)
    }

    pub async fn ensure_session_secret(&mut self, pool: &PgPool) -> Result<String, String> {
        let secret = match self.session_secret.clone() {
            Some(secret) => secret,
            None => generate_session_secret(),
        };
        if secret.len() < MIN_SESSION_SECRET_LEN {
            return Err(format!(
                "session secret must be at least {MIN_SESSION_SECRET_LEN} bytes, got {}",
                secret.len()
            ));
        }
        let value = serde_json::to_value(SecuritySettings {
            session_secret: secret.clone(),
        })
        .map_err(|err| format!("failed to encode session secret: {err}"))?;
        sqlx::query(
            "INSERT INTO settings (key, value, updated_at) VALUES ($1, $2, now())
             ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()",
        )
        .bind(SECURITY_KEY)
        .bind(value)
        .execute(pool)
        .await
        .map_err(|err| format!("failed to store session secret: {err}"))?;
        self.session_secret = Some(secret.clone());
        Ok(secret)
    }
}

fn parse_row<T: serde::de::DeserializeOwned>(
    key: &str,
    value: serde_json::Value,
) -> Result<T, String> {
    serde_json::from_value(value).map_err(|err| format!("invalid settings row \"{key}\": {err}"))
}

fn generate_session_secret() -> String {
    let mut bytes = [0u8; SESSION_SECRET_BYTES];
    OsRng
        .try_fill_bytes(&mut bytes)
        .expect("os rng is infallible");
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_json_defaults_optional_fields() {
        let llm: LlmSettings = serde_json::from_value(
            serde_json::json!({"base_url": "b", "api_key": "k", "model": "m"}),
        )
        .unwrap();
        assert_eq!(llm.base_url, "b");
        assert_eq!(llm.max_tokens, None);
        assert_eq!(llm.reasoning_effort, None);
    }

    #[test]
    fn tts_json_requires_the_resource_id() {
        let err = serde_json::from_value::<TtsSettings>(serde_json::json!({
            "base_url": "wss://tts",
            "api_key": "k",
            "speaker": "s"
        }))
        .unwrap_err();
        assert!(err.to_string().contains("resource_id"), "{err}");
    }

    #[test]
    fn asr_json_requires_the_resource_id() {
        let err = serde_json::from_value::<AsrSettings>(serde_json::json!({
            "base_url": "wss://asr",
            "api_key": "k"
        }))
        .unwrap_err();
        assert!(err.to_string().contains("resource_id"), "{err}");
    }

    #[test]
    fn ota_json_defaults_timezone_offset() {
        let ota: OtaSettings = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(ota.websocket_url, None);
        assert_eq!(ota.timezone_offset, 480);
    }

    #[test]
    fn llm_json_requires_required_fields() {
        let err = serde_json::from_value::<LlmSettings>(serde_json::json!({"base_url": "b"}))
            .unwrap_err();
        assert!(err.to_string().contains("api_key"), "{err}");
    }

    #[test]
    fn generated_session_secret_is_long_hex() {
        let secret = generate_session_secret();
        assert_eq!(secret.len(), SESSION_SECRET_BYTES * 2);
        assert!(secret.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn short_session_secret_is_rejected() {
        let mut settings = Settings {
            session_secret: Some("too-short".into()),
            ..Settings::default()
        };
        let pool = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgres://unused@localhost/unused")
            .unwrap();
        let err = settings.ensure_session_secret(&pool).await.unwrap_err();
        assert!(err.contains("at least 64"), "unexpected error: {err}");
    }
}
