use std::fmt;

use rand::TryRngCore;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

const ASR_PROVIDER_ID_KEY: &str = "asr_provider_id";
const LLM_PROVIDER_ID_KEY: &str = "llm_provider_id";
const TTS_PROVIDER_ID_KEY: &str = "tts_provider_id";
const OTA_KEY: &str = "ota";
const SECURITY_KEY: &str = "security";

const SESSION_SECRET_BYTES: usize = 64;
const MIN_SESSION_SECRET_LEN: usize = 64;

/// The `providers` catalog row a stage uses, referenced by its `id` (stable
/// across renames; unique per row). The `{asr,llm,tts}_provider_id` settings
/// rows hold this id; activating a provider rewrites its type's row.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProviderSelections {
    pub asr: Option<i64>,
    pub llm: Option<i64>,
    pub tts: Option<i64>,
}

impl ProviderSelections {
    pub fn selected(&self, r#type: &str) -> Option<i64> {
        match r#type {
            "asr" => self.asr,
            "llm" => self.llm,
            "tts" => self.tts,
            _ => None,
        }
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
    pub asr: Option<i64>,
    pub llm: Option<i64>,
    pub tts: Option<i64>,
    pub ota: Option<OtaSettings>,
    session_secret: Option<String>,
}

impl fmt::Debug for Settings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Settings")
            .field("asr", &self.asr)
            .field("llm", &self.llm)
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
                ASR_PROVIDER_ID_KEY => {
                    settings.asr = Some(parse_row::<i64>(ASR_PROVIDER_ID_KEY, value)?)
                }
                LLM_PROVIDER_ID_KEY => {
                    settings.llm = Some(parse_row::<i64>(LLM_PROVIDER_ID_KEY, value)?)
                }
                TTS_PROVIDER_ID_KEY => {
                    settings.tts = Some(parse_row::<i64>(TTS_PROVIDER_ID_KEY, value)?)
                }
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

/// Stores a type's provider selection — the `activate` path of the
/// provider admin API. The value is the selected provider's id (a JSONB
/// number, stable across renames).
pub async fn set_provider_selection(pool: &PgPool, r#type: &str, id: i64) -> Result<(), String> {
    let key = provider_key(r#type).ok_or_else(|| format!("unknown provider type \"{type}\""))?;
    sqlx::query(
        "INSERT INTO settings (key, value, updated_at) VALUES ($1, to_jsonb($2::bigint), now()) \
         ON CONFLICT (key) DO UPDATE SET value = to_jsonb($2::bigint), updated_at = now()",
    )
    .bind(key)
    .bind(id)
    .execute(pool)
    .await
    .map_err(|err| format!("failed to store the {type} provider selection: {err}"))?;
    Ok(())
}

fn provider_key(r#type: &str) -> Option<&'static str> {
    match r#type {
        "asr" => Some(ASR_PROVIDER_ID_KEY),
        "llm" => Some(LLM_PROVIDER_ID_KEY),
        "tts" => Some(TTS_PROVIDER_ID_KEY),
        _ => None,
    }
}

/// Reads just the three type selections (as provider ids), for resolving
/// the generation to assemble.
pub async fn load_provider_selections(pool: &PgPool) -> Result<ProviderSelections, String> {
    let rows: Vec<(String, serde_json::Value)> = sqlx::query_as(
        "SELECT key, value FROM settings WHERE key IN ('asr_provider_id', 'llm_provider_id', 'tts_provider_id')",
    )
    .fetch_all(pool)
    .await
    .map_err(|err| format!("failed to load provider selections: {err}"))?;
    let mut selections = ProviderSelections::default();
    for (key, value) in rows {
        let id = parse_row::<i64>(&key, value)?;
        match key.as_str() {
            ASR_PROVIDER_ID_KEY => selections.asr = Some(id),
            LLM_PROVIDER_ID_KEY => selections.llm = Some(id),
            TTS_PROVIDER_ID_KEY => selections.tts = Some(id),
            _ => {}
        }
    }
    Ok(selections)
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
    fn ota_json_defaults_timezone_offset() {
        let ota: OtaSettings = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(ota.websocket_url, None);
        assert_eq!(ota.timezone_offset, 480);
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
