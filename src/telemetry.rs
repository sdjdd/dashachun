//! The boot subscriber: human-readable by default, JSON under
//! `TELEMETRY_JSON`. The conventions spans (telemetry.md) are plain
//! `tracing` spans — this module is where the future
//! `tracing-opentelemetry` bridge mounts, beside the same subscriber. It
//! also owns the boot env flags the span record sites read:
//! `TELEMETRY_CONTENT` gates the opt-in content attributes.

use std::env;

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

/// Installs the global subscriber from the boot environment. `RUST_LOG`
/// filters both modes identically; unset (or empty/invalid-only) it falls
/// back to an `info` default directive — a successful server must not be
/// silent. `TELEMETRY_JSON=true` switches the formatter to one JSON line
/// per event, each line carrying the enclosing span stack with its
/// recorded fields — the machine-readable stopgap until a collector exists
/// (spans without inner events render no line; the per-span export is the
/// bridge's job). Boot-level env: a restart applies it.
pub fn init() {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    if flag_enabled(env::var("TELEMETRY_JSON").ok()) {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

/// Boot env booleans parse as a plain `bool` — an unset or unparseable
/// value counts as the default `false`, the same pattern as the other
/// boot envs.
fn flag_enabled(value: Option<String>) -> bool {
    value.and_then(|value| value.parse().ok()).unwrap_or(false)
}

/// `TELEMETRY_CONTENT` gates the opt-in content attributes (telemetry.md):
/// the speech spans' `asr.transcript` / `tts.input.text` and, when they
/// arrive, the chat span's message attributes. Read at the record sites
/// across the process; boot-level env — a restart applies it.
pub(crate) fn content_enabled() -> bool {
    flag_enabled(env::var("TELEMETRY_CONTENT").ok())
}

#[cfg(test)]
mod tests {
    use super::flag_enabled;

    #[test]
    fn unset_or_garbage_stays_off() {
        assert!(!flag_enabled(None));
        assert!(!flag_enabled(Some("false".into())));
        assert!(!flag_enabled(Some("garbage".into())));
        assert!(!flag_enabled(Some("".into())));
    }

    #[test]
    fn true_switches_on() {
        assert!(flag_enabled(Some("true".into())));
    }
}
