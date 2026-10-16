//! The boot subscriber: human-readable by default, JSON under
//! `TELEMETRY_JSON`. The conventions spans (telemetry.md) are plain
//! `tracing` spans — this module is where the future
//! `tracing-opentelemetry` bridge mounts, beside the same subscriber.

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
    if json_enabled(env::var("TELEMETRY_JSON").ok()) {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

/// `TELEMETRY_JSON` parses as a plain `bool` — an unset or unparseable
/// value counts as the default `false` (human-readable), the same pattern
/// as the other boot envs.
fn json_enabled(value: Option<String>) -> bool {
    value.and_then(|value| value.parse().ok()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::json_enabled;

    #[test]
    fn unset_or_garbage_stays_human_readable() {
        assert!(!json_enabled(None));
        assert!(!json_enabled(Some("false".into())));
        assert!(!json_enabled(Some("garbage".into())));
        assert!(!json_enabled(Some("".into())));
    }

    #[test]
    fn true_switches_to_json() {
        assert!(json_enabled(Some("true".into())));
    }
}
