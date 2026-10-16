//! The boot subscriber: human-readable by default, JSON under
//! `TELEMETRY_JSON`, with the `tracing-opentelemetry` bridge mounted beside
//! the same registry under `OTEL_EXPORTER_OTLP_ENDPOINT` — the conventions
//! spans (telemetry.md) are plain `tracing` spans, and this is where they
//! become real OTel spans batch-exported over OTLP. The module also owns
//! the boot env flags the span record sites read: `TELEMETRY_CONTENT`
//! gates the opt-in content attributes.

use std::env;
use std::sync::OnceLock;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

/// The bridge's provider handle, set at boot when the bridge is on;
/// [`shutdown`] flushes it.
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

/// Installs the global subscriber from the boot environment. `RUST_LOG`
/// filters every layer identically; unset (or empty/invalid-only) it falls
/// back to an `info` default directive — a successful server must not be
/// silent. `TELEMETRY_JSON=true` switches the formatter to one JSON line
/// per event, each line carrying the enclosing span stack with its
/// recorded fields.
///
/// `OTEL_EXPORTER_OTLP_ENDPOINT` (set and non-empty) mounts the
/// `tracing-opentelemetry` bridge on the same registry: the conventions
/// spans export unchanged — fields become attributes, `otel.name` /
/// `otel.status_code` map to the OTel span name and status — over
/// OTLP/HTTP (protobuf) through a batch exporter on its own thread. The
/// exporter reuses the crate's reqwest/rustls stack (no second TLS
/// dependency; `cargo tree -i ring` stays empty) and reads the standard
/// `OTEL_EXPORTER_OTLP_*` env family itself (target, headers, protocol,
/// timeout); `OTEL_SERVICE_NAME` names the service in the collector,
/// defaulting to the crate name. The bridge is best-effort telemetry: a
/// failed exporter build only warns and the server runs without a
/// collector, and runtime export failures surface as internal warnings —
/// never down the server. Boot-level env: a restart applies it. Call
/// [`shutdown`] once at the end of `main` so in-flight spans reach the
/// collector before exit.
pub fn init() {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let fmt = if flag_enabled(env::var("TELEMETRY_JSON").ok()) {
        tracing_subscriber::fmt::layer().json().boxed()
    } else {
        tracing_subscriber::fmt::layer().boxed()
    };
    let bridge = bridge_endpoint(env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok())
        .map(|endpoint| (endpoint, bridge_provider()));
    let provider = match &bridge {
        Some((_, Ok(provider))) => Some(provider.clone()),
        _ => None,
    };
    if let Some(provider) = &provider {
        let _ = TRACER_PROVIDER.set(provider.clone());
    }
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .with(provider.map(|provider| {
            tracing_opentelemetry::layer().with_tracer(provider.tracer(env!("CARGO_PKG_NAME")))
        }))
        .init();
    // The bridge's outcome logs land after the subscriber exists, so they
    // surface like any other boot event.
    match bridge {
        Some((endpoint, Ok(_))) => {
            tracing::info!(endpoint = %endpoint, "opentelemetry bridge on");
        }
        Some((endpoint, Err(err))) => {
            tracing::warn!(
                endpoint = %endpoint,
                %err,
                "opentelemetry bridge failed to build; continuing without a collector"
            );
        }
        None => {}
    }
}

/// Flushes and stops the bridge's batch exporter, if the bridge booted.
/// Called once at the end of `main` so in-flight spans reach the collector
/// before the process exits; the forced-exit watchdog path skips this and
/// may drop the last few seconds of spans.
pub fn shutdown() {
    if let Some(provider) = TRACER_PROVIDER.get()
        && let Err(err) = provider.shutdown()
    {
        tracing::warn!(%err, "opentelemetry bridge flush failed on shutdown");
    }
}

/// Builds the bridge's provider: an OTLP/HTTP (protobuf) exporter on the
/// crate's reqwest stack (the exporter resolves its target and protocol
/// from the standard `OTEL_EXPORTER_OTLP_*` envs, default
/// `http://localhost:4318`) and a batch processor on the resource named by
/// `OTEL_SERVICE_NAME`.
fn bridge_provider() -> Result<SdkTracerProvider, String> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
        .map_err(|err| err.to_string())?;
    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name(service_name(env::var("OTEL_SERVICE_NAME").ok()))
                .build(),
        )
        .build())
}

/// The bridge gate: `OTEL_EXPORTER_OTLP_ENDPOINT` set and non-empty — an
/// empty value counts as unset, the same pattern as the other boot envs.
fn bridge_endpoint(value: Option<String>) -> Option<String> {
    value.filter(|endpoint| !endpoint.trim().is_empty())
}

/// The bridge's resource service name: the standard SDK-spec
/// `OTEL_SERVICE_NAME`, defaulting to the crate name.
fn service_name(value: Option<String>) -> String {
    value
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| env!("CARGO_PKG_NAME").to_string())
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
    use super::{bridge_endpoint, flag_enabled, service_name};

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

    #[test]
    fn empty_endpoint_counts_as_unset() {
        assert_eq!(bridge_endpoint(None), None);
        assert_eq!(bridge_endpoint(Some(String::new())), None);
        assert_eq!(bridge_endpoint(Some("   ".into())), None);
        assert_eq!(
            bridge_endpoint(Some("http://localhost:4317".into())),
            Some("http://localhost:4317".into())
        );
    }

    #[test]
    fn service_name_defaults_to_the_crate() {
        assert_eq!(service_name(None), "dashachun");
        assert_eq!(service_name(Some("   ".into())), "dashachun");
        assert_eq!(service_name(Some("dsc".into())), "dsc");
    }
}
