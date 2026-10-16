//! The boot subscriber: human-readable by default, JSON under
//! `TELEMETRY_JSON`, with the `tracing-opentelemetry` bridge mounted beside
//! the same registry under `OTEL_EXPORTER_OTLP_ENDPOINT` — the conventions
//! spans (telemetry.md) are plain `tracing` spans, and this is where they
//! become real OTel spans batch-exported over OTLP. The same gate also
//! boots the SDK meter provider behind the GenAI inference metrics the
//! chat loop records. The module also owns the boot env flags the span
//! record sites read: `TELEMETRY_CONTENT` gates the opt-in content
//! attributes.

use std::env;
use std::sync::OnceLock;
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

/// The bridge's provider handle, set at boot when the bridge is on;
/// [`shutdown`] flushes it.
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

/// The bridge's meter provider handle, set at boot when the bridge is on;
/// [`shutdown`] flushes it.
static METER_PROVIDER: OnceLock<SdkMeterProvider> = OnceLock::new();

/// The GenAI inference metric instruments, created once at boot from the
/// global meter — real when the bridge's meter provider is installed
/// first, no-ops otherwise.
static INFERENCE_METRICS: OnceLock<InferenceMetrics> = OnceLock::new();

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
/// same gate installs the SDK meter provider behind the GenAI inference
/// metrics ([`record_chat_round`]) with a periodic reader; the instruments
/// are created regardless and record as no-ops without it. The exporters
/// reuse the crate's reqwest/rustls stack (no second TLS dependency;
/// `cargo tree -i ring` stays empty) and read the standard
/// `OTEL_EXPORTER_OTLP_*` env family themselves (target, headers,
/// protocol, timeout); `OTEL_SERVICE_NAME` names the service in the
/// collector, defaulting to the crate name. The bridge is best-effort
/// telemetry: a failed exporter build only warns and the server runs
/// without a collector, and runtime export failures surface as internal
/// warnings — never down the server. Boot-level env: a restart applies
/// it. Call [`shutdown`] once at the end of `main` so in-flight spans and
/// the last metrics collection reach the collector before exit.
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
        .map(|endpoint| (endpoint, bridge_providers()));
    let tracer = match &bridge {
        Some((_, Ok((tracer, _)))) => Some(tracer.clone()),
        _ => None,
    };
    if let Some(provider) = &tracer {
        let _ = TRACER_PROVIDER.set(provider.clone());
    }
    if let Some((_, Ok((_, meter)))) = &bridge {
        opentelemetry::global::set_meter_provider(meter.clone());
    }
    // The instruments exist regardless of the bridge: real when the meter
    // provider was just installed, no-ops otherwise — the record sites are
    // unconditional.
    let _ = INFERENCE_METRICS.set(InferenceMetrics::from_meter(&opentelemetry::global::meter(
        env!("CARGO_PKG_NAME"),
    )));
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .with(tracer.map(|provider| {
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

/// Flushes and stops the bridge's exporters, if the bridge booted. Called
/// once at the end of `main` so in-flight spans and the final metrics
/// collection reach the collector before the process exits; the
/// forced-exit watchdog path skips this and may drop the last few seconds
/// of telemetry.
pub fn shutdown() {
    if let Some(provider) = TRACER_PROVIDER.get()
        && let Err(err) = provider.shutdown()
    {
        tracing::warn!(%err, "opentelemetry bridge flush failed on shutdown");
    }
    if let Some(provider) = METER_PROVIDER.get()
        && let Err(err) = provider.shutdown()
    {
        tracing::warn!(%err, "opentelemetry metrics flush failed on shutdown");
    }
}

/// Builds the bridge's providers on the shared resource: an OTLP/HTTP
/// (protobuf) span exporter with a batch processor, and a metric exporter
/// with a periodic reader (60s interval, `OTEL_METRIC_EXPORT_INTERVAL`
/// overrides) — both on the crate's reqwest stack, both resolving their
/// target and protocol from the standard `OTEL_EXPORTER_OTLP_*` envs
/// (default `http://localhost:4318`), both named by `OTEL_SERVICE_NAME`.
fn bridge_providers() -> Result<(SdkTracerProvider, SdkMeterProvider), String> {
    let resource = opentelemetry_sdk::Resource::builder()
        .with_service_name(service_name(env::var("OTEL_SERVICE_NAME").ok()))
        .build();
    let span_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
        .map_err(|err| err.to_string())?;
    let metric_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .build()
        .map_err(|err| err.to_string())?;
    let tracer_provider = SdkTracerProvider::builder()
        .with_batch_exporter(span_exporter)
        .with_resource(resource.clone())
        .build();
    let meter_provider = SdkMeterProvider::builder()
        .with_periodic_exporter(metric_exporter)
        .with_resource(resource)
        .build();
    Ok((tracer_provider, meter_provider))
}

/// The advisory explicit bucket boundaries the conventions give
/// `gen_ai.client.inference.duration`.
const INFERENCE_DURATION_BUCKETS: [f64; 14] = [
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];

/// The GenAI inference metric instruments for one `Llm::chat` round
/// (telemetry.md, audited against the conventions repo @ main 2026-10-17):
/// the end-to-end duration histogram and the token usage counters — the
/// primary instruments for token consumption. The per-operation token
/// histograms and the cache/reasoning subsets are deliberately not
/// recorded: the chat span attributes already carry the per-round counts,
/// and no provider exposes cache/reasoning tokens today.
struct InferenceMetrics {
    duration: Histogram<f64>,
    input_tokens: Counter<u64>,
    output_tokens: Counter<u64>,
}

impl InferenceMetrics {
    fn from_meter(meter: &Meter) -> Self {
        Self {
            duration: meter
                .f64_histogram("gen_ai.client.inference.duration")
                .with_unit("s")
                .with_description("GenAI client inference operation duration.")
                .with_boundaries(INFERENCE_DURATION_BUCKETS.to_vec())
                .build(),
            input_tokens: meter
                .u64_counter("gen_ai.client.inference.usage.input_tokens")
                .with_unit("{token}")
                .with_description(
                    "The number of input (prompt) tokens used, including cached tokens.",
                )
                .build(),
            output_tokens: meter
                .u64_counter("gen_ai.client.inference.usage.output_tokens")
                .with_unit("{token}")
                .with_description(
                    "The number of output (completion) tokens used, including reasoning tokens.",
                )
                .build(),
        }
    }

    /// Records one round onto all three instruments, each carrying exactly
    /// its conventioned attribute set — the duration adds `error.type` when
    /// the round failed, the counters add the `gen_ai.token.modality`
    /// dimension: the provider-supplied counts under `text`, or `0` under
    /// `unknown` when no usage arrived (the record-every-round rule,
    /// failed and cancelled rounds included).
    fn record(
        &self,
        provider: &str,
        model: &str,
        response_model: Option<&str>,
        duration: Duration,
        error: Option<&str>,
        usage: Option<(u64, u64)>,
    ) {
        let base = [
            KeyValue::new("gen_ai.operation.name", "chat"),
            KeyValue::new("gen_ai.provider.name", provider.to_string()),
            KeyValue::new("gen_ai.request.model", model.to_string()),
        ];
        let mut duration_attrs = base.to_vec();
        if let Some(model) = response_model {
            duration_attrs.push(KeyValue::new("gen_ai.response.model", model.to_string()));
        }
        if let Some(error) = error {
            duration_attrs.push(KeyValue::new("error.type", error.to_string()));
        }
        self.duration
            .record(duration.as_secs_f64(), &duration_attrs);
        let mut usage_attrs = base.to_vec();
        usage_attrs.push(KeyValue::new(
            "gen_ai.token.modality",
            if usage.is_some() { "text" } else { "unknown" },
        ));
        let (input, output) = usage.unwrap_or((0, 0));
        self.input_tokens.add(input, &usage_attrs);
        self.output_tokens.add(output, &usage_attrs);
    }
}

/// Records one `Llm::chat` round onto the inference metrics (telemetry.md).
/// A no-op until [`init`] created the instruments — and forever a no-op
/// without the bridge's collector, whose absence leaves the global meter a
/// no-op.
pub(crate) fn record_chat_round(
    provider: &str,
    model: &str,
    response_model: Option<&str>,
    duration: Duration,
    error: Option<&str>,
    usage: Option<(u64, u64)>,
) {
    if let Some(metrics) = INFERENCE_METRICS.get() {
        metrics.record(provider, model, response_model, duration, error, usage);
    }
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
    use super::{InferenceMetrics, bridge_endpoint, flag_enabled, service_name};

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

    /// The instruments carry exactly their conventioned names, units and
    /// attribute sets: the duration histogram takes the error class when
    /// the round failed, the usage counters ride the modality dimension —
    /// provider counts under `text`, `0` under `unknown` (the
    /// record-every-round rule, audited @ main 2026-10-17).
    #[test]
    fn inference_metrics_record_the_conventioned_shape() {
        use opentelemetry::KeyValue;
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
        use opentelemetry_sdk::metrics::{InMemoryMetricExporter, SdkMeterProvider};
        use std::collections::BTreeMap;
        use std::time::Duration;

        fn attrs<'a>(points: impl Iterator<Item = &'a KeyValue>) -> BTreeMap<String, String> {
            points
                .map(|kv| (kv.key.as_str().to_string(), kv.value.as_str().to_string()))
                .collect()
        }

        let exporter = InMemoryMetricExporter::default();
        let provider = SdkMeterProvider::builder()
            .with_periodic_exporter(exporter.clone())
            .build();
        let metrics = InferenceMetrics::from_meter(&provider.meter("test"));
        metrics.record(
            "volcengine",
            "doubao-seed",
            Some("doubao-pro"),
            Duration::from_millis(1500),
            None,
            Some((11, 7)),
        );
        metrics.record(
            "volcengine",
            "doubao-seed",
            None,
            Duration::from_millis(50),
            Some("failed"),
            None,
        );
        provider.force_flush().unwrap();

        let mut durations = Vec::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for resource in exporter.get_finished_metrics().unwrap() {
            for scope in resource.scope_metrics() {
                for metric in scope.metrics() {
                    match (metric.name(), metric.data()) {
                        (
                            "gen_ai.client.inference.duration",
                            AggregatedMetrics::F64(MetricData::Histogram(data)),
                        ) => {
                            assert_eq!(metric.unit(), "s");
                            for point in data.data_points() {
                                durations.push((
                                    attrs(point.attributes()),
                                    point.count(),
                                    point.sum(),
                                ));
                            }
                        }
                        (
                            "gen_ai.client.inference.usage.input_tokens",
                            AggregatedMetrics::U64(MetricData::Sum(data)),
                        ) => {
                            assert_eq!(metric.unit(), "{token}");
                            for point in data.data_points() {
                                inputs.push((attrs(point.attributes()), point.value()));
                            }
                        }
                        (
                            "gen_ai.client.inference.usage.output_tokens",
                            AggregatedMetrics::U64(MetricData::Sum(data)),
                        ) => {
                            assert_eq!(metric.unit(), "{token}");
                            for point in data.data_points() {
                                outputs.push((attrs(point.attributes()), point.value()));
                            }
                        }
                        (name, _) => panic!("unexpected metric {name}"),
                    }
                }
            }
        }

        assert_eq!(durations.len(), 2);
        let (clean, clean_count, clean_sum) = durations
            .iter()
            .find(|(attrs, _, _)| !attrs.contains_key("error.type"))
            .unwrap();
        assert_eq!(
            clean.get("gen_ai.operation.name").map(String::as_str),
            Some("chat")
        );
        assert_eq!(
            clean.get("gen_ai.provider.name").map(String::as_str),
            Some("volcengine")
        );
        assert_eq!(
            clean.get("gen_ai.request.model").map(String::as_str),
            Some("doubao-seed")
        );
        assert_eq!(
            clean.get("gen_ai.response.model").map(String::as_str),
            Some("doubao-pro")
        );
        assert_eq!(*clean_count, 1);
        assert!((*clean_sum - 1.5).abs() < 0.1);
        let (failed, _, _) = durations
            .iter()
            .find(|(attrs, _, _)| attrs.contains_key("error.type"))
            .unwrap();
        assert_eq!(failed.get("error.type").map(String::as_str), Some("failed"));
        assert!(!failed.contains_key("gen_ai.response.model"));

        for (points, expected) in [(&inputs, 11u64), (&outputs, 7u64)] {
            assert_eq!(points.len(), 2);
            let text = points.iter().find(|(_, value)| *value == expected).unwrap();
            let unknown = points.iter().find(|(_, value)| *value == 0).unwrap();
            assert_eq!(
                text.0.get("gen_ai.token.modality").map(String::as_str),
                Some("text")
            );
            assert_eq!(
                unknown.0.get("gen_ai.token.modality").map(String::as_str),
                Some("unknown")
            );
        }
    }
}
