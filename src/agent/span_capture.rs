//! The span-assertion kit, shared by the crate's span-asserting tests: a
//! `tracing_subscriber` layer that captures span names, parents and fields
//! into a shared vec — no new dependencies, no otel SDK (telemetry.md's
//! Testing note). Tests install it through the thread-local
//! `tracing::subscriber::set_default` after `install_global_test_registry`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use tracing::Id;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Record};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// Captures span names, parents and fields through a tracing-subscriber
/// layer — the assertion surface for the conventions spans.
#[derive(Clone)]
pub(crate) struct SpanCapture(Arc<Mutex<Vec<CapturedSpan>>>);

#[derive(Debug, Clone)]
pub(crate) struct CapturedSpan {
    pub(crate) id: u64,
    pub(crate) parent: Option<u64>,
    pub(crate) name: String,
    pub(crate) fields: BTreeMap<String, String>,
}

impl SpanCapture {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }

    pub(crate) fn spans(&self, name: &str) -> Vec<CapturedSpan> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|span| span.name == name)
            .cloned()
            .collect()
    }
}

/// Installs a bare global `Registry` once per test process. The callsite
/// interest cache is process-global and a callsite's first registration is
/// decided by whichever thread reaches it first: a thread without any
/// subscriber answers `Interest::never`, which poisons the callsite for
/// every later span — scoped dispatchers never trigger a rebuild. The bare
/// registry answers `always`, discards everything, and leaves each test's
/// scoped capture in charge.
pub(crate) fn install_global_test_registry() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::subscriber::set_global_default(tracing_subscriber::registry())
            .expect("global test registry");
    });
}

struct FieldWriter<'a>(&'a mut BTreeMap<String, String>);

impl Visit for FieldWriter<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

impl<S> Layer<S> for SpanCapture
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = BTreeMap::new();
        attrs.record(&mut FieldWriter(&mut fields));
        // The resolved parent, not `attrs.parent()` — that one only carries
        // an explicit `parent:`; a contextual parent (the ambient context an
        // instrumented task carries) is resolved by the registry and stored
        // on the span data.
        let parent = ctx
            .span(id)
            .and_then(|span| span.parent())
            .map(|parent| parent.id().into_u64());
        self.0.lock().unwrap().push(CapturedSpan {
            id: id.into_u64(),
            parent,
            name: attrs.metadata().name().to_string(),
            fields,
        });
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut spans = self.0.lock().unwrap();
        if let Some(span) = spans.iter_mut().find(|span| span.id == id.into_u64()) {
            values.record(&mut FieldWriter(&mut span.fields));
        }
    }
}

#[cfg(test)]
mod repro {
    use super::*;
    use tracing::Instrument;
    use tracing_subscriber::layer::SubscriberExt;

    /// Pins the layer's parent capture for the contextual case: a span
    /// created inside an `Instrument`ed spawned task takes the instrumented
    /// span as its parent — the registry resolves it, `attrs.parent()` would
    /// only carry an explicit one.
    #[tokio::test]
    async fn capture_records_contextual_parents() {
        install_global_test_registry();
        let capture = SpanCapture::new();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));
        let span = tracing::info_span!("probe_outer");
        tokio::spawn(
            async move {
                drop(tracing::info_span!("probe_inner"));
            }
            .instrument(span.clone()),
        )
        .await
        .unwrap();
        drop(_guard);
        let outers = capture.spans("probe_outer");
        let inners = capture.spans("probe_inner");
        assert_eq!(outers.len(), 1);
        assert_eq!(inners.len(), 1);
        assert_eq!(inners[0].parent, Some(outers[0].id), "ambient parenting");
    }
}
