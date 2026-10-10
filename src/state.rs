use std::sync::{Arc, RwLock};

use arc_swap::ArcSwapOption;
use tokio::sync::{Mutex, watch};

use crate::agent::{AgentFactory, AudioStorage, ToolRegistry};
use crate::config::AppConfig;
use crate::device::DeviceStore;
use crate::provider::ProviderRegistry;
use crate::vad::VadFactory;

#[derive(Clone)]
pub struct ServerState {
    pub config: AppConfig,
    pub registry: Arc<ProviderRegistry>,
    pub vad: Arc<dyn VadFactory>,
    pub tools: Arc<ToolRegistry>,
    audio_storage: Arc<RwLock<Option<Arc<dyn AudioStorage>>>>,
    factories: Arc<ArcSwapOption<AgentFactory>>,
    pub swap_lock: Arc<Mutex<()>>,
    pub devices: DeviceStore,
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
}

impl ServerState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: AppConfig,
        registry: Arc<ProviderRegistry>,
        vad: Arc<dyn VadFactory>,
        tools: Arc<ToolRegistry>,
        audio_storage: Option<Arc<dyn AudioStorage>>,
        factories: Arc<ArcSwapOption<AgentFactory>>,
        devices: DeviceStore,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            config,
            registry,
            vad,
            tools,
            audio_storage: Arc::new(RwLock::new(audio_storage)),
            factories,
            swap_lock: Arc::new(Mutex::new(())),
            devices,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
        }
    }

    /// A snapshot of the current audio storage — `None` while the feature
    /// is disabled. Factories assemble with it; in-flight sessions keep
    /// their clone across a swap, exactly like provider generations.
    pub fn audio_storage(&self) -> Option<Arc<dyn AudioStorage>> {
        self.audio_storage.read().unwrap().clone()
    }

    /// Replaces the audio storage slot — the `/api/settings/audio-capture` apply
    /// path. The previous storage drains with the sessions that hold it.
    pub fn set_audio_storage(&self, next: Option<Arc<dyn AudioStorage>>) {
        *self.audio_storage.write().unwrap() = next;
    }

    /// A snapshot of the current agent generation — `None` until a provider
    /// is selected for every type; the gateway rejects devices with 503
    /// while it is.
    pub fn agent_factory(&self) -> Option<Arc<AgentFactory>> {
        self.factories.load_full()
    }

    /// Publishes a new generation. Callers hold `swap_lock` across the
    /// write-then-swap sequence so activations serialize.
    pub fn swap_agent_factory(&self, next: Arc<AgentFactory>) {
        let previous = self.factories.swap(Some(next));
        tracing::info!("agent factory swapped");
        if let Some(previous) = previous {
            previous.drain_when_orphaned();
        }
    }

    pub fn shutdown_sender(&self) -> Arc<watch::Sender<bool>> {
        self.shutdown_tx.clone()
    }

    pub fn shutdown_signal(&self) -> watch::Receiver<bool> {
        self.shutdown_rx.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentStore;
    use crate::config::{DeviceConfig, ServerConfig};
    use crate::provider::ProviderSpec;
    use crate::settings::OtaSettings;
    use crate::vad::{SileroVadFactory, VadConfig};

    fn test_config() -> AppConfig {
        AppConfig {
            server: ServerConfig {
                bind_addr: "127.0.0.1:0".into(),
                playback_prebuffer_ms: 180,
                shutdown_grace_ms: 5000,
            },
            ota: OtaSettings {
                websocket_url: None,
                timezone_offset: 480,
            },
            device: DeviceConfig {
                activation_ttl_secs: 600,
            },
        }
    }

    fn stub_parts() -> (
        Arc<ProviderRegistry>,
        Arc<dyn VadFactory>,
        Arc<ToolRegistry>,
    ) {
        (
            Arc::new(ProviderRegistry::with_stubs()),
            Arc::new(SileroVadFactory::new(VadConfig::default())),
            Arc::new(ToolRegistry::new(Vec::new())),
        )
    }

    fn stub_factory() -> AgentFactory {
        let pool = sqlx::PgPool::connect_lazy("postgres://unused@localhost/unused").unwrap();
        let (registry, vad, tools) = stub_parts();
        AgentFactory::assemble(
            &registry,
            &ProviderSpec::stub(),
            &ProviderSpec::stub(),
            &ProviderSpec::stub(),
            vad,
            tools,
            AgentStore::new(pool.clone()),
            pool,
        )
        .unwrap()
    }

    fn test_state(generation: Option<AgentFactory>) -> ServerState {
        let pool = sqlx::PgPool::connect_lazy("postgres://unused@localhost/unused").unwrap();
        let (registry, vad, tools) = stub_parts();
        let factories = match generation {
            Some(factory) => Arc::new(ArcSwapOption::from_pointee(factory)),
            None => Arc::new(ArcSwapOption::empty()),
        };
        ServerState::new(
            test_config(),
            registry,
            vad,
            tools,
            None,
            factories,
            DeviceStore::new(
                pool,
                redis::Client::open("redis://127.0.0.1:6379").unwrap(),
                "state-test",
            ),
        )
    }

    #[tokio::test]
    async fn swap_publishes_the_new_generation() {
        let state = test_state(Some(stub_factory()));
        let before = state.agent_factory().unwrap();
        let next = Arc::new(stub_factory());
        state.swap_agent_factory(next.clone());
        let after = state.agent_factory().unwrap();
        assert!(Arc::ptr_eq(&after, &next));
        assert!(!Arc::ptr_eq(&after, &before));
    }

    #[tokio::test]
    async fn empty_state_publishes_none_until_swapped() {
        let state = test_state(None);
        assert!(state.agent_factory().is_none());
        let next = Arc::new(stub_factory());
        state.swap_agent_factory(next.clone());
        assert!(Arc::ptr_eq(&state.agent_factory().unwrap(), &next));
    }
}
