use std::sync::Arc;

use arc_swap::ArcSwap;
use tokio::sync::watch;

use crate::agent::AgentFactory;
use crate::config::AppConfig;
use crate::device::DeviceStore;

#[derive(Clone)]
pub struct ServerState {
    pub config: AppConfig,
    factories: Arc<ArcSwap<AgentFactory>>,
    pub devices: DeviceStore,
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
}

impl ServerState {
    pub fn new(
        config: AppConfig,
        factories: Arc<ArcSwap<AgentFactory>>,
        devices: DeviceStore,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            config,
            factories,
            devices,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
        }
    }

    pub fn agent_factory(&self) -> Arc<AgentFactory> {
        self.factories.load_full()
    }

    pub fn swap_agent_factory(&self, next: Arc<AgentFactory>) {
        let previous = self.factories.swap(next);
        tracing::info!("agent factory swapped");
        previous.drain_when_orphaned();
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
    use crate::agent::{AgentStore, ToolRegistry};
    use crate::config::{DeviceConfig, ServerConfig};
    use crate::provider::{ProviderRegistry, ProviderSpec};
    use crate::settings::OtaSettings;
    use crate::vad::{SileroVadFactory, VadConfig};

    fn stub_factory() -> AgentFactory {
        let pool = sqlx::PgPool::connect_lazy("postgres://unused@localhost/unused").unwrap();
        AgentFactory::assemble(
            &ProviderRegistry::with_stubs(),
            &ProviderSpec::stub(),
            &ProviderSpec::stub(),
            &ProviderSpec::stub(),
            Arc::new(SileroVadFactory::new(VadConfig::default())),
            Arc::new(ToolRegistry::new(Vec::new())),
            AgentStore::new(pool.clone()),
            pool.clone(),
        )
        .unwrap()
    }

    fn test_state() -> ServerState {
        let pool = sqlx::PgPool::connect_lazy("postgres://unused@localhost/unused").unwrap();
        ServerState::new(
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
            },
            Arc::new(ArcSwap::from_pointee(stub_factory())),
            DeviceStore::new(pool),
        )
    }

    #[tokio::test]
    async fn swap_publishes_the_new_generation() {
        let state = test_state();
        let next = Arc::new(stub_factory());
        state.swap_agent_factory(next.clone());
        assert!(Arc::ptr_eq(&state.agent_factory(), &next));
    }
}
