use std::sync::Arc;

use tokio::sync::watch;

use crate::agent::Agent;
use crate::config::AppConfig;
use crate::device::DeviceStore;

#[derive(Clone)]
pub struct ServerState {
    pub config: AppConfig,
    pub agent: Arc<dyn Agent>,
    pub devices: Option<DeviceStore>,
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
}

impl ServerState {
    pub fn new(config: AppConfig, agent: Arc<dyn Agent>) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            config,
            agent,
            devices: None,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
        }
    }

    pub fn shutdown_sender(&self) -> Arc<watch::Sender<bool>> {
        self.shutdown_tx.clone()
    }

    pub fn shutdown_signal(&self) -> watch::Receiver<bool> {
        self.shutdown_rx.clone()
    }
}
