use std::sync::Arc;

use crate::agent::Agent;
use crate::config::AppConfig;

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub agent: Arc<dyn Agent>,
}
