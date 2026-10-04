use std::sync::Arc;

use crate::asr::Asr;
use crate::config::AppConfig;

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub asr: Arc<dyn Asr>,
}
