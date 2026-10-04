use crate::config::AppConfig;

#[derive(Clone, Debug)]
pub struct AppState {
    pub config: AppConfig,
}
