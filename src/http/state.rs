//! The per-route-group router states: each group needs the session auth
//! plus a slice of the server wiring, and `FromRef` resolution is how the
//! shared extractors (`AuthUser`/`AdminUser`/`ClientIp`) work under every
//! state shape.

use axum::extract::FromRef;

use crate::state::ServerState;

use super::session::AuthState;

/// The agent REST routes need both halves: session auth and the server
/// whose audio-storage and voiceprint-model slots the enrollment POST reads
/// and whose device store the device sub-routes read (the speakers,
/// message-audio and device sub-routes merge in here, so one state shape
/// carries both).
#[derive(Clone)]
pub struct AgentsState {
    pub auth: AuthState,
    pub server: ServerState,
}

impl FromRef<AgentsState> for AuthState {
    fn from_ref(state: &AgentsState) -> Self {
        state.auth.clone()
    }
}

impl FromRef<AgentsState> for ServerState {
    fn from_ref(state: &AgentsState) -> Self {
        state.server.clone()
    }
}

/// Router state for the provider admin API: session auth plus the server
/// wiring (registry, shared VAD/tools and the hot-swap slot) that
/// activation needs to assemble and publish a new agent generation.
#[derive(Clone)]
pub struct ProvidersState {
    pub auth: AuthState,
    pub server: ServerState,
}

impl FromRef<ProvidersState> for AuthState {
    fn from_ref(state: &ProvidersState) -> Self {
        state.auth.clone()
    }
}

impl FromRef<ProvidersState> for ServerState {
    fn from_ref(state: &ProvidersState) -> Self {
        state.server.clone()
    }
}

/// Router state for the settings API (`/api/settings/*`): session auth plus
/// the server handle whose slots the apply paths swap (today audio-capture;
/// the ota section only writes the row).
#[derive(Clone)]
pub struct SettingsState {
    pub auth: AuthState,
    pub server: ServerState,
}

impl FromRef<SettingsState> for AuthState {
    fn from_ref(state: &SettingsState) -> Self {
        state.auth.clone()
    }
}

impl FromRef<SettingsState> for ServerState {
    fn from_ref(state: &SettingsState) -> Self {
        state.server.clone()
    }
}
