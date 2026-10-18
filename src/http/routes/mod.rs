//! The axum handlers, one module per route group. Handlers are thin: they
//! extract (session, ownership, validated args), call the domain stores
//! and map the outcome to status codes. Router states live in
//! [`crate::http::state`].

pub mod agents;
pub mod audio_capture;
pub mod auth;
pub mod conversations;
pub mod devices;
pub mod gateway;
pub mod health;
pub mod memories;
pub mod ota;
pub mod providers;
pub mod settings;
pub mod speakers;
pub mod voiceprint;
