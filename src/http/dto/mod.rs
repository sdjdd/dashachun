//! HTTP wire types: one module per API group. Every type here is serde +
//! utoipa (`ToSchema`) material — request args (garde-validated at the
//! `Validated<T>` extractor) and response bodies — converted from the
//! domain records at the edge. The OpenAPI root lists them all.

pub mod agent;
pub mod auth;
pub mod device;
pub mod ota;
pub mod provider;
pub mod speaker;
