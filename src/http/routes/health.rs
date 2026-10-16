//! The load-balancer health probe. Infrastructure surface, deliberately
//! outside the OpenAPI document like the gateway.

use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::Serialize;

use crate::state::ServerState;

const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

pub fn routes() -> Router<ServerState> {
    Router::new().route("/", get(health))
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
}

/// Readiness for LB health checks: one cheap Postgres probe, because every
/// serving surface runs through the pool. Redis is deliberately not probed:
/// its runtime failures fail open (the rate limiter serves through them and
/// only activation codes are lost), so letting a Redis blip mark every
/// replica unhealthy would amplify a partial degradation into a full
/// outage. The agent generation's absence is configuration state, not
/// health — the gateway answers 503 for it by itself.
async fn health(State(server): State<ServerState>) -> Response {
    let probe = tokio::time::timeout(
        PROBE_TIMEOUT,
        sqlx::query("SELECT 1").execute(server.devices.pool()),
    )
    .await;
    let status = match probe {
        Ok(Ok(_)) => Health { status: "ok" },
        _ => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(Health {
                    status: "unavailable",
                }),
            )
                .into_response();
        }
    };
    (StatusCode::OK, Json(status)).into_response()
}
