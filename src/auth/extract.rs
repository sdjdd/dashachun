use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum_extra::extract::cookie::SignedCookieJar;
use uuid::Uuid;

use crate::error::AppError;

use super::session;
use super::state::AuthState;

pub struct AuthUser {
    pub id: i64,
    pub session_id: Uuid,
}

impl FromRequestParts<AuthState> for AuthUser {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AuthState,
    ) -> Result<Self, Self::Rejection> {
        let jar = SignedCookieJar::from_headers(&parts.headers, state.key.clone());
        let session_id = jar
            .get(&state.cookie_name)
            .and_then(|cookie| Uuid::parse_str(cookie.value()).ok())
            .ok_or(AppError::Unauthorized)?;

        let user_id = session::user_id(&state.pool, session_id)
            .await?
            .ok_or(AppError::Unauthorized)?;

        Ok(AuthUser {
            id: user_id,
            session_id,
        })
    }
}
