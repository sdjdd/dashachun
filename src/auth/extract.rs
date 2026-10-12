use core::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use axum::extract::{ConnectInfo, FromRef, FromRequestParts};
use axum::http::request::Parts;
use axum_extra::extract::cookie::SignedCookieJar;
use uuid::Uuid;

use crate::error::AppError;

use super::{Role, session, state::AuthState};

pub struct AuthUser {
    pub id: i64,
    pub session_id: Uuid,
}

impl<S> FromRequestParts<S> for AuthUser
where
    AuthState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let state = AuthState::from_ref(state);
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

/// The session-authenticated operator: [`AuthUser`]'s session verification
/// plus one role lookup. Non-admins get 403 before the handler runs. The
/// operator surface — `/api/providers` and `/api/settings/*` — takes this
/// extractor instead of [`AuthUser`]; both reads and writes sit behind it,
/// because provider rows return their config (api keys included) verbatim.
pub struct AdminUser {
    pub id: i64,
    pub session_id: Uuid,
}

impl<S> FromRequestParts<S> for AdminUser
where
    AuthState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let user = AuthUser::from_request_parts(parts, state).await?;
        let state = AuthState::from_ref(state);
        let role = super::user_role(&state.pool, user.id).await?;
        if role != Role::Admin {
            return Err(AppError::Forbidden);
        }
        Ok(Self {
            id: user.id,
            session_id: user.session_id,
        })
    }
}

/// The client IP for rate limiting: with `TRUST_PROXY` the rightmost
/// `X-Forwarded-For` entry (the one the trusted edge appended), otherwise
/// the socket peer. Falls back to `0.0.0.0` when neither is present (e.g.
/// the `oneshot` test path, which has no connect info) — never fails.
pub struct ClientIp(pub IpAddr);

impl<S> FromRequestParts<S> for ClientIp
where
    AuthState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let state = AuthState::from_ref(state);
        if state.trust_proxy
            && let Some(forwarded) = parts.headers.get("x-forwarded-for")
            && let Ok(raw) = forwarded.to_str()
            && let Some(ip) = raw.rsplit(',').next().and_then(|s| s.trim().parse().ok())
        {
            return Ok(ClientIp(ip));
        }
        let ip = parts
            .extensions
            .get::<ConnectInfo<SocketAddr>>()
            .map(|info| info.0.ip())
            .unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        Ok(ClientIp(ip))
    }
}
