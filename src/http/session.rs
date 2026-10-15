//! The cookie session state of the HTTP surface: the signed-cookie key,
//! the session TTL and cookie attributes, the client-IP rate limiter and
//! the proxy-trust switch. Account storage and password/session operations
//! live in the domain module (`crate::auth`).

use axum_extra::extract::cookie::{Cookie, Key, SameSite};
use sqlx::PgPool;
use uuid::Uuid;

use crate::ratelimit::RedisLimiter;

#[derive(Clone)]
pub struct AuthState {
    pub pool: PgPool,
    pub key: Key,
    pub ttl_secs: i64,
    pub cookie_name: String,
    pub cookie_secure: bool,
    pub limiter: RedisLimiter,
    pub trust_proxy: bool,
}

impl AuthState {
    pub fn session_cookie(&self, id: Uuid) -> Cookie<'static> {
        Cookie::build((self.cookie_name.clone(), id.to_string()))
            .path("/")
            .http_only(true)
            .secure(self.cookie_secure)
            .same_site(SameSite::Lax)
            .max_age(time::Duration::seconds(self.ttl_secs))
            .build()
    }

    pub fn removal_cookie(&self) -> Cookie<'static> {
        Cookie::build(self.cookie_name.clone()).path("/").build()
    }
}

impl axum::extract::FromRef<AuthState> for Key {
    fn from_ref(state: &AuthState) -> Self {
        state.key.clone()
    }
}
