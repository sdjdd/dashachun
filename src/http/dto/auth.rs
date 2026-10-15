use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use utoipa::ToSchema;
use uuid::Uuid;

use garde::Validate;

use crate::auth::Role;

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct RegisterArgs {
    #[garde(pattern(r"^[A-Za-z0-9_-]{3,32}$"))]
    pub username: String,
    #[garde(length(min = 8, max = 128))]
    pub password: String,
}

#[derive(Deserialize, Validate, ToSchema)]
pub struct LoginArgs {
    #[garde(pattern(r"^[A-Za-z0-9_-]{3,32}$"))]
    pub username: String,
    #[garde(skip)]
    pub password: String,
}

impl std::fmt::Debug for LoginArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoginArgs")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}

#[derive(Deserialize, Validate, ToSchema)]
pub struct ChangePasswordArgs {
    #[garde(skip)]
    pub current_password: String,
    #[garde(length(min = 8, max = 128))]
    pub new_password: String,
}

impl std::fmt::Debug for ChangePasswordArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChangePasswordArgs")
            .field("current_password", &"<redacted>")
            .field("new_password", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct UserResponse {
    pub id: i64,
    pub username: String,
    pub role: Role,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SessionResponse {
    pub id: Uuid,
    pub ip: String,
    pub user_agent: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub last_active_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub expires_at: OffsetDateTime,
    /// Whether this is the session the request itself arrived on.
    pub current: bool,
}
