use serde::{Deserialize, Serialize};

use garde::Validate;

#[derive(Debug, Deserialize, Validate)]
pub struct RegisterArgs {
    #[garde(pattern(r"^[A-Za-z0-9_-]{3,32}$"))]
    pub username: String,
    #[garde(length(min = 8, max = 128))]
    pub password: String,
}

#[derive(Deserialize, Validate)]
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

#[derive(Deserialize, Validate)]
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

#[derive(Debug, Serialize)]
pub struct UserResponse {
    pub id: i64,
    pub username: String,
}
