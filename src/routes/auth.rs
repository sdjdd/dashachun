use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::{Json, Router, routing::get, routing::post};
use axum_extra::extract::cookie::SignedCookieJar;

use crate::auth::dto::{ChangePasswordArgs, LoginArgs, RegisterArgs, UserResponse};
use crate::auth::extract::{AuthUser, ClientIp};
use crate::auth::state::AuthState;
use crate::auth::{self, password, session};
use crate::error::{AppError, ErrorResponse};
use crate::extract::Validated;
use crate::ratelimit::Decision;

/// Rate limits are a fixed window per (endpoint, client IP).
pub const WINDOW: Duration = Duration::from_secs(60);
pub const REGISTER_LIMIT: u32 = 10;
pub const LOGIN_LIMIT: u32 = 10;
pub const CHANGE_PASSWORD_LIMIT: u32 = 5;

pub fn routes() -> Router<AuthState> {
    Router::new()
        .route("/register", post(register))
        .route("/login", post(login))
        .route("/logout", post(logout))
        .route("/change-password", post(change_password))
        .route("/me", get(me))
}

async fn enforce(
    state: &AuthState,
    bucket: &str,
    ip: &ClientIp,
    limit: u32,
) -> Result<(), AppError> {
    match state
        .limiter
        .hit(bucket, &ip.0.to_string(), limit, WINDOW)
        .await
    {
        Decision::Allowed => Ok(()),
        Decision::Limited { retry_after } => {
            Err(AppError::TooManyRequests(retry_after.as_secs().max(1)))
        }
    }
}

#[utoipa::path(
    post,
    path = "/api/auth/register",
    tag = "auth",
    request_body = RegisterArgs,
    responses(
        (status = 201, body = UserResponse, description = "Account created; session cookie set"),
        (status = 400, body = ErrorResponse, description = "Invalid payload"),
        (status = 409, body = ErrorResponse, description = "Username taken"),
        (status = 429, body = ErrorResponse, description = "Rate limited"),
    ),
)]
pub(crate) async fn register(
    State(state): State<AuthState>,
    ip: ClientIp,
    jar: SignedCookieJar,
    Validated(args): Validated<RegisterArgs>,
) -> Result<(StatusCode, SignedCookieJar, Json<UserResponse>), AppError> {
    enforce(&state, "register", &ip, REGISTER_LIMIT).await?;
    let (user_id, role) = auth::create_user(&state.pool, &args.username, &args.password).await?;
    let session_id = session::create(&state.pool, user_id, state.ttl_secs).await?;
    let user = UserResponse {
        id: user_id,
        username: args.username,
        role,
    };
    Ok((
        StatusCode::CREATED,
        jar.add(state.session_cookie(session_id)),
        Json(user),
    ))
}

#[utoipa::path(
    post,
    path = "/api/auth/login",
    tag = "auth",
    request_body = LoginArgs,
    responses(
        (status = 200, body = UserResponse, description = "Session cookie set"),
        (status = 400, body = ErrorResponse, description = "Invalid payload"),
        (status = 401, body = ErrorResponse, description = "Wrong credentials"),
        (status = 429, body = ErrorResponse, description = "Rate limited"),
    ),
)]
pub(crate) async fn login(
    State(state): State<AuthState>,
    ip: ClientIp,
    jar: SignedCookieJar,
    Validated(args): Validated<LoginArgs>,
) -> Result<(SignedCookieJar, Json<UserResponse>), AppError> {
    enforce(&state, "login", &ip, LOGIN_LIMIT).await?;
    let user = auth::find_user(&state.pool, &args.username).await?;
    let Some(user) = user else {
        password::verify_dummy(&args.password);
        return Err(AppError::Unauthorized);
    };
    if !password::verify(&args.password, &user.password_hash) {
        return Err(AppError::Unauthorized);
    }
    let session_id = session::create(&state.pool, user.id, state.ttl_secs).await?;
    let response = UserResponse {
        id: user.id,
        username: user.username,
        role: user.role,
    };
    Ok((jar.add(state.session_cookie(session_id)), Json(response)))
}

#[utoipa::path(
    post,
    path = "/api/auth/logout",
    tag = "auth",
    security(("cookieAuth" = [])),
    responses(
        (status = 204, description = "Session deleted; cookie removed"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
    ),
)]
pub(crate) async fn logout(
    State(state): State<AuthState>,
    user: AuthUser,
    jar: SignedCookieJar,
) -> Result<(StatusCode, SignedCookieJar), AppError> {
    session::delete(&state.pool, user.session_id).await?;
    Ok((StatusCode::NO_CONTENT, jar.remove(state.removal_cookie())))
}

#[utoipa::path(
    post,
    path = "/api/auth/change-password",
    tag = "auth",
    security(("cookieAuth" = [])),
    request_body = ChangePasswordArgs,
    responses(
        (status = 204, description = "Password changed; other sessions revoked"),
        (status = 400, body = ErrorResponse, description = "Invalid payload"),
        (status = 401, body = ErrorResponse, description = "Not authenticated or wrong current password"),
        (status = 429, body = ErrorResponse, description = "Rate limited"),
    ),
)]
pub(crate) async fn change_password(
    State(state): State<AuthState>,
    ip: ClientIp,
    user: AuthUser,
    Validated(args): Validated<ChangePasswordArgs>,
) -> Result<StatusCode, AppError> {
    enforce(&state, "change-password", &ip, CHANGE_PASSWORD_LIMIT).await?;
    let record = auth::find_user_by_id(&state.pool, user.id).await?;
    let Some(record) = record else {
        return Err(AppError::Unauthorized);
    };
    if !password::verify(&args.current_password, &record.password_hash) {
        return Err(AppError::Unauthorized);
    }
    auth::update_password(&state.pool, user.id, &args.new_password).await?;
    session::delete_others(&state.pool, user.id, user.session_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[utoipa::path(
    get,
    path = "/api/auth/me",
    tag = "auth",
    security(("cookieAuth" = [])),
    responses(
        (status = 200, body = UserResponse, description = "The authenticated account"),
        (status = 401, body = ErrorResponse, description = "Not authenticated"),
    ),
)]
pub(crate) async fn me(
    State(state): State<AuthState>,
    user: AuthUser,
) -> Result<Json<UserResponse>, AppError> {
    let record = auth::find_user_by_id(&state.pool, user.id)
        .await?
        .ok_or(AppError::Unauthorized)?;
    Ok(Json(UserResponse {
        id: record.id,
        username: record.username,
        role: record.role,
    }))
}
