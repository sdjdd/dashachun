use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[derive(Debug)]
pub enum AppError {
    MissingHeader(&'static str),
    InvalidHeader(&'static str),
    BadRequest(String),
    Unauthorized,
    TooManyRequests(u64),
    Conflict(String),
    NotFound,
    Internal(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::MissingHeader(name) => write!(f, "Missing {name} header"),
            AppError::InvalidHeader(name) => write!(f, "Invalid {name} header"),
            AppError::BadRequest(msg) => write!(f, "Bad request: {msg}"),
            AppError::Unauthorized => write!(f, "Unauthorized"),
            AppError::TooManyRequests(retry_after) => {
                write!(f, "Too many requests, retry after {retry_after} seconds")
            }
            AppError::Conflict(msg) => write!(f, "Conflict: {msg}"),
            AppError::NotFound => write!(f, "Not found"),
            AppError::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for AppError {}

impl From<sqlx::Error> for AppError {
    fn from(err: sqlx::Error) -> Self {
        AppError::Internal(format!("database error: {err}"))
    }
}

impl From<redis::RedisError> for AppError {
    fn from(err: redis::RedisError) -> Self {
        AppError::Internal(format!("redis error: {err}"))
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        if let AppError::TooManyRequests(retry_after) = &self {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [("retry-after", retry_after.to_string())],
                self.to_string(),
            )
                .into_response();
        }
        let status = match self {
            AppError::MissingHeader(_) | AppError::InvalidHeader(_) | AppError::BadRequest(_) => {
                StatusCode::BAD_REQUEST
            }
            AppError::Unauthorized => StatusCode::UNAUTHORIZED,
            AppError::Conflict(_) => StatusCode::CONFLICT,
            AppError::NotFound => StatusCode::NOT_FOUND,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::TooManyRequests(_) => unreachable!(),
        };
        (status, self.to_string()).into_response()
    }
}
