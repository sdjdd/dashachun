use axum::extract::{FromRequest, Request};
use garde::Validate;

use crate::error::AppError;

pub struct Validated<T>(pub T);

impl<S, T> FromRequest<S> for Validated<T>
where
    S: Send + Sync,
    T: serde::de::DeserializeOwned + Validate<Context = ()>,
{
    type Rejection = AppError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let axum::Json(value) = axum::Json::<T>::from_request(req, state)
            .await
            .map_err(|err| AppError::BadRequest(err.to_string()))?;
        value
            .validate()
            .map_err(|report| AppError::BadRequest(report.to_string()))?;
        Ok(Validated(value))
    }
}
