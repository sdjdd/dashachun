pub mod validated;

pub use validated::Validated;

use axum::extract::FromRequestParts;

use crate::error::AppError;

macro_rules! header_extractor {
    ($name:ident, $header:literal) => {
        pub struct $name(pub String);

        impl<S> FromRequestParts<S> for $name
        where
            S: Send + Sync,
        {
            type Rejection = AppError;

            async fn from_request_parts(
                parts: &mut axum::http::request::Parts,
                _state: &S,
            ) -> Result<Self, Self::Rejection> {
                let value = parts
                    .headers
                    .get($header)
                    .ok_or(AppError::MissingHeader($header))?
                    .to_str()
                    .map_err(|_| AppError::InvalidHeader($header))?;

                Ok($name(value.to_owned()))
            }
        }
    };
}

header_extractor!(ClientId, "client-id");
header_extractor!(DeviceId, "device-id");
