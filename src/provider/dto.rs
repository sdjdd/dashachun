use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use garde::Validate;

use super::store::ProviderRecord;

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct CreateProviderArgs {
    #[garde(length(min = 1))]
    pub r#type: String,
    #[garde(length(min = 1, max = 64))]
    pub name: String,
    #[garde(length(min = 1))]
    pub kind: Option<String>,
    #[garde(skip)]
    pub config: serde_json::Value,
}

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct UpdateProviderArgs {
    #[garde(length(min = 1, max = 64))]
    pub name: Option<String>,
    #[garde(length(min = 1))]
    pub kind: Option<String>,
    #[garde(skip)]
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ProviderResponse {
    pub id: i64,
    pub r#type: String,
    pub name: String,
    pub kind: String,
    pub config: serde_json::Value,
    pub selected: bool,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: time::OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: time::OffsetDateTime,
}

impl ProviderResponse {
    /// `selected` is resolved by the caller from the type selections —
    /// the response reports catalog state plus whether the server uses
    /// this row right now.
    pub fn new(record: ProviderRecord, selected: bool) -> Self {
        Self {
            id: record.id,
            r#type: record.r#type,
            name: record.name,
            kind: record.kind,
            config: record.config,
            selected,
            created_at: record.created_at,
            updated_at: record.updated_at,
        }
    }
}
