use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use garde::Validate;

use super::store::VoiceprintRecord;

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct CreateVoiceprintArgs {
    #[garde(length(min = 1, max = 64))]
    pub label: String,
    #[garde(length(max = 200))]
    pub description: Option<String>,
    #[garde(length(min = 1, max = 10))]
    pub message_ids: Vec<i64>,
}

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct UpdateVoiceprintArgs {
    #[garde(length(min = 1, max = 64))]
    pub label: Option<String>,
    #[garde(length(max = 200))]
    pub description: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct VoiceprintResponse {
    pub id: i64,
    pub agent_id: i64,
    pub label: String,
    pub description: String,
    pub sample_count: i32,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: time::OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: time::OffsetDateTime,
}

impl From<VoiceprintRecord> for VoiceprintResponse {
    fn from(record: VoiceprintRecord) -> Self {
        Self {
            id: record.id,
            agent_id: record.agent_id,
            label: record.label,
            description: record.description,
            sample_count: record.sample_count,
            created_at: record.created_at,
            updated_at: record.updated_at,
        }
    }
}
