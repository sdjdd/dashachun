use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use garde::Validate;

use crate::voiceprint::SpeakerRecord;

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct CreateSpeakerArgs {
    #[garde(length(min = 1, max = 64))]
    pub name: String,
    #[garde(length(max = 200))]
    pub description: Option<String>,
    #[garde(length(min = 1, max = 10))]
    pub message_ids: Vec<i64>,
}

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct UpdateSpeakerArgs {
    #[garde(length(min = 1, max = 64))]
    pub name: Option<String>,
    #[garde(length(max = 200))]
    pub description: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SpeakerResponse {
    pub id: i64,
    pub agent_id: i64,
    pub name: String,
    pub description: String,
    pub sample_count: i32,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: time::OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: time::OffsetDateTime,
}

impl From<SpeakerRecord> for SpeakerResponse {
    fn from(record: SpeakerRecord) -> Self {
        Self {
            id: record.id,
            agent_id: record.agent_id,
            name: record.name,
            description: record.description,
            sample_count: record.sample_count,
            created_at: record.created_at,
            updated_at: record.updated_at,
        }
    }
}
