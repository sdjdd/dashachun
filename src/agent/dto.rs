use serde::{Deserialize, Serialize};

use garde::Validate;

use super::AgentRecord;

#[derive(Debug, Deserialize, Validate)]
pub struct CreateAgentArgs {
    #[garde(length(min = 1, max = 64))]
    pub name: String,
    #[garde(length(max = 4096))]
    pub persona_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub id: i64,
    pub name: String,
    pub persona_prompt: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: time::OffsetDateTime,
}

impl From<AgentRecord> for AgentResponse {
    fn from(record: AgentRecord) -> Self {
        Self {
            id: record.id,
            name: record.name,
            persona_prompt: record.persona_prompt,
            created_at: record.created_at,
        }
    }
}
