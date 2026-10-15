use serde::Serialize;
use utoipa::ToSchema;

use crate::agent::MemoryEntry;

#[derive(Debug, Serialize, ToSchema)]
pub struct MemoryResponse {
    pub mem_no: i32,
    pub content: String,
}

impl From<MemoryEntry> for MemoryResponse {
    fn from(entry: MemoryEntry) -> Self {
        Self {
            mem_no: entry.mem_no,
            content: entry.content,
        }
    }
}
