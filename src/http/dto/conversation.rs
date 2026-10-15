//! The conversation transcript read API: paged conversation and message
//! listings plus the wire shape of the captured-audio read's scoping
//! (the audio bytes themselves stream raw, not through these DTOs).

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use utoipa::ToSchema;

use crate::agent::TranscriptMessage;
use crate::device::conversation::Conversation;

/// The shared query args of the paged reads: keyset pagination on the row
/// id, newest first — `before_id` is the previous page's last item id and
/// `limit` the page size.
#[derive(Debug, Deserialize)]
pub struct PageArgs {
    pub before_id: Option<i64>,
    pub limit: Option<i32>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ConversationResponse {
    pub id: i64,
    /// The wire id the device saw (hello ack + echoes), recorded for
    /// correlation with device-side logs.
    pub session_id: String,
    pub device_id: i64,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// NULL while the connection lives and after a server crash.
    #[serde(with = "time::serde::rfc3339::option")]
    pub ended_at: Option<OffsetDateTime>,
}

impl From<Conversation> for ConversationResponse {
    fn from(record: Conversation) -> Self {
        Self {
            id: record.id,
            session_id: record.session_id,
            device_id: record.device_id,
            created_at: record.created_at,
            ended_at: record.ended_at,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ConversationPage {
    pub items: Vec<ConversationResponse>,
    /// Whether older conversations remain — continue with the last item's
    /// id as `before_id`.
    pub has_more: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct MessageResponse {
    pub id: i64,
    /// The stored message's serde tag — `user`, `assistant` or `tool`.
    pub role: String,
    /// The stored AI SDK ModelMessage JSON verbatim (tool-call requests
    /// and results included).
    pub message: serde_json::Value,
    /// The voiceprint attribution on user rows — the matched `speakers`
    /// row id, NULL when the utterance identified no one; dangling after
    /// a speaker delete, resolve against the live roster.
    pub speaker_id: Option<i64>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

impl From<TranscriptMessage> for MessageResponse {
    fn from(row: TranscriptMessage) -> Self {
        Self {
            id: row.id,
            role: row
                .message
                .get("role")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            message: row.message,
            speaker_id: row.speaker_id,
            created_at: row.created_at,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct MessagePage {
    pub items: Vec<MessageResponse>,
    /// Whether older messages remain — continue with the last item's id
    /// as `before_id`.
    pub has_more: bool,
}
