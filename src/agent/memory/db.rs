use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use sqlx::PgPool;
use tracing::debug;
use tracing::warn;

use crate::agent::AgentSession;
use crate::llm::{ChatItem, ToolCall};

use super::{HISTORY_LIMIT, Memory, MemoryHook, MemoryOwner, trim};

/// The conversation memory backed by the durable `messages` log. The recent
/// window is loaded once per connection — the log is the source of truth
/// across sessions and devices, `session_id` is provenance only — and kept
/// in process memory afterwards: every turn is appended here by the driver
/// and, in parallel, written durably by the store methods, so per-turn
/// requests never wait on Postgres. Committed rows are announced to the
/// [`MemoryHook`]s.
pub struct DbMemory {
    pool: PgPool,
    owner: MemoryOwner,
    hooks: Vec<Arc<dyn MemoryHook>>,
    messages: Mutex<Vec<ChatItem>>,
}

impl DbMemory {
    /// Loads the recent history for the owner (user + agent), oldest item
    /// first. An error fails the connection — the database is mandatory.
    pub async fn load(
        pool: PgPool,
        owner: MemoryOwner,
        hooks: Vec<Arc<dyn MemoryHook>>,
    ) -> Result<Self, sqlx::Error> {
        let mut rows: Vec<(String, Option<String>, Option<serde_json::Value>)> = sqlx::query_as(
            "SELECT role, content, parts FROM messages \
             WHERE user_id = $1 AND agent_id = $2 \
             ORDER BY id DESC LIMIT $3",
        )
        .bind(owner.user_id)
        .bind(owner.agent_id)
        .bind(HISTORY_LIMIT as i64)
        .fetch_all(&pool)
        .await?;
        rows.reverse();
        let mut messages = rows_to_items(rows);
        trim(&mut messages);
        Ok(Self {
            pool,
            owner,
            hooks,
            messages: Mutex::new(messages),
        })
    }
}

#[async_trait]
impl Memory for DbMemory {
    async fn history(&self) -> Vec<ChatItem> {
        self.messages.lock().unwrap().clone()
    }

    async fn append(&self, items: Vec<ChatItem>) {
        let mut messages = self.messages.lock().unwrap();
        messages.extend(items);
        trim(&mut messages);
    }

    async fn store_utterance(&self, session: &AgentSession, text: &str) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        let message_id = store_utterance_row(&self.pool, &self.owner, &session.id, text).await?;
        for hook in &self.hooks {
            hook.utterance_stored(session, message_id);
        }
        debug!(session_id = %session.id, message_id, "utterance stored");
        Ok(())
    }

    fn log_items(&self, session: &AgentSession, items: Vec<ChatItem>) {
        if items.is_empty() {
            return;
        }
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let hooks = self.hooks.clone();
        let session = session.clone();
        tokio::spawn(async move {
            match store_turn_row(&pool, &owner, &session.id, &items).await {
                Ok(Some(message_id)) => {
                    for hook in &hooks {
                        hook.turn_stored(&session, message_id);
                    }
                    debug!(session_id = %session.id, message_id, "turn stored");
                }
                Ok(None) => {}
                Err(err) => warn!(session_id = %session.id, %err, "failed to store turn items"),
            }
        });
    }

    fn store_partial_reply(&self, session: &AgentSession, text: &str) {
        if text.is_empty() {
            return;
        }
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let hooks = self.hooks.clone();
        let session = session.clone();
        let text = text.to_string();
        tokio::spawn(async move {
            match store_partial_reply_row(&pool, &owner, &session.id, &text).await {
                Ok(message_id) => {
                    for hook in &hooks {
                        hook.turn_stored(&session, message_id);
                    }
                    debug!(session_id = %session.id, message_id, "partial reply stored");
                }
                Err(err) => warn!(session_id = %session.id, %err, "failed to store partial reply"),
            }
        });
    }
}

struct TurnRow {
    content: Option<String>,
    parts: Option<serde_json::Value>,
}

/// Folds one turn's conversation items — the tool exchange plus the final
/// reply — into a single message row, in execution order. Arguments and
/// results stay verbatim strings; round boundaries are recoverable from the
/// part order (a `tool_call` following a `tool_result` opens the next
/// round). `System`/`User` items are never stored here: the user message is
/// written by [`Memory::store_utterance`], the system prompt is prepended
/// every turn.
fn turn_row(items: &[ChatItem]) -> Option<TurnRow> {
    let mut parts = Vec::new();
    let mut content = None;
    for item in items {
        match item {
            ChatItem::System { .. } | ChatItem::User { .. } => {}
            ChatItem::Assistant {
                content: text,
                tool_calls,
            } => {
                for call in tool_calls {
                    parts.push(serde_json::json!({
                        "type": "tool_call",
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }));
                }
                if text.as_deref().is_some_and(|text| !text.is_empty()) {
                    content = text.clone();
                }
            }
            ChatItem::Tool {
                tool_call_id,
                content,
            } => {
                parts.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }));
            }
        }
    }
    if parts.is_empty() && content.is_none() {
        return None;
    }
    Some(TurnRow {
        content,
        parts: (!parts.is_empty()).then_some(serde_json::Value::Array(parts)),
    })
}

/// Rebuilds the conversation items from message rows, oldest first — the
/// inverse of [`turn_row`]. A user row becomes a `User` item; an assistant
/// row unfolds its `parts` in order (consecutive `tool_call` parts merge
/// into one round's `assistant_tool_calls` item, each `tool_result` becomes
/// a `Tool` item) and its non-empty `content` closes the turn as the final
/// reply. Rows are atomic, so a tool exchange is never split.
fn rows_to_items(rows: Vec<(String, Option<String>, Option<serde_json::Value>)>) -> Vec<ChatItem> {
    let mut items = Vec::new();
    for (role, content, parts) in rows {
        if role == "user" {
            if let Some(content) = content.filter(|content| !content.is_empty()) {
                items.push(ChatItem::user(content));
            }
            continue;
        }
        let Some(parts) = parts.as_ref().and_then(serde_json::Value::as_array) else {
            if let Some(content) = content.filter(|content| !content.is_empty()) {
                items.push(ChatItem::assistant(content));
            }
            continue;
        };
        let mut round = Vec::new();
        for part in parts {
            match part.get("type").and_then(serde_json::Value::as_str) {
                Some("tool_call") => round.push(ToolCall {
                    id: part
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .into(),
                    name: part
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .into(),
                    arguments: part
                        .get("arguments")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .into(),
                }),
                Some("tool_result") => {
                    if !round.is_empty() {
                        items.push(ChatItem::assistant_tool_calls(std::mem::take(&mut round)));
                    }
                    items.push(ChatItem::tool(
                        part.get("tool_call_id")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default(),
                        part.get("content")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default(),
                    ));
                }
                _ => {}
            }
        }
        if !round.is_empty() {
            items.push(ChatItem::assistant_tool_calls(round));
        }
        if let Some(content) = content.filter(|content| !content.is_empty()) {
            items.push(ChatItem::assistant(content));
        }
    }
    items
}

async fn store_utterance_row(
    pool: &PgPool,
    owner: &MemoryOwner,
    session_id: &str,
    text: &str,
) -> Result<i64, String> {
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO messages (session_id, user_id, agent_id, client_id, device_id, role, content) \
         VALUES ($1, $2, $3, $4, $5, 'user', $6) RETURNING id",
    )
    .bind(session_id)
    .bind(owner.user_id)
    .bind(owner.agent_id)
    .bind(owner.client_id)
    .bind(&owner.device_id)
    .bind(text)
    .fetch_one(pool)
    .await
    .map_err(|err| err.to_string())?;
    Ok(id)
}

async fn store_turn_row(
    pool: &PgPool,
    owner: &MemoryOwner,
    session_id: &str,
    items: &[ChatItem],
) -> Result<Option<i64>, String> {
    let Some(row) = turn_row(items) else {
        return Ok(None);
    };
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO messages (session_id, user_id, agent_id, client_id, device_id, \
         role, content, parts) \
         VALUES ($1, $2, $3, $4, $5, 'assistant', $6, $7) RETURNING id",
    )
    .bind(session_id)
    .bind(owner.user_id)
    .bind(owner.agent_id)
    .bind(owner.client_id)
    .bind(&owner.device_id)
    .bind(&row.content)
    .bind(&row.parts)
    .fetch_one(pool)
    .await
    .map_err(|err| err.to_string())?;
    Ok(Some(id))
}

async fn store_partial_reply_row(
    pool: &PgPool,
    owner: &MemoryOwner,
    session_id: &str,
    text: &str,
) -> Result<i64, String> {
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO messages (session_id, user_id, agent_id, client_id, device_id, role, content) \
         VALUES ($1, $2, $3, $4, $5, 'assistant', $6) RETURNING id",
    )
    .bind(session_id)
    .bind(owner.user_id)
    .bind(owner.agent_id)
    .bind(owner.client_id)
    .bind(&owner.device_id)
    .bind(text)
    .fetch_one(pool)
    .await
    .map_err(|err| err.to_string())?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_items_fold_into_one_turn_row() {
        use crate::llm::ToolCall;

        let row = turn_row(&[
            ChatItem::system("system prompt"),
            ChatItem::user("hello"),
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"hangzhou\"}".into(),
            }]),
            ChatItem::tool("call_x", "sunny 26 degrees"),
            ChatItem::assistant("🙂sunny today"),
        ])
        .expect("the turn produced items");

        assert_eq!(row.content.as_deref(), Some("🙂sunny today"));
        let parts = row.parts.unwrap().as_array().unwrap().clone();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "tool_call");
        assert_eq!(parts[0]["id"], "call_x");
        assert_eq!(parts[0]["name"], "get_weather");
        assert_eq!(parts[0]["arguments"], "{\"city\":\"hangzhou\"}");
        assert_eq!(parts[1]["type"], "tool_result");
        assert_eq!(parts[1]["tool_call_id"], "call_x");
        assert_eq!(parts[1]["content"], "sunny 26 degrees");
    }

    #[test]
    fn turn_row_stores_nothing_without_parts_or_text() {
        use crate::llm::ToolCall;

        assert!(turn_row(&[]).is_none());
        assert!(turn_row(&[ChatItem::system("system prompt")]).is_none());
        // An empty final reply over no tool exchange stores nothing either.
        assert!(turn_row(&[ChatItem::assistant("")]).is_none());
        // A tool round without a final reply still stores.
        let row = turn_row(&[
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "call_x".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("call_x", "result"),
        ])
        .expect("tool parts are stored");
        assert_eq!(row.content, None);
        assert_eq!(row.parts.unwrap().as_array().unwrap().len(), 2);
    }

    #[test]
    fn stored_rows_replay_back_to_items() {
        use crate::llm::ToolCall;

        let calls = |id: &str| {
            vec![ToolCall {
                id: id.into(),
                name: "lookup".into(),
                arguments: format!("{{\"q\":\"{id}\"}}"),
            }]
        };
        let assistant = |items: &[ChatItem]| {
            let row = turn_row(items).expect("turn produced items");
            ("assistant".to_string(), row.content, row.parts)
        };
        let user = |text: &str| ("user".to_string(), Some(text.to_string()), None);

        let rows = vec![
            user("hello"),
            assistant(&[
                ChatItem::assistant_tool_calls(calls("call_a")),
                ChatItem::tool("call_a", "result a"),
                ChatItem::assistant_tool_calls(calls("call_b")),
                ChatItem::tool("call_b", "result b"),
                ChatItem::assistant("done"),
            ]),
        ];

        assert_eq!(
            rows_to_items(rows),
            vec![
                ChatItem::user("hello"),
                ChatItem::assistant_tool_calls(calls("call_a")),
                ChatItem::tool("call_a", "result a"),
                ChatItem::assistant_tool_calls(calls("call_b")),
                ChatItem::tool("call_b", "result b"),
                ChatItem::assistant("done"),
            ]
        );
    }

    #[test]
    fn tool_only_rows_replay_without_a_final_reply() {
        use crate::llm::ToolCall;

        let row = turn_row(&[
            ChatItem::assistant_tool_calls(vec![
                ToolCall {
                    id: "1".into(),
                    name: "a".into(),
                    arguments: "{}".into(),
                },
                ToolCall {
                    id: "2".into(),
                    name: "b".into(),
                    arguments: "{}".into(),
                },
            ]),
            ChatItem::tool("1", "one"),
            ChatItem::tool("2", "two"),
        ])
        .expect("tool parts are stored");

        let items = rows_to_items(vec![("assistant".into(), row.content, row.parts)]);
        assert_eq!(
            items,
            vec![
                ChatItem::assistant_tool_calls(vec![
                    ToolCall {
                        id: "1".into(),
                        name: "a".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "2".into(),
                        name: "b".into(),
                        arguments: "{}".into(),
                    },
                ]),
                ChatItem::tool("1", "one"),
                ChatItem::tool("2", "two"),
            ]
        );
    }
}
