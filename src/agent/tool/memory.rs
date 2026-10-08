use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::agent::ToolSpec;
use crate::agent::memory::{AddError, EntryMemory, MAX_ENTRIES, entry_id};

use super::{ToolHandler, ToolOutcome, params_schema};

const MAX_CONTENT_CHARS: usize = 512;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MemoryAddArgs {
    /// The fact to remember, short and self-contained.
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MemoryUpdateArgs {
    /// The ID of the entry to replace, e.g. "mem_03".
    id: String,
    /// The replacement text, short and self-contained.
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MemoryDeleteArgs {
    /// The ID of the entry to delete, e.g. "mem_03".
    id: String,
}

fn check_content(content: &str) -> Result<String, String> {
    let content = content.trim();
    if content.is_empty() {
        return Err("content is required".into());
    }
    if content.chars().count() > MAX_CONTENT_CHARS {
        return Err(format!(
            "content is too long (max {MAX_CONTENT_CHARS} chars); keep the entry short"
        ));
    }
    Ok(content.to_string())
}

/// Accepts the id exactly as the prompt shows it ("mem_03") and tolerates
/// the bare number.
fn parse_id(id: &str) -> Option<i32> {
    let id = id.trim();
    let id = id.strip_prefix("mem_").unwrap_or(id);
    id.parse().ok().filter(|mem_no| *mem_no > 0)
}

pub struct MemoryAdd {
    memory: Arc<dyn EntryMemory>,
}

impl MemoryAdd {
    pub fn new(memory: Arc<dyn EntryMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait::async_trait]
impl ToolHandler for MemoryAdd {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "memory_add".to_string(),
            description: "Add a durable fact about the user to your long-term memory. The \
                call is rejected while the memory is full."
                .to_string(),
            parameters: params_schema::<MemoryAddArgs>(),
        }
    }

    async fn call(&self, args: &Value) -> Result<ToolOutcome, String> {
        let args: MemoryAddArgs = serde_json::from_value(args.clone())
            .map_err(|err| format!("invalid arguments: {err}"))?;
        let content = check_content(&args.content)?;
        match self.memory.add(&content).await {
            Ok(entry) => Ok(ToolOutcome {
                content: format!("stored as [{}]", entry_id(entry.mem_no)),
                output: None,
                needs_reply: false,
            }),
            Err(AddError::Full) => Ok(ToolOutcome {
                content: format!(
                    "memory is full ({MAX_ENTRIES} entries); update or delete an existing entry instead"
                ),
                output: None,
                needs_reply: true,
            }),
            Err(AddError::Failed(err)) => Err(err),
        }
    }
}

pub struct MemoryUpdate {
    memory: Arc<dyn EntryMemory>,
}

impl MemoryUpdate {
    pub fn new(memory: Arc<dyn EntryMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait::async_trait]
impl ToolHandler for MemoryUpdate {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "memory_update".to_string(),
            description: "Replace the text of a remembered fact by its ID.".to_string(),
            parameters: params_schema::<MemoryUpdateArgs>(),
        }
    }

    async fn call(&self, args: &Value) -> Result<ToolOutcome, String> {
        let args: MemoryUpdateArgs = serde_json::from_value(args.clone())
            .map_err(|err| format!("invalid arguments: {err}"))?;
        let mem_no = parse_id(&args.id).ok_or_else(|| {
            format!(
                "invalid id `{}`; use the [mem_NN] shown in the prompt",
                args.id
            )
        })?;
        let content = check_content(&args.content)?;
        let id = entry_id(mem_no);
        if self.memory.update(mem_no, &content).await? {
            Ok(ToolOutcome {
                content: format!("updated [{id}]"),
                output: None,
                needs_reply: false,
            })
        } else {
            Ok(ToolOutcome {
                content: format!("no memory entry [{id}]"),
                output: None,
                needs_reply: false,
            })
        }
    }
}

pub struct MemoryDelete {
    memory: Arc<dyn EntryMemory>,
}

impl MemoryDelete {
    pub fn new(memory: Arc<dyn EntryMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait::async_trait]
impl ToolHandler for MemoryDelete {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "memory_delete".to_string(),
            description: "Remove a fact from your long-term memory by its ID.".to_string(),
            parameters: params_schema::<MemoryDeleteArgs>(),
        }
    }

    async fn call(&self, args: &Value) -> Result<ToolOutcome, String> {
        let args: MemoryDeleteArgs = serde_json::from_value(args.clone())
            .map_err(|err| format!("invalid arguments: {err}"))?;
        let mem_no = parse_id(&args.id).ok_or_else(|| {
            format!(
                "invalid id `{}`; use the [mem_NN] shown in the prompt",
                args.id
            )
        })?;
        let id = entry_id(mem_no);
        if self.memory.delete(mem_no).await? {
            Ok(ToolOutcome {
                content: format!("deleted [{id}]"),
                output: None,
                needs_reply: false,
            })
        } else {
            Ok(ToolOutcome {
                content: format!("no memory entry [{id}]"),
                output: None,
                needs_reply: false,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::InMemEntryMemory;
    use serde_json::json;

    fn memory() -> Arc<dyn EntryMemory> {
        Arc::new(InMemEntryMemory::default())
    }

    #[tokio::test]
    async fn add_stores_and_reports_the_new_id() {
        let memory = memory();
        let tool = MemoryAdd::new(memory.clone());
        let outcome = tool
            .call(&json!({ "content": " likes Rust " }))
            .await
            .unwrap();
        assert_eq!(outcome.content, "stored as [mem_01]");
        let outcome = tool
            .call(&json!({ "content": "building an ESP32 assistant" }))
            .await
            .unwrap();
        assert_eq!(outcome.content, "stored as [mem_02]");
        assert_eq!(memory.list().await.len(), 2);
        assert_eq!(memory.list().await[0].content, "likes Rust");
    }

    #[tokio::test]
    async fn add_rejects_empty_and_overlong_content() {
        let tool = MemoryAdd::new(memory());
        assert!(tool.call(&json!({ "content": "  " })).await.is_err());
        let long = "长".repeat(MAX_CONTENT_CHARS + 1);
        assert!(tool.call(&json!({ "content": long })).await.is_err());
    }

    #[tokio::test]
    async fn add_reports_a_full_memory_without_failing() {
        let memory = memory();
        let tool = MemoryAdd::new(memory.clone());
        for i in 0..MAX_ENTRIES {
            tool.call(&json!({ "content": format!("fact {i}") }))
                .await
                .unwrap();
        }
        let outcome = tool.call(&json!({ "content": "one more" })).await.unwrap();
        assert!(
            outcome.content.contains("memory is full"),
            "{}",
            outcome.content
        );
        assert_eq!(memory.list().await.len(), MAX_ENTRIES);
    }

    #[tokio::test]
    async fn update_replaces_by_id() {
        let memory = memory();
        memory.add("old").await.unwrap();
        let tool = MemoryUpdate::new(memory.clone());
        let outcome = tool
            .call(&json!({ "id": "mem_01", "content": "new" }))
            .await
            .unwrap();
        assert_eq!(outcome.content, "updated [mem_01]");
        assert_eq!(memory.list().await[0].content, "new");
        // Bare numbers work too; unknown ids report instead of failing.
        let outcome = tool
            .call(&json!({ "id": "9", "content": "x" }))
            .await
            .unwrap();
        assert_eq!(outcome.content, "no memory entry [mem_09]");
        assert!(
            tool.call(&json!({ "id": "abc", "content": "x" }))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn delete_removes_by_id() {
        let memory = memory();
        memory.add("one").await.unwrap();
        let tool = MemoryDelete::new(memory.clone());
        let outcome = tool.call(&json!({ "id": "mem_01" })).await.unwrap();
        assert_eq!(outcome.content, "deleted [mem_01]");
        assert!(memory.list().await.is_empty());
        let outcome = tool.call(&json!({ "id": "mem_01" })).await.unwrap();
        assert_eq!(outcome.content, "no memory entry [mem_01]");
    }

    #[test]
    fn parse_id_forms() {
        assert_eq!(parse_id("mem_03"), Some(3));
        assert_eq!(parse_id("3"), Some(3));
        assert_eq!(parse_id("  mem_12 "), Some(12));
        assert_eq!(parse_id("0"), None);
        assert_eq!(parse_id("-1"), None);
        assert_eq!(parse_id("abc"), None);
        assert_eq!(parse_id(""), None);
    }

    #[test]
    fn specs_declare_closed_object_parameters() {
        let memory = memory();
        for spec in [
            MemoryAdd::new(memory.clone()).spec(),
            MemoryUpdate::new(memory.clone()).spec(),
            MemoryDelete::new(memory).spec(),
        ] {
            assert!(spec.name.starts_with("memory_"));
            assert_eq!(spec.parameters["type"], "object");
            assert_eq!(spec.parameters["additionalProperties"], false);
        }
    }
}
