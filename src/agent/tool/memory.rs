use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::agent::generate::{StepResult, StopCondition};
use crate::agent::memory::{AddError, EntryMemory, MAX_ENTRIES, entry_id};

use super::Tool;

const MAX_CONTENT_CHARS: usize = 512;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MemoryAddArgs {
    /// The fact to remember, short and self-contained.
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MemoryUpdateArgs {
    /// The ID of the entry to replace, e.g. "mem_03".
    id: String,
    /// The replacement text, short and self-contained.
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MemoryDeleteArgs {
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
impl Tool for MemoryAdd {
    const NAME: &'static str = "memory_add";
    type Args = MemoryAddArgs;
    type Output = String;
    type Error = String;

    fn description(&self) -> String {
        "Add a durable fact about the user to your long-term memory. The call is \
         rejected while the memory is full."
            .to_string()
    }

    async fn call(&self, args: MemoryAddArgs) -> Result<String, String> {
        let content = check_content(&args.content)?;
        match self.memory.add(&content).await {
            Ok(entry) => Ok(format!("stored as [{}]", entry_id(entry.mem_no))),
            Err(AddError::Full) => Err(format!(
                "memory is full ({MAX_ENTRIES} entries); update or delete an existing entry instead"
            )),
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
impl Tool for MemoryUpdate {
    const NAME: &'static str = "memory_update";
    type Args = MemoryUpdateArgs;
    type Output = String;
    type Error = String;

    fn description(&self) -> String {
        "Replace the text of a remembered fact by its ID.".to_string()
    }

    async fn call(&self, args: MemoryUpdateArgs) -> Result<String, String> {
        let mem_no = parse_id(&args.id).ok_or_else(|| {
            format!(
                "invalid id `{}`; use the [mem_NN] shown in the prompt",
                args.id
            )
        })?;
        let content = check_content(&args.content)?;
        let id = entry_id(mem_no);
        if self.memory.update(mem_no, &content).await? {
            Ok(format!("updated [{id}]"))
        } else {
            Ok(format!("no memory entry [{id}]"))
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
impl Tool for MemoryDelete {
    const NAME: &'static str = "memory_delete";
    type Args = MemoryDeleteArgs;
    type Output = String;
    type Error = String;

    fn description(&self) -> String {
        "Remove a fact from your long-term memory by its ID.".to_string()
    }

    async fn call(&self, args: MemoryDeleteArgs) -> Result<String, String> {
        let mem_no = parse_id(&args.id).ok_or_else(|| {
            format!(
                "invalid id `{}`; use the [mem_NN] shown in the prompt",
                args.id
            )
        })?;
        let id = entry_id(mem_no);
        if self.memory.delete(mem_no).await? {
            Ok(format!("deleted [{id}]"))
        } else {
            Ok(format!("no memory entry [{id}]"))
        }
    }
}

const MEMORY_TOOLS: [&str; 3] = [MemoryAdd::NAME, MemoryUpdate::NAME, MemoryDelete::NAME];

/// The voice turn's stop condition: once a round produced text, a round of
/// only successful memory writes ends the turn — a second round could only
/// re-confirm what was just said. A failed call (a full memory, an invalid
/// id, malformed arguments) keeps the loop running for a correction.
pub fn stop_after_memory_writes() -> StopCondition {
    Box::new(|steps: &[StepResult]| {
        steps.last().is_some_and(|step| {
            !step.text.is_empty()
                && step.ok
                && !step.tool_calls.is_empty()
                && step
                    .tool_calls
                    .iter()
                    .all(|call| MEMORY_TOOLS.contains(&call.name.as_str()))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ToolCall;
    use crate::agent::memory::InMemEntryMemory;
    use crate::agent::tool::ToolEntry;

    fn memory() -> Arc<dyn EntryMemory> {
        Arc::new(InMemEntryMemory::default())
    }

    #[tokio::test]
    async fn add_stores_and_reports_the_new_id() {
        let memory = memory();
        let tool = MemoryAdd::new(memory.clone());
        let outcome = tool
            .call(MemoryAddArgs {
                content: " likes Rust ".into(),
            })
            .await
            .unwrap();
        assert_eq!(outcome, "stored as [mem_01]");
        let outcome = tool
            .call(MemoryAddArgs {
                content: "building an ESP32 assistant".into(),
            })
            .await
            .unwrap();
        assert_eq!(outcome, "stored as [mem_02]");
        assert_eq!(memory.list().await.len(), 2);
        assert_eq!(memory.list().await[0].content, "likes Rust");
    }

    #[tokio::test]
    async fn add_rejects_empty_and_overlong_content() {
        let tool = MemoryAdd::new(memory());
        assert!(
            tool.call(MemoryAddArgs {
                content: "  ".into()
            })
            .await
            .is_err()
        );
        let long = "长".repeat(MAX_CONTENT_CHARS + 1);
        assert!(tool.call(MemoryAddArgs { content: long }).await.is_err());
    }

    #[tokio::test]
    async fn add_reports_a_full_memory_without_failing_the_scope() {
        let memory = memory();
        let tool = MemoryAdd::new(memory.clone());
        for i in 0..MAX_ENTRIES {
            tool.call(MemoryAddArgs {
                content: format!("fact {i}"),
            })
            .await
            .unwrap();
        }
        let err = tool
            .call(MemoryAddArgs {
                content: "one more".into(),
            })
            .await
            .unwrap_err();
        assert!(err.contains("memory is full"), "{err}");
        assert_eq!(memory.list().await.len(), MAX_ENTRIES);
    }

    #[tokio::test]
    async fn update_replaces_by_id() {
        let memory = memory();
        memory.add("old").await.unwrap();
        let tool = MemoryUpdate::new(memory.clone());
        let outcome = tool
            .call(MemoryUpdateArgs {
                id: "mem_01".into(),
                content: "new".into(),
            })
            .await
            .unwrap();
        assert_eq!(outcome, "updated [mem_01]");
        assert_eq!(memory.list().await[0].content, "new");
        // Bare numbers work too; unknown ids report instead of failing.
        let outcome = tool
            .call(MemoryUpdateArgs {
                id: "9".into(),
                content: "x".into(),
            })
            .await
            .unwrap();
        assert_eq!(outcome, "no memory entry [mem_09]");
        assert!(
            tool.call(MemoryUpdateArgs {
                id: "abc".into(),
                content: "x".into(),
            })
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn delete_removes_by_id() {
        let memory = memory();
        memory.add("one").await.unwrap();
        let tool = MemoryDelete::new(memory.clone());
        let outcome = tool
            .call(MemoryDeleteArgs {
                id: "mem_01".into(),
            })
            .await
            .unwrap();
        assert_eq!(outcome, "deleted [mem_01]");
        assert!(memory.list().await.is_empty());
        let outcome = tool
            .call(MemoryDeleteArgs {
                id: "mem_01".into(),
            })
            .await
            .unwrap();
        assert_eq!(outcome, "no memory entry [mem_01]");
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
        for entry in [
            ToolEntry::new(MemoryAdd::new(memory())),
            ToolEntry::new(MemoryUpdate::new(memory())),
            ToolEntry::new(MemoryDelete::new(memory())),
        ] {
            let spec = entry.spec();
            assert!(spec.name.starts_with("memory_"));
            assert_eq!(spec.parameters["type"], "object");
            assert_eq!(spec.parameters["additionalProperties"], false);
        }
    }

    fn step(tool_calls: Vec<ToolCall>, ok: bool, text: &str) -> StepResult {
        StepResult {
            tool_calls,
            ok,
            text: text.into(),
        }
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: "1".into(),
            name: name.into(),
            arguments: "{}".into(),
        }
    }

    #[test]
    fn stop_condition_fires_only_on_spoken_successful_memory_rounds() {
        let condition = stop_after_memory_writes();
        // A silent round — nothing spoken yet — never stops the turn.
        assert!(!condition(&[step(vec![call("memory_add")], true, "")]));
        // A failed memory write keeps the loop running for a correction.
        assert!(!condition(&[step(
            vec![call("memory_add")],
            false,
            "记好了"
        )]));
        // A query tool round is not a memory-only round.
        assert!(!condition(&[step(
            vec![call("get_weather")],
            true,
            "北京今天晴"
        )]));
        // Mixed calls are not memory-only either.
        assert!(!condition(&[step(
            vec![call("memory_add"), call("get_weather")],
            true,
            "记好了，北京今天晴"
        )]));
        // The firing shape: the round spoke, every call succeeded, and only
        // memory tools ran.
        assert!(condition(&[step(vec![call("memory_add")], true, "记好了")]));
        assert!(condition(&[step(
            vec![call("memory_delete")],
            true,
            "删掉了"
        )]));
    }
}
