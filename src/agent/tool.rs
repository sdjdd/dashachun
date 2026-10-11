use std::fmt::Display;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use schemars::JsonSchema;
use schemars::generate::{SchemaGenerator, SchemaSettings};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::{debug, warn};

use crate::agent::{ToolCall, ToolSpec};

mod datetime;
mod memory;
mod weather;

pub use datetime::GetDateTime;
pub use memory::{MemoryAdd, MemoryDelete, MemoryUpdate, stop_after_memory_writes};
pub use weather::GetWeather;

pub(crate) fn params_schema<T: JsonSchema>() -> Value {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    SchemaGenerator::new(settings)
        .root_schema_for::<T>()
        .to_value()
}

/// The result of one tool call as the erased layer hands it to the loop:
/// the content fed back to the model as the tool message, and whether the
/// call succeeded — a degraded call (invalid arguments, unknown tool, tool
/// failure) is not ok. The bit feeds [`StepResult::ok`]; it does not steer
/// the loop by itself.
pub struct ToolOutcome {
    pub content: String,
    pub ok: bool,
}

/// The arguments type for tools that take no arguments: the schema
/// advertises a strict empty object, and `{}` — also what an empty
/// argument string degrades to — deserializes into it.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoArgs {}

/// The typed tool surface: one impl per tool with arguments and results as
/// real types. `Args` deserializes from the model's JSON arguments, `Output`
/// renders as the tool message content, `Error` renders as the failure
/// content. Whether the loop ends after a tool round is the caller's call —
/// `stop_when` conditions see every completed [`StepResult`], including
/// whether its calls succeeded. Tools that need per-connection state hold it
/// at construction time (the memory tools take the scope's `EntryMemory`
/// handle in `new`), so `call` needs no ambient context.
#[async_trait::async_trait]
pub trait Tool: Send + Sync + 'static {
    const NAME: &'static str;
    type Args: DeserializeOwned + JsonSchema;
    type Output: Display;
    type Error: Display;

    fn description(&self) -> String;

    fn parameters(&self) -> Value {
        params_schema::<Self::Args>()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;
}

/// The erased call surface: the raw JSON arguments in, the rendered tool
/// message out.
type ErasedCall =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = ToolOutcome> + Send>> + Send + Sync>;

/// One registered tool as the runtime sees it: the precomputed spec (the
/// schemars schema runs once at construction, not per round or per lookup)
/// plus the erased call surface. Built only through [`ToolEntry::new`]; the
/// fields stay private, so tool authors keep working against the typed
/// [`Tool`] alone.
#[derive(Clone)]
pub struct ToolEntry {
    name: &'static str,
    spec: ToolSpec,
    call: ErasedCall,
}

impl ToolEntry {
    /// Wraps a typed [`Tool`] into an erasable entry for a `Vec<ToolEntry>`.
    pub fn new<T: Tool>(tool: T) -> Self {
        let spec = ToolSpec {
            name: T::NAME.to_string(),
            description: tool.description(),
            parameters: tool.parameters(),
        };
        let inner = Arc::new(tool);
        ToolEntry {
            name: T::NAME,
            spec,
            call: Arc::new(move |args| {
                let inner = inner.clone();
                Box::pin(async move { call_typed(&*inner, args).await })
            }),
        }
    }

    pub(crate) fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, args: Value) -> ToolOutcome {
        (self.call)(args).await
    }
}

async fn call_typed<T: Tool>(tool: &T, args: Value) -> ToolOutcome {
    let args: T::Args = match T::Args::deserialize(&args) {
        Ok(args) => args,
        Err(err) => {
            warn!(tool = T::NAME, %err, "invalid tool arguments");
            return ToolOutcome {
                content: format!("error: invalid arguments: {err}"),
                ok: false,
            };
        }
    };
    match tool.call(args).await {
        Ok(output) => {
            debug!(tool = T::NAME, output = %output, "tool result");
            ToolOutcome {
                content: output.to_string(),
                ok: true,
            }
        }
        Err(err) => {
            warn!(tool = T::NAME, %err, "tool failed");
            ToolOutcome {
                content: format!("error: {err}"),
                ok: false,
            }
        }
    }
}

/// Degrades everything that can go wrong in a call — unknown name, malformed
/// argument JSON, tool failure — into an error-string tool message, so a
/// misbehaving model or a broken tool can never crash the loop.
pub(crate) async fn execute_tool(tools: &[ToolEntry], call: &ToolCall) -> ToolOutcome {
    debug!(tool = %call.name, args = %call.arguments, "tool call");
    let Some(entry) = tools.iter().find(|entry| entry.name == call.name) else {
        warn!(tool = %call.name, "unknown tool requested");
        return ToolOutcome {
            content: format!("error: unknown tool `{}`", call.name),
            ok: false,
        };
    };
    let args = if call.arguments.trim().is_empty() {
        Value::Object(serde_json::Map::new())
    } else {
        match serde_json::from_str(&call.arguments) {
            Ok(args) => args,
            Err(err) => {
                warn!(tool = %call.name, %err, "invalid tool arguments");
                return ToolOutcome {
                    content: format!("error: invalid arguments: {err}"),
                    ok: false,
                };
            }
        }
    };
    let result = entry.execute(args).await;
    debug!(tool = %call.name, result = %result.content, "tool result");
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct AddArgs {
        x: i64,
        y: i64,
    }

    struct Add;

    #[async_trait::async_trait]
    impl Tool for Add {
        const NAME: &'static str = "add";
        type Args = AddArgs;
        type Output = i64;
        type Error = Infallible;

        fn description(&self) -> String {
            "Add x and y together".to_string()
        }

        async fn call(&self, args: AddArgs) -> Result<i64, Infallible> {
            Ok(args.x + args.y)
        }
    }

    struct Failing;

    #[async_trait::async_trait]
    impl Tool for Failing {
        const NAME: &'static str = "fail";
        type Args = AddArgs;
        type Output = i64;
        type Error = String;

        fn description(&self) -> String {
            "Always fails".to_string()
        }

        async fn call(&self, _args: AddArgs) -> Result<i64, String> {
            Err("boom".to_string())
        }
    }

    /// A no-argument tool: the model sends an empty argument string, which
    /// degrades to `{}` and must deserialize into [`NoArgs`].
    struct Ping;

    #[async_trait::async_trait]
    impl Tool for Ping {
        const NAME: &'static str = "ping";
        type Args = NoArgs;
        type Output = &'static str;
        type Error = Infallible;

        fn description(&self) -> String {
            "Returns pong".to_string()
        }

        async fn call(&self, _args: NoArgs) -> Result<&'static str, Infallible> {
            Ok("pong")
        }
    }

    fn registry() -> Vec<ToolEntry> {
        vec![
            ToolEntry::new(Add),
            ToolEntry::new(Failing),
            ToolEntry::new(Ping),
        ]
    }

    #[test]
    fn typed_tool_joins_the_specs() {
        let entries = registry();
        let add = entries.iter().find(|entry| entry.name == "add").unwrap();
        assert_eq!(add.spec().description, "Add x and y together");
        assert_eq!(add.spec().parameters, params_schema::<AddArgs>());
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn typed_tool_renders_output_as_content() {
        let outcome = execute_tool(
            &registry(),
            &ToolCall {
                id: "1".into(),
                name: "add".into(),
                arguments: r#"{"x": 1, "y": 2}"#.into(),
            },
        )
        .await;
        assert_eq!(outcome.content, "3");
        assert!(outcome.ok);
    }

    #[tokio::test]
    async fn failing_tool_degrades_to_error_content_with_follow_up() {
        let outcome = execute_tool(
            &registry(),
            &ToolCall {
                id: "1".into(),
                name: "fail".into(),
                arguments: r#"{"x": 1, "y": 2}"#.into(),
            },
        )
        .await;
        assert_eq!(outcome.content, "error: boom");
        assert!(!outcome.ok);
    }

    #[tokio::test]
    async fn malformed_typed_arguments_degrade_with_follow_up() {
        let outcome = execute_tool(
            &registry(),
            &ToolCall {
                id: "1".into(),
                name: "add".into(),
                arguments: r#"{"x": 1}"#.into(),
            },
        )
        .await;
        assert!(outcome.content.starts_with("error: invalid arguments"));
        assert!(!outcome.ok);
    }

    #[tokio::test]
    async fn empty_arguments_reach_a_no_arg_tool() {
        let outcome = execute_tool(
            &registry(),
            &ToolCall {
                id: "1".into(),
                name: "ping".into(),
                arguments: String::new(),
            },
        )
        .await;
        assert_eq!(outcome.content, "pong");
        assert!(outcome.ok);
    }

    #[tokio::test]
    async fn unknown_name_degrades_with_follow_up() {
        let outcome = execute_tool(
            &registry(),
            &ToolCall {
                id: "1".into(),
                name: "nope".into(),
                arguments: "{}".into(),
            },
        )
        .await;
        assert_eq!(outcome.content, "error: unknown tool `nope`");
        assert!(!outcome.ok);
    }
}
