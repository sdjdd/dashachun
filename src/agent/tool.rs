use std::fmt::Display;
use std::sync::Arc;

use schemars::JsonSchema;
use schemars::generate::{SchemaGenerator, SchemaSettings};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::{debug, warn};

use crate::agent::{AgentOutput, ToolCall, ToolSpec};

mod datetime;
mod memory;
mod weather;

pub use datetime::GetDateTime;
pub use memory::{MemoryAdd, MemoryDelete, MemoryUpdate};
pub use weather::GetWeather;

pub(crate) fn params_schema<T: JsonSchema>() -> Value {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    SchemaGenerator::new(settings)
        .root_schema_for::<T>()
        .to_value()
}

/// The result of one tool call: the content fed back to the model as the
/// tool message, an optional side-channel output for the device, and
/// whether the model must produce a follow-up reply after seeing the
/// result. Query tools and failures need one — the reply depends on the
/// result or must correct a claim the model already made; a memory write
/// that succeeded does not, which lets the tool loop close a turn that
/// already spoke instead of asking the model again (a second round could
/// only re-confirm what was just said).
pub struct ToolOutcome {
    pub content: String,
    pub output: Option<AgentOutput>,
    pub needs_reply: bool,
}

#[async_trait::async_trait]
pub trait ToolHandler: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, args: &serde_json::Value) -> Result<ToolOutcome, String>;
}

/// The per-call environment handed to a tool. It carries only tool-calling
/// concerns — session or transport specifics never enter this layer. Fields
/// appear here when a real tool needs them; today the context is empty.
#[derive(Clone, Copy, Default)]
pub struct ToolContext {}

/// The typed tool surface: one impl per tool with arguments and results as
/// real types. `Args` deserializes from the model's JSON arguments, `Output`
/// renders as the tool message content, `Error` renders as the failure
/// content — a failure always asks the model for a follow-up. `NEEDS_REPLY`
/// marks tools whose *successful* result is silent plumbing: the loop may
/// then close a turn that already answered instead of asking the model again
/// (a second round could only re-confirm what was just said).
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

    const NEEDS_REPLY: bool = true;

    async fn call(&self, ctx: &ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error>;
}

/// The type-erased execution surface the registry stores. Sealed: only the
/// typed [`Tool`] blanket impl and the legacy [`ToolHandler`] passthrough
/// live here, so one registry holds both styles and consumers never see
/// either shape directly.
#[async_trait::async_trait]
trait ErasedTool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn execute(&self, ctx: &ToolContext, args: &Value) -> ToolOutcome;
}

#[async_trait::async_trait]
impl<T: Tool> ErasedTool for T {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: T::NAME.to_string(),
            description: self.description(),
            parameters: self.parameters(),
        }
    }

    async fn execute(&self, ctx: &ToolContext, args: &Value) -> ToolOutcome {
        let args: T::Args = match T::Args::deserialize(args) {
            Ok(args) => args,
            Err(err) => {
                warn!(tool = T::NAME, %err, "invalid tool arguments");
                return ToolOutcome {
                    content: format!("error: invalid arguments: {err}"),
                    output: None,
                    needs_reply: true,
                };
            }
        };
        match T::call(self, ctx, args).await {
            Ok(output) => {
                debug!(tool = T::NAME, output = %output, "tool result");
                ToolOutcome {
                    content: output.to_string(),
                    output: None,
                    needs_reply: T::NEEDS_REPLY,
                }
            }
            Err(err) => {
                warn!(tool = T::NAME, %err, "tool failed");
                ToolOutcome {
                    content: format!("error: {err}"),
                    output: None,
                    needs_reply: true,
                }
            }
        }
    }
}

/// Passthrough for the legacy [`ToolHandler`] impls — untouched until they
/// migrate to [`Tool`].
struct HandlerTool(Arc<dyn ToolHandler>);

#[async_trait::async_trait]
impl ErasedTool for HandlerTool {
    fn spec(&self) -> ToolSpec {
        self.0.spec()
    }

    async fn execute(&self, _ctx: &ToolContext, args: &Value) -> ToolOutcome {
        match self.0.call(args).await {
            Ok(outcome) => outcome,
            Err(err) => {
                warn!(tool = %self.0.spec().name, %err, "tool failed");
                ToolOutcome {
                    content: format!("error: {err}"),
                    output: None,
                    needs_reply: true,
                }
            }
        }
    }
}

pub struct ToolRegistry {
    tools: Vec<Arc<dyn ErasedTool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn ToolHandler>>) -> Self {
        Self {
            tools: tools
                .into_iter()
                .map(|handler| Arc::new(HandlerTool(handler)) as Arc<dyn ErasedTool>)
                .collect(),
        }
    }

    /// The shared registry plus per-connection handlers, built fresh by the
    /// factory for every device: tools that need the connection's owner
    /// (the memory tools) cannot live in the process-wide registry.
    pub fn extended(&self, extra: Vec<Arc<dyn ToolHandler>>) -> Self {
        let mut tools = self.tools.clone();
        tools.extend(
            extra
                .into_iter()
                .map(|handler| Arc::new(HandlerTool(handler)) as Arc<dyn ErasedTool>),
        );
        Self { tools }
    }

    /// Registers a typed [`Tool`] alongside any legacy handlers.
    pub fn register_tool<T: Tool>(&mut self, tool: T) {
        self.tools.push(Arc::new(tool));
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|tool| tool.spec()).collect()
    }

    fn get(&self, name: &str) -> Option<&Arc<dyn ErasedTool>> {
        self.tools.iter().find(|tool| tool.spec().name == name)
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Degrades everything that can go wrong in a call — unknown name, malformed
/// argument JSON, handler failure — into an error-string tool message, so a
/// misbehaving model or a broken tool can never crash the loop.
pub(crate) async fn execute_tool(
    tools: &ToolRegistry,
    ctx: &ToolContext,
    call: &ToolCall,
) -> ToolOutcome {
    debug!(tool = %call.name, args = %call.arguments, "tool call");
    let Some(tool) = tools.get(&call.name) else {
        warn!(tool = %call.name, "unknown tool requested");
        return ToolOutcome {
            content: format!("error: unknown tool `{}`", call.name),
            output: None,
            needs_reply: true,
        };
    };
    let args = if call.arguments.trim().is_empty() {
        serde_json::Value::Null
    } else {
        match serde_json::from_str(&call.arguments) {
            Ok(args) => args,
            Err(err) => {
                warn!(tool = %call.name, %err, "invalid tool arguments");
                return ToolOutcome {
                    content: format!("error: invalid arguments: {err}"),
                    output: None,
                    needs_reply: true,
                };
            }
        }
    };
    let result = tool.execute(ctx, &args).await;
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

        async fn call(&self, _ctx: &ToolContext, args: AddArgs) -> Result<i64, Infallible> {
            Ok(args.x + args.y)
        }
    }

    /// A successful call that is silent plumbing — the loop may close the
    /// turn without asking the model again.
    struct SilentAdd;

    #[async_trait::async_trait]
    impl Tool for SilentAdd {
        const NAME: &'static str = "silent_add";
        const NEEDS_REPLY: bool = false;
        type Args = AddArgs;
        type Output = i64;
        type Error = Infallible;

        fn description(&self) -> String {
            "Add x and y together".to_string()
        }

        async fn call(&self, _ctx: &ToolContext, args: AddArgs) -> Result<i64, Infallible> {
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

        async fn call(&self, _ctx: &ToolContext, _args: AddArgs) -> Result<i64, String> {
            Err("boom".to_string())
        }
    }

    /// A no-argument tool: the model sends an empty argument string, which
    /// degrades to `Value::Null` and must deserialize into `()`.
    struct Ping;

    #[async_trait::async_trait]
    impl Tool for Ping {
        const NAME: &'static str = "ping";
        type Args = ();
        type Output = &'static str;
        type Error = Infallible;

        fn description(&self) -> String {
            "Returns pong".to_string()
        }

        async fn call(&self, _ctx: &ToolContext, _args: ()) -> Result<&'static str, Infallible> {
            Ok("pong")
        }
    }

    struct Legacy;

    #[async_trait::async_trait]
    impl ToolHandler for Legacy {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "legacy".to_string(),
                description: "legacy tool".to_string(),
                parameters: Value::Null,
            }
        }

        async fn call(&self, _args: &Value) -> Result<ToolOutcome, String> {
            Ok(ToolOutcome {
                content: "legacy-ok".to_string(),
                output: None,
                needs_reply: false,
            })
        }
    }

    fn registry() -> ToolRegistry {
        let mut tools = ToolRegistry::new(vec![Arc::new(Legacy)]);
        tools.register_tool(Add);
        tools.register_tool(SilentAdd);
        tools.register_tool(Failing);
        tools.register_tool(Ping);
        tools
    }

    #[test]
    fn typed_tool_joins_the_specs() {
        let specs = registry().specs();
        let add = specs.iter().find(|spec| spec.name == "add").unwrap();
        assert_eq!(add.description, "Add x and y together");
        assert_eq!(add.parameters, params_schema::<AddArgs>());
        assert!(specs.iter().any(|spec| spec.name == "legacy"));
        assert_eq!(specs.len(), 5);
    }

    #[tokio::test]
    async fn typed_tool_renders_output_as_content() {
        let outcome = execute_tool(
            &registry(),
            &ToolContext {},
            &ToolCall {
                id: "1".into(),
                name: "add".into(),
                arguments: r#"{"x": 1, "y": 2}"#.into(),
            },
        )
        .await;
        assert_eq!(outcome.content, "3");
        assert!(outcome.needs_reply);
    }

    #[tokio::test]
    async fn successful_silent_tool_keeps_needs_reply_false() {
        let outcome = execute_tool(
            &registry(),
            &ToolContext {},
            &ToolCall {
                id: "1".into(),
                name: "silent_add".into(),
                arguments: r#"{"x": 1, "y": 2}"#.into(),
            },
        )
        .await;
        assert_eq!(outcome.content, "3");
        assert!(!outcome.needs_reply);
    }

    #[tokio::test]
    async fn failing_tool_degrades_to_error_content_with_follow_up() {
        let outcome = execute_tool(
            &registry(),
            &ToolContext {},
            &ToolCall {
                id: "1".into(),
                name: "fail".into(),
                arguments: r#"{"x": 1, "y": 2}"#.into(),
            },
        )
        .await;
        assert_eq!(outcome.content, "error: boom");
        assert!(outcome.needs_reply);
    }

    #[tokio::test]
    async fn malformed_typed_arguments_degrade_with_follow_up() {
        let outcome = execute_tool(
            &registry(),
            &ToolContext {},
            &ToolCall {
                id: "1".into(),
                name: "add".into(),
                arguments: r#"{"x": 1}"#.into(),
            },
        )
        .await;
        assert!(outcome.content.starts_with("error: invalid arguments"));
        assert!(outcome.needs_reply);
    }

    #[tokio::test]
    async fn empty_arguments_reach_a_unit_args_tool_as_null() {
        let outcome = execute_tool(
            &registry(),
            &ToolContext {},
            &ToolCall {
                id: "1".into(),
                name: "ping".into(),
                arguments: String::new(),
            },
        )
        .await;
        assert_eq!(outcome.content, "pong");
        assert!(outcome.needs_reply);
    }

    #[tokio::test]
    async fn unknown_name_degrades_with_follow_up() {
        let outcome = execute_tool(
            &registry(),
            &ToolContext {},
            &ToolCall {
                id: "1".into(),
                name: "nope".into(),
                arguments: "{}".into(),
            },
        )
        .await;
        assert_eq!(outcome.content, "error: unknown tool `nope`");
        assert!(outcome.needs_reply);
    }

    #[tokio::test]
    async fn legacy_handler_still_executes_alongside_typed_tools() {
        let outcome = execute_tool(
            &registry(),
            &ToolContext {},
            &ToolCall {
                id: "1".into(),
                name: "legacy".into(),
                arguments: "{}".into(),
            },
        )
        .await;
        assert_eq!(outcome.content, "legacy-ok");
        assert!(!outcome.needs_reply);
    }
}
