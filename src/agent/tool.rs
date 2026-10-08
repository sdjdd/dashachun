use std::sync::Arc;

use schemars::JsonSchema;
use schemars::generate::{SchemaGenerator, SchemaSettings};
use serde_json::Value;

use crate::agent::AgentOutput;
use crate::agent::ToolSpec;

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

pub struct ToolOutcome {
    pub content: String,
    pub output: Option<AgentOutput>,
}

#[async_trait::async_trait]
pub trait ToolHandler: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, args: &serde_json::Value) -> Result<ToolOutcome, String>;
}

pub struct ToolRegistry {
    tools: Vec<Arc<dyn ToolHandler>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn ToolHandler>>) -> Self {
        Self { tools }
    }

    /// The shared registry plus per-connection handlers, built fresh by the
    /// factory for every device: tools that need the connection's owner
    /// (the memory tools) cannot live in the process-wide registry.
    pub fn extended(&self, extra: Vec<Arc<dyn ToolHandler>>) -> Self {
        let mut tools = self.tools.clone();
        tools.extend(extra);
        Self { tools }
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|tool| tool.spec()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ToolHandler>> {
        self.tools.iter().find(|tool| tool.spec().name == name)
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}
