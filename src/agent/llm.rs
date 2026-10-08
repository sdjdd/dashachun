use std::fmt;
use std::pin::Pin;

use futures_util::Stream;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub type LlmEvents<'a> = Pin<Box<dyn Stream<Item = Result<LlmEvent, LlmError>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatItem {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

impl ChatItem {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant {
            content: None,
            tool_calls,
        }
    }

    pub fn assistant_text_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            tool_calls,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmEvent {
    Delta { text: String },
    ToolCall(ToolCall),
    Done,
}

#[derive(Debug)]
pub enum LlmError {
    Failed(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Failed(message) => write!(f, "llm failed: {message}"),
        }
    }
}

impl std::error::Error for LlmError {}

impl From<String> for LlmError {
    fn from(message: String) -> Self {
        LlmError::Failed(message)
    }
}

impl From<&str> for LlmError {
    fn from(message: &str) -> Self {
        LlmError::Failed(message.to_string())
    }
}

pub trait Llm: Send + Sync {
    fn chat(
        &self,
        history: Vec<ChatItem>,
        tools: Vec<ToolSpec>,
        cancel: CancellationToken,
    ) -> LlmEvents<'_>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_items_serialize_with_ai_sdk_roles() {
        assert_eq!(
            serde_json::to_value(ChatItem::user("hello")).unwrap(),
            json!({"role": "user", "content": "hello"})
        );
        assert_eq!(
            serde_json::to_value(ChatItem::assistant("hi")).unwrap(),
            json!({"role": "assistant", "content": "hi"})
        );
        let round = ChatItem::assistant_text_tool_calls(
            "好的，我查一下",
            vec![ToolCall {
                id: "call_x".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
            }],
        );
        assert_eq!(
            serde_json::to_value(round).unwrap(),
            json!({
                "role": "assistant",
                "content": "好的，我查一下",
                "tool_calls": [{"id": "call_x", "name": "lookup", "arguments": "{}"}]
            })
        );
        assert_eq!(
            serde_json::to_value(ChatItem::tool("call_x", "result")).unwrap(),
            json!({"role": "tool", "tool_call_id": "call_x", "content": "result"})
        );
    }

    #[test]
    fn assistant_json_omits_absent_fields() {
        let calls = vec![ToolCall {
            id: "call_x".into(),
            name: "lookup".into(),
            arguments: "{}".into(),
        }];
        assert_eq!(
            serde_json::to_value(ChatItem::assistant_tool_calls(calls)).unwrap(),
            json!({
                "role": "assistant",
                "tool_calls": [{"id": "call_x", "name": "lookup", "arguments": "{}"}]
            })
        );
    }

    #[test]
    fn chat_item_json_round_trips() {
        let items = vec![
            ChatItem::system("system prompt"),
            ChatItem::user("hello"),
            ChatItem::assistant("hi"),
            ChatItem::assistant_text_tool_calls(
                "好的",
                vec![ToolCall {
                    id: "1".into(),
                    name: "a".into(),
                    arguments: "{}".into(),
                }],
            ),
            ChatItem::assistant_tool_calls(vec![ToolCall {
                id: "2".into(),
                name: "b".into(),
                arguments: "{}".into(),
            }]),
            ChatItem::tool("1", "one"),
        ];
        for item in items {
            let value = serde_json::to_value(&item).unwrap();
            assert_eq!(serde_json::from_value::<ChatItem>(value).unwrap(), item);
        }
    }
}
