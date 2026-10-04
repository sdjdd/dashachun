use std::fmt;
use std::pin::Pin;

use futures_util::Stream;

pub mod openai;

pub use openai::{OpenAiConfig, OpenAiLlm};

pub type LlmEvents<'a> = Pin<Box<dyn Stream<Item = Result<LlmEvent, LlmError>> + Send + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmEvent {
    Delta { text: String },
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
    fn chat(&self, history: Vec<ChatMessage>) -> LlmEvents<'_>;
}
