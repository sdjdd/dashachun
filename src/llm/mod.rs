#[derive(Debug)]
pub enum LlmEvent {}

#[derive(Debug)]
pub enum LlmError {}

pub trait Llm: Send + Sync {}
