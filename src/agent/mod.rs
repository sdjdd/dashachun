mod composite;

use std::pin::Pin;

use futures_util::Stream;

pub use composite::CompositeAgent;

pub type AgentInputStream = Pin<Box<dyn Stream<Item = AgentInput> + Send>>;
pub type AgentOutputStream = Pin<Box<dyn Stream<Item = AgentOutput> + Send>>;

#[derive(Debug, Clone)]
pub struct AgentSession {
    pub id: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_duration_ms: u32,
}

#[derive(Debug, Clone)]
pub enum AgentInput {
    ListenStart { mode: Option<String> },
    ListenStop,
    Audio(Vec<f32>),
    Interrupt { reason: Option<String> },
    Mcp(serde_json::Value),
}

#[derive(Debug, Clone)]
pub enum AgentOutput {
    Stt { text: String, is_final: bool },
    TtsStart,
    TtsSentence { text: String },
    TtsSubtitle { subtitle: crate::tts::Subtitle },
    Audio(Vec<f32>),
    TtsStop,
    TtsAbort,
    Mcp(serde_json::Value),
    Error { message: String },
}

pub trait Agent: Send + Sync {
    fn run(&self, session: AgentSession, input: AgentInputStream) -> AgentOutputStream;
}
