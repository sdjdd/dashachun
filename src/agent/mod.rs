mod asr;
mod capture;
mod composite;
mod emotion;
mod factory;
mod generate;
mod llm;
mod memory;
mod prompt;
mod store;
pub mod tool;
mod tts;

pub use asr::{Asr, AsrError, AsrEvent, AsrEvents, AudioStream};
pub use capture::{
    AudioCapture, AudioStorage, Capture, Frame, ReplyCapture, UtteranceCapture, collect,
};
pub use composite::CompositeAgent;
pub use factory::AgentFactory;
pub use generate::{
    StepResult, StopCondition, StreamEvent, StreamEvents, StreamResult, StreamTextParams,
    generate_text, stream_text,
};
pub use llm::{
    ChatItem, Llm, LlmError, LlmEvent, LlmEvents, ReasoningEffort, ToolCall, ToolChoice, ToolSpec,
};
pub use memory::{
    AddError, DbEntryMemory, DbMemory, EntryMemory, EntryOp, EntryOpResult, Extractor,
    InMemEntryMemory, InMemMemory, MAX_ENTRIES, Memory, MemoryEntry, MemoryHook, OnCommit,
    TranscriptMessage, read_transcript,
};
pub use prompt::{Speaker, SystemPrompt, labelled_utterance};
pub use store::{AgentRecord, AgentStore};
pub use tool::{NoArgs, Tool, ToolOutcome, ToolSet};
pub use tts::{Subtitle, TextStream, Tts, TtsError, TtsEvent, TtsEvents};

use std::pin::Pin;

use futures_util::Stream;

pub type AgentInputStream = Pin<Box<dyn Stream<Item = AgentInput> + Send>>;
pub type AgentOutputStream = Pin<Box<dyn Stream<Item = AgentOutput> + Send>>;

/// The agent's view of one conversation. `conversation_id` is the durable
/// row id the connection's message writes hang off; `sample_rate`/`channels`/
/// `frame_duration_ms` describe the device's uplink audio stream; the
/// downlink format is the server-owned [`crate::audio::DOWNLINK`] and never
/// reaches the agent.
#[derive(Debug, Clone)]
pub struct AgentSession {
    pub conversation_id: i64,
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
    TtsSubtitle { subtitle: Subtitle },
    Audio(Vec<f32>),
    TtsStop,
    TtsAbort,
    Emotion { emotion: String },
    Mcp(serde_json::Value),
    Error { message: String },
}

pub trait Agent: Send + Sync {
    fn run(&self, session: AgentSession, input: AgentInputStream) -> AgentOutputStream;
}
