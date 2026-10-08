//! Concrete agent-stage providers. Each provider knows only the
//! [`crate::agent`] trait boundary types — never the agent's internal wiring.

pub mod asr;
pub mod llm;
pub mod tts;
