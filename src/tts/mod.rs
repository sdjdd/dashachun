#[derive(Debug)]
pub enum TtsEvent {}

#[derive(Debug)]
pub enum TtsError {}

pub trait Tts: Send + Sync {}
