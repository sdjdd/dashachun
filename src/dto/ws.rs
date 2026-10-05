use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ClientHello {
    pub version: u32,
    pub transport: String,
    #[serde(default)]
    pub features: Features,
    #[serde(default)]
    pub audio_params: Option<AudioParams>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Features {
    pub mcp: bool,
    pub aec: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct AudioParams {
    pub format: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub frame_duration: u32,
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            format: "opus".into(),
            sample_rate: 16000,
            channels: 1,
            frame_duration: 60,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ServerHello {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub transport: &'static str,
    pub session_id: String,
    pub audio_params: AudioParams,
}

impl ServerHello {
    pub fn new(session_id: String, audio_params: AudioParams) -> Self {
        Self {
            type_: "hello",
            transport: "websocket",
            session_id,
            audio_params,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum InboundMessage {
    #[serde(rename = "hello")]
    Hello(ClientHello),
    #[serde(rename = "listen")]
    Listen(Listen),
    #[serde(rename = "abort")]
    Abort(Abort),
    #[serde(rename = "mcp")]
    Mcp(Mcp),
}

#[derive(Debug, Deserialize)]
pub struct Listen {
    pub state: String,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Abort {
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Mcp {
    pub payload: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct SttMessage {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub session_id: String,
    pub text: String,
}

impl SttMessage {
    pub fn new(session_id: String, text: String) -> Self {
        Self {
            type_: "stt",
            session_id,
            text,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct LlmMessage {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub session_id: String,
    pub emotion: String,
}

impl LlmMessage {
    pub fn new(session_id: String, emotion: String) -> Self {
        Self {
            type_: "llm",
            session_id,
            emotion,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TtsMessage {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub session_id: String,
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl TtsMessage {
    pub fn start(session_id: String) -> Self {
        Self {
            type_: "tts",
            session_id,
            state: "start",
            text: None,
        }
    }

    pub fn stop(session_id: String) -> Self {
        Self {
            type_: "tts",
            session_id,
            state: "stop",
            text: None,
        }
    }

    pub fn sentence_start(session_id: String, text: String) -> Self {
        Self {
            type_: "tts",
            session_id,
            state: "sentence_start",
            text: Some(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_message_serializes_emotion() {
        let value = serde_json::to_value(LlmMessage::new("s1".into(), "happy".into())).unwrap();
        assert_eq!(value["type"], "llm");
        assert_eq!(value["session_id"], "s1");
        assert_eq!(value["emotion"], "happy");
        assert!(value.get("text").is_none());
    }
}
