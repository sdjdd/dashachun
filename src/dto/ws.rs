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
