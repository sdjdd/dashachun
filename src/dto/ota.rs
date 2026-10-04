use serde::{Deserialize, Serialize};

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct OtaRequest {
    pub application: Application,
    pub board: Board,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct Application {
    pub version: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct Board {
    #[serde(rename = "type")]
    pub board_type: String,
}

#[derive(Serialize)]
pub struct OtaResponse {
    pub server_time: ServerTime,
    pub firmware: Firmware,
    pub websocket: Websocket,
}

#[derive(Serialize)]
pub struct ServerTime {
    pub timestamp: u64,
    pub timezone_offset: i32,
}

#[derive(Serialize)]
pub struct Firmware {
    pub version: String,
    pub url: String,
}

#[derive(Serialize)]
pub struct Websocket {
    pub url: String,
    pub token: String,
}
