use serde::{Deserialize, Serialize};

use garde::Validate;

use super::DeviceRecord;

#[derive(Debug, Deserialize, Validate)]
pub struct ActivateDeviceArgs {
    #[garde(pattern(r"^[0-9]{6}$"))]
    pub code: String,
}

#[derive(Debug, Serialize)]
pub struct DeviceResponse {
    pub client_id: String,
    pub device_id: Option<String>,
    pub board_type: Option<String>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub activated_at: Option<time::OffsetDateTime>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: time::OffsetDateTime,
}

impl From<DeviceRecord> for DeviceResponse {
    fn from(record: DeviceRecord) -> Self {
        Self {
            client_id: record.client_id.to_string(),
            device_id: record.device_id,
            board_type: record.board_type,
            activated_at: record.activated_at,
            created_at: record.created_at,
        }
    }
}
