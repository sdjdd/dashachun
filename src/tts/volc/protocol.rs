use serde_json::Value;

pub const PROTOCOL_VERSION: u8 = 0b0001;
pub const HEADER_SIZE: u8 = 1;

pub const MSG_FULL_CLIENT_REQUEST: u8 = 0b0001;
pub const MSG_FULL_SERVER_RESPONSE: u8 = 0b1001;
pub const MSG_AUDIO_ONLY_SERVER: u8 = 0b1011;
pub const MSG_SERVER_ERROR_RESPONSE: u8 = 0b1111;

pub const FLAG_WITH_EVENT: u8 = 0b0100;

pub const SERIALIZATION_JSON: u8 = 0b0001;
pub const COMPRESSION_NONE: u8 = 0b0000;

pub const EVENT_START_CONNECTION: i32 = 1;
pub const EVENT_FINISH_CONNECTION: i32 = 2;
pub const EVENT_CONNECTION_STARTED: i32 = 50;
pub const EVENT_CONNECTION_FAILED: i32 = 51;
pub const EVENT_CONNECTION_FINISHED: i32 = 52;
pub const EVENT_START_SESSION: i32 = 100;
pub const EVENT_FINISH_SESSION: i32 = 102;
pub const EVENT_SESSION_STARTED: i32 = 150;
pub const EVENT_SESSION_FINISHED: i32 = 152;
pub const EVENT_SESSION_FAILED: i32 = 153;
pub const EVENT_TASK_REQUEST: i32 = 200;
pub const EVENT_TTS_SENTENCE_START: i32 = 350;
pub const EVENT_TTS_SENTENCE_END: i32 = 351;
pub const EVENT_TTS_RESPONSE: i32 = 352;
pub const EVENT_TTS_SUBTITLE: i32 = 364;

fn is_connection_event(event: i32) -> bool {
    matches!(
        event,
        EVENT_START_CONNECTION
            | EVENT_FINISH_CONNECTION
            | EVENT_CONNECTION_STARTED
            | EVENT_CONNECTION_FAILED
            | EVENT_CONNECTION_FINISHED
    )
}

fn is_connection_id_event(event: i32) -> bool {
    matches!(
        event,
        EVENT_CONNECTION_STARTED | EVENT_CONNECTION_FAILED | EVENT_CONNECTION_FINISHED
    )
}

fn header(message_type: u8, flags: u8, serialization: u8, compression: u8) -> [u8; 4] {
    [
        (PROTOCOL_VERSION << 4) | HEADER_SIZE,
        (message_type << 4) | flags,
        (serialization << 4) | compression,
        0x00,
    ]
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    push_u32(out, bytes.len() as u32);
    out.extend_from_slice(bytes);
}

fn event_frame(event: i32, session_id: &str, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(16 + session_id.len() + payload.len());
    frame.extend_from_slice(&header(
        MSG_FULL_CLIENT_REQUEST,
        FLAG_WITH_EVENT,
        SERIALIZATION_JSON,
        COMPRESSION_NONE,
    ));
    frame.extend_from_slice(&event.to_be_bytes());
    if !is_connection_event(event) {
        push_bytes(&mut frame, session_id.as_bytes());
    }
    push_bytes(&mut frame, payload);
    frame
}

pub fn start_connection() -> Vec<u8> {
    event_frame(EVENT_START_CONNECTION, "", b"{}")
}

pub fn finish_connection() -> Vec<u8> {
    event_frame(EVENT_FINISH_CONNECTION, "", b"{}")
}

pub fn start_session(session_id: &str, payload: &Value) -> Vec<u8> {
    event_frame(
        EVENT_START_SESSION,
        session_id,
        payload.to_string().as_bytes(),
    )
}

pub fn task_request(session_id: &str, text: &str) -> Vec<u8> {
    let payload = serde_json::json!({ "req_params": { "text": text } });
    event_frame(
        EVENT_TASK_REQUEST,
        session_id,
        payload.to_string().as_bytes(),
    )
}

pub fn finish_session(session_id: &str) -> Vec<u8> {
    event_frame(EVENT_FINISH_SESSION, session_id, b"{}")
}

#[derive(Debug, Default)]
pub struct Message {
    pub message_type: u8,
    pub event: i32,
    pub session_id: String,
    pub connect_id: String,
    pub error_code: Option<u32>,
    pub payload: Vec<u8>,
}

impl Message {
    pub fn json(&self) -> Option<Value> {
        serde_json::from_slice(&self.payload).ok()
    }

    pub fn audio(&self) -> &[u8] {
        &self.payload
    }
}

#[derive(Debug)]
pub enum ProtocolError {
    Truncated,
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::Truncated => write!(f, "truncated message frame"),
        }
    }
}

impl std::error::Error for ProtocolError {}

fn take<'a>(data: &mut &'a [u8], n: usize) -> Result<&'a [u8], ProtocolError> {
    if data.len() < n {
        return Err(ProtocolError::Truncated);
    }
    let (head, rest) = data.split_at(n);
    *data = rest;
    Ok(head)
}

fn take_u32(data: &mut &[u8]) -> Result<u32, ProtocolError> {
    let bytes = take(data, 4)?;
    Ok(u32::from_be_bytes(bytes.try_into().expect("4 bytes")))
}

fn take_i32(data: &mut &[u8]) -> Result<i32, ProtocolError> {
    let bytes = take(data, 4)?;
    Ok(i32::from_be_bytes(bytes.try_into().expect("4 bytes")))
}

fn take_bytes<'a>(data: &mut &'a [u8]) -> Result<&'a [u8], ProtocolError> {
    let len = take_u32(data)? as usize;
    take(data, len)
}

pub fn parse(data: &[u8]) -> Result<Message, ProtocolError> {
    if data.len() < 4 {
        return Err(ProtocolError::Truncated);
    }
    let header_size = (data[0] & 0x0f) as usize;
    let message_type = data[1] >> 4;
    let flags = data[1] & 0x0f;

    let mut rest = data
        .get(header_size * 4..)
        .ok_or(ProtocolError::Truncated)?;
    let mut message = Message {
        message_type,
        ..Default::default()
    };

    let has_event = flags & FLAG_WITH_EVENT != 0;
    if has_event {
        message.event = take_i32(&mut rest)?;
    }
    if message_type == MSG_SERVER_ERROR_RESPONSE {
        message.error_code = Some(take_u32(&mut rest)?);
    }
    if has_event && !is_connection_event(message.event) {
        message.session_id = String::from_utf8_lossy(take_bytes(&mut rest)?).into_owned();
    }
    if has_event && is_connection_id_event(message.event) {
        message.connect_id = String::from_utf8_lossy(take_bytes(&mut rest)?).into_owned();
    }
    if !rest.is_empty() {
        message.payload = take_bytes(&mut rest)?.to_vec();
    }
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn start_connection_has_no_session_id() {
        let frame = start_connection();
        assert_eq!(frame[0], 0x11);
        assert_eq!(frame[1], (MSG_FULL_CLIENT_REQUEST << 4) | FLAG_WITH_EVENT);
        assert_eq!(frame[2], (SERIALIZATION_JSON << 4) | COMPRESSION_NONE);
        assert_eq!(i32::from_be_bytes(frame[4..8].try_into().unwrap()), 1);
        assert_eq!(&frame[8..12], &2u32.to_be_bytes());
        assert_eq!(&frame[12..], b"{}");
        assert_eq!(frame.len(), 14);
    }

    #[test]
    fn start_session_carries_session_id_then_payload() {
        let frame = start_session("abc", &json!({"user": {"uid": "1"}}));
        assert_eq!(i32::from_be_bytes(frame[4..8].try_into().unwrap()), 100);
        assert_eq!(&frame[8..12], &3u32.to_be_bytes());
        assert_eq!(&frame[12..15], b"abc");
        let payload_len = u32::from_be_bytes(frame[15..19].try_into().unwrap()) as usize;
        assert_eq!(frame.len(), 19 + payload_len);
    }

    #[test]
    fn task_request_uses_event_200() {
        let frame = task_request("abc", "你好");
        assert_eq!(i32::from_be_bytes(frame[4..8].try_into().unwrap()), 200);
        let payload = serde_json::from_slice::<Value>(&frame[19..]).unwrap();
        assert_eq!(payload["req_params"]["text"], "你好");
    }

    #[test]
    fn parse_connection_started_reads_connect_id() {
        let mut frame = Vec::new();
        frame.extend_from_slice(&header(
            MSG_FULL_SERVER_RESPONSE,
            FLAG_WITH_EVENT,
            SERIALIZATION_JSON,
            COMPRESSION_NONE,
        ));
        frame.extend_from_slice(&50i32.to_be_bytes());
        frame.extend_from_slice(&3u32.to_be_bytes());
        frame.extend_from_slice(b"cid");
        frame.extend_from_slice(&2u32.to_be_bytes());
        frame.extend_from_slice(b"{}");

        let message = parse(&frame).unwrap();
        assert_eq!(message.message_type, MSG_FULL_SERVER_RESPONSE);
        assert_eq!(message.event, EVENT_CONNECTION_STARTED);
        assert_eq!(message.connect_id, "cid");
        assert_eq!(message.session_id, "");
    }

    #[test]
    fn parse_audio_only_server_returns_raw_payload() {
        let mut frame = Vec::new();
        frame.extend_from_slice(&header(
            MSG_AUDIO_ONLY_SERVER,
            FLAG_WITH_EVENT,
            0b0000,
            COMPRESSION_NONE,
        ));
        frame.extend_from_slice(&352i32.to_be_bytes());
        frame.extend_from_slice(&3u32.to_be_bytes());
        frame.extend_from_slice(b"sid");
        frame.extend_from_slice(&4u32.to_be_bytes());
        frame.extend_from_slice(&[1, 2, 3, 4]);

        let message = parse(&frame).unwrap();
        assert_eq!(message.message_type, MSG_AUDIO_ONLY_SERVER);
        assert_eq!(message.event, EVENT_TTS_RESPONSE);
        assert_eq!(message.session_id, "sid");
        assert_eq!(message.audio(), &[1, 2, 3, 4]);
    }

    #[test]
    fn parse_error_frame_reads_code() {
        let mut frame = Vec::new();
        frame.extend_from_slice(&header(
            MSG_SERVER_ERROR_RESPONSE,
            FLAG_WITH_EVENT,
            SERIALIZATION_JSON,
            COMPRESSION_NONE,
        ));
        frame.extend_from_slice(&0i32.to_be_bytes());
        frame.extend_from_slice(&45000001u32.to_be_bytes());
        frame.extend_from_slice(&2u32.to_be_bytes());
        frame.extend_from_slice(b"{}");

        let message = parse(&frame).unwrap();
        assert_eq!(message.message_type, MSG_SERVER_ERROR_RESPONSE);
        assert_eq!(message.error_code, Some(45000001));
    }

    #[test]
    fn truncated_frame_is_rejected() {
        assert!(matches!(
            parse(&[0x11, 0x14]),
            Err(ProtocolError::Truncated)
        ));
    }
}
