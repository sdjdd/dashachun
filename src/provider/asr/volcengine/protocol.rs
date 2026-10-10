use std::io::Write;

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde_json::Value;

pub const PROTOCOL_VERSION: u8 = 0b0001;
pub const HEADER_SIZE: u8 = 1;

pub const MSG_FULL_CLIENT_REQUEST: u8 = 0b0001;
pub const MSG_AUDIO_ONLY_REQUEST: u8 = 0b0010;
pub const MSG_FULL_SERVER_RESPONSE: u8 = 0b1001;
pub const MSG_SERVER_ERROR_RESPONSE: u8 = 0b1111;

pub const FLAG_POS_SEQUENCE: u8 = 0b0001;
pub const FLAG_NEG_WITH_SEQUENCE: u8 = 0b0011;

pub const SERIALIZATION_JSON: u8 = 0b0001;

pub const COMPRESSION_GZIP: u8 = 0b0001;

#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub message_type: u8,
    pub flags: u8,
    pub serialization: u8,
    pub compression: u8,
}

impl Header {
    pub fn new(message_type: u8, flags: u8, serialization: u8, compression: u8) -> Self {
        Self {
            message_type,
            flags,
            serialization,
            compression,
        }
    }

    fn to_bytes(self) -> [u8; 4] {
        [
            (PROTOCOL_VERSION << 4) | HEADER_SIZE,
            (self.message_type << 4) | self.flags,
            (self.serialization << 4) | self.compression,
            0x00,
        ]
    }
}

pub fn gzip_compress(data: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).expect("gzip write");
    encoder.finish().expect("gzip finish")
}

pub fn gzip_decompress(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(data);
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut out)?;
    Ok(out)
}

pub fn build_full_client_request(seq: i32, payload: &Value) -> Vec<u8> {
    let header = Header::new(
        MSG_FULL_CLIENT_REQUEST,
        FLAG_POS_SEQUENCE,
        SERIALIZATION_JSON,
        COMPRESSION_GZIP,
    );
    let body = gzip_compress(payload.to_string().as_bytes());
    let mut frame = Vec::with_capacity(8 + body.len());
    frame.extend_from_slice(&header.to_bytes());
    frame.extend_from_slice(&seq.to_be_bytes());
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(&body);
    frame
}

pub fn build_audio_request(seq: i32, audio: &[u8], last: bool) -> Vec<u8> {
    let header = Header::new(
        MSG_AUDIO_ONLY_REQUEST,
        if last {
            FLAG_NEG_WITH_SEQUENCE
        } else {
            FLAG_POS_SEQUENCE
        },
        SERIALIZATION_JSON,
        COMPRESSION_GZIP,
    );
    let seq = if last { -seq } else { seq };
    let body = gzip_compress(audio);
    let mut frame = Vec::with_capacity(8 + body.len());
    frame.extend_from_slice(&header.to_bytes());
    frame.extend_from_slice(&seq.to_be_bytes());
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(&body);
    frame
}

#[derive(Debug, Default)]
pub struct Response {
    pub code: i32,
    pub is_last: bool,
    pub sequence: i32,
    pub payload: Option<Value>,
}

#[derive(Debug)]
pub enum ProtocolError {
    Truncated,
    Gzip(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::Truncated => write!(f, "truncated response frame"),
            ProtocolError::Gzip(err) => write!(f, "gzip error: {err}"),
            ProtocolError::Json(err) => write!(f, "json error: {err}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

pub fn parse_response(msg: &[u8]) -> Result<Response, ProtocolError> {
    if msg.len() < 4 {
        return Err(ProtocolError::Truncated);
    }
    let header_size = (msg[0] & 0x0f) as usize;
    let message_type = msg[1] >> 4;
    let flags = msg[1] & 0x0f;
    let serialization = msg[2] >> 4;
    let compression = msg[2] & 0x0f;

    let mut payload = msg.get(header_size * 4..).ok_or(ProtocolError::Truncated)?;
    let mut response = Response::default();

    if flags & 0x01 != 0 {
        let (seq, rest) = split::<4>(payload)?;
        response.sequence = i32::from_be_bytes(seq);
        payload = rest;
    }
    if flags & 0x02 != 0 {
        response.is_last = true;
    }

    match message_type {
        MSG_FULL_SERVER_RESPONSE => {
            let (_size, rest) = split::<4>(payload)?;
            payload = rest;
        }
        MSG_SERVER_ERROR_RESPONSE => {
            let (code, rest) = split::<4>(payload)?;
            response.code = i32::from_be_bytes(code);
            let (_size, rest) = split::<4>(rest)?;
            payload = rest;
        }
        _ => {}
    }

    if payload.is_empty() {
        return Ok(response);
    }

    let body = if compression == COMPRESSION_GZIP {
        gzip_decompress(payload).map_err(ProtocolError::Gzip)?
    } else {
        payload.to_vec()
    };

    if serialization == SERIALIZATION_JSON {
        response.payload = Some(serde_json::from_slice(&body).map_err(ProtocolError::Json)?);
    }
    Ok(response)
}

fn split<const N: usize>(data: &[u8]) -> Result<([u8; N], &[u8]), ProtocolError> {
    if data.len() < N {
        return Err(ProtocolError::Truncated);
    }
    let mut head = [0u8; N];
    head.copy_from_slice(&data[..N]);
    Ok((head, &data[N..]))
}

pub fn audio_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn full_client_request_roundtrip_header() {
        let frame = build_full_client_request(1, &json!({"a": 1}));
        assert_eq!(frame[0], 0x11);
        assert_eq!(frame[1], (MSG_FULL_CLIENT_REQUEST << 4) | FLAG_POS_SEQUENCE);
        assert_eq!(frame[2], (SERIALIZATION_JSON << 4) | COMPRESSION_GZIP);
        assert_eq!(
            i32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]),
            1
        );
    }

    #[test]
    fn audio_last_frame_negates_sequence() {
        let frame = build_audio_request(3, &[0, 1, 2], true);
        assert_eq!(frame[1] & 0x0f, FLAG_NEG_WITH_SEQUENCE);
        assert_eq!(
            i32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]),
            -3
        );
    }

    #[test]
    fn parse_full_server_response() {
        let payload = json!({"result": {"text": "hello"}});
        let body = gzip_compress(payload.to_string().as_bytes());
        let mut frame = vec![
            0x11,
            (MSG_FULL_SERVER_RESPONSE << 4) | FLAG_POS_SEQUENCE,
            (SERIALIZATION_JSON << 4) | COMPRESSION_GZIP,
            0x00,
        ];
        frame.extend_from_slice(&2i32.to_be_bytes());
        frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
        frame.extend_from_slice(&body);

        let response = parse_response(&frame).unwrap();
        assert_eq!(response.sequence, 2);
        assert_eq!(response.payload.unwrap()["result"]["text"], "hello");
    }

    #[test]
    fn pcm16_encodes_le() {
        assert_eq!(
            audio_to_pcm16(&[0.0, 1.0, -1.0]),
            vec![0, 0, 0xff, 0x7f, 0x01, 0x80]
        );
    }
}
