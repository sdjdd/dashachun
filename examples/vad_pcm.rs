use std::path::PathBuf;

use xiaozhi_server_rs::vad::{Vad, VadConfig, VadEvent};

const FRAME_SAMPLES: usize = 960;

fn read_wav(path: &std::path::Path) -> Result<(u32, Vec<f32>), String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut rate = None;
    let mut pcm = None;
    let mut off = 12;
    while off + 8 <= data.len() {
        let id = &data[off..off + 4];
        let size = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
        let body = off + 8;
        if body + size > data.len() {
            return Err("truncated chunk".into());
        }
        match id {
            b"fmt " => {
                let format = u16::from_le_bytes(data[body..body + 2].try_into().unwrap());
                let channels = u16::from_le_bytes(data[body + 2..body + 4].try_into().unwrap());
                if format != 1 || channels != 1 {
                    return Err(format!(
                        "need mono PCM16, got format={format} channels={channels}"
                    ));
                }
                rate = Some(u32::from_le_bytes(
                    data[body + 4..body + 8].try_into().unwrap(),
                ));
            }
            b"data" => {
                let (chunks, _) = data[body..body + size].as_chunks::<2>();
                pcm = Some(
                    chunks
                        .iter()
                        .map(|c| i16::from_le_bytes(*c) as f32 / 32768.0)
                        .collect(),
                );
            }
            _ => {}
        }
        off = body + size + (size & 1);
    }
    match (rate, pcm) {
        (Some(rate), Some(pcm)) => Ok((rate, pcm)),
        _ => Err("missing fmt or data chunk".into()),
    }
}

fn main() {
    let Some(path) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: cargo run --example vad_pcm -- <16k-mono-pcm16.wav> [threshold]");
        std::process::exit(2);
    };

    let (rate, samples) = match read_wav(&path) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("failed to read {}: {err}", path.display());
            std::process::exit(1);
        }
    };
    println!(
        "rate={rate} samples={} duration={:.2}s",
        samples.len(),
        samples.len() as f32 / rate as f32
    );

    let mut config = VadConfig::default();
    if let Some(threshold) = std::env::args().nth(2).and_then(|v| v.parse::<f32>().ok()) {
        config.speech_threshold = threshold;
    }

    let mut vad = match Vad::new(rate, config) {
        Ok(vad) => vad,
        Err(err) => {
            eprintln!("failed to create vad: {err}");
            std::process::exit(1);
        }
    };

    let mut events = Vec::new();
    let mut speech_ms = 0u64;
    let mut speech_start = 0u64;
    for frame in samples.chunks(FRAME_SAMPLES) {
        vad.push(frame, &mut events);
        for event in events.drain(..) {
            match event {
                VadEvent::SpeechStart { at_ms } => {
                    speech_start = at_ms;
                    println!("[{at_ms:>7} ms] speech start");
                }
                VadEvent::SpeechEnd { at_ms } => {
                    speech_ms += at_ms - speech_start;
                    println!(
                        "[{at_ms:>7} ms] speech end (segment {:.2}s)",
                        (at_ms - speech_start) as f32 / 1000.0
                    );
                }
            }
        }
    }
    vad.flush(&mut events);
    for event in events {
        if let VadEvent::SpeechEnd { at_ms } = event {
            speech_ms += at_ms - speech_start;
            println!(
                "[{at_ms:>7} ms] speech end (segment {:.2}s, flushed)",
                (at_ms - speech_start) as f32 / 1000.0
            );
        }
    }
    println!("speech total: {:.2}s", speech_ms as f32 / 1000.0);
}
