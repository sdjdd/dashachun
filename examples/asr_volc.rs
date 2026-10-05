use std::path::PathBuf;
use std::time::{Duration, Instant};

use dashachun::asr::volc::VolcAsr;
use dashachun::asr::{Asr, AsrEvent, AudioStream};
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;

const CHUNK_SAMPLES: usize = 960;
const CHUNK_INTERVAL: Duration = Duration::from_millis(60);

fn read_wav(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pcm = None;
    let mut off = 12;
    while off + 8 <= data.len() {
        let id = &data[off..off + 4];
        let size = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
        let body = off + 8;
        if body + size > data.len() {
            return Err("truncated chunk".into());
        }
        if id == b"data" {
            let (chunks, _) = data[body..body + size].as_chunks::<2>();
            pcm = Some(
                chunks
                    .iter()
                    .map(|c| i16::from_le_bytes(*c) as f32 / 32768.0)
                    .collect(),
            );
        }
        off = body + size + (size & 1);
    }
    pcm.ok_or_else(|| "missing data chunk".to_string())
}

fn audio_stream(samples: Vec<f32>) -> AudioStream {
    let chunks = samples
        .chunks(CHUNK_SAMPLES)
        .map(|c| c.to_vec())
        .collect::<Vec<_>>();
    Box::pin(futures_util::stream::unfold(
        chunks.into_iter(),
        |mut iter| async move {
            let chunk = iter.next()?;
            tokio::time::sleep(CHUNK_INTERVAL).await;
            Some((chunk, iter))
        },
    ))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let _ = dotenvy::dotenv();

    let Some(path) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: cargo run --example asr_volc -- <16k-mono-pcm16.wav>");
        std::process::exit(2);
    };
    let samples = match read_wav(&path) {
        Ok(samples) => samples,
        Err(err) => {
            eprintln!("failed to read {}: {err}", path.display());
            std::process::exit(1);
        }
    };
    println!(
        "{}: {} samples ({:.2}s)",
        path.display(),
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    let Some(asr) = VolcAsr::from_env() else {
        eprintln!("set VOLC_ASR_API_KEY and VOLC_ASR_BASE_URL");
        std::process::exit(1);
    };

    let start = Instant::now();
    let mut events = asr.transcribe(audio_stream(samples), CancellationToken::new());
    while let Some(event) = events.next().await {
        let elapsed = start.elapsed().as_millis();
        match event {
            Ok(AsrEvent::Partial { text }) => println!("[{elapsed:>6} ms] partial: {text}"),
            Ok(AsrEvent::Final { text }) => {
                println!("[{elapsed:>6} ms] final:   {text}");
                break;
            }
            Err(err) => {
                eprintln!("[{elapsed:>6} ms] error:   {err}");
                std::process::exit(1);
            }
        }
    }
}
