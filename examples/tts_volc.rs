use std::time::Instant;

use dashachun::audio::DOWNLINK;
use dashachun::tts::volc::VolcTts;
use dashachun::tts::{TextStream, Tts, TtsEvent};
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;

fn text_stream(text: String) -> TextStream {
    let chunks = text.chars().map(|c| c.to_string()).collect::<Vec<_>>();
    Box::pin(futures_util::stream::iter(chunks))
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    let mut out = Vec::new();
    let data_len = (samples.len() * 2) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16000u32.to_le_bytes());
    out.extend_from_slice(&32000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, out)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let _ = dotenvy::dotenv();

    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "你好，我是小智，很高兴见到你。".to_string());
    let out_path = std::env::args()
        .nth(2)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("tts_out.wav"));

    let tts = match VolcTts::from_env(DOWNLINK) {
        Ok(Some(tts)) => tts,
        Ok(None) => {
            eprintln!(
                "set VOLC_TTS_API_KEY, VOLC_TTS_BASE_URL and VOLC_TTS_SPEAKER (optional VOLC_TTS_RESOURCE_ID)"
            );
            std::process::exit(1);
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let start = Instant::now();
    let mut samples = Vec::new();
    let mut events = tts.synthesize(text_stream(text), CancellationToken::new());
    while let Some(event) = events.next().await {
        let elapsed = start.elapsed().as_millis();
        match event {
            Ok(TtsEvent::SentenceStart { text }) => println!("[{elapsed:>6} ms] sentence: {text}"),
            Ok(TtsEvent::Subtitle(subtitle)) => println!(
                "[{elapsed:>6} ms] subtitle: {}..{} ms  {}",
                subtitle.start_ms, subtitle.end_ms, subtitle.text
            ),
            Ok(TtsEvent::Audio(chunk)) => {
                println!("[{elapsed:>6} ms] audio:    {} samples", chunk.len());
                samples.extend_from_slice(&chunk);
            }
            Ok(TtsEvent::Done) => {
                println!("[{elapsed:>6} ms] done",);
                break;
            }
            Err(err) => {
                eprintln!("[{elapsed:>6} ms] error:    {err}");
                std::process::exit(1);
            }
        }
    }

    if let Err(err) = write_wav(&out_path, &samples) {
        eprintln!("failed to write {}: {err}", out_path.display());
        std::process::exit(1);
    }
    println!(
        "wrote {} ({:.2}s)",
        out_path.display(),
        samples.len() as f32 / 16000.0
    );
}
