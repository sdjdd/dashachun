# Voiceprint — integration notes

`mod.rs` is the capability (PCM → embedding); this file records what a future
agent integration needs to know. Deployment (which model, where it lives) is
in `docs/architecture/voiceprint.md`; only the integration-relevant summary
is repeated here.

## Module surface and operational facts

- `Voiceprint::load(path)` → `embed(&[i16]) -> Vec<f32>` (192-dim) +
  `cosine(a, b)`. Constants: `MODEL_SHA256`, `EMBEDDING_DIM = 192`,
  `SAMPLE_RATE = 16_000`, `DEFAULT_MODEL_PATH`.
- `embed` needs `&mut self` (the ort session `run` takes `&mut`). One ORT session load
  costs ~100 ms — share one `Voiceprint` (e.g. `Arc<Mutex<_>>`), never
  per-connection. Per-utterance inference is tens of ms on CPU
  (demo: whole process incl. ORT init + 3 embeddings = 0.43 s).
- `embed` is synchronous CPU work: call it via `spawn_blocking` from async
  code, do not run it on a runtime worker.
- CPU ORT inference is deterministic (same PCM → same vector) — usable in
  tests and for embedding dedup.
- The pinned model outputs 192 dims; swapping to another export (e.g.
  ERes2NetV2, 512-dim) changes the dim and **invalidates every stored
  embedding** — re-enrollment is mandatory on model change. Store the dim
  alongside stored vectors.
- Device uplink is already 16 kHz mono — no resampling on the embed path.

## Firmware/protocol facts that shape the design

All verified against the `xiaozhi-esp32` checkout (protocol reference only):

- `CONFIG_SEND_WAKE_WORD_DATA` is **default-on**: the device sends the
  wake-word Opus frames (~1 s window including pre-roll) **before** the
  `listen state=detect` message. Wire order is
  `hello → binary opus frames → listen detect → listen start`.
- Our gateway today only tolerates this ordering: binary frames arriving
  before the decoder is ready are dropped, and `listen state=detect` is
  only logged (`src/gateway/session.rs`). Capturing the pre-`detect` audio
  window is the first integration task if identification should happen at
  wake time.
- Wake during speaking/listening: the firmware first sends
  `abort` (`reason: "wake_word_detected"`) and clears its send queue —
  the wake-word audio itself still arrives before the abort.
- The device **never learns who spoke**: downstream message types are
  `notify/tts/stt/llm/mcp/system/alert/custom` only. Identification is
  purely server-side and reaches the user exclusively through reply
  content (TTS/subtitles) — no protocol extension needed or possible.

## Reference design (xinnan-tech/xiaozhi-esp32-server + voiceprint-api)

The ecosystem's proven shape, worth copying or consciously rejecting:

- Voiceprint runs **in parallel with ASR on the same PCM buffer** — it
  never replaces or delays ASR (they `asyncio.gather` the two).
- Per-utterance identify: embed the utterance, cosine against enrolled
  candidates, take the max, threshold **0.4** (below = unidentified).
- Prompt injection: the identified speaker's first appearance is passed to
  the LLM as `{"speaker": "name", "content": "text"}` so the model greets
  that person naturally **once**; later turns carry plain text only (a
  connection-scoped `introduced_speakers` set prevents repetitive naming).
- Enrollment: recording upload → embed → store the vector. Candidates are
  scoped **per agent** in their model. At enrollment counts that matter
  (tens), brute-force cosine in the application layer is entirely
  sufficient — no vector index, no pgvector.

## Open decisions for dashachun (not yet made)

- When to identify: at the wake-word window, at the first utterance, or
  both; whether the result is cached per connection.
- Identity scope: enrolled voices per agent or per user; which candidate
  set a given device's audio is matched against (multi-user privacy).
- Enrollment surface: REST on the agent, and whether the existing audio
  captures (FLAC in S3) can double as the enrollment source.
- Storage: Postgres (`BYTEA` + dim column) vs anywhere else.
- Boot wiring: `VOICEPRINT_MODEL` env is the contract; unset = voiceprint
  off. Failure behavior when set but broken should follow the audio-capture
  fail-open precedent (warn + continue without voiceprint) — decide at
  integration.

## Validation assets

- The ignored model test (`embeds_deterministically_at_the_pinned_dimension`)
  asserts dim, finiteness and determinism; run after `just model` via
  `cargo test --lib voiceprint -- --ignored`.
- Separation baseline (sherpa-onnx reference wavs, this exact pipeline):
  same speaker 0.6495, cross speakers −0.08 / 0.01. Upstream
  `scripts/3dspeaker/test-onnx.py` is the diff baseline if numbers drift.

## ort/ndarray history: the rc.10 pin and the rc.13 bump (2026-10)

`ort` ran pinned `=2.0.0-rc.10` + `ndarray 0.16` until 2026-10. Bumping both
to `ort 2.0.0-rc.13` + `ndarray 0.17` was attempted (2025-10) and reverted —
the full picture, kept as the record:

- **The original blocker was `voice_activity_detector`** (then our Silero
  VAD). Its 0.2.0 `ort` requirement was a loose `^2.0.0-rc.9` — cargo
  resolved it to whatever `ort` we picked (pre-release tuple rule) — while
  its own code passed ndarray **0.16** views into `ort::inputs!`. On rc.13
  (ndarray feature 0.17) the compile broke **inside its source**, not ours.
  Upstream 0.2.1 responded by hard-locking `=2.0.0-rc.10`; the repo has no
  rc.13 follow-up (last commit 2025-08).
- **A fork patch worked and was prototyped**: vendored
  `voice_activity_detector` with `ndarray = "0.17"` + `ort = "=2.0.0-rc.13"`
  compiles after a trivial Cargo.toml-only diff. Rejected: a
  `[patch.crates-io]` entry plus the whole crate vendored in-tree —
  including its 2.3 MB `silero_vad.onnx` baked in via `include_bytes!` —
  and a fork to maintain forever.
- **rc.13 API breaks catalogued** (all now applied): `ort::Error` becomes
  tag-generic `Error<R>` — `#[from] ort::Error` no longer fits, so both
  audio modules carry `String` variants (`VoiceprintError::Onnx`,
  `VadError::Build`) via `map_err` at the call site (its `Display` is
  message-only anyway); `try_extract_tensor` returns `(&Shape, &[T])`
  (the `.1` indexing already matched); `run` rejects a bare `Value` — keep
  the `ort::inputs![...]` macro.
- **Patch footgun** (still worth knowing): a `[patch.crates-io]` source
  whose `[package]` version does not exactly match the version resolved in
  the graph is **silently unused** ("patch ... was not used in the crate
  graph") and the registry copy compiles instead — the failure looks like
  the patch never happened.

**The bump was completed 2026-10** after the VAD swap removed the blocker:
`ort =2.0.0-rc.13` (ORT 1.28) + `ndarray 0.17` (the single version in the
tree — `ort`'s ndarray feature shares it). The knf-rs ndarray-0.16 shim
that was going to be the remaining constraint became unnecessary: the
frontend moved from the C++ `knf-rs` binding (which pinned a second
ndarray 0.16 copy and pulled `eyre`) to the pure-Rust `kaldi-fbank` 0.1.0
— a faithful port of kaldi-native-fbank with no ndarray dependency at
all. The frontend contract is two overrides on the crate defaults
(`dither = 0.0`, `mel_opts.num_bins = 80`); the per-utterance mel-mean
subtraction moved out of `knf_rs::compute_fbank` into our `embed()`
explicitly.

Frontend parity was verified by a throwaway A/B (both extractors compiled,
same inputs): synthetic sine mixes and a real 16 kHz wav produce equal
frame counts with max abs diff ≈ 1.6e-4, mean abs diff ≈ 1e-6 — FFT
rounding level, so the separation baselines below carry over. The model
test (`embeds_deterministically_at_the_pinned_dimension`) passes on rc.13.
