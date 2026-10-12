# Voiceprint — integration notes

`mod.rs` is the capability (PCM → embedding); this file records what a future
agent integration needs to know. Deployment (which model, where it lives) is
in `docs/architecture/voiceprint.md`; only the integration-relevant summary
is repeated here.

## Module surface and operational facts

- `Voiceprint::load(path)` → `embed(&[i16]) -> Vec<f32>` (192-dim) +
  `cosine(a, b)`. Constants: `MODEL_SHA256`, `EMBEDDING_DIM = 192`,
  `SAMPLE_RATE = 16_000`, `DEFAULT_MODEL_PATH`.
- `embed` needs `&mut self` (ort rc.10 session API). One ORT session load
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

## Why `ort`/`ndarray` are pinned (do not bump casually)

`ort` is pinned `=2.0.0-rc.10` and `ndarray` `0.16` **because of the VAD
crate, not because of this module**. Bumping both to `ort 2.0.0-rc.13` +
`ndarray 0.17` was attempted (2025-10) and reverted; the full picture:

- **The actual blocker is `voice_activity_detector`** (our Silero VAD).
  Its 0.2.0 `ort` requirement is a loose `^2.0.0-rc.9` — cargo resolves it
  to whatever `ort` we pick (pre-release tuple rule) — while its own code
  passes ndarray **0.16** views into `ort::inputs!`. On rc.13 (ndarray
  feature 0.17) the compile breaks **inside its source**, not ours. Upstream
  0.2.1 responds by hard-locking `=2.0.0-rc.10` (see the comment in its
  `Cargo.toml`); the repo has no rc.13 follow-up (last commit 2025-08).
- **A fork patch works and was prototyped**: vendored
  `voice_activity_detector` with `ndarray = "0.17"` + `ort = "=2.0.0-rc.13"`
  compiles after a trivial Cargo.toml-only diff (its `Tensor::from_array` /
  `try_extract_array` usage is already rc.10-style), our model test passed.
  Costs: a `[patch.crates-io]` entry plus the whole crate vendored in-tree —
  including its 2.3 MB `silero_vad.onnx` baked in via `include_bytes!` —
  and a fork to maintain forever. Judged not worth it.
- **rc.13-specific API breaks** hit in this module (for whoever retries):
  `ort::Error` becomes tag-generic `Error<R>` — `#[from] ort::Error` no
  longer fits, errors must be carried as `Onnx(String)` via `map_err`
  (its `Display` is message-only anyway); `try_extract_tensor` returns
  `(&Shape, &[T])` (`.1` still works); `run` rejects a bare `Value` —
  keep the `ort::inputs![...]` macro; and knf-rs still returns an ndarray
  0.16 `Array2`, which needs
  `into_raw_vec_and_offset()` + `Array2::from_shape_vec` to cross into
  0.17 before `Tensor::from_array`.
- **Patch footgun**: a `[patch.crates-io]` source whose `[package]`
  version does not exactly match the version resolved in the graph is
  **silently unused** ("patch ... was not used in the crate graph") and the
  registry copy compiles instead — the failure looks like the patch never
  happened.

Reopen condition: upstream `voice_activity_detector` ships a version
supporting a newer `ort` (check its `Cargo.toml` first), or dashachun
replaces/writes its own VAD — at which point bump `ort` + `ndarray`
together and re-run the model test.
