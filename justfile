# dev-stack connection defaults (compose.yaml); values already present in the
# environment win.
export DATABASE_URL := env_var_or_default("DATABASE_URL", "postgres://postgres:dev@localhost:5432/dsc")
export REDIS_URL := env_var_or_default("REDIS_URL", "redis://localhost:6379")
export S3_ENDPOINT := env_var_or_default("S3_ENDPOINT", "http://localhost:9000")
export S3_ACCESS_KEY_ID := env_var_or_default("S3_ACCESS_KEY_ID", "rustfsadmin")
export S3_SECRET_ACCESS_KEY := env_var_or_default("S3_SECRET_ACCESS_KEY", "rustfsadmin")

# the pinned CAM++ voiceprint model (src/voiceprint/mod.rs holds the same
# URL + sha; keep the two in sync)
model_url := "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
model_sha := "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11"
model_path := env_var_or_default("VOICEPRINT_MODEL", "models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")

# the pinned Silero VAD v6 model (src/vad/mod.rs holds the same URL + sha;
# keep the two in sync)
vad_model_url := "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx"
vad_model_sha := "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
vad_model_path := env_var_or_default("SILERO_VAD_MODEL", "models/silero_vad.onnx")

suites := "api auth devices memory capture agents providers audio_capture settings oss speakers"

# start the compose dev stack and wait for its healthchecks
up:
    docker compose up -d --wait

# reformat in place
fmt:
    cargo fmt

# one integration suite, against the dev stack
test name:
    cargo test --test {{name}} -- --ignored

# every integration suite, against the dev stack
tests: up
    fail=0; for t in {{suites}}; do cargo test --test $t -- --ignored || fail=1; done; exit $fail

# everything that must pass before a commit; CI (.github/workflows/ci.yml) runs exactly this
ci:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test --lib
    just tests

# run the server against the dev stack (it reads .env itself)
run: up
    cargo run

# download the pinned models (sha256-verified) into models/
model:
    mkdir -p "$(dirname {{model_path}})"
    curl -fL --retry 3 -o {{model_path}} {{model_url}}
    echo "{{model_sha}}  {{model_path}}" | sha256sum -c -
    mkdir -p "$(dirname {{vad_model_path}})"
    curl -fL --retry 3 -o {{vad_model_path}} {{vad_model_url}}
    echo "{{vad_model_sha}}  {{vad_model_path}}" | sha256sum -c -

# psql shell in the dev postgres
psql:
    docker compose exec postgres psql -U postgres -d dsc

# follow the dev stack logs (optionally: just logs postgres)
logs *services:
    docker compose logs -f {{services}}
