# Dev-stack connection defaults (compose.yaml); values already present in the
# environment win.
export DATABASE_URL := env_var_or_default("DATABASE_URL", "postgres://postgres:dev@localhost:5432/dsc")
export REDIS_URL := env_var_or_default("REDIS_URL", "redis://localhost:6379")
export S3_ENDPOINT := env_var_or_default("S3_ENDPOINT", "http://localhost:9000")
export S3_ACCESS_KEY_ID := env_var_or_default("S3_ACCESS_KEY_ID", "rustfsadmin")
export S3_SECRET_ACCESS_KEY := env_var_or_default("S3_SECRET_ACCESS_KEY", "rustfsadmin")

suites := "api auth devices memory capture agents providers audio_capture settings oss"

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

# psql shell in the dev postgres
psql:
    docker compose exec postgres psql -U postgres -d dsc

# follow the dev stack logs (optionally: just logs postgres)
logs *services:
    docker compose logs -f {{services}}
