-- Early development: the whole schema lives in this single migration and is
-- amended in place while the project has not shipped yet.
CREATE TABLE users (
    id            BIGSERIAL PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Login sessions: the bearer tokens / cookies issued by /api/auth,
-- validated on every authenticated request by absolute expiry.
CREATE TABLE auth_sessions (
    id         UUID PRIMARY KEY,
    user_id    BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX auth_sessions_user_id_idx ON auth_sessions (user_id);

CREATE TABLE agents (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL,
    name            TEXT NOT NULL,
    persona_prompt  TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, name)
);

CREATE INDEX agents_user_id_idx ON agents (user_id);

CREATE TABLE devices (
    id                          BIGSERIAL PRIMARY KEY,
    client_id                   UUID NOT NULL UNIQUE,
    device_id                   TEXT,
    board_type                  TEXT,
    user_id                     BIGINT,
    agent_id                    BIGINT,
    token_hash                  TEXT,
    activated_at                TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX devices_user_id_idx ON devices (user_id);
CREATE INDEX devices_agent_id_idx ON devices (agent_id);

-- Device conversation sessions: one row per accepted gateway connection —
-- created by the upgrade route with the server-assigned `session_id` (the
-- wire id echoed by the device), closed when the device goes offline.
-- `ended_at` stays NULL only when the server dies mid-session. `device_id`
-- and `agent_id` are the binding snapshot at hello: ownership of a session
-- never drifts when the device is re-bound afterwards. Integrity is
-- enforced in the app layer; no foreign keys.
CREATE TABLE device_sessions (
    id         TEXT PRIMARY KEY,
    device_id  BIGINT NOT NULL,
    agent_id   BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at   TIMESTAMPTZ
);

CREATE INDEX device_sessions_device_id_idx ON device_sessions (device_id);

-- The durable conversation log: one row per message, the JSON form of the
-- agent's `ChatItem` in the AI SDK ModelMessage shape (`role` as the serde
-- tag; user rows carry the utterance, assistant rows text and/or tool-call
-- requests, tool rows one result each — `ORDER BY id` rebuilds the
-- conversation). `System` messages are never stored (the agent prepends its
-- prompt every turn). `speaker_id` is the attribution the voiceprint
-- identification matched for a user row: the matched `speakers` row id,
-- NULL when the utterance identified no one (below threshold, failed, not
-- ready in time) or the agent runs no identification. A deleted speaker
-- leaves old rows pointing nowhere — consumers resolve the id against the
-- agent's live roster and treat a miss as unidentified. Only user rows
-- carry it. Ownership derives through `device_sessions`; no foreign keys.
CREATE TABLE messages (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    message    JSONB NOT NULL,
    speaker_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX messages_session_id_idx ON messages (session_id, created_at);

-- FLAC capture (16-bit PCM, metadata blocks + frames) stored as one S3
-- object per row, referenced by its `bucket` + `object_key`
-- (`audios/{YYYYMMDD}/{client_id}/{uuid}.flac` under that bucket); attached
-- to its message (the user message for uplink, the turn's last message for
-- downlink). The bucket is recorded per row because the capture settings
-- can switch it at runtime — old rows keep reading from their own bucket.
-- Rows exist only when audio storage is configured — the message row is
-- the primary record, the object is best-effort.
CREATE TABLE message_audios (
    id          BIGSERIAL PRIMARY KEY,
    message_id  BIGINT NOT NULL,
    bucket      TEXT NOT NULL,
    object_key  TEXT NOT NULL,
    sample_rate INT NOT NULL,
    channels    INT NOT NULL,
    duration_ms INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX message_audios_message_id_idx ON message_audios (message_id);

-- Per-agent speakers: one row per (agent, name) — an enrolled speaker
-- identity whose `embedding` holds the element-wise SUM of the
-- L2-normalized per-segment voiceprint vectors (cosine is scale-invariant,
-- so scoring the sum equals scoring the mean, and enrolling more audio is
-- a pure vector addition; the source audio is never retained). The row id
-- is the stable speaker id the rest of the system speaks: the prompt's
-- roster, the messages' `speaker_id`. The 192 dimension is the pinned
-- CAM++ model's output width (src/voiceprint); candidates are enumerated
-- per agent, so there is no ANN index by design. Integrity is enforced in
-- the app layer; no foreign keys.
CREATE TABLE speakers (
    id          BIGSERIAL PRIMARY KEY,
    agent_id    BIGINT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    embedding   REAL[] NOT NULL CHECK (cardinality(embedding) = 192),
    sample_count INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (agent_id, name)
);

CREATE INDEX speakers_agent_id_idx ON speakers (agent_id);

-- Entry-based user memory: short durable facts the LLM maintains through the
-- memory tools, injected into the system prompt each turn. `mem_no` is
-- monotonic per (user_id, agent_id) and never reused — deletions are soft
-- (`deleted_at`) so a stale prompt reference can never point at a different
-- entry. Integrity is enforced in the app layer; no foreign keys.
CREATE TABLE memory_entries (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL,
    agent_id   BIGINT NOT NULL,
    mem_no     INT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    UNIQUE (user_id, agent_id, mem_no)
);

-- Runtime settings: one row per section
-- (`{asr,llm,tts}_provider_id`/`ota`/`security`), `value` is the typed JSON
-- of the matching struct in `src/settings.rs`. This table is the single
-- source for these sections — no env fallback (boot-level items like
-- DATABASE_URL/BIND_ADDR stay in env); the `security` row is written back
-- on every boot (generated on first run), so the values the web wizard will
-- manage live here. The `{type}_provider_id` rows do not carry provider
-- configs; they select which `providers` row each type uses (the row id as
-- a JSONB number) — activating a provider rewrites its type's row.
CREATE TABLE settings (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Provider catalog: one row per configured provider instance, identified
-- by UNIQUE (type, name) — e.g. type=tts, name=volcengine. `type` is what
-- the pipeline stage consumes (asr/llm/tts) and `kind` selects the
-- implementation through the provider registry; `config` is the raw JSON
-- the matching builder parses (api keys included). Both are validated in
-- the app layer on write (the type must be one of asr/llm/tts and the
-- kind must be registered for it). Which provider each type uses is NOT
-- recorded here — the `{type}_provider_id` settings rows hold that
-- selection. Integrity is enforced in the app layer; no foreign keys.
CREATE TABLE providers (
    id         BIGSERIAL PRIMARY KEY,
    type       TEXT NOT NULL,
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    config     JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (type, name)
);
