-- Early development: the whole schema lives in this single migration and is
-- amended in place while the project has not shipped yet.
CREATE TABLE users (
    id            BIGSERIAL PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
    id           UUID PRIMARY KEY,
    user_id      BIGINT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX sessions_user_id_idx ON sessions (user_id);

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
    activation_code             TEXT UNIQUE,
    activation_code_expires_at  TIMESTAMPTZ,
    activated_at                TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX devices_user_id_idx ON devices (user_id);
CREATE INDEX devices_agent_id_idx ON devices (agent_id);

-- The durable conversation log: one row per turn. A user row carries the
-- utterance text; an assistant row carries the turn's final reply text plus
-- the whole tool exchange as an ordered JSONB `parts` array (call arguments
-- and results are kept verbatim as strings — no parsing, no type guessing;
-- the round boundaries are recoverable from the part order). `System` items
-- are never stored (the agent prepends its prompt every turn). Integrity is
-- enforced in the app layer; no foreign keys.
CREATE TABLE messages (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT NOT NULL,
    user_id      BIGINT NOT NULL,
    agent_id     BIGINT NOT NULL,
    client_id    UUID NOT NULL,
    device_id    TEXT,
    role         TEXT NOT NULL,          -- 'user' | 'assistant'
    content      TEXT,                   -- user: utterance text; assistant: final reply text (NULL when the turn ended without one)
    parts        JSONB,                  -- assistant: [{type:'tool_call',id,name,arguments},{type:'tool_result',tool_call_id,content}]
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX messages_user_id_idx ON messages (user_id, created_at);
CREATE INDEX messages_session_id_idx ON messages (session_id, created_at);

-- FLAC capture (16-bit PCM, metadata blocks + frames) attached to its user
-- message; playback and voice-cloning pipelines read it with any decoder.
CREATE TABLE message_audios (
    id          BIGSERIAL PRIMARY KEY,
    message_id  BIGINT NOT NULL,
    audio       BYTEA NOT NULL,
    sample_rate INT NOT NULL,
    channels    INT NOT NULL,
    duration_ms INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX message_audios_message_id_idx ON message_audios (message_id);

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
