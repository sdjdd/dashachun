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

-- The durable conversation log: one row per ChatItem. `System` items are
-- never stored (the agent prepends its prompt every turn). Integrity is
-- enforced in the app layer; no foreign keys.
CREATE TABLE messages (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT NOT NULL,
    user_id      BIGINT NOT NULL,
    agent_id     BIGINT NOT NULL,
    client_id    UUID NOT NULL,
    device_id    TEXT,
    role         TEXT NOT NULL,          -- 'user' | 'assistant' | 'tool'
    content      TEXT,                   -- user/assistant/tool text; NULL for tool-call-only assistant items
    tool_call_id TEXT,                   -- role='tool': id of the originating call
    tool_calls   JSONB,                  -- assistant tool-call items: [{id,name,arguments}]
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
