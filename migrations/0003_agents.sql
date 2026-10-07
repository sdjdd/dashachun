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

ALTER TABLE devices ADD COLUMN agent_id BIGINT;

CREATE INDEX devices_agent_id_idx ON devices (agent_id);
