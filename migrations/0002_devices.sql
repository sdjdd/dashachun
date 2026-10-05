CREATE TABLE devices (
    id                          BIGSERIAL PRIMARY KEY,
    client_id                   UUID NOT NULL UNIQUE,
    device_id                   TEXT,
    board_type                  TEXT,
    user_id                     BIGINT,
    token_hash                  TEXT,
    activation_code             TEXT UNIQUE,
    activation_code_expires_at  TIMESTAMPTZ,
    activated_at                TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX devices_user_id_idx ON devices (user_id);
