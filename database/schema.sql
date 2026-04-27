-- Wafer die-level records
CREATE TABLE IF NOT EXISTS wafer_history (
    id          SERIAL PRIMARY KEY,
    lot_id      VARCHAR(100) NOT NULL,
    wafer_id    INTEGER NOT NULL,
    die_x       INTEGER NOT NULL,
    die_y       INTEGER NOT NULL,
    pass_fail   INTEGER NOT NULL CHECK (pass_fail IN (0, 1)),
    defect_code VARCHAR(100) DEFAULT 'none',
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Lot-level aggregated summary
CREATE TABLE IF NOT EXISTS lot_summary (
    id               SERIAL PRIMARY KEY,
    lot_id           VARCHAR(100) UNIQUE NOT NULL,
    total_dies       INTEGER NOT NULL,
    passed_dies      INTEGER NOT NULL,
    failed_dies      INTEGER NOT NULL,
    yield_rate       FLOAT NOT NULL,
    dominant_defect  VARCHAR(100) DEFAULT 'none',
    created_at       TIMESTAMP DEFAULT NOW()
);

-- Conversation chat history
CREATE TABLE IF NOT EXISTS chat_history (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(100) NOT NULL,
    lot_id      VARCHAR(100) DEFAULT NULL,
    role        VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Engineer-uploaded session data (separate from machine-generated wafer_history)
CREATE TABLE IF NOT EXISTS session_wafer_data (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(100) NOT NULL,
    lot_id      VARCHAR(100) NOT NULL,
    wafer_id    INTEGER NOT NULL,
    die_x       INTEGER NOT NULL,
    die_y       INTEGER NOT NULL,
    pass_fail   INTEGER NOT NULL CHECK (pass_fail IN (0, 1)),
    defect_code VARCHAR(100) DEFAULT 'none',
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_wh_lot_id     ON wafer_history(lot_id);
CREATE INDEX IF NOT EXISTS idx_wh_defect     ON wafer_history(defect_code);
CREATE INDEX IF NOT EXISTS idx_wh_pass_fail  ON wafer_history(pass_fail);
CREATE INDEX IF NOT EXISTS idx_wh_wafer      ON wafer_history(lot_id, wafer_id);
CREATE INDEX IF NOT EXISTS idx_ls_yield      ON lot_summary(yield_rate);
CREATE INDEX IF NOT EXISTS idx_ls_created    ON lot_summary(created_at);
CREATE INDEX IF NOT EXISTS idx_swd_session   ON session_wafer_data(session_id);
CREATE INDEX IF NOT EXISTS idx_swd_lot       ON session_wafer_data(session_id, lot_id);