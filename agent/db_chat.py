"""
agent/db_chat.py
Persist and retrieve chat history from PostgreSQL.
"""

import os
from sqlalchemy import create_engine, text


def _get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise EnvironmentError("DATABASE_URL not set.")
    return create_engine(url)


def save_message(session_id: str, role: str, content: str, lot_id: str = None) -> None:
    try:
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO chat_history (session_id, lot_id, role, content)
                    VALUES (:session_id, :lot_id, :role, :content)
                """),
                {"session_id": session_id, "lot_id": lot_id, "role": role, "content": content},
            )
    except Exception:
        pass  # never crash the app over logging


def load_sessions(limit: int = 20) -> list[dict]:
    """Return the most recent sessions with their first user message."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT ON (session_id)
                    session_id, lot_id, created_at,
                    content AS first_message
                FROM chat_history
                WHERE role = 'user'
                ORDER BY session_id, created_at ASC
            """)).fetchall()
        sessions = [dict(r._mapping) for r in rows]
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions[:limit]
    except Exception:
        return []


def load_session_messages(session_id: str) -> list[dict]:
    """Return all messages for a given session ordered by time."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT role, content, created_at
                    FROM chat_history
                    WHERE session_id = :sid
                    ORDER BY created_at ASC
                """),
                {"sid": session_id},
            ).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception:
        return []


def restore_memory(session_id: str, memory) -> None:
    """Repopulate a ConversationBufferMemory from a saved session."""
    messages = load_session_messages(session_id)
    for msg in messages:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])


def restore_dataframe(session_id: str):
    """Reconstruct current_df from session_wafer_data for the given session."""
    import pandas as pd
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(
                text("""
                    SELECT lot_id, wafer_id, die_x, die_y, pass_fail, defect_code
                    FROM session_wafer_data
                    WHERE session_id = :sid
                """),
                conn,
                params={"sid": session_id},
            )
        return df if not df.empty else None
    except Exception:
        return None


def ensure_table() -> None:
    """Create all required tables if they don't exist."""
    try:
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id          SERIAL PRIMARY KEY,
                    session_id  VARCHAR(100) NOT NULL,
                    lot_id      VARCHAR(100) DEFAULT NULL,
                    role        VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                    content     TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.execute(text("""
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
                )
            """))
    except Exception:
        pass
