"""
Tool 1: DataIngestionTool
Parse and validate uploaded wafer batch CSV.
"""

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from langchain_core.tools import tool

REQUIRED_COLUMNS = ["lot_id", "wafer_id", "die_x", "die_y", "pass_fail", "defect_code"]

WM811K_DEFECT_LABELS = {
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none"
}


def _save_to_db(df: pd.DataFrame, yield_rate: float, passed: int, failed: int, total: int) -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        return
    try:
        session_id = st.session_state.get("session_id")
        if not session_id:
            return
        lot_id = str(df["lot_id"].iloc[0])
        engine = create_engine(url)
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM session_wafer_data WHERE session_id = :sid AND lot_id = :lot_id"),
                {"sid": session_id, "lot_id": lot_id},
            )
            records = [
                {**row, "session_id": session_id}
                for row in df[["lot_id", "wafer_id", "die_x", "die_y", "pass_fail", "defect_code"]].to_dict("records")
            ]
            conn.execute(
                text("""
                    INSERT INTO session_wafer_data
                        (session_id, lot_id, wafer_id, die_x, die_y, pass_fail, defect_code)
                    VALUES
                        (:session_id, :lot_id, :wafer_id, :die_x, :die_y, :pass_fail, :defect_code)
                """),
                records,
            )
    except Exception:
        pass  # never crash the agent over a DB write


@tool
def data_ingestion_tool(file_path: str) -> str:
    """
    Parse and validate an uploaded wafer batch CSV file into the analysis session.
    Must be called first before any other analysis tool.
    Input: absolute file path to the uploaded CSV
    Returns: summary statistics confirming data is loaded
    """
    try:
        df = pd.read_csv(file_path)

        # Column validation
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            return f"ERROR: Missing required columns: {missing_cols}. Required: {REQUIRED_COLUMNS}"

        # Type coercion
        df["die_x"] = pd.to_numeric(df["die_x"], errors="coerce")
        df["die_y"] = pd.to_numeric(df["die_y"], errors="coerce")
        df["pass_fail"] = pd.to_numeric(df["pass_fail"], errors="coerce")
        df = df.dropna(subset=["die_x", "die_y", "pass_fail"])
        df["pass_fail"] = df["pass_fail"].astype(int)

        # Normalise defect codes
        df["defect_code"] = df["defect_code"].fillna("none").astype(str)

        # Stats
        total = len(df)
        passed = int(df["pass_fail"].sum())
        failed = total - passed
        yield_rate = round(passed / total * 100, 2) if total > 0 else 0.0
        wafers = df["wafer_id"].nunique()
        lots = df["lot_id"].nunique()
        defect_types = sorted(df["defect_code"].unique().tolist())
        unknown = [d for d in defect_types if d not in WM811K_DEFECT_LABELS]

        # Store in session state for subsequent tools
        st.session_state["current_df"] = df
        st.session_state["current_file_path"] = file_path

        # Persist to DB so the data can be restored in future sessions
        _save_to_db(df, yield_rate, passed, failed, total)

        summary = f"""Data ingestion successful ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total dies    : {total:,}
Passed dies   : {passed:,}
Failed dies   : {failed:,}
Yield rate    : {yield_rate:.2f}%
Wafer count   : {wafers}
Lot count     : {lots}
Defect types  : {defect_types}"""

        if unknown:
            summary += f"\n⚠ Non-standard defect labels detected: {unknown}"

        return summary

    except FileNotFoundError:
        return f"ERROR: File not found at path: {file_path}"
    except Exception as e:
        return f"ERROR in data ingestion: {type(e).__name__}: {str(e)}"