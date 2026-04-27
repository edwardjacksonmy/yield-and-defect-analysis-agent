"""
Tool 2: YieldCalculatorTool
Compute yield rate overall and per-wafer.
"""

import os
import pandas as pd
import streamlit as st
from langchain_core.tools import tool
from sqlalchemy import create_engine, text


def _fetch_from_db(lot_id: str) -> pd.DataFrame:
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM wafer_history WHERE lot_id = :lot_id"),
            conn,
            params={"lot_id": lot_id},
        )
    return df


@tool
def yield_calculator_tool(lot_id: str = "current") -> str:
    """
    Calculate overall yield rate and per-wafer breakdown.
    Input: lot_id — use 'current' for the uploaded batch, or a specific lot_id to query from PostgreSQL history.
    Returns: yield percentage, pass/fail counts, best and worst performing wafers.
    """
    try:
        if lot_id.strip().lower() == "current":
            df = st.session_state.get("current_df")
            if df is None:
                return "ERROR: No batch loaded. Run data_ingestion_tool first."
            label = "current batch"
        else:
            df = _fetch_from_db(lot_id)
            if df.empty:
                return f"ERROR: No data found in database for lot_id '{lot_id}'."
            label = lot_id

        total = len(df)
        passed = int(df["pass_fail"].sum())
        failed = total - passed
        yield_rate = round(passed / total * 100, 4) if total > 0 else 0.0

        # Per-wafer breakdown
        per_wafer = (
            df.groupby("wafer_id")
            .apply(lambda g: round(g["pass_fail"].sum() / len(g) * 100, 2))
            .reset_index(name="yield_pct")
        )
        per_wafer = per_wafer.sort_values("yield_pct")
        worst = per_wafer.iloc[0]
        best = per_wafer.iloc[-1]

        wafer_table = "\n".join(
            f"  Wafer {int(r.wafer_id):>3}: {r.yield_pct:>6.2f}%"
            for _, r in per_wafer.iterrows()
        )

        flag = " ⚠ BELOW TARGET" if yield_rate < 80 else " ✅ ON TARGET"

        return f"""Yield Analysis — {label}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall yield : {yield_rate:.2f}%{flag}
Total dies    : {total:,}
Passed        : {passed:,}
Failed        : {failed:,}
━━━━━━━━━━━━━━
Per-wafer breakdown:
{wafer_table}
━━━━━━━━━━━━━━
Best wafer  : Wafer {int(best.wafer_id)} ({best.yield_pct:.2f}%)
Worst wafer : Wafer {int(worst.wafer_id)} ({worst.yield_pct:.2f}%)"""

    except Exception as e:
        return f"ERROR in yield calculation: {type(e).__name__}: {str(e)}"