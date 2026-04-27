"""
Tool 5: HistoricalQueryTool
Query PostgreSQL for historical lot data to enable trend comparison.
"""

import os
import pandas as pd
import streamlit as st
from langchain_core.tools import tool
from sqlalchemy import create_engine, text


def _get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise EnvironmentError("DATABASE_URL not set in environment.")
    return create_engine(url)


@tool
def historical_query_tool(input: str = "yield_rate") -> str:
    """
    Query PostgreSQL historical database to retrieve past lot performance for trend analysis.
    Input: metric name as a plain string — one of: 'yield_rate' | 'defect_breakdown' | 'lot_summary' (default: yield_rate).
    Returns: historical trend data with computed delta vs current batch where applicable.
    """
    try:
        import re as _re
        metric = "yield_rate"
        limit = 5
        for m in ["yield_rate", "defect_breakdown", "lot_summary"]:
            if m in str(input):
                metric = m
                break
        num = _re.search(r'\d+', str(input))
        if num:
            limit = int(num.group())
        engine = _get_engine()

        if metric == "yield_rate":
            query = text("""
                SELECT lot_id, yield_rate, dominant_defect, total_dies, failed_dies, created_at
                FROM lot_summary
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            with engine.connect() as conn:
                hist_df = pd.read_sql(query, conn, params={"limit": limit})

            if hist_df.empty:
                return "No historical data in database. Run db_seeder.py first."

            avg_yield = round(hist_df["yield_rate"].mean(), 2)
            current_df = st.session_state.get("current_df")

            lines = [
                f"Historical Yield Trend (last {len(hist_df)} lots)",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ]

            for _, row in hist_df.iterrows():
                ts = str(row["created_at"])[:10] if row["created_at"] else "N/A"
                lines.append(
                    f"  {row['lot_id']:<20} | "
                    f"Yield: {row['yield_rate']:>6.2f}% | "
                    f"Dominant: {row['dominant_defect']:<12} | "
                    f"Date: {ts}"
                )

            if len(hist_df) > 1:
                first_yield = hist_df["yield_rate"].iloc[-1]
                last_yield = hist_df["yield_rate"].iloc[0]
                delta = round(last_yield - first_yield, 2)
                trend = f"↑ +{delta}% improving" if delta > 0 else f"↓ {delta}% declining"
                lines += [
                    "",
                    f"Historical avg yield : {avg_yield:.2f}%",
                    f"Trend                : {trend}",
                ]

            # Delta vs current batch
            if current_df is not None:
                curr_yield = round(current_df["pass_fail"].mean() * 100, 2)
                delta_curr = round(curr_yield - avg_yield, 2)
                delta_str = f"+{delta_curr}%" if delta_curr >= 0 else f"{delta_curr}%"
                lines += [
                    "",
                    f"Current batch yield  : {curr_yield:.2f}%",
                    f"vs historical avg    : {delta_str} {'✅' if delta_curr >= 0 else '⚠'}",
                ]

            return "\n".join(lines)

        elif metric == "defect_breakdown":
            query = text("""
                SELECT defect_code,
                       COUNT(*) AS count,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
                FROM wafer_history
                WHERE pass_fail = 0
                GROUP BY defect_code
                ORDER BY count DESC
            """)
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)

            if df.empty:
                return "No historical defect data found."

            lines = ["Historical Defect Breakdown (all lots in DB)", "━━━━━━━━━━━━━━━━━━━━"]
            for _, row in df.iterrows():
                lines.append(f"  {row['defect_code']:<12}: {int(row['count']):>7,} dies ({row['pct']}%)")
            return "\n".join(lines)

        elif metric == "lot_summary":
            query = text("""
                SELECT lot_id, total_dies, passed_dies, failed_dies, yield_rate, dominant_defect, created_at
                FROM lot_summary
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"limit": limit})

            if df.empty:
                return "No lot summary data found."

            lines = [f"Lot Summary (last {len(df)} lots)", "━━━━━━━━━━━━━━━━━━━━"]
            for _, row in df.iterrows():
                lines.append(
                    f"  {row['lot_id']}: {row['yield_rate']:.2f}% yield | "
                    f"{row['total_dies']:,} dies | {row['dominant_defect']}"
                )
            return "\n".join(lines)

        else:
            return (
                f"Unknown metric '{metric}'. "
                "Use one of: 'yield_rate', 'defect_breakdown', 'lot_summary'."
            )

    except Exception as e:
        return f"ERROR in historical query: {type(e).__name__}: {str(e)}"