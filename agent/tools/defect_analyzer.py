"""
Tool 3: DefectAnalyzerTool
Analyze defect pattern distribution using WM-811K labels.
"""

import pandas as pd
import streamlit as st
from langchain_core.tools import tool

DEFECT_INSIGHTS = {
    "Center":    "Center cluster → likely CMP non-uniformity or spin-coat irregularity at wafer center.",
    "Donut":     "Donut ring → spin coating donut effect; solvent evaporation creates mid-radius ring pattern.",
    "Edge-Loc":  "Localized edge → edge bead removal (EBR) malfunction or wafer chuck clamping damage.",
    "Edge-Ring": "Full edge ring → photolithography edge focus gradient or RTP edge cooling effect.",
    "Loc":       "Localized cluster → particle contamination, reticle defect, or chuck particle transfer.",
    "Near-full": "Near-full failure → major process excursion; chemical contamination or tool alarm likely.",
    "Random":    "Random distribution → airborne particle contamination or DI water quality issue.",
    "Scratch":   "Linear scratch → wafer handling robot malfunction or CMP pad conditioner damage.",
    "none":      "No specific pattern — random noise or passing dies with minor parametric variation.",
}


@tool
def defect_analyzer_tool(input: str = "3") -> str:
    """
    Analyze defect type distribution in the current wafer batch using WM-811K defect taxonomy.
    Identifies dominant failure modes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch.
    Input: number of top defect types to highlight as a plain integer string, e.g. "3".
    Returns: ranked defect breakdown with counts, percentages, and process insights.
    """
    try:
        try:
            import re
            match = re.search(r'\d+', str(input))
            top_n = int(match.group()) if match else 3
        except Exception:
            top_n = 3
        df = st.session_state.get("current_df")
        if df is None:
            return "ERROR: No batch loaded. Run data_ingestion_tool first."

        total_dies = len(df)
        failed_df = df[df["pass_fail"] == 0]
        total_fails = len(failed_df)

        if total_fails == 0:
            return "No failed dies found. Yield is 100% — no defect analysis required."

        defect_counts = failed_df["defect_code"].value_counts()
        pass_count = int(df["pass_fail"].sum())

        lines = [
            f"Defect Pattern Analysis",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Total dies    : {total_dies:,}",
            f"Failed dies   : {total_fails:,} ({total_fails/total_dies*100:.1f}% of all dies)",
            f"",
            f"Defect Breakdown (failed dies only):",
        ]

        for rank, (defect, count) in enumerate(defect_counts.items(), start=1):
            pct_of_fails = count / total_fails * 100
            pct_of_total = count / total_dies * 100
            marker = " ◀ DOMINANT" if rank == 1 else ""
            lines.append(
                f"  #{rank:>2}. {defect:<12} : {count:>5,} dies "
                f"({pct_of_fails:>5.1f}% of failures | {pct_of_total:.1f}% of wafer){marker}"
            )

        dominant = defect_counts.index[0]
        lines += [
            f"",
            f"━━━━━━━━━━━━━━",
            f"Dominant failure: {dominant}",
            f"Process insight : {DEFECT_INSIGHTS.get(dominant, 'Unknown pattern — manual review required.')}",
        ]

        if len(defect_counts) > 1:
            lines.append(f"")
            lines.append(f"Top {min(top_n, len(defect_counts))} defects account for "
                         f"{defect_counts.head(top_n).sum() / total_fails * 100:.1f}% of all failures.")

        return "\n".join(lines)

    except Exception as e:
        return f"ERROR in defect analysis: {type(e).__name__}: {str(e)}"