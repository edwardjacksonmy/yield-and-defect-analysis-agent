"""
Tool 7: ReportGeneratorTool
Compile all analysis results into a structured markdown report.
"""

import os
import pandas as pd
import streamlit as st
from datetime import datetime
from langchain_core.tools import tool
from sqlalchemy import create_engine, text


@tool
def report_generator_tool(sections: str = "all") -> str:
    """
    Generate a comprehensive structured markdown report summarising yield, defects, clusters, and recommendations.
    Input: sections — comma-separated list of sections to include, or 'all'.
           Options: 'summary', 'defects', 'clusters', 'history', 'recommendations'
    Returns: full markdown report string; also stored in session state for download.
    """
    try:
        df = st.session_state.get("current_df")
        if df is None:
            return "ERROR: No batch loaded. Run data_ingestion_tool first."

        include = [s.strip() for s in sections.split(",")] if sections != "all" else [
            "summary", "defects", "clusters", "history", "recommendations"
        ]

        lot_id = df["lot_id"].iloc[0] if "lot_id" in df.columns else "Unknown"
        total = len(df)
        passed = int(df["pass_fail"].sum())
        failed = total - passed
        yield_rate = round(passed / total * 100, 2) if total > 0 else 0.0
        failed_df = df[df["pass_fail"] == 0]
        defect_counts = failed_df["defect_code"].value_counts() if len(failed_df) > 0 else pd.Series()
        dominant = defect_counts.index[0] if len(defect_counts) > 0 else "none"
        wafer_count = df["wafer_id"].nunique()

        report_lines = [
            "# Wafer Yield & Defect Analysis Report",
            f"**Generated :** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Lot ID     :** {lot_id}",
            f"**Analyst    :** Yield & Defect Analysis Agent (Agentic AI)",
            "",
            "---",
            "",
        ]

        # --- SUMMARY ---
        if "summary" in include:
            status = "🔴 BELOW TARGET" if yield_rate < 80 else "🟢 ON TARGET"
            report_lines += [
                "## 1. Executive Summary",
                "",
                f"| Metric | Value | Status |",
                f"|--------|-------|--------|",
                f"| Lot ID | {lot_id} | — |",
                f"| Total Dies | {total:,} | — |",
                f"| Passed Dies | {passed:,} | — |",
                f"| Failed Dies | {failed:,} | — |",
                f"| **Yield Rate** | **{yield_rate:.2f}%** | {status} |",
                f"| Wafer Count | {wafer_count} | — |",
                f"| Dominant Defect | {dominant} | — |",
                "",
            ]

        # --- DEFECTS ---
        if "defects" in include and len(defect_counts) > 0:
            report_lines += [
                "## 2. Defect Pattern Breakdown",
                "",
                "| Rank | Defect Type | Count | % of Failures | % of Total Dies |",
                "|------|-------------|-------|----------------|-----------------|",
            ]
            for rank, (defect, count) in enumerate(defect_counts.items(), 1):
                pct_fail = round(count / failed * 100, 1) if failed > 0 else 0
                pct_total = round(count / total * 100, 1)
                report_lines.append(
                    f"| #{rank} | {defect} | {count:,} | {pct_fail}% | {pct_total}% |"
                )
            report_lines.append("")

        # --- CLUSTERS ---
        if "clusters" in include:
            clustered_df = st.session_state.get("clustered_df")
            if clustered_df is not None and "cluster" in clustered_df.columns:
                n_clusters = len(set(clustered_df["cluster"]) - {-1})
                noise = int((clustered_df["cluster"] == -1).sum())
                report_lines += [
                    "## 3. Spatial Cluster Analysis",
                    "",
                    f"- Clusters detected: **{n_clusters}**",
                    f"- Isolated noise dies: {noise}",
                    "",
                ]
                for cid in sorted(set(clustered_df["cluster"]) - {-1}):
                    cg = clustered_df[clustered_df["cluster"] == cid]
                    report_lines.append(
                        f"- **Cluster {cid+1}**: {len(cg)} dies, "
                        f"center=({cg['die_x'].mean():.1f}, {cg['die_y'].mean():.1f}), "
                        f"pattern={cg['defect_code'].mode().iloc[0]}"
                    )
                report_lines.append("")
            else:
                report_lines += ["## 3. Spatial Cluster Analysis", "", "_Run spatial_clustering_tool to populate this section._", ""]

        # --- HISTORY ---
        if "history" in include:
            try:
                engine = create_engine(os.getenv("DATABASE_URL"))
                with engine.connect() as conn:
                    hist = pd.read_sql(
                        text("SELECT lot_id, yield_rate, dominant_defect FROM lot_summary ORDER BY created_at DESC LIMIT 5"),
                        conn,
                    )
                if not hist.empty:
                    avg_hist = round(hist["yield_rate"].mean(), 2)
                    delta = round(yield_rate - avg_hist, 2)
                    delta_str = f"+{delta}%" if delta >= 0 else f"{delta}%"
                    report_lines += [
                        "## 4. Historical Comparison",
                        "",
                        f"| Lot ID | Yield Rate | Dominant Defect |",
                        f"|--------|------------|-----------------|",
                    ]
                    for _, row in hist.iterrows():
                        report_lines.append(f"| {row['lot_id']} | {row['yield_rate']:.2f}% | {row['dominant_defect']} |")
                    report_lines += [
                        "",
                        f"**Historical average yield**: {avg_hist:.2f}%",
                        f"**Current batch delta**: {delta_str} {'✅' if delta >= 0 else '⚠'}",
                        "",
                    ]
            except Exception:
                report_lines += ["## 4. Historical Comparison", "", "_Database unavailable._", ""]

        # --- RECOMMENDATIONS ---
        if "recommendations" in include:
            rec_map = {
                "Center":    "Check CMP tool uniformity map and spin-coat recipe RPM profile.",
                "Donut":     "Adjust spin coat exhaust flow and review developer spray pattern.",
                "Edge-Loc":  "Inspect EBR nozzle; review chuck contact pressure settings.",
                "Edge-Ring": "Recalibrate edge focus correction; review RTP lamp uniformity.",
                "Loc":       "Run particle map overlay; inspect reticle under SEM; clean chuck.",
                "Near-full": "Quarantine lot; trace process logs for tool alarms; check chemical bath dates.",
                "Random":    "Check cleanroom particle monitor; review ioniser; inspect DI water quality.",
                "Scratch":   "Inspect robot end effector; check cassette slots; review CMP conditioner logs.",
                "none":      "No specific action required. Continue routine monitoring.",
            }
            report_lines += [
                "## 5. Recommendations",
                "",
                f"1. **Immediate**: Investigate **{dominant}** defects — highest occurrence rate ({defect_counts.iloc[0] if len(defect_counts) > 0 else 0:,} dies).",
                f"2. **Process Action**: {rec_map.get(dominant, 'Perform root cause analysis.')}",
                f"3. **Monitoring**: Increase inspection frequency if yield remains below 80%.",
                f"4. **Escalation**: If no improvement within 2 lots, escalate to process engineering.",
                "",
            ]

        report_lines += ["---", "*Report generated by Yield & Defect Analysis Agent — Agentic AI System*"]

        report = "\n".join(report_lines)
        st.session_state["last_report"] = report
        return report

    except Exception as e:
        return f"ERROR in report generation: {type(e).__name__}: {str(e)}"