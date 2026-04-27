"""
Tool 6: WaferMapVisualizerTool
Generate Plotly wafer map and defect distribution charts.
Renders directly into Streamlit session state for display.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import tool

DEFECT_COLOR_MAP = {
    "none":      "#22c55e",   # green  — pass
    "Center":    "#ef4444",   # red
    "Donut":     "#f97316",   # orange
    "Edge-Loc":  "#eab308",   # yellow
    "Edge-Ring": "#a855f7",   # purple
    "Loc":       "#06b6d4",   # cyan
    "Near-full": "#ec4899",   # pink
    "Random":    "#94a3b8",   # slate
    "Scratch":   "#dc2626",   # dark red
}

BG = "#0d1117"
FONT_COLOR = "#e2e8f0"
PAPER_BG = "#060a12"


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=FONT_COLOR, size=14)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=BG,
        font=dict(color=FONT_COLOR, family="Courier New"),
        margin=dict(l=40, r=20, t=50, b=40),
    )


@tool
def wafer_visualizer_tool(input: str = "wafer_map") -> str:
    """
    Generate wafer visualisation charts from the current batch data.
    Input: a string with chart_type and optional wafer_id, e.g. "wafer_map", "wafer_map 12", "defect_bar", "yield_heatmap", "cluster_map".
    chart_type — one of: 'wafer_map' | 'defect_bar' | 'yield_heatmap' | 'cluster_map'
    Returns: confirmation string; the figure is stored in session state for Streamlit rendering.
    """
    try:
        import re as _re
        parts = str(input).strip().split()
        chart_type = parts[0] if parts else "wafer_map"
        wafer_id = -1
        if len(parts) > 1:
            try:
                wafer_id = int(parts[1])
            except Exception:
                pass
        # also handle "chart_type=wafer_map" style
        kv = _re.search(r'chart_type[=:\s]+(\w+)', str(input))
        if kv:
            chart_type = kv.group(1)
        kv2 = _re.search(r'wafer_id[=:\s]+(\d+)', str(input))
        if kv2:
            wafer_id = int(kv2.group(1))

        df = st.session_state.get("current_df")
        if df is None:
            return "ERROR: No batch loaded. Run data_ingestion_tool first."

        if chart_type == "wafer_map":
            target_id = wafer_id if wafer_id >= 0 else int(df["wafer_id"].iloc[0])
            wafer_df = df[df["wafer_id"] == target_id].copy()
            if wafer_df.empty:
                return f"ERROR: Wafer {target_id} not found in current batch."

            lot = wafer_df["lot_id"].iloc[0]
            total = len(wafer_df)
            fails = int((wafer_df["pass_fail"] == 0).sum())
            yield_pct = round((total - fails) / total * 100, 2)

            fig = px.scatter(
                wafer_df,
                x="die_x", y="die_y",
                color="defect_code",
                color_discrete_map=DEFECT_COLOR_MAP,
                title=f"Wafer Map — Wafer {target_id} | Lot: {lot} | Yield: {yield_pct:.2f}%",
                labels={"die_x": "Die X", "die_y": "Die Y", "defect_code": "Defect Type"},
                height=480,
            )
            fig.update_traces(marker=dict(size=7, symbol="square", opacity=0.9))
            fig.update_layout(**_base_layout(
                f"Wafer Map — Wafer {target_id} | Lot {lot} | Yield {yield_pct:.2f}%"
            ))
            st.session_state["last_figure"] = fig
            st.session_state["last_chart_type"] = "wafer_map"
            return f"Wafer map rendered for Wafer {target_id} (yield: {yield_pct:.2f}%)."

        elif chart_type == "defect_bar":
            failed = df[df["pass_fail"] == 0]
            counts = failed["defect_code"].value_counts().reset_index()
            counts.columns = ["defect_code", "count"]
            counts["pct"] = (counts["count"] / len(failed) * 100).round(1)

            fig = px.bar(
                counts, x="defect_code", y="count",
                color="defect_code",
                color_discrete_map=DEFECT_COLOR_MAP,
                text=counts["pct"].astype(str) + "%",
                title="Defect Type Distribution",
                labels={"defect_code": "Defect Type", "count": "Failed Die Count"},
                height=420,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**_base_layout("Defect Type Distribution"))
            st.session_state["last_figure"] = fig
            st.session_state["last_chart_type"] = "defect_bar"
            return "Defect distribution bar chart rendered."

        elif chart_type == "yield_heatmap":
            pivot = (
                df.groupby(["die_x", "die_y"])["pass_fail"]
                .mean()
                .reset_index()
                .pivot(index="die_y", columns="die_x", values="pass_fail")
            )
            fig = go.Figure(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale=[
                        [0.0, "#ef4444"],
                        [0.5, "#eab308"],
                        [1.0, "#22c55e"],
                    ],
                    zmin=0, zmax=1,
                    colorbar=dict(title="Pass Rate"),
                )
            )
            fig.update_layout(**_base_layout("Yield Heatmap (Aggregated All Wafers)"))
            st.session_state["last_figure"] = fig
            st.session_state["last_chart_type"] = "yield_heatmap"
            return "Yield heatmap rendered (aggregated across all wafers in batch)."

        elif chart_type == "cluster_map":
            clustered_df = st.session_state.get("clustered_df")
            if clustered_df is None:
                return "ERROR: Run spatial_clustering_tool first to generate cluster data."

            all_pass = df[df["pass_fail"] == 1].copy()
            all_pass["cluster_label"] = "Pass"
            clustered_df = clustered_df.copy()
            clustered_df["cluster_label"] = clustered_df["cluster"].apply(
                lambda c: f"Cluster {int(c)+1}" if c >= 0 else "Noise"
            )

            combined = pd.concat([
                all_pass[["die_x", "die_y", "cluster_label"]],
                clustered_df[["die_x", "die_y", "cluster_label"]],
            ])

            fig = px.scatter(
                combined, x="die_x", y="die_y",
                color="cluster_label",
                title="Spatial Cluster Map",
                labels={"die_x": "Die X", "die_y": "Die Y", "cluster_label": "Group"},
                height=480,
            )
            fig.update_traces(marker=dict(size=7, symbol="square", opacity=0.85))
            fig.update_layout(**_base_layout("DBSCAN Spatial Cluster Map"))
            st.session_state["last_figure"] = fig
            st.session_state["last_chart_type"] = "cluster_map"
            return "DBSCAN cluster map rendered."

        else:
            return (
                f"Unknown chart_type '{chart_type}'. "
                "Use: 'wafer_map', 'defect_bar', 'yield_heatmap', 'cluster_map'."
            )

    except Exception as e:
        return f"ERROR in visualizer: {type(e).__name__}: {str(e)}"