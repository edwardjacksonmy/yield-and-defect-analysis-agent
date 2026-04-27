"""
Tool 4: SpatialClusteringTool
DBSCAN clustering on failed die coordinates to identify spatial defect clusters.
"""

import pandas as pd
import numpy as np
import streamlit as st
from langchain_core.tools import tool
from sklearn.cluster import DBSCAN


@tool
def spatial_clustering_tool(input: str = "") -> str:
    """
    Perform DBSCAN spatial clustering on failed dies to detect defect cluster patterns on the wafer map.
    Clustered failures indicate systematic process issues; scattered failures indicate random contamination.
    Input: optional string with eps and min_samples, e.g. "2.0, 5" or leave empty for defaults (eps=2.0, min_samples=5).
    Returns: cluster count, spatial locations, sizes, and dominant defect type per cluster.
    """
    try:
        import re
        eps, min_samples = 2.0, 5
        nums = re.findall(r'[\d.]+', str(input))
        if len(nums) >= 2:
            try:
                eps = float(nums[0]); min_samples = int(float(nums[1]))
            except Exception:
                pass
        elif len(nums) == 1:
            try:
                eps = float(nums[0])
            except Exception:
                pass
        df = st.session_state.get("current_df")
        if df is None:
            return "ERROR: No batch loaded. Run data_ingestion_tool first."

        failed_df = df[df["pass_fail"] == 0].copy()
        total_fails = len(failed_df)

        if total_fails == 0:
            return "No failed dies to cluster — yield is 100%."

        if total_fails < min_samples:
            return (
                f"Insufficient failed dies ({total_fails}) for DBSCAN "
                f"with min_samples={min_samples}. Reduce min_samples or check data."
            )

        coords = failed_df[["die_x", "die_y"]].values
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(coords)
        failed_df = failed_df.copy()
        failed_df["cluster"] = db.labels_

        cluster_ids = set(db.labels_) - {-1}
        n_clusters = len(cluster_ids)
        n_noise = int((db.labels_ == -1).sum())
        noise_pct = n_noise / total_fails * 100

        lines = [
            "Spatial Clustering Results (DBSCAN)",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Parameters    : eps={eps}, min_samples={min_samples}",
            f"Failed dies   : {total_fails:,}",
            f"Clusters found: {n_clusters}",
            f"Noise (isolated dies): {n_noise} ({noise_pct:.1f}%)",
            "",
        ]

        if n_clusters == 0:
            lines.append(
                "→ No spatial clusters detected. Defect distribution is random/scattered.\n"
                "  Likely cause: environmental contamination (particles, static)."
            )
        else:
            lines.append("Cluster Details:")
            for cid in sorted(cluster_ids):
                cluster_dies = failed_df[failed_df["cluster"] == cid]
                cx = round(cluster_dies["die_x"].mean(), 1)
                cy = round(cluster_dies["die_y"].mean(), 1)
                size = len(cluster_dies)
                span_x = int(cluster_dies["die_x"].max() - cluster_dies["die_x"].min())
                span_y = int(cluster_dies["die_y"].max() - cluster_dies["die_y"].min())
                dominant_defect = (
                    cluster_dies["defect_code"].mode().iloc[0]
                    if len(cluster_dies) > 0 else "unknown"
                )
                lines.append(
                    f"  Cluster {cid + 1}: {size} dies | "
                    f"center=({cx}, {cy}) | "
                    f"span={span_x}x{span_y} dies | "
                    f"pattern={dominant_defect}"
                )

            if n_clusters >= 3:
                lines.append("\n⚠ Multiple clusters detected → Systematic process issue likely.")
            elif n_clusters == 1:
                lines.append("\n→ Single cluster → Localised contamination or tool-specific issue.")

        # Save cluster labels for visualizer
        st.session_state["cluster_labels"] = (
            failed_df.set_index(failed_df.index)["cluster"].to_dict()
        )
        st.session_state["clustered_df"] = failed_df

        return "\n".join(lines)

    except Exception as e:
        return f"ERROR in spatial clustering: {type(e).__name__}: {str(e)}"