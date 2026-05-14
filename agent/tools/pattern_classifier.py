"""
Tool 9: PatternClassifierTool
Predict WM-811K defect pattern for each wafer from spatial fail-die distribution.

Requires: data/wafer_pattern_model.joblib
Train it with: python data/train_classifier.py data/LSWMD.pkl
"""

import os
from pathlib import Path

import numpy as np
import streamlit as st
from langchain_core.tools import tool
from sklearn.cluster import DBSCAN

_MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "wafer_pattern_model.joblib"
_model_cache: dict | None = None


def _load_model() -> dict | None:
    global _model_cache
    if _model_cache is None:
        if not _MODEL_PATH.exists():
            return None
        import joblib
        _model_cache = joblib.load(_MODEL_PATH)
    return _model_cache


def _extract_features(wafer_df) -> dict | None:
    """
    Derive the same 16 spatial features used during training from a
    single wafer's die-level DataFrame (columns: die_x, die_y, pass_fail).
    """
    total_dies  = len(wafer_df)
    total_fails = int((wafer_df["pass_fail"] == 0).sum())
    if total_dies == 0:
        return None

    fail_rate = total_fails / total_dies

    cx = (wafer_df["die_x"].max() + wafer_df["die_x"].min()) / 2.0
    cy = (wafer_df["die_y"].max() + wafer_df["die_y"].min()) / 2.0
    max_r = max(
        float(np.sqrt((wafer_df["die_x"] - cx) ** 2 + (wafer_df["die_y"] - cy) ** 2).max()),
        1e-9,
    )

    all_radii = np.sqrt((wafer_df["die_x"] - cx) ** 2 + (wafer_df["die_y"] - cy) ** 2) / max_r

    fail_df = wafer_df[wafer_df["pass_fail"] == 0]
    if total_fails > 0:
        fail_radii = np.sqrt(
            (fail_df["die_x"] - cx) ** 2 + (fail_df["die_y"] - cy) ** 2
        ) / max_r
        fail_angles = np.arctan2(fail_df["die_y"] - cy, fail_df["die_x"] - cx).values
    else:
        fail_radii  = np.array([])
        fail_angles = np.array([])

    # Zone fail rates — must match training feature definitions exactly
    def _zone(r_min: float, r_max: float) -> float:
        mask  = (all_radii >= r_min) & (all_radii < r_max)
        total = int(mask.sum())
        if total == 0:
            return 0.0
        fails = int((wafer_df.loc[mask.index[mask], "pass_fail"] == 0).sum())
        return fails / total

    center_fail_rate     = _zone(0.00, 0.30)
    mid_fail_rate        = _zone(0.30, 0.60)
    ring_fail_rate       = _zone(0.50, 0.80)
    edge_fail_rate       = _zone(0.70, 1.01)
    outer_edge_fail_rate = _zone(0.85, 1.01)

    mean_fail_radius = float(fail_radii.mean()) if len(fail_radii) > 0 else 0.0
    std_fail_radius  = float(fail_radii.std())  if len(fail_radii) > 0 else 0.0

    if len(fail_angles) > 0:
        sector_idx    = ((fail_angles + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        sector_counts = np.bincount(sector_idx, minlength=8).astype(float)
        sector_fracs  = sector_counts / (total_fails + 1e-9)
        max_angular_sector = float(sector_fracs.max())
        nz = sector_fracs[sector_fracs > 0]
        angular_entropy = float(-np.sum(nz * np.log(nz)))
    else:
        max_angular_sector = 0.0
        angular_entropy    = 0.0

    if total_fails >= 5:
        coords = fail_df[["die_x", "die_y"]].values.astype(float)
        cov    = np.cov((coords - coords.mean(axis=0)).T)
        if cov.ndim == 2:
            evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            pca_linearity = float(evals[0] / (evals[1] + 1e-9))
        else:
            pca_linearity = 1.0
    else:
        pca_linearity = 1.0

    n_clusters = 0
    clustered_fraction       = 0.0
    largest_cluster_fraction = 0.0
    if total_fails >= 5:
        span = max(
            float(wafer_df["die_x"].max() - wafer_df["die_x"].min()),
            float(wafer_df["die_y"].max() - wafer_df["die_y"].min()),
            1.0,
        )
        eps    = max(2.0, 0.07 * span)
        coords = fail_df[["die_x", "die_y"]].values
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(coords)
        cluster_ids = set(labels) - {-1}
        n_clusters  = len(cluster_ids)
        if n_clusters > 0:
            clustered_fraction       = int((labels >= 0).sum()) / total_fails
            largest_cluster_fraction = max(
                int((labels == c).sum()) for c in cluster_ids
            ) / total_fails

    return {
        "fail_rate":              fail_rate,
        "center_fail_rate":       center_fail_rate,
        "mid_fail_rate":          mid_fail_rate,
        "ring_fail_rate":         ring_fail_rate,
        "edge_fail_rate":         edge_fail_rate,
        "outer_edge_fail_rate":   outer_edge_fail_rate,
        "mean_fail_radius":       mean_fail_radius,
        "std_fail_radius":        std_fail_radius,
        "max_angular_sector":     max_angular_sector,
        "angular_entropy":        angular_entropy,
        "pca_linearity":          pca_linearity,
        "n_clusters":             n_clusters,
        "clustered_fraction":     clustered_fraction,
        "largest_cluster_fraction": largest_cluster_fraction,
        "center_to_total":        center_fail_rate / (fail_rate + 1e-9),
        "edge_to_total":          edge_fail_rate   / (fail_rate + 1e-9),
    }


@tool
def pattern_classifier_tool(input: str = "") -> str:
    """
    Predict WM-811K defect pattern for each wafer using a trained spatial ML classifier.
    Classifies: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none.
    Does NOT require a pre-labeled defect_code column — infers pattern purely from the
    spatial distribution of failing dies (die_x, die_y, pass_fail).
    Updates the defect_code column in session data so downstream tools use ML predictions.
    Input: "" (reads from the already-loaded session data — no argument needed).
    Returns: per-wafer predicted pattern with confidence, and lot-level dominant pattern.
    """
    bundle = _load_model()
    if bundle is None:
        return (
            "ERROR: Trained model not found.\n"
            f"Expected path: {_MODEL_PATH}\n"
            "Train it first with:\n"
            "  python data/train_classifier.py data/LSWMD.pkl"
        )

    clf           = bundle["model"]
    le            = bundle["label_encoder"]
    feature_names = bundle["feature_names"]

    df = st.session_state.get("current_df")
    if df is None:
        return "ERROR: No batch loaded. Run data_ingestion_tool first."

    df = df.copy()
    results: list[tuple] = []

    for wafer_id, wafer_df in df.groupby("wafer_id"):
        feats = _extract_features(wafer_df)
        if feats is None:
            results.append((wafer_id, "unknown", 0.0))
            continue

        X           = np.array([[feats[f] for f in feature_names]])
        pred_enc    = clf.predict(X)[0]
        pred_label  = le.inverse_transform([pred_enc])[0]
        confidence  = float(clf.predict_proba(X)[0].max())
        results.append((wafer_id, pred_label, confidence))

        df.loc[df["wafer_id"] == wafer_id, "defect_code"] = pred_label

    st.session_state["current_df"] = df

    # ── Report ────────────────────────────────────────────────────────────────
    col_w = max(len(str(r[0])) for r in results) if results else 8
    lines = [
        "Defect Pattern Classification (ML)",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"{'Wafer ID':<{col_w}}  {'Predicted Pattern':<18}  {'Confidence':>10}",
        "─" * (col_w + 33),
    ]
    for wid, label, conf in results:
        lines.append(f"{str(wid):<{col_w}}  {label:<18}  {conf * 100:>9.1f}%")

    pattern_counts: dict[str, int] = {}
    for _, label, _ in results:
        pattern_counts[label] = pattern_counts.get(label, 0) + 1

    dominant     = max(pattern_counts, key=pattern_counts.get)
    n_wafers     = len(results)
    low_conf     = [r for r in results if r[2] < 0.6]

    lines += [
        "─" * (col_w + 33),
        f"Lot dominant pattern : {dominant}  "
        f"({pattern_counts[dominant]}/{n_wafers} wafers)",
    ]
    if low_conf:
        low_ids = ", ".join(str(r[0]) for r in low_conf)
        lines.append(
            f"⚠  Low-confidence predictions (< 60%): wafer(s) {low_ids} — "
            "consider manual review."
        )
    lines += [
        "",
        "defect_code column updated in session data.",
        "Run defect_analyzer_tool or root_cause_tool to continue analysis.",
    ]

    return "\n".join(lines)
