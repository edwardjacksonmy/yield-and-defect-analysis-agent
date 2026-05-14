"""
Train a Random Forest classifier to predict WM-811K defect patterns
from spatial fail-die distributions — no defect label used as a feature.

The WM-811K labels (Center, Donut, Edge-Ring, …) serve as ground truth only.
Features are derived purely from the 2D spatial arrangement of failing dies.

Usage:
    python data/train_classifier.py data/LSWMD.pkl
    python data/train_classifier.py data/LSWMD.pkl --max-wafers 5000
    python data/train_classifier.py data/LSWMD.pkl --output data/wafer_pattern_model.joblib

Saves:
    data/wafer_pattern_model.joblib  (model + label encoder + feature names)
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ── Label extraction ─────────────────────────────────────────────────────────

def _extract_failure_type(raw) -> str:
    """Safely extract the WM-811K failureType string from its nested array."""
    try:
        if raw is None:
            return "none"
        if isinstance(raw, str):
            return raw if raw else "none"
        if hasattr(raw, "__len__"):
            if len(raw) == 0:
                return "none"
            inner = raw[0]
            ft = inner[0] if hasattr(inner, "__len__") and len(inner) > 0 else inner
            return str(ft) if ft else "none"
    except Exception:
        return "none"
    return "none"


# ── Spatial feature extraction ────────────────────────────────────────────────

def extract_wafer_features(wafer_map: np.ndarray) -> dict | None:
    """
    Derive 16 spatial features from a 2D wafer map.

    wafer_map encoding:  0 = no die (untested),  1 = pass,  2 = fail

    Returns None for empty / degenerate maps.
    """
    if wafer_map.ndim != 2:
        return None

    rows, cols = wafer_map.shape
    cx, cy = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_r = np.sqrt(cx ** 2 + cy ** 2) + 1e-9

    die_mask  = wafer_map > 0   # valid die positions
    fail_mask = wafer_map == 2

    total_dies  = int(die_mask.sum())
    total_fails = int(fail_mask.sum())
    if total_dies == 0:
        return None

    fail_rate = total_fails / total_dies

    # Row/col indices for all valid dies
    xi_all, yi_all = np.where(die_mask)
    all_radii = np.sqrt((xi_all - cx) ** 2 + (yi_all - cy) ** 2) / max_r

    # Row/col indices for failing dies
    xi_f, yi_f = np.where(fail_mask)
    fail_radii = np.sqrt((xi_f - cx) ** 2 + (yi_f - cy) ** 2) / max_r if total_fails > 0 else np.array([])
    fail_angles = np.arctan2(yi_f - cy, xi_f - cx) if total_fails > 0 else np.array([])

    # ── Zone-based fail rates (concentric rings) ──────────────────────────────
    def _zone_fail_rate(r_min: float, r_max: float) -> float:
        zone_idx = np.where((all_radii >= r_min) & (all_radii < r_max))[0]
        if len(zone_idx) == 0:
            return 0.0
        zone_fails = int(fail_mask[xi_all[zone_idx], yi_all[zone_idx]].sum())
        return zone_fails / len(zone_idx)

    center_fail_rate     = _zone_fail_rate(0.00, 0.30)
    mid_fail_rate        = _zone_fail_rate(0.30, 0.60)
    ring_fail_rate       = _zone_fail_rate(0.50, 0.80)  # donut zone
    edge_fail_rate       = _zone_fail_rate(0.70, 1.01)
    outer_edge_fail_rate = _zone_fail_rate(0.85, 1.01)

    mean_fail_radius = float(fail_radii.mean()) if len(fail_radii) > 0 else 0.0
    std_fail_radius  = float(fail_radii.std())  if len(fail_radii) > 0 else 0.0

    # ── Angular distribution (8 × 45° sectors) ────────────────────────────────
    if len(fail_angles) > 0:
        sector_idx    = ((fail_angles + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        sector_counts = np.bincount(sector_idx, minlength=8).astype(float)
        sector_fracs  = sector_counts / (total_fails + 1e-9)
        max_angular_sector = float(sector_fracs.max())
        # Shannon entropy — low → concentrated (localised/scratch), high → uniform
        nz = sector_fracs[sector_fracs > 0]
        angular_entropy = float(-np.sum(nz * np.log(nz)))
    else:
        max_angular_sector = 0.0
        angular_entropy    = 0.0

    # ── Linearity score via PCA (high ratio → scratch / linear pattern) ───────
    if total_fails >= 5:
        coords = np.column_stack([xi_f, yi_f]).astype(float)
        cov    = np.cov((coords - coords.mean(axis=0)).T)
        if cov.ndim == 2:
            evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            pca_linearity = float(evals[0] / (evals[1] + 1e-9))
        else:
            pca_linearity = 1.0
    else:
        pca_linearity = 1.0

    # ── DBSCAN cluster features ───────────────────────────────────────────────
    n_clusters = 0
    clustered_fraction       = 0.0
    largest_cluster_fraction = 0.0
    if total_fails >= 5:
        eps    = max(2.0, 0.07 * max(rows, cols))
        coords = np.column_stack([xi_f, yi_f])
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


# ── Training pipeline ─────────────────────────────────────────────────────────

def main(pkl_path: str, max_wafers: int | None, output_path: str) -> None:
    print(f"Loading WM-811K from {pkl_path} …")
    df_raw = pd.read_pickle(pkl_path)
    if max_wafers:
        df_raw = df_raw.head(max_wafers)
    print(f"Total wafers to process: {len(df_raw):,}")

    records: list[dict] = []
    skipped = 0

    for idx, row in df_raw.iterrows():
        wafer_map = row["waferMap"]
        label     = _extract_failure_type(row.get("failureType"))

        if not isinstance(wafer_map, np.ndarray):
            skipped += 1
            continue

        feats = extract_wafer_features(wafer_map)
        if feats is None:
            skipped += 1
            continue

        feats["label"] = label
        records.append(feats)

        if (idx + 1) % 2000 == 0:
            print(f"  … {idx + 1:,} / {len(df_raw):,} wafers processed")

    print(f"\nFeature extraction complete: {len(records):,} usable wafers, {skipped} skipped")

    df = pd.DataFrame(records)
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}\n")

    feature_names = [c for c in df.columns if c != "label"]
    X = df[feature_names].values
    y = df["label"].values

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"Training Random Forest …  ({len(X_train):,} train / {len(X_test):,} test)")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    importances = sorted(
        zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]
    )
    print("Top feature importances:")
    for name, imp in importances[:8]:
        print(f"  {name:<30}  {imp:.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": clf, "label_encoder": le, "feature_names": feature_names},
        output_path,
    )
    print(f"\nModel saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train WM-811K defect pattern classifier from spatial die data"
    )
    parser.add_argument("pkl_path", help="Path to LSWMD.pkl")
    parser.add_argument("--max-wafers", type=int, default=None,
                        help="Limit wafers processed (default: all)")
    parser.add_argument("--output", default="data/wafer_pattern_model.joblib",
                        help="Output path for saved model")
    args = parser.parse_args()
    main(args.pkl_path, args.max_wafers, args.output)
