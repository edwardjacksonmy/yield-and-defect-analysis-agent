"""
preprocessing.py
Convert WM-811K raw pickle dataset into die-level tabular DataFrame.

WM-811K waferMap encoding:
    0 = no die (edge / untested area)
    1 = good die (pass)
    2 = failed die (fail)

WM-811K failureType labels:
    Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def _extract_failure_type(raw) -> str:
    """Safely extract failure type string from WM-811K nested array."""
    try:
        if raw is None:
            return "none"
        if isinstance(raw, str):
            return raw if raw else "none"
        if hasattr(raw, "__len__"):
            if len(raw) == 0:
                return "none"
            inner = raw[0]
            if hasattr(inner, "__len__"):
                ft = inner[0] if len(inner) > 0 else "none"
            else:
                ft = inner
            return str(ft) if ft else "none"
    except Exception:
        return "none"
    return "none"


def load_wm811k(pkl_path: str, max_wafers: int = None) -> pd.DataFrame:
    """
    Load WM-811K pickle and flatten into die-level DataFrame.

    Args:
        pkl_path: Path to LSWMD.pkl file
        max_wafers: Limit number of wafers (None = load all)

    Returns:
        DataFrame with columns:
            lot_id, wafer_id, die_x, die_y, pass_fail, defect_code
    """
    print(f"Loading WM-811K from: {pkl_path}")
    df_raw = pd.read_pickle(pkl_path)

    if max_wafers:
        df_raw = df_raw.head(max_wafers)

    print(f"Total wafers in dataset: {len(df_raw)}")

    records = []
    for idx, row in df_raw.iterrows():
        wafer_map = row["waferMap"]
        lot_id = str(row.get("lotName", f"LOT_{idx:06d}"))
        wafer_id = int(row.get("waferIndex", idx))
        defect_code = _extract_failure_type(row.get("failureType"))

        if not isinstance(wafer_map, np.ndarray):
            continue

        for x in range(wafer_map.shape[0]):
            for y in range(wafer_map.shape[1]):
                val = int(wafer_map[x][y])
                if val == 0:          # no die — skip
                    continue
                pass_fail = 1 if val == 1 else 0
                records.append({
                    "lot_id":      lot_id,
                    "wafer_id":    wafer_id,
                    "die_x":       x,
                    "die_y":       y,
                    "pass_fail":   pass_fail,
                    "defect_code": defect_code,
                })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1} wafers...")

    df = pd.DataFrame(records)
    print(f"Preprocessing complete: {len(df):,} die records from {df['lot_id'].nunique()} lots")
    return df


def export_batch_csv(df: pd.DataFrame, lot_id: str, output_path: str) -> str:
    """Export a single lot as CSV for demo upload."""
    batch = df[df["lot_id"] == lot_id].copy()
    batch.to_csv(output_path, index=False)
    print(f"Exported {len(batch):,} dies for lot '{lot_id}' to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    pkl = sys.argv[1] if len(sys.argv) > 1 else "data/LSWMD.pkl"
    df = load_wm811k(pkl, max_wafers=200)
    print(df.head())
    print(df["defect_code"].value_counts())