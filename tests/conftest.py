"""
tests/conftest.py
Shared pytest fixtures for tool and agent tests.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile


@pytest.fixture
def sample_wafer_df():
    """Standard test DataFrame with mixed defect types — mirrors WM-811K structure."""
    np.random.seed(42)
    n = 600
    defect_codes = ["Center", "Edge-Ring", "Random", "Scratch", "none"]
    weights = [0.07, 0.09, 0.06, 0.05, 0.73]
    defects = np.random.choice(defect_codes, size=n, p=weights)

    return pd.DataFrame({
        "lot_id":      ["LOT_TEST_001"] * n,
        "wafer_id":    np.random.randint(1, 6, n),
        "die_x":       np.random.randint(0, 25, n),
        "die_y":       np.random.randint(0, 25, n),
        "pass_fail":   [1 if d == "none" else 0 for d in defects],
        "defect_code": defects,
    })


@pytest.fixture
def perfect_yield_df():
    """All-passing wafer batch — used to verify 100% yield edge case."""
    n = 200
    return pd.DataFrame({
        "lot_id":      ["LOT_PERFECT"] * n,
        "wafer_id":    [1] * n,
        "die_x":       list(range(n)),
        "die_y":       [0] * n,
        "pass_fail":   [1] * n,
        "defect_code": ["none"] * n,
    })


@pytest.fixture
def clustered_fail_df():
    """DataFrame with a deliberate 5x5 spatial cluster of failures at top-left."""
    records = []
    # Fail cluster: (0-4, 0-4)
    for x in range(5):
        for y in range(5):
            records.append({
                "lot_id": "LOT_CLUSTER", "wafer_id": 1,
                "die_x": x, "die_y": y,
                "pass_fail": 0, "defect_code": "Center",
            })
    # Pass remainder
    for x in range(5, 20):
        for y in range(5, 20):
            records.append({
                "lot_id": "LOT_CLUSTER", "wafer_id": 1,
                "die_x": x, "die_y": y,
                "pass_fail": 1, "defect_code": "none",
            })
    return pd.DataFrame(records)


@pytest.fixture
def missing_cols_df():
    """DataFrame missing required columns — used to test validation error handling."""
    return pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})


@pytest.fixture
def sample_csv_path(sample_wafer_df, tmp_path):
    """Write sample_wafer_df to a temp CSV and return the path."""
    path = tmp_path / "test_batch.csv"
    sample_wafer_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def bad_csv_path(missing_cols_df, tmp_path):
    """Write missing_cols_df to a temp CSV and return the path."""
    path = tmp_path / "bad_batch.csv"
    missing_cols_df.to_csv(path, index=False)
    return str(path)