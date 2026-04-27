"""
tests/test_tools.py
Unit tests for all 8 agent tools — TC-01 through TC-06 + extended coverage.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ─── Helper to mock streamlit session_state ──────────────────────────────────
class MockSessionState(dict):
    """Dict subclass that behaves like st.session_state for tool tests."""
    pass


# ─── TC-01: DataIngestionTool — valid CSV ────────────────────────────────────
def test_TC01_data_ingestion_valid(sample_csv_path):
    """TC-01: Valid CSV loads successfully and returns correct summary."""
    mock_state = MockSessionState()
    with patch("streamlit.session_state", mock_state):
        from agent.tools.data_ingestion import data_ingestion_tool
        result = data_ingestion_tool.invoke(sample_csv_path)

    assert "Data ingestion successful" in result
    assert "Total dies" in result
    assert "Yield rate" in result
    assert "current_df" in mock_state
    assert mock_state["current_df"] is not None


# ─── TC-02: DataIngestionTool — missing columns ──────────────────────────────
def test_TC02_data_ingestion_missing_columns(bad_csv_path):
    """TC-02: CSV with missing required columns returns descriptive error."""
    mock_state = MockSessionState()
    with patch("streamlit.session_state", mock_state):
        from agent.tools.data_ingestion import data_ingestion_tool
        result = data_ingestion_tool.invoke(bad_csv_path)

    assert "ERROR" in result
    assert "Missing required columns" in result


# ─── TC-03: DataIngestionTool — file not found ───────────────────────────────
def test_TC03_data_ingestion_file_not_found():
    """TC-03: Non-existent path returns file not found error."""
    mock_state = MockSessionState()
    with patch("streamlit.session_state", mock_state):
        from agent.tools.data_ingestion import data_ingestion_tool
        result = data_ingestion_tool.invoke("/nonexistent/path/batch.csv")

    assert "ERROR" in result


# ─── TC-04: YieldCalculatorTool — known yield ───────────────────────────────
def test_TC04_yield_calculator_known_data(sample_wafer_df):
    """TC-04: YieldCalculatorTool computes correct yield for known DataFrame."""
    expected_yield = round(sample_wafer_df["pass_fail"].mean() * 100, 4)
    mock_state = MockSessionState({"current_df": sample_wafer_df})

    with patch("streamlit.session_state", mock_state):
        from agent.tools.yield_calculator import yield_calculator_tool
        result = yield_calculator_tool.invoke("current")

    assert f"{expected_yield:.2f}%" in result
    assert "Overall yield" in result
    assert "Best wafer" in result
    assert "Worst wafer" in result


# ─── TC-05: YieldCalculatorTool — 100% yield ────────────────────────────────
def test_TC05_yield_calculator_perfect_yield(perfect_yield_df):
    """TC-05: 100% pass rate correctly reported as 100.0% yield."""
    mock_state = MockSessionState({"current_df": perfect_yield_df})

    with patch("streamlit.session_state", mock_state):
        from agent.tools.yield_calculator import yield_calculator_tool
        result = yield_calculator_tool.invoke("current")

    assert "100.00%" in result


# ─── TC-06: YieldCalculatorTool — no data ───────────────────────────────────
def test_TC06_yield_calculator_no_data():
    """TC-06: YieldCalculatorTool returns error when no data is loaded."""
    mock_state = MockSessionState()  # no current_df

    with patch("streamlit.session_state", mock_state):
        from agent.tools.yield_calculator import yield_calculator_tool
        result = yield_calculator_tool.invoke("current")

    assert "ERROR" in result


# ─── TC-07: DefectAnalyzerTool — WM-811K labels ──────────────────────────────
def test_TC07_defect_analyzer_wm811k_labels(sample_wafer_df):
    """TC-07: DefectAnalyzerTool returns WM-811K taxonomy labels in results."""
    mock_state = MockSessionState({"current_df": sample_wafer_df})
    wm811k_labels = ["Center", "Edge-Ring", "Random", "Scratch"]

    with patch("streamlit.session_state", mock_state):
        from agent.tools.defect_analyzer import defect_analyzer_tool
        result = defect_analyzer_tool.invoke("3")

    assert "Defect Pattern Analysis" in result
    assert any(label in result for label in wm811k_labels)
    assert "Dominant failure" in result
    assert "Process insight" in result


# ─── TC-08: DefectAnalyzerTool — perfect yield ───────────────────────────────
def test_TC08_defect_analyzer_no_failures(perfect_yield_df):
    """TC-08: DefectAnalyzerTool correctly handles 100% yield — no failures."""
    mock_state = MockSessionState({"current_df": perfect_yield_df})

    with patch("streamlit.session_state", mock_state):
        from agent.tools.defect_analyzer import defect_analyzer_tool
        result = defect_analyzer_tool.invoke("3")

    assert "100%" in result or "No failed dies" in result


# ─── TC-09: SpatialClusteringTool — detects cluster ─────────────────────────
def test_TC09_spatial_clustering_detects_cluster(clustered_fail_df):
    """TC-09: DBSCAN detects the intentional 5x5 spatial cluster of failures."""
    mock_state = MockSessionState({"current_df": clustered_fail_df})

    with patch("streamlit.session_state", mock_state):
        from agent.tools.spatial_clustering import spatial_clustering_tool
        result = spatial_clustering_tool.invoke({"eps": 2.0, "min_samples": 3})

    assert "Clusters found:" in result
    assert "Cluster 1:" in result
    import re
    match = re.search(r"Clusters found:\s*(\d+)", result)
    assert match and int(match.group(1)) >= 1


# ─── TC-10: SpatialClusteringTool — random fails → no cluster ────────────────
def test_TC10_spatial_clustering_random_no_cluster():
    """TC-10: Sparse random failures should not form clusters."""
    np.random.seed(99)
    sparse_df = pd.DataFrame({
        "lot_id":      ["LOT_RANDOM"] * 300,
        "wafer_id":    [1] * 300,
        "die_x":       np.random.randint(0, 50, 300),
        "die_y":       np.random.randint(0, 50, 300),
        "pass_fail":   [0 if i < 5 else 1 for i in range(300)],  # only 5 sparse fails
        "defect_code": ["Random"] * 5 + ["none"] * 295,
    })
    mock_state = MockSessionState({"current_df": sparse_df})

    with patch("streamlit.session_state", mock_state):
        from agent.tools.spatial_clustering import spatial_clustering_tool
        result = spatial_clustering_tool.invoke({"eps": 2.0, "min_samples": 5})

    assert "Clusters found: 0" in result or "Insufficient failed dies" in result or "No spatial clusters" in result


# ─── TC-11: HistoricalQueryTool — empty database ─────────────────────────────
def test_TC11_historical_query_empty_db():
    """TC-11: HistoricalQueryTool handles empty database gracefully."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    empty_df = pd.DataFrame(columns=["lot_id", "yield_rate", "dominant_defect", "total_dies", "failed_dies", "created_at"])

    with patch("agent.tools.historical_query._get_engine", return_value=mock_engine), \
         patch("pandas.read_sql", return_value=empty_df), \
         patch("streamlit.session_state", MockSessionState()):
        from agent.tools.historical_query import historical_query_tool
        result = historical_query_tool.invoke("yield_rate")

    assert "No historical data" in result


# ─── TC-12: RootCauseTool — known WM-811K pattern ────────────────────────────
def test_TC12_root_cause_known_pattern():
    """TC-12: RootCauseTool returns hypotheses for each known WM-811K pattern."""
    from agent.tools.root_cause import root_cause_tool
    known_patterns = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]

    for pattern in known_patterns:
        result = root_cause_tool.invoke(pattern)
        assert "Ranked Hypotheses" in result, f"Failed for pattern: {pattern}"
        assert "Recommended Corrective Actions" in result, f"No actions for pattern: {pattern}"
        assert "#1." in result


# ─── TC-13: RootCauseTool — unknown pattern ──────────────────────────────────
def test_TC13_root_cause_unknown_pattern():
    """TC-13: RootCauseTool handles unrecognised defect pattern gracefully."""
    from agent.tools.root_cause import root_cause_tool
    result = root_cause_tool.invoke("UnknownPattern_XYZ")

    assert "not in WM-811K taxonomy" in result
    assert "Generic recommendation" in result


# ─── TC-14: RootCauseTool — auto mode ────────────────────────────────────────
def test_TC14_root_cause_auto_mode(sample_wafer_df):
    """TC-14: RootCauseTool auto-detects dominant defect from session state."""
    mock_state = MockSessionState({"current_df": sample_wafer_df})

    with patch("streamlit.session_state", mock_state):
        from agent.tools.root_cause import root_cause_tool
        result = root_cause_tool.invoke("auto")

    assert "Root Cause Analysis" in result
    assert "Ranked Hypotheses" in result


# ─── TC-15: ReportGeneratorTool — generates full report ──────────────────────
def test_TC15_report_generator_full(sample_wafer_df):
    """TC-15: ReportGeneratorTool produces all required report sections."""
    mock_state = MockSessionState({"current_df": sample_wafer_df})

    with patch("streamlit.session_state", mock_state), \
         patch("agent.tools.report_generator.create_engine") as mock_eng:
        mock_conn = MagicMock()
        mock_eng.return_value.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_eng.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pandas.read_sql", return_value=pd.DataFrame()):
            from agent.tools.report_generator import report_generator_tool
            result = report_generator_tool.invoke("all")

    assert "# Wafer Yield & Defect Analysis Report" in result
    assert "Executive Summary" in result
    assert "Defect Pattern Breakdown" in result
    assert "Recommendations" in result
    assert "last_report" in mock_state
