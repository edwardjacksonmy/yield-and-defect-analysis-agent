"""
tests/test_agent.py
Integration and end-to-end tests for the LangChain ReAct agent — TC-16 to TC-22.

Note: These tests make real LLM API calls.
      Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env before running.
      Use pytest -m "not e2e" to skip LLM tests in CI.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from langchain_classic.memory import ConversationBufferMemory


class MockSessionState(dict):
    pass


# ─── TC-16: Agent routes yield query to YieldCalculatorTool ──────────────────
@pytest.mark.e2e
def test_TC16_agent_routes_yield_query(sample_wafer_df):
    """TC-16: Agent correctly routes a yield query to YieldCalculatorTool."""
    mock_state = MockSessionState({"current_df": sample_wafer_df})

    with patch("streamlit.session_state", mock_state):
        from agent.core import create_agent_executor
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        executor = create_agent_executor(memory)

        result = executor.invoke({"input": "What is the current yield rate of this batch?"})

    tool_names = [step[0].tool for step in result.get("intermediate_steps", [])]
    assert "yield_calculator_tool" in tool_names, (
        f"Expected yield_calculator_tool in tool chain, got: {tool_names}"
    )
    assert result["output"]  # Non-empty response


# ─── TC-17: Agent chains multiple tools for complex query ─────────────────────
@pytest.mark.e2e
def test_TC17_agent_chains_multiple_tools(sample_wafer_df, sample_csv_path):
    """TC-17: Agent invokes ≥3 tools for a comprehensive analysis query."""
    mock_state = MockSessionState({
        "current_df": sample_wafer_df,
        "uploaded_file_path": sample_csv_path
    })

    with patch("streamlit.session_state", mock_state):
        from agent.core import create_agent_executor
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        executor = create_agent_executor(memory)

        result = executor.invoke({
            "input": (
                "Give me a complete analysis: yield rate, top defect patterns, "
                "spatial clustering, and what the root cause might be."
            )
        })

    steps = result.get("intermediate_steps", [])
    assert len(steps) >= 3, f"Expected ≥3 tool calls, got {len(steps)}"


# ─── TC-18: Agent retains memory across turns ─────────────────────────────────
@pytest.mark.e2e
def test_TC18_agent_memory_retention(sample_wafer_df):
    """TC-18: Agent references prior yield value in follow-up question."""
    mock_state = MockSessionState({"current_df": sample_wafer_df})

    with patch("streamlit.session_state", mock_state):
        from agent.core import create_agent_executor
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        executor = create_agent_executor(memory)

        # Turn 1
        executor.invoke({"input": "What is the yield rate?"})

        # Turn 2 — should reference prior result
        result_2 = executor.invoke({"input": "Based on that yield, is it above or below the 80% target?"})

    assert len(memory.chat_memory.messages) >= 4  # 2 human + 2 AI turns
    assert "80" in result_2["output"] or "target" in result_2["output"].lower()


# ─── TC-19: E2E — upload CSV → ask → get response ─────────────────────────────
@pytest.mark.e2e
def test_TC19_e2e_full_pipeline(sample_wafer_df, sample_csv_path):
    """TC-19: Full pipeline: provide file path → ask defect question → get structured response."""
    mock_state = MockSessionState()

    with patch("streamlit.session_state", mock_state):
        from agent.core import create_agent_executor
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        executor = create_agent_executor(memory)

        result = executor.invoke({
            "input": (
                f"[Uploaded file path: {sample_csv_path}] "
                "Load this batch and tell me the top 3 defect patterns."
            )
        })

    assert result["output"]
    assert len(result["output"]) > 50  # Meaningful response, not empty
    assert "current_df" in mock_state or len(result.get("intermediate_steps", [])) > 0


# ─── TC-20: Agent handles missing data gracefully ────────────────────────────
@pytest.mark.e2e
def test_TC20_agent_handles_no_data():
    """TC-20: Agent returns informative error when no data is loaded and no file provided."""
    mock_state = MockSessionState()  # no data, no file

    with patch("streamlit.session_state", mock_state):
        from agent.core import create_agent_executor
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        executor = create_agent_executor(memory)

        result = executor.invoke({"input": "What is the yield rate?"})

    output = result["output"].lower()
    assert any(phrase in output for phrase in [
        "no data", "no batch", "ingest", "upload", "file", "error"
    ]), f"Unexpected response: {result['output']}"


# ─── Mock-based integration tests (no LLM call) ──────────────────────────────
def test_TC21_agent_registers_all_tools():
    """TC-21 (mock): create_agent_executor wires all 8 expected tools into ALL_TOOLS."""
    from agent.core import ALL_TOOLS
    registered = {t.name for t in ALL_TOOLS}
    expected = {
        "data_ingestion_tool",
        "yield_calculator_tool",
        "defect_analyzer_tool",
        "spatial_clustering_tool",
        "historical_query_tool",
        "wafer_visualizer_tool",
        "report_generator_tool",
        "root_cause_tool",
    }
    assert registered == expected, f"Tool mismatch — diff: {registered ^ expected}"


# ─── TC-22: ReAct prompt template declares all required input variables ────────
def test_TC22_agent_prompt_input_variables():
    """TC-22 (mock): ReAct prompt template declares all variables the executor injects."""
    from agent.core import REACT_PROMPT_TEMPLATE

    for var in ["tools", "tool_names", "agent_scratchpad", "input", "chat_history"]:
        assert f"{{{var}}}" in REACT_PROMPT_TEMPLATE, f"Missing prompt placeholder: '{{{var}}}'"
