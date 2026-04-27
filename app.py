"""
app.py
Streamlit frontend for the Yield & Defect Analysis Agent.

Run:
    streamlit run app.py
"""

import os
import uuid
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferMemory

from agent.core import create_agent_executor
from agent.db_chat import (
    save_message, load_sessions, load_session_messages,
    restore_memory, restore_dataframe, ensure_table,
)

load_dotenv()
ensure_table()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yield & Defect Analysis Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #060a12; }
  [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1f2937; }
  .stChatMessage { background: #0d1117; border: 1px solid #1f2937; border-radius: 8px; margin-bottom: 0.5rem; }
  .stChatMessage p { color: #e2e8f0; }
  h1, h2, h3 { color: #e2e8f0 !important; }
  .metric-label { color: #6b7280 !important; font-size: 0.75rem !important; }
  .stButton > button {
    background: #0d1117; color: #e2e8f0;
    border: 1px solid #374151; border-radius: 6px;
    font-size: 0.75rem; transition: all 0.2s;
  }
  .stButton > button:hover { border-color: #38bdf8; color: #38bdf8; }
  code { color: #34d399 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session state init ──────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":          [],
        "current_df":        None,
        "last_figure":       None,
        "last_report":       None,
        "uploaded_file_path": None,
        "clustered_df":      None,
        "cluster_labels":    None,
        "last_chart_type":   None,
        "quick_query":       None,
        "session_id":        str(uuid.uuid4()),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    if "agent_executor" not in st.session_state:
        st.session_state["agent_executor"] = create_agent_executor(
            st.session_state["memory"]
        )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 Yield & Defect Agent")
        st.caption("IC Manufacturing · WM-811K · Agentic AI")
        st.divider()

        # File upload
        uploaded = st.file_uploader(
            "Upload Wafer Batch CSV",
            type=["csv"],
            help="Required columns: lot_id, wafer_id, die_x, die_y, pass_fail, defect_code",
        )

        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            df = pd.read_csv(tmp_path)
            st.session_state["current_df"] = df
            st.session_state["uploaded_file_path"] = tmp_path

            st.success(f"✅ {uploaded.name} loaded")

            if "pass_fail" in df.columns:
                yield_rate = df["pass_fail"].mean() * 100
                c1, c2 = st.columns(2)
                c1.metric("Yield", f"{yield_rate:.1f}%",
                          delta=f"{yield_rate - 80:.1f}% vs target",
                          delta_color="normal")
                c2.metric("Dies", f"{len(df):,}")

                wafers = df["wafer_id"].nunique()
                fails = int((df["pass_fail"] == 0).sum())
                st.caption(f"{wafers} wafers · {fails:,} failed dies")

        st.divider()

        # Quick-fire queries
        st.markdown("**Quick Queries**")
        quick_queries = [
            ("📊", "What is the yield rate?"),
            ("🔍", "Show top 3 defect patterns"),
            ("🗺", "Generate wafer map"),
            ("🎯", "Run spatial clustering"),
            ("🕐", "Compare to historical data"),
            ("💡", "What is the root cause?"),
            ("📝", "Generate full analysis report"),
        ]
        for icon, query in quick_queries:
            if st.button(f"{icon} {query}", use_container_width=True):
                st.session_state["quick_query"] = query

        st.divider()

        # Session controls
        col_a, col_b = st.columns(2)
        if col_a.button("🗑 Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["memory"] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            st.session_state["agent_executor"] = create_agent_executor(
                st.session_state["memory"]
            )
            st.rerun()

        if col_b.button("🔄 New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()

        # Chat history viewer
        with st.expander("🕐 Chat History", expanded=False):
            sessions = load_sessions(limit=20)
            if not sessions:
                st.caption("No history saved yet.")
            else:
                for s in sessions:
                    ts = str(s["created_at"])[:16]
                    label = f"{ts} | {s['lot_id'] or 'no lot'} — {s['first_message'][:40]}..."
                    if st.button(label, key=f"hist_{s['session_id']}"):
                        msgs = load_session_messages(s["session_id"])
                        st.session_state["messages"] = [
                            {"role": m["role"], "content": m["content"]} for m in msgs
                        ]
                        new_memory = ConversationBufferMemory(
                            memory_key="chat_history", return_messages=True
                        )
                        restore_memory(s["session_id"], new_memory)
                        st.session_state["memory"] = new_memory
                        st.session_state["agent_executor"] = create_agent_executor(new_memory)
                        st.session_state["session_id"] = s["session_id"]
                        restored_df = restore_dataframe(s["session_id"])
                        if restored_df is not None:
                            st.session_state["current_df"] = restored_df
                        st.rerun()

        st.caption("Built with LangChain · Claude · PostgreSQL · Streamlit")


# ─── Main layout ──────────────────────────────────────────────────────────────
def _run_agent(user_prompt: str):
    """Invoke agent executor and return response."""
    file_path = st.session_state.get("uploaded_file_path", "")

    # Auto-prepend file path if data not yet ingested
    messages_so_far = " ".join(m["content"] for m in st.session_state["messages"])
    needs_ingestion = (
        file_path and
        "data ingestion successful" not in messages_so_far.lower() and
        "data_ingestion_tool" not in messages_so_far.lower()
    )

    if needs_ingestion:
        full_prompt = f"[Uploaded file path: {file_path}] {user_prompt}"
    else:
        full_prompt = user_prompt

    result = st.session_state["agent_executor"].invoke({"input": full_prompt})
    return result.get("output", "Analysis complete."), result.get("intermediate_steps", [])


def main():
    _init_state()
    _render_sidebar()

    # Title
    st.markdown(
        "<h2 style='margin-bottom:0; color:#e2e8f0; font-family:Courier New'>🔬 Yield & Defect Analysis Agent</h2>",
        unsafe_allow_html=True,
    )
    st.caption("WM-811K · IC Manufacturing Agentic AI · LangChain ReAct")
    st.divider()

    # Two-column layout
    col_chat, col_viz = st.columns([1.1, 1], gap="medium")

    # ── Chat panel ─────────────────────────────────────────────────────
    with col_chat:
        st.markdown("### 💬 Analysis Chat")
        chat_container = st.container(height=520)

        with chat_container:
            if not st.session_state["messages"]:
                st.info(
                    "👋 Upload a wafer batch CSV in the sidebar, then ask me anything about "
                    "yield, defects, spatial patterns, or root causes."
                )
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Resolve prompt (quick query or typed)
        quick = st.session_state.pop("quick_query", None)
        typed = st.chat_input("Ask about yield, defects, clusters, root cause...")
        prompt = quick or typed

        if prompt:
            session_id = st.session_state["session_id"]
            lot_id = None
            df = st.session_state.get("current_df")
            if df is not None and "lot_id" in df.columns:
                lot_id = str(df["lot_id"].iloc[0])
            st.session_state["messages"].append({"role": "user", "content": prompt})
            save_message(session_id, "user", prompt, lot_id)

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("🤖 Agent reasoning..."):
                        try:
                            response, steps = _run_agent(prompt)
                            st.markdown(response)

                            # Show tool trace in expander
                            if steps:
                                with st.expander(f"🔧 Agent used {len(steps)} tool(s)", expanded=False):
                                    for i, (action, obs) in enumerate(steps, 1):
                                        st.code(
                                            f"Tool {i}: {action.tool}\n"
                                            f"Input: {action.tool_input}\n"
                                            f"Output: {str(obs)[:300]}...",
                                            language="text",
                                        )

                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                            )
                            save_message(session_id, "assistant", response, lot_id)
                        except Exception as e:
                            err = f"⚠️ Agent error: {type(e).__name__}: {str(e)}"
                            st.error(err)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": err}
                            )
                            save_message(session_id, "assistant", err, lot_id)
            st.rerun()

    # ── Visualisation panel ────────────────────────────────────────────
    with col_viz:
        st.markdown("### 📊 Visualisation Panel")

        fig = st.session_state.get("last_figure")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key="main_plot")
        else:
            st.markdown(
                """
                <div style='height:340px; display:flex; align-items:center; justify-content:center;
                            background:#0d1117; border:1px dashed #374151; border-radius:8px;
                            color:#374151; font-family:Courier New; font-size:0.8rem;'>
                    Wafer map will appear here after analysis
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Chart type selector
        if st.session_state.get("current_df") is not None:
            chart_col1, chart_col2 = st.columns(2)
            if chart_col1.button("🗺 Wafer Map", use_container_width=True):
                st.session_state["quick_query"] = "Generate wafer map visualisation"
                st.rerun()
            if chart_col2.button("📊 Defect Bar", use_container_width=True):
                st.session_state["quick_query"] = "Show defect type bar chart"
                st.rerun()
            chart_col3, chart_col4 = st.columns(2)
            if chart_col3.button("🌡 Yield Heatmap", use_container_width=True):
                st.session_state["quick_query"] = "Show yield heatmap for all wafers"
                st.rerun()
            if chart_col4.button("🎯 Cluster Map", use_container_width=True):
                st.session_state["quick_query"] = "Show DBSCAN cluster map"
                st.rerun()

        st.divider()

        # Data preview
        df = st.session_state.get("current_df")
        if df is not None:
            with st.expander("📋 Data Preview (first 100 rows)"):
                st.dataframe(df.head(100), use_container_width=True, height=220)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Dies", f"{len(df):,}")
            c2.metric("Wafers", df["wafer_id"].nunique())
            c3.metric("Defect Types", df["defect_code"].nunique())

        # Report download
        report = st.session_state.get("last_report")
        if report:
            st.download_button(
                label="📄 Download Analysis Report (.md)",
                data=report,
                file_name="wafer_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()