# Yield & Defect Analysis Agent

An AI-powered agentic system for IC (Integrated Circuit) manufacturing wafer yield and defect analysis. Engineers upload wafer batch data and ask natural language questions — the agent autonomously selects and chains specialized analysis tools to deliver insights, visualizations, and root-cause recommendations.

---

## Overview

This system combines a **Streamlit web dashboard**, a **LangChain ReAct agent** powered by Claude (Anthropic) or GPT-4o (OpenAI), and a **PostgreSQL database** for historical data and session persistence. It is built around the [WM-811K wafer defect dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) taxonomy, which defines 9 standard defect pattern classifications used in semiconductor manufacturing.

### Key Capabilities

- Upload wafer batch CSV files and analyze die-level pass/fail data
- Ask natural language questions and receive AI-generated insights with IC manufacturing terminology
- Detect spatial defect clusters (systematic vs. random failure patterns)
- Compare current batch yield against historical lot trends
- Generate downloadable markdown reports with engineering recommendations
- Replay past chat sessions via PostgreSQL-backed conversation history

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                 │
│     Chat UI · Wafer Map · Metrics · Report Download │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            LangChain ReAct Agent (core.py)           │
│   Claude Sonnet 4.6 (primary) · GPT-4o (fallback)   │
└──────────────────────┬──────────────────────────────┘
                       │ routes to
    ┌──────────────────┼──────────────────┐
    │                  │                  │
┌───▼────┐       ┌─────▼─────┐     ┌─────▼──────┐
│Analysis│       │Spatial /  │     │ Reporting  │
│Tools   │       │Visual     │     │ Tools      │
│        │       │Tools      │     │            │
│• Yield │       │• DBSCAN   │     │• Root Cause│
│• Defect│       │  Cluster  │     │• Report Gen│
│• Ingest│       │• Wafer Map│     │• Historical│
└───┬────┘       └─────┬─────┘     └─────┬──────┘
    │                  │                  │
┌───▼──────────────────▼──────────────────▼──────────┐
│                 PostgreSQL Database                  │
│  wafer_history · lot_summary · chat_history         │
│  session_wafer_data                                  │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Yield and Defect Analysis Agent/
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── .env                         # Environment variables (API keys, DB credentials)
│
├── agent/
│   ├── core.py                  # LangChain ReAct agent setup
│   ├── db_chat.py               # Chat history persistence (PostgreSQL)
│   └── tools/
│       ├── data_ingestion.py    # CSV parsing & validation
│       ├── yield_calculator.py  # Yield rate calculations
│       ├── defect_analyzer.py   # WM-811K defect pattern analysis
│       ├── spatial_clustering.py# DBSCAN spatial cluster detection
│       ├── wafer_visualizer.py  # Interactive Plotly chart generation
│       ├── historical_query.py  # Historical lot trend queries
│       ├── root_cause.py        # Root cause hypothesis engine
│       └── report_generator.py  # Markdown report compilation
│
├── data/
│   ├── preprocessing.py         # WM-811K pickle → DataFrame converter
│   ├── db_seeder.py             # Historical data loader for PostgreSQL
│   ├── LSWMD.pkl                # WM-811K wafer map dataset (2.1 GB)
│   └── demo_batches/
│       ├── lot11_batch.csv      # Sample wafer batch (lot 11)
│       └── lot12_batch.csv      # Sample wafer batch (lot 12)
│
├── database/
│   └── schema.sql               # PostgreSQL table definitions
│
└── tests/
    ├── conftest.py              # Shared test fixtures
    ├── test_agent.py            # Agent routing & tool-chaining tests
    └── test_tools.py            # Individual tool unit tests
```

---

## Agent Tools

| Tool | Description |
|---|---|
| `data_ingestion_tool` | Parses and validates uploaded CSV; stores to session state and DB |
| `yield_calculator_tool` | Computes overall and per-wafer yield rates; flags critical batches (< 80%) |
| `defect_analyzer_tool` | Ranks defect types using WM-811K taxonomy; surfaces process-specific insights |
| `spatial_clustering_tool` | Runs DBSCAN on failed die coordinates to distinguish systematic vs. random defects |
| `wafer_visualizer_tool` | Generates Plotly wafer maps, defect bar charts, and yield heatmaps |
| `historical_query_tool` | Queries PostgreSQL for yield trends and lot comparisons |
| `root_cause_tool` | Returns ranked hypotheses and recommended actions per defect pattern |
| `report_generator_tool` | Compiles all analysis sections into a downloadable markdown report |

---

## WM-811K Defect Classifications

The system recognizes the 9 standard defect pattern types from the WM-811K taxonomy:

| Pattern | Typical Root Cause |
|---|---|
| **Center** | CMP non-uniformity, spin-coat centre effect, furnace temperature hotspot |
| **Donut** | Spin coating edge bead, etch chamber ring contamination |
| **Edge-Loc** | Edge exposure variation, mechanical chuck damage |
| **Edge-Ring** | Etch ring artifact, deposition non-uniformity at wafer periphery |
| **Loc** | Localized contamination particle, photomask defect |
| **Near-full** | Global process excursion (pressure, temperature, chemistry) |
| **Random** | Particle contamination, electrostatic discharge |
| **Scratch** | Handling damage, robotic arm contact |
| **none** | No detectable defect pattern |

---

## Prerequisites

- Python 3.10+
- PostgreSQL 14+
- An Anthropic API key (preferred) or OpenAI API key (fallback)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/edwardjackson_my/yield-defect-analysis-agent.git
cd yield-defect-analysis-agent
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# LLM API (at least one required)
ANTHROPIC_API_KEY=sk-ant-api03-...     # Preferred — Claude Sonnet 4.6
OPENAI_API_KEY=sk-...                  # Fallback — GPT-4o

# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=wafer_db
DB_USER=postgres
DB_PASSWORD=your_password
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/wafer_db
```

> **Security note**: Never commit `.env` to version control. Add it to `.gitignore`.

### 5. Initialize the Database

```bash
psql -U postgres -d wafer_db -f database/schema.sql
```

### 6. (Optional) Seed Historical Data

This loads WM-811K wafer records into the database for historical trend analysis:

```bash
python data/db_seeder.py data/LSWMD.pkl --lots 15
```

---

## Running the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Usage Workflow

1. **Upload** a wafer batch CSV using the sidebar file uploader.
2. **Ask questions** in the chat box — for example:
   - *"What is the yield rate for this batch?"*
   - *"Show me the top defects and their process implications."*
   - *"Are the failures spatially clustered or random?"*
   - *"Compare this lot to historical performance."*
   - *"Generate a full analysis report."*
3. **View** the AI's reasoning, interactive Plotly wafer maps, and metrics.
4. **Download** the markdown report from the sidebar.
5. **Reload** past sessions from the chat history panel.

### CSV Format

Uploaded CSV files must contain the following columns:

| Column | Type | Values |
|---|---|---|
| `lot_id` | string | Lot identifier |
| `wafer_id` | string/int | Wafer identifier within lot |
| `die_x` | int | Die X coordinate |
| `die_y` | int | Die Y coordinate |
| `pass_fail` | int | `1` = pass, `0` = fail |
| `defect_code` | string | One of the 9 WM-811K pattern labels |

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Skip LLM-dependent end-to-end tests (faster, no API key needed)
pytest -m "not e2e" tests/

# Verbose output
pytest -v tests/
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **AI / Agent** | LangChain ReAct, Anthropic Claude Sonnet 4.6, OpenAI GPT-4o |
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Clustering** | scikit-learn (DBSCAN) |
| **Visualization** | Plotly, Matplotlib |
| **Database** | PostgreSQL, SQLAlchemy, psycopg2 |
| **Configuration** | python-dotenv |
| **Testing** | pytest, pytest-mock |

---

## License

This project is for educational and research purposes. The WM-811K dataset is sourced from publicly available semiconductor research. See the dataset's original license for usage terms.

---

## Acknowledgements

- [WM-811K Wafer Map Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) — Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015)
- [LangChain](https://www.langchain.com/) — Agent framework
- [Anthropic Claude](https://www.anthropic.com/) — Primary LLM
