"""
db_seeder.py
Load WM-811K historical lots into PostgreSQL.
Run once before starting the Streamlit app.

Usage:
    python data/db_seeder.py data/LSWMD.pkl --lots 15
"""

import os
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.preprocessing import load_wm811k

load_dotenv()


def create_tables(engine):
    schema_path = os.path.join(os.path.dirname(__file__), "..", "database", "schema.sql")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    with engine.connect() as conn:
        for statement in schema_sql.split(";"):
            s = statement.strip()
            if s:
                conn.execute(text(s))
        conn.commit()
    print("✅ Tables created / verified")


def compute_lot_summary(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for lot_id, group in df.groupby("lot_id"):
        total = len(group)
        passed = int(group["pass_fail"].sum())
        failed = total - passed
        yield_rate = round(passed / total * 100, 4) if total > 0 else 0.0
        fail_group = group[group["pass_fail"] == 0]
        if len(fail_group) > 0:
            mode_val = fail_group["defect_code"].mode()
            dominant = mode_val.iloc[0] if len(mode_val) > 0 else "none"
        else:
            dominant = "none"
        summaries.append({
            "lot_id":          lot_id,
            "total_dies":      total,
            "passed_dies":     passed,
            "failed_dies":     failed,
            "yield_rate":      yield_rate,
            "dominant_defect": dominant,
        })
    return pd.DataFrame(summaries)


def seed(pkl_path: str, n_historical_lots: int = 10, n_demo_lots: int = 2):
    """
    Seed PostgreSQL with historical lots from WM-811K.

    Args:
        pkl_path: Path to LSWMD.pkl
        n_historical_lots: Number of lots to load as historical data
        n_demo_lots: Additional lots exported as CSV files for demo upload
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL not set in .env")

    engine = create_engine(database_url)
    create_tables(engine)

    total_wafers = n_historical_lots + n_demo_lots
    print(f"\nLoading {total_wafers} wafers from WM-811K...")
    df = load_wm811k(pkl_path, max_wafers=total_wafers * 25)  # 25 wafers per lot approx

    all_lots = df["lot_id"].unique()
    historical_lots = all_lots[:n_historical_lots]
    demo_lots = all_lots[n_historical_lots: n_historical_lots + n_demo_lots]

    # Load historical into PostgreSQL
    hist_df = df[df["lot_id"].isin(historical_lots)].copy()
    print(f"\nInserting {len(hist_df):,} die records ({n_historical_lots} lots) into wafer_history...")
    hist_df.to_sql("wafer_history", engine, if_exists="append", index=False, chunksize=2000)

    summary_df = compute_lot_summary(hist_df)
    summary_df.to_sql("lot_summary", engine, if_exists="append", index=False)
    print(f"✅ Historical data seeded: {n_historical_lots} lots, {len(hist_df):,} dies")

    # Export demo CSVs for upload
    demo_dir = os.path.join(os.path.dirname(__file__), "..", "data", "demo_batches")
    os.makedirs(demo_dir, exist_ok=True)
    for lot in demo_lots:
        batch = df[df["lot_id"] == lot].copy()
        csv_path = os.path.join(demo_dir, f"{lot}_batch.csv")
        batch.to_csv(csv_path, index=False)
        print(f"📄 Demo CSV exported: {csv_path} ({len(batch):,} dies)")

    print("\n🎉 Database seeding complete. Ready to run the agent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed wafer_db with WM-811K data")
    parser.add_argument("pkl_path", help="Path to LSWMD.pkl")
    parser.add_argument("--lots", type=int, default=10, help="Number of historical lots to load")
    args = parser.parse_args()
    seed(args.pkl_path, n_historical_lots=args.lots)