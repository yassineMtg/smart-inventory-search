import logging
import pandas as pd
import re
from database.db_connection import get_db_engine
from sqlalchemy.sql import text
from sqlalchemy import create_engine, text
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_data() -> pd.DataFrame:
    """Fetch raw data from MySQL using the centralized connection."""
    logging.info("📥 [STEP 1] Fetching raw data from MySQL...")
    engine = get_db_engine()
    
    with engine.connect() as connection:
        df = pd.read_sql(text("SELECT * FROM raw_data"), connection)
    
    logging.info(f"✅ [STEP 1] Fetched {df.shape[0]} rows and {df.shape[1]} columns.")
    logging.info(f"🔍 [STEP 1] Sample raw data:\n{df.head()}")

    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess text and transform features"""
    logging.info("⚙️ [STEP 2] Starting data preprocessing...")

    if df.empty:
        logging.warning("⚠️ [STEP 2] DataFrame is empty! Skipping preprocessing.")
        return df

    logging.info(f"📊 [STEP 2] Dataset shape before preprocessing: {df.shape}")

    df["event_timestamp"] = pd.to_datetime(datetime.utcnow())
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(lambda x: re.sub(r"\W+", " ", x))

    logging.info(f"✅ [STEP 2] Dataset shape after preprocessing: {df.shape}")
    logging.info(f"🔍 [STEP 2] Sample preprocessed data:\n{df.head()}")

    return df

DB_URI = "mysql+pymysql://root:8520@localhost:3306/mlflow_metadata"

PARQUET_PATH = "./feature_store/data/processed_data.parquet"

def store_processed_data(df: pd.DataFrame, table_name: str = "processed_data"):
    """Store processed data into MySQL and update Parquet without duplication."""
    engine = create_engine(DB_URI)

    with engine.connect() as conn:
        existing_df = pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)

        # ✅ Ensure event_timestamp exists and has valid values
        if "event_timestamp" not in df.columns or df["event_timestamp"].isna().all():
            df["event_timestamp"] = pd.to_datetime(datetime.utcnow())

        # Remove duplicates before inserting
        new_data = df[~df["text"].isin(existing_df["text"])]

        if new_data.empty:
            logging.info("⚠️ No new processed data to insert, skipping.")
            return

        # ✅ Append new data to MySQL
        new_data.to_sql(table_name, con=engine, if_exists="append", index=False)
        logging.info("✅ Processed data successfully stored in MySQL!")

        # ✅ Fix: Ensure timestamps are included in Parquet
        if os.path.exists(PARQUET_PATH):
            existing_parquet = pd.read_parquet(PARQUET_PATH)
            updated_parquet = pd.concat([existing_parquet, new_data], ignore_index=True)
        else:
            updated_parquet = new_data

        updated_parquet.to_parquet(PARQUET_PATH, index=False)
        logging.info(f"✅ Processed data successfully stored in Parquet: {PARQUET_PATH}")


def run_preprocessing_pipeline():
    """Runs the full preprocessing pipeline"""
    logging.info("🚀 Running Preprocessing Pipeline...")
    raw_data = fetch_data()
    processed_data = preprocess_data(raw_data)
    store_processed_data(processed_data)
    logging.info("✅ Preprocessing Pipeline Completed Successfully!")
