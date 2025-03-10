import logging
import pandas as pd
import pandera as pa
from pandera.typing import Series
import pymysql
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Database connection details
DB_URI = "mysql+pymysql://root:8520@localhost:3306/mlflow_metadata"
TABLE_NAME = "processed_data"


class DataSchema(pa.DataFrameModel):
    text: Series[str] = pa.Field(str_length={"min_value": 1}, nullable=False)
    labels: Series[int] = pa.Field(isin={0, 1, 2}, nullable=False)


# Function to fetch processed data
def fetch_processed_data():
    logging.info("📥 Fetching processed data from MySQL...")
    engine = create_engine(DB_URI)
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql(query, engine)
    logging.info(f"✅ Retrieved {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to validate data using Pandera
def validate_data(df):
    logging.info("🧪 Validating data using Pandera...")
    try:
        validated_df = DataSchema.validate(df)
        logging.info("✅ Data validation passed!")
        return validated_df
    except pa.errors.SchemaErrors as e:
        logging.error("❌ Data validation failed!")
        logging.error(e)
        return None


# MySQL Database Config
DB_URI = "mysql+pymysql://root:8520@localhost:3306/mlflow_metadata"

# Function to store validated data
def store_validated_data(df: pd.DataFrame):
    """Store validated data into MySQL while ensuring no redundant versions."""
    engine = create_engine(DB_URI)

    with engine.connect() as conn:
        # Check if this version of data already exists
        existing_df = pd.read_sql(text("SELECT * FROM validated_data"), engine)

        new_data = df[~df["text"].isin(existing_df["text"])]

        if new_data.empty:
            logging.info("⚠️ No new validated data to insert, skipping.")
            return

        # Get latest version number
        result = conn.execute(text("SELECT MAX(version) FROM validated_data")).fetchone()
        latest_version = result[0] if result[0] else 0
        new_version = latest_version + 1

        new_data["version"] = new_version
        new_data.to_sql("validated_data", con=engine, if_exists="append", index=False)

        logging.info(f"✅ Stored validated data with version: {new_version}")


# Run validation pipeline
def validation_pipeline():
    logging.info("🚀 Running Data Validation Pipeline...")
    df = fetch_processed_data()
    validated_df = validate_data(df)
    if validated_df is not None:
        store_validated_data(validated_df)
        logging.info("🎉 Validation Pipeline Completed Successfully!")
    else:
        logging.error("❌ Validation pipeline failed due to schema errors.")