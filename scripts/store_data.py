import os
import sys
import logging
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text

# Ensure Python can find the database module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the database engine function
from database.db_connection import get_db_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def store_raw_data(file_path: str, table_name: str = "raw_data"):
    """
    Load dataset from CSV and store it into MySQL without duplicating existing data.
    
    :param file_path: Path to the dataset CSV file.
    :param table_name: Name of the MySQL table where data will be stored.
    """
    try:
        logging.info("Loading dataset from %s", file_path)
        df = pd.read_csv(file_path)

        # Get database engine
        engine = get_db_engine()

        logging.info("Checking for duplicate records in MySQL table: %s", table_name)

        # Read existing data from MySQL
        existing_df = pd.read_sql(text(f"SELECT * FROM {table_name}"), engine)

        # Remove duplicates by checking if the text column already exists
        new_data = df[~df["text"].isin(existing_df["text"])]

        if new_data.empty:
            logging.info("⚠️ No new data to insert, skipping insertion.")
            return

        logging.info("Storing %d new records into MySQL table: %s", len(new_data), table_name)
        new_data.to_sql(table_name, engine, if_exists="append", index=False)

        logging.info("✅ Data successfully stored in MySQL!")

    except FileNotFoundError:
        logging.error("Dataset file not found at %s", file_path)
    except SQLAlchemyError as e:
        logging.error("Database error: %s", e)
    except Exception as e:
        logging.error("Unexpected error: %s", e)


if __name__ == "__main__":
    DATASET_PATH = "./datasets/dataset-all.csv"  # Update if necessary
    store_raw_data(DATASET_PATH)