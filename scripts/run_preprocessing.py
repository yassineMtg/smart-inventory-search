import logging
import sys
import os

# Ensure Python can find the `database` module
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from pipelines.preprocessing_pipeline import run_preprocessing_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info("✅ Running Data Preprocessing Pipeline...")
    run_preprocessing_pipeline()
    logging.info("✅ Data Preprocessing Pipeline Completed Successfully!")
