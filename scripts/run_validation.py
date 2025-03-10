import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from pipelines.validation_pipeline import validation_pipeline

if __name__ == "__main__":
    print("✅ Running Data Validation Pipeline...")
    validation_pipeline()
    print("🎉 Validation Completed Successfully! Check MySQL for the stored version.")
