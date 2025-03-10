import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from pipelines.ml_pipeline import ml_pipeline

if __name__ == "__main__":
    print("🚀 Running Full ML Pipeline...")
    ml_pipeline()
    print("✅ ML Pipeline Execution Complete!")
