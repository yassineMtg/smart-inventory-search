import mlflow
import mlflow.sklearn
from prefect import flow, task
from scripts.store_data import store_raw_data
from scripts.run_preprocessing import run_preprocessing_pipeline
from scripts.run_validation import validation_pipeline
from feast import FeatureStore
from datetime import datetime

# Set up MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ML_Pipeline_Experiment")


@task
def ingest_data():
    """Task to ingest raw data into MySQL and track it in MLflow."""
    with mlflow.start_run(nested=True, run_name="Data_Ingestion"):
        mlflow.log_param("step", "ingestion")
        store_raw_data(file_path="./datasets/dataset-all.csv")
        mlflow.log_artifact("./datasets/dataset-all.csv")
        print("✅ Data Ingestion Complete")


@task
def preprocess_data():
    """Task to preprocess raw data and store processed data in MySQL with MLflow tracking."""
    with mlflow.start_run(nested=True, run_name="Data_Preprocessing"):
        mlflow.log_param("step", "preprocessing")

        processed_rows = run_preprocessing_pipeline()  # This must return a valid number
        if processed_rows is None:
            processed_rows = 0  # Ensure a default value if None

        mlflow.log_metric("processed_rows", int(processed_rows))  # Ensure it's an integer
        print(f"✅ Data Preprocessing Complete: {processed_rows} rows processed")


@task
def validate_data():
    """Task to validate the preprocessed data and store validated versions with MLflow tracking."""
    with mlflow.start_run(nested=True, run_name="Data_Validation"):
        mlflow.log_param("step", "validation")

        result = validation_pipeline()  # ✅ Call function

        if result is None:  # ✅ Ensure it never unpacks None
            validation_version, validated_rows = 0, 0
        else:
            validation_version, validated_rows = result

        mlflow.log_metric("validation_version", int(validation_version))
        mlflow.log_metric("validated_rows", int(validated_rows))

        print(f"✅ Data Validation Complete: Version {validation_version}, Rows: {validated_rows}")

@task
def materialize_features():
    """Task to materialize features into the online store."""
    store = FeatureStore(repo_path="feature_store/feature_repo")  # ✅ Correct path
    end_date = datetime.utcnow()  # ✅ Ensure it's a datetime object

    print(f"📥 Materializing Features from {end_date.replace(microsecond=0)}")

    try:
        store.materialize_incremental(end_date)  # ✅ No need for `.astimezone()`
        print("✅ Feature materialization complete")
    except Exception as e:
        print(f"❌ Feature materialization failed: {e}")



@flow(name="ML Data Processing Pipeline")
def ml_pipeline():
    """Prefect flow to run the full ML data pipeline with Feature Store integration."""
    with mlflow.start_run(run_name="ML_Pipeline"):
        ingest_data()
        preprocess_data()
        validate_data()
        materialize_features()


if __name__ == "__main__":
    ml_pipeline()
