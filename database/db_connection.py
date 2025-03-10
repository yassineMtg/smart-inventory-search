from sqlalchemy import create_engine

# Database connection details
DB_NAME = "mlflow_metadata"
DB_USER = "root"
DB_PASS = "8520"
DB_HOST = "localhost"
DB_PORT = "3306"

def get_db_engine():
    """Create and return a database engine."""
    return create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
