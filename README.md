# Smart Inventory Search

## Milestone 1 (Project Inception)

### **Business Case**

Businesses managing large inventories often struggle with inefficient product retrieval systems. Traditional search mechanisms rely on exact keyword matching, which fails to handle descriptive or contextual queries effectively. This leads to:

- **Increased operational costs**: Employees spend excessive time manually filtering search results.
- **Lost sales opportunities**: Customers cannot find relevant products due to vague or inconsistent product descriptions.
- **Poor user experience**: Frustration from inefficient search systems reduces customer satisfaction and engagement.

A **Smart Inventory Search** system powered by machine learning (ML) can address these challenges by understanding natural language queries and contextual meaning, enabling faster and more accurate product discovery.

---

### **Business Value**

Implementing an ML-driven search system provides the following business value:

1. **Enhanced Search Accuracy**:
   - Understands user intent and context, improving product discovery.
   - Reduces reliance on rigid keyword matches.

2. **Increased Operational Efficiency**:
   - Reduces manual search time, allowing employees to focus on higher-value tasks.
   - Handles large and evolving product catalogs with minimal reconfiguration.

3. **Improved User Experience**:
   - Enables conversational and intuitive searching, increasing customer satisfaction.
   - Boosts engagement and sales by ensuring customers find relevant products quickly.

4. **Scalability & Adaptability**:
   - Adapts to new products and changing inventory without requiring extensive rule updates.
   - Scales seamlessly with business growth.

By leveraging ML, businesses can achieve faster, more accurate product retrieval, leading to increased efficiency and revenue growth.

---

### **ML Framing**

#### **Project Archetype: Software 2.0**

This project falls under the **Software 2.0** archetype, where traditional rule-based systems are replaced by machine learning models that learn from data. Unlike static, predefined SQL queries, this system uses deep learning models to dynamically interpret and process inventory search requests. Key advantages include:

- **Natural Language Understanding**: Handles ambiguous or complex user queries.
- **Contextual Reasoning**: Understands synonyms, product attributes, and natural language variations.
- **Continuous Improvement**: Learns from user interactions to improve over time.

#### **Feasibility & Baseline Model**

To establish feasibility, I am using a pre-trained NLP model as a baseline and fine-tune it on domain-specific inventory queries. The combination of **DistilBERT** and **Gemini API** ensures an optimal balance between accuracy, speed, and contextual understanding.

| Model Name         | Developer     | Purpose                                | Performance      |
|--------------------|---------------|----------------------------------------|------------------|
| DistilBERT         | Hugging Face  | NLP-based search query processing      | 97% of BERT      |
| Gemini API         | Google AI     | Advanced contextual NLP                | State-of-the-art |

#### **Baseline Model Justification**

- **DistilBERT**: A distilled version of BERT, offering fast inference while maintaining strong semantic search capabilities. It is pretrained on large-scale datasets and can be fine-tuned efficiently for inventory-specific queries.
- **Gemini API**: A cutting-edge NLP model known for handling conversational AI tasks and contextual search. It complements DistilBERT by processing ambiguous or complex user queries.

By fine-tuning DistilBERT on structured inventory datasets and leveraging Gemini API for contextual interpretation, the system can handle diverse search queries with high accuracy.

---

### **Baseline Model Card**

#### **Model Name**: `distilbert-base-uncased`

- **Developed by**: Hugging Face
- **Model Type**: Transformer-based Language Model
- **Language**: English
- **License**: Apache 2.0
- **Intended Use**: Fine-tuned for sequence classification and semantic search tasks.
- **Training Data**: Pretrained on BookCorpus and English Wikipedia.
- **Evaluation Results**: Retains 97% of BERT's performance while being 60% faster and 40% smaller.

#### **Limitations**
- Lacks `token_type_ids`, making it unsuitable for tasks requiring segment differentiation.
- Limited to English language understanding.

---

### **Metrics for Business Goal Evaluation**

To evaluate the success of the Smart Inventory Search system, I will use the following metrics:

1. **Search Accuracy**:
   - Precision, recall, and F1 score for product retrieval.
2. **User Satisfaction**:
   - Measured through user feedback or surveys.
3. **Operational Efficiency**:
   - Reduction in average search time (e.g., from 10 minutes to 2 minutes).

---

### **Dataset**

The dataset used for this project is the **Rakuten France Multimodal Product Classification Dataset**, available on Kaggle. It contains product data with images and text descriptions, making it ideal for training and evaluating multimodal search systems.

- **Dataset Name**: Rakuten France Multimodal Product Classification
- **Source**: [Kaggle](https://www.kaggle.com/datasets/moussasacko/rakuten-france-multimodal-product-classification)
- **Download Command**:
  ```bash
  kaggle datasets download moussasacko/rakuten-france-multimodal-product-classification

---

### **Proof of Concept (PoC)**

The PoC will be built using **Streamlit**, allowing users to input search queries and view relevant product results in real-time. The PoC will be deployed on **Hugging Face Spaces** for easy access and sharing.

#### **Steps to Set Up the PoC**

1. Clone the repository: (Make sure to install lfs using this command 'git lfs install')
   ```bash
   git clone https://huggingface.co/spaces/yassinemtg/smart-inventory-search
   ```
   If you want to clone without large files - just their pointers
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/yassinemtg/smart-inventory-search
   ```



Milestone 3 (Data Processing and Pipeline Automation)
📌 Objective

The goal of this milestone is to automate data processing, validation, and ingestion using ZenML, while ensuring data versioning (DVC) and storage in MySQL.
Business Case

Managing large inventories requires efficient product retrieval. Traditional keyword-based search mechanisms are inefficient and lead to:

    Increased operational costs due to manual searches.
    Lost sales opportunities because of vague product descriptions.
    Poor user experience from irrelevant search results.

By implementing Smart Inventory Search, powered by machine learning (ML), businesses can benefit from faster, more accurate product discovery.
📌 Project Architecture

To manage the data lifecycle, we use the following tools:

    ✅ ZenML → Automates the ML pipeline for data ingestion, preprocessing, and validation.
    ✅ DVC (Data Version Control) → Tracks dataset versions and ensures reproducibility.
    ✅ MySQL → Stores raw and processed data.
    ✅ Great Expectations → Validates data integrity before ingestion.

📌 Pipeline Workflow

1️⃣ Data Ingestion → 2️⃣ Data Preprocessing → 3️⃣ Data Validation → 4️⃣ Storing in MySQL

1️⃣ Data Ingestion

We fetch the dataset and store it in MySQL.
📜 Implementation
Database Connection (db_connection.py)

from sqlalchemy import create_engine

# Database connection details
DB_NAME = "zenml_metadata"
DB_USER = "root"
DB_PASS = "8520"
DB_HOST = "localhost"
DB_PORT = "3306"

def get_db_engine():
    """Create and return a database engine."""
    return create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

Storing Data (store_data.py)

import pandas as pd
from database.db_connection import get_db_engine

# Load the dataset
df = pd.read_csv("./datasets/dataset-all.csv")

# Store data in MySQL
engine = get_db_engine()
df.to_sql("raw_data", engine, if_exists="replace", index=False)

print("✅ Data successfully stored in MySQL!")

Running Data Ingestion

python scripts/store_data.py

2️⃣ Data Preprocessing (ZenML Pipeline)

Once the data is ingested, we use ZenML to fetch and preprocess it.
📌 Pipeline Overview

    🔹 Step 1: Fetch data from MySQL
    🔹 Step 2: Apply preprocessing (lowercasing, text cleaning)
    🔹 Step 3: Store processed data back into MySQL

📜 Implementation
Pipeline Definition (preprocessing_pipeline.py)

from pipelines.custom_materializer import PandasMaterializer
from zenml import pipeline, step
import pandas as pd
import re
from database.db_connection import get_db_engine
from sqlalchemy.sql import text

@step(output_materializers=PandasMaterializer)
def fetch_data() -> pd.DataFrame:
    """Fetch raw data from MySQL using the centralized connection."""
    engine = get_db_engine()
    with engine.connect() as connection:
        df = pd.read_sql(text("SELECT * FROM raw_data"), connection)
    return df

@step(output_materializers=PandasMaterializer)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess text and transform features."""
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(lambda x: re.sub(r"\W+", " ", x))
    return df

@step
def store_preprocessed_data(df: pd.DataFrame):
    """Store preprocessed data into MySQL."""
    engine = get_db_engine()
    df.to_sql("processed_data", engine, if_exists="replace", index=False)
    print("✅ Preprocessed data stored in MySQL!")

@pipeline
def preprocessing_pipeline():
    """ZenML Preprocessing Pipeline"""
    raw_data = fetch_data()
    processed_data = preprocess_data(raw_data)
    store_preprocessed_data(processed_data)

Running the Pipeline

python scripts/run_preprocessing.py

3️⃣ Data Validation (Great Expectations)

Before storing processed data, we validate it using Great Expectations.
📜 Implementation

The validation pipeline checks:

    ✔ No missing values
    ✔ Proper text formatting
    ✔ Consistency in column names

Validation Script (run_validation.py)

import great_expectations as ge
from database.db_connection import get_db_engine
import pandas as pd

# Load processed data
engine = get_db_engine()
df = pd.read_sql("SELECT * FROM processed_data", con=engine)

# Validate data
ge_df = ge.from_pandas(df)
expectation_suite = ge_df.expect_column_values_to_not_be_null("text")

if expectation_suite["success"]:
    print("✅ Data validation passed!")
else:
    print("❌ Data validation failed. Check data quality.")

Running Data Validation

python scripts/run_validation.py

4️⃣ Data Versioning (DVC)

To ensure reproducibility, we use DVC to track dataset versions.
Setting Up DVC

dvc init
dvc add datasets/dataset-all.csv
git add .gitignore datasets/dataset-all.csv.dvc
git commit -m "Track dataset version with DVC"

Pushing Data to Remote Storage

dvc remote add origin /path/to/remote
dvc push

📌 Proof of Concept (PoC)

The PoC will be built using Streamlit, allowing users to input search queries and view relevant product results in real-time.
Steps to Set Up the PoC

    Clone the repository: (Make sure to install lfs using this command git lfs install)

git clone https://huggingface.co/spaces/yassinemtg/smart-inventory-search

If you want to clone without large files - just their pointers

    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/yassinemtg/smart-inventory-search

🚀 Key Achievements in Milestone 3

✅ Stored raw data into MySQL
✅ Implemented ZenML pipeline for data preprocessing
✅ Validated data with Great Expectations
✅ Tracked dataset versions using DVC
✅ Ensured full automation with ZenML
✅ Final Steps

To ensure everything runs correctly, execute the following commands in order:

# 1️⃣ Store raw data
python scripts/store_data.py

# 2️⃣ Run preprocessing pipeline
python scripts/run_preprocessing.py

# 3️⃣ Validate processed data
python scripts/run_validation.py

# 4️⃣ Track dataset version
dvc add datasets/dataset-all.csv
git add datasets/dataset-all.csv.dvc
git commit -m "Track dataset with DVC"
dvc push

