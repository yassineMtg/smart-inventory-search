# Smart Inventory Search

---

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

## **Milestone 2: Proof of Concept (PoC)**

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
   
---


# Milestone 3 (Data Acquisition Validation and Preparation)

## **Introduction**

This milestone focuses on data acquisition, validation, and preparation to ensure high-quality data for machine learning model development. The entire pipeline is built using TensorFlow Extended (TFX) and integrates a feature store using TFX Transform.

The key components of this pipeline are:
1. **Data Ingestion** – Loading and storing raw data into a PostgreSQL database.
2. **Data Validation** – Ensuring consistency, correctness, and completeness of data.
3. **Data Preprocessing** – Applying text transformation and normalization techniques.
4. **Feature Store** – Managing features for real-time and batch ML applications.
5. **Data Versioning** – Tracking dataset changes for reproducibility.

Each of these steps plays a crucial role in ensuring a **robust and automated ML pipeline,**, allowing for seamless data tracking and transformation.

---

## Installation & Setup:

### Prerequisites:

- Python 3.9

- Conda (Miniconda or Anaconda)

- PostgreSQL (For metadata and raw data storage)

- TFX 1.15.0 (TensorFlow Extended for ML pipelines)

### Setting up the Environment:

Run the following commands to set up the environment:

```bash
# Step 1: Create and activate Conda environment
conda create --name smartsearch python=3.9 -y
conda activate smartsearch

# Step 2: Install required dependencies
pip install -r requirement.txt

# Step 3: Setup PostgreSQL connection
export DATABASE_URL='postgresql://yassine:8520@localhost:5432/tfx_metadata'
```

![TFX version](./images/tfx_version.png)

---

## Pipeline Execution:

### Step-by-step Execution:

To run the complete pipeline, execute the following command:

```bash
python pipeline.py
```

This script automates the execution of the following steps:

1. **Data Ingestion**

- Extracts raw data and stores it in PostgreSQL.

- Saves a local CSV copy under data/raw_data.csv.

2. **Data Validation**

- Generates statistics and schema validation.

- Stores the schema at data/schema/schema.pbtxt.

3. **Data Preprocessing**

- Applies text transformations (lowercasing, normalization).

- Stores transformed examples in tfx_pipeline/Transform/transformed_examples/.

4. **Feature Store**

- Loads transformed data from preprocessing.

- Stores processed features in tfx_pipeline/FeatureStore/transformed_features.parquet.

---

### **Step 1: Data Ingestion**

- Connects to PostgreSQL.
- Loads raw data from a CSV file.
- Stores data in PostgreSQL while handling versioning.

```python
from database.db_connection import get_db_engine
import pandas as pd

def store_raw_data(file_path: str, table_name="raw_data"):
    """Load raw dataset into PostgreSQL with version tracking."""
    engine = get_db_engine()
    df = pd.read_csv(file_path)

    with engine.connect() as conn:
        df.to_sql(table_name, con=conn, if_exists="replace", index=False)

store_raw_data("./datasets/dataset-all.csv")
```

---

Run the script with:

```bash
python scripts/data_ingestion.py
```

Output:

![TFX version](./images/data_ingestion.png)

---

### **Step 2: Data Validation**

- Validates schema (data types, missing values).
- Generates statistics to detect anomalies.
- Saves schema to data/schema/schema.pbtxt.

```python
import tensorflow_data_validation as tfdv

# Load raw data
data_path = "./data/raw_data.csv"
df = pd.read_csv(data_path)

# Compute statistics
stats = tfdv.generate_statistics_from_dataframe(df)

# Infer schema
schema = tfdv.infer_schema(stats)

# Save schema
tfdv.write_schema_text(schema, "./data/schema/schema.pbtxt")
```

---

Run the script with:

```bash
python scripts/validation_pipeline.py
```

Output:

![Validation output](./images/validation_output.png)

---

### **Step 3: Data Preprocessing**

- Cleans text (lowercasing, removing special characters).
- Normalizes labels for ML model training.
- Stores transformed data in tfx_pipeline/Transform/transformed_examples/.

```python
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Applies transformations to input features."""
    outputs = {}
    outputs['text'] = tf.strings.lower(inputs['text'])
    outputs['labels'] = tft.scale_to_0_1(tf.cast(inputs['labels'], tf.float32))
    return outputs
```

---

Run the script with:

```bash
python 
```

Output:

![Preprocessing output](./images/preprocessing.png)
![Preprocessing output](./images/preprocessing2.png)

---

### **Step 4: Feature Store**

- Retrieves preprocessed features using TFX Transform.
- Stores transformed data in FeatureStore/transformed_features.parquet.

```python
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.saved.saved_transform_io as tft_io

TRANSFORM_FN_PATH = "tfx_pipeline/Transform/transform_graph/42"
loaded_model = tft_io.load_transform_fn(TRANSFORM_FN_PATH)

# Example input
example_input = {
    "text": tf.constant([["Example text for transformation"]], dtype=tf.string),
    "labels": tf.constant([[1]], dtype=tf.int64)
}

# Apply transformation
transformed_output = loaded_model.transform_features_layer()(example_input)

# Convert to Pandas DataFrame & Save
df = pd.DataFrame.from_dict({k: v.numpy().tolist() for k, v in transformed_output.items()})
df.to_parquet("tfx_pipeline/FeatureStore/transformed_features.parquet")
```

---

Run the script with:

```bash
python scripts/feature_store_pipeline.py
```

Output:

![Feature Store Output](./images/feature_store.png)

---

### **Step 5: Data Versioning**

- Tracks dataset versions in PostgreSQL.
- Ensures reproducibility for future training.

```sql
-- Table for version tracking
CREATE TABLE IF NOT EXISTS dataset_versioning (
    version_id SERIAL PRIMARY KEY,
    dataset_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### **Step 6: Full Pipeline Execution**

- Runs all steps automatically.
- Checks if the Feature Store exists before running preprocessing.
- Uses TFX to orchestrate pipeline steps.

```python
import os

if not os.path.exists("tfx_pipeline/FeatureStore/transformed_features.parquet"):
    os.system("python scripts/preprocessing_pipeline.py")

os.system("python scripts/data_ingestion.py")
os.system("python scripts/validation_pipeline.py")
os.system("python scripts/feature_store_pipeline.py")
```

---

Run the script with:

```bash
python pipeline.py
```

Output:

![Whole Pipeline Output](./images/pipeline_output.png)

---
