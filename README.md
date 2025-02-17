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

To establish feasibility, we use a pre-trained NLP model as a baseline and fine-tune it on domain-specific inventory queries. The combination of **DistilBERT** and **Gemini API** ensures an optimal balance between accuracy, speed, and contextual understanding.

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

To evaluate the success of the Smart Inventory Search system, we will use the following metrics:

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
