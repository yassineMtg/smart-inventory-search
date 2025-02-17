import os
import streamlit as st
import pandas as pd
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi
import json

# Authenticate with Kaggle API (using secrets stored in Hugging Face)
kaggle_json = os.getenv('KAGGLE_JSON')
if kaggle_json:
    # Parse the Kaggle JSON credentials stored in the Hugging Face secret
    credentials = json.loads(kaggle_json)
    # Write the credentials to a kaggle.json file
    with open('kaggle.json', 'w') as f:
        json.dump(credentials, f)
    os.environ['KAGGLE_CONFIG_DIR'] = './'  # Set the directory where kaggle.json is located
else:
    st.error("Kaggle API credentials not found")

# Function to download dataset from Kaggle
def download_dataset():
    dataset_name = 'moussasacko/rakuten-france-multimodal-product-classification'  # Correct dataset identifier
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')

    api = KaggleApi()
    api.authenticate()  # Authenticate with the Kaggle API

    api.dataset_download_files(dataset_name, path='./datasets', unzip=True)

# Load the dataset
@st.cache_data
def load_data():
    data_path = "./datasets/X_train_update.csv"  # Update with the correct path to your data
    return pd.read_csv(data_path)

# Search function
def search_products(query, data):
    # Handle NaN values in the 'description' column
    data["description"] = data["description"].fillna("")  # Replace NaN with empty strings
    
    # Perform the search
    query = query.lower()
    results = data[data["description"].str.lower().str.contains(query)]
    return results

# Streamlit app
def main():
    st.title("Smart Inventory Search")
    st.write("Enter a search query to find relevant products.")

    # Check if dataset is available, else download
    if not os.path.exists("./datasets"):
        st.write("Downloading dataset from Kaggle...")
        download_dataset()

    # Load data
    data = load_data()

    # Input: Search query
    query = st.text_input("Search for a product:")

    if query:
        # Perform search
        results = search_products(query, data)

        if not results.empty:
            st.write(f"Found {len(results)} results:")
            for _, row in results.iterrows():
                st.subheader(row["designation"])  # Changed from 'product_name' to 'designation'
                st.write(f"**Description:** {row['description']}")

                # Display product image
                image_name = f"image_{row['imageid']}_product_{row['productid']}.jpg"  # Adjusted image name format
                image_path = os.path.join("./datasets/images/images/image_train", image_name)
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption=row["designation"], width=300)  # Changed from 'product_name' to 'designation'
                else:
                    st.write("Image not found.")
        else:
            st.write("No results found.")

if __name__ == "__main__":
    main()
