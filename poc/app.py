import streamlit as st
import pandas as pd
from PIL import Image
import os

# Load the dataset
@st.cache_data
def load_data():
    data_path = "./datasets/rakuten-france-multimodal-product-classification/X_train_update.csv"  # Update with the correct path
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
                #st.write(f"**Category:** {row['category'] if 'category' in row else 'N/A'}")  # Adjusted in case 'category' doesn't exist
                st.write(f"**Description:** {row['description']}")

                # Display product image
                image_name = f"image_{row['imageid']}_product_{row['productid']}.jpg"  # Adjusted image name format
                image_path = os.path.join("./datasets/rakuten-france-multimodal-product-classification/images/image_train", image_name)
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption=row["designation"], width=300)  # Changed from 'product_name' to 'designation'
                else:
                    st.write("Image not found.")
        else:
            st.write("No results found.")

if __name__ == "__main__":
    main()
