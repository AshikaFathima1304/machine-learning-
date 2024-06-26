import streamlit as st
import pandas as pd
import numpy as np

# Iris dataset
iris_data = {
    'Sepal_Length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1],
    'Sepal_Width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8],
    'Petal_Length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5],
    'Petal_Width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3],
    'Targets': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

# Create DataFrame
iris_df = pd.DataFrame(iris_data)

# Define colormap
colormap = np.array(['red', 'lime', 'black'])

# Streamlit app title
st.title("Iris Dataset Clustering")

# Sidebar for dataset info
st.sidebar.subheader("Dataset Info")
st.sidebar.write("Number of samples:", len(iris_df))
st.sidebar.write("Number of features:", iris_df.shape[1] - 1)
st.sidebar.write("Number of classes:", len(np.unique(iris_df['Targets'])))

# Display the dataset
st.write("Iris Dataset:", iris_df)

# Display placeholders for clustering results
st.write("KMeans Clustering Results:")
st.dataframe(iris_df[['Petal_Length', 'Petal_Width']])
st.write("GMM Clustering Results:")
st.dataframe(iris_df[['Petal_Length', 'Petal_Width']])
