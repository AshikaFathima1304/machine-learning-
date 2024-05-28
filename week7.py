import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Plotting method
def plot_clusters():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Real Plot
    axes[0].scatter(iris_df['Petal_Length'], iris_df['Petal_Width'], c=colormap[iris_df['Targets']], s=40)
    axes[0].set_title('Real')

    # KMeans Plot
    predY_kmeans = [0] * len(iris_df)
    axes[1].scatter(iris_df['Petal_Length'], iris_df['Petal_Width'], c=colormap[predY_kmeans], s=40)
    axes[1].set_title('KMeans')

    # GMM Plot
    y_cluster_gmm = [0] * len(iris_df)
    axes[2].scatter(iris_df['Petal_Length'], iris_df['Petal_Width'], c=colormap[y_cluster_gmm], s=40)
    axes[2].set_title('GMM Classification')

    # Set common labels
    for ax in axes:
        ax.set_xlabel('Petal Length')
        ax.set_ylabel('Petal Width')

    return fig

# Display plots in Streamlit
st.pyplot(plot_clusters())
