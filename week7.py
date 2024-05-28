import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

# Load the dataset
dataset = load_iris()

# Prepare the DataFrame
X = pd.DataFrame(dataset.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(dataset.target, columns=['Targets'])

# Define colormap
colormap = np.array(['red', 'lime', 'black'])

# Streamlit app title
st.title("Iris Dataset Clustering")

# Sidebar for dataset info
st.sidebar.subheader("Dataset Info")
st.sidebar.write("Number of samples:", X.shape[0])
st.sidebar.write("Number of features:", X.shape[1])
st.sidebar.write("Number of classes:", len(np.unique(y)))

# Plotting method
def plot_clusters():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Real Plot
    axes[0].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
    axes[0].set_title('Real')

    # KMeans Plot
    model_kmeans = KMeans(n_clusters=3)
    model_kmeans.fit(X)
    predY_kmeans = np.choose(model_kmeans.labels_, [0, 1, 2]).astype(np.int64)
    axes[1].scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY_kmeans], s=40)
    axes[1].set_title('KMeans')

    # GMM Plot
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    xsa = scaler.transform(X)
    xs = pd.DataFrame(xsa, columns=X.columns)
    gmm = GaussianMixture(n_components=3)
    gmm.fit(xs)
    y_cluster_gmm = gmm.predict(xs)
    axes[2].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
    axes[2].set_title('GMM Classification')

    # Set common labels
    for ax in axes:
        ax.set_xlabel('Petal Length')
        ax.set_ylabel('Petal Width')

    return fig

# Display plots in Streamlit
st.pyplot(plot_clusters())

