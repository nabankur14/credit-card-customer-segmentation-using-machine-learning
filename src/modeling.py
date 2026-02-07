from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def train_kmeans(data, k=3, random_state=1):
    """
    Train K-Means clustering model.
    
    Args:
        data (pd.DataFrame): Preprocessed data.
        k (int): Number of clusters.
        random_state (int): Seed for reproducibility.
        
    Returns:
        KMeans: Fitted model.
        np.array: Cluster labels.
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(data)
    return kmeans, kmeans.labels_

def train_hierarchical(data, method='ward', n_clusters=3):
    """
    Train Hierarchical Clustering model.
    
    Args:
        data (pd.DataFrame): Preprocessed data.
        method (str): Linkage method (e.g., 'ward').
        n_clusters (int): Number of clusters to cut.
        
    Returns:
        np.array: Cluster labels.
        np.array: Linkage matrix.
    """
    mergings = linkage(data, method=method)
    
    # We can perform Agglomerative Clustering to get labels directly consistent with sklearn API
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = hc.fit_predict(data)
    
    return labels, mergings
