import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans

def plot_elbow_method(data, k_range=(2, 10), out_path=None):
    """
    Plot Elbow Method using Yellowbrick.
    
    Args:
        data (pd.DataFrame): Scaled data.
        k_range (tuple): Range of k to test.
        out_path (str): Path to save the plot.
    """
    model = KMeans(random_state=1)
    visualizer = KElbowVisualizer(model, k=k_range, timings=False)
    visualizer.fit(data)
    if out_path:
        visualizer.show(outpath=out_path)
    else:
        visualizer.show()
    plt.close()

def plot_silhouette_analysis(data, k, out_path=None):
    """
    Plot Silhouette Analysis for a specific k using Yellowbrick.
    
    Args:
        data (pd.DataFrame): Scaled data.
        k (int): Number of clusters.
        out_path (str): Path to save the plot.
    """
    model = KMeans(n_clusters=k, random_state=1)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(data)
    if out_path:
        visualizer.show(outpath=out_path)
    else:
        visualizer.show()
    plt.close()

def calculate_silhouette_score(data, labels):
    """
    Calculate average silhouette score.
    """
    return silhouette_score(data, labels)
