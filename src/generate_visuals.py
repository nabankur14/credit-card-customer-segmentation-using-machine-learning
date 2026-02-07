import os
import sys
import matplotlib.pyplot as plt

# Add current directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.modeling import train_kmeans, train_hierarchical
from src.evaluation import plot_elbow_method, plot_silhouette_analysis

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'raw', 'Credit_Card_Customer_Data.xlsx')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    # Ensure visuals directory exists
    os.makedirs(visuals_dir, exist_ok=True)
    
    print("Loading data...")
    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print("Preprocessing data...")
    df_scaled, df_original = preprocess_data(df)
    
    print("Generating Elbow Plot...")
    plot_elbow_method(df_scaled, out_path=os.path.join(visuals_dir, 'elbow_curve.png'))
    
    print("Generating Silhouette Plot (k=3)...")
    plot_silhouette_analysis(df_scaled, k=3, out_path=os.path.join(visuals_dir, 'silhouette_plot.png'))
    
    # K-Means
    print("Training K-Means (k=3)...")
    kmeans, labels = train_kmeans(df_scaled, k=3)
    
    # You might want to plot clusters here if it was 2D, but it's high dimensional.
    # Keep it simple for now, focusing on evaluation metrics plots which are most important.
    
    print("Visuals generated successfully in 'visuals/' directory.")

if __name__ == "__main__":
    main()
