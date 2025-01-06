import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to prepare the data for clustering
def prepare_clustering_data(electricity):
    # Group consumption by postal code and hour of the day
    daily_patterns = electricity.groupby(['postalcode', electricity['time'].dt.hour])['consumption'].mean().unstack()
    daily_patterns.fillna(0, inplace=True)  # Replace missing values with 0
    return daily_patterns

# Function to apply clustering
def perform_clustering(data, n_clusters=3):
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Return results
    return clusters, kmeans

# Function to visualize clustering results
def visualize_clusters(data, clusters):
    plt.figure(figsize=(10, 6))
    for cluster in np.unique(clusters):
        cluster_data = data[clusters == cluster].mean(axis=0)
        plt.plot(cluster_data, label=f'Cluster {cluster}')
    plt.title('Average daily patterns by cluster')
    plt.xlabel('Hour of the day')
    plt.ylabel('Average consumption (kWh)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load data (this should integrate with the main data loading module)
    electricity = pd.read_parquet('all_data/electricity_consumption.parquet')

    # Prepare the data
    clustering_data = prepare_clustering_data(electricity)

    # Perform clustering
    clusters, kmeans_model = perform_clustering(clustering_data, n_clusters=4)

    # Visualize results
    visualize_clusters(clustering_data, clusters)
