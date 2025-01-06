import pandas as pd
from clustering import prepare_clustering_data, perform_clustering, visualize_clusters
from classification import prepare_classification_data, train_classifier
from forecasting import prepare_forecasting_data, train_forecasting_model
import joblib

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    electricity = pd.read_parquet('all_data/electricity_consumption.parquet')
    weather = pd.read_parquet('all_data/weather.parquet')
    socioeconomic = pd.read_parquet('all_data/socioeconomic.parquet')

    # TASK 1: Clustering
    print("Performing clustering...")
    electricity['hour'] = electricity['time'].dt.hour
    clustering_data = prepare_clustering_data(electricity)

    # Ensure that the postalcode and hour columns are present
    clustering_data = clustering_data.reset_index()  # Reset index to recover key columns
    if 'postalcode' not in clustering_data.columns:
        clustering_data['postalcode'] = electricity['postalcode']
    if 'hour' not in clustering_data.columns:
        clustering_data['hour'] = electricity['hour']

    # Check columns before dropping
    drop_columns = [col for col in ['postalcode', 'hour'] if col in clustering_data.columns]
    clusters, kmeans_model = perform_clustering(clustering_data.drop(columns=drop_columns), n_clusters=4)

    # Map clusters to the original dataset
    clustering_data['cluster'] = clusters
    electricity = pd.merge(electricity, clustering_data[['postalcode', 'hour', 'cluster']],
                           on=['postalcode', 'hour'], how='left')

    visualize_clusters(clustering_data.drop(columns=drop_columns).values, clusters)
    joblib.dump(kmeans_model, 'models/kmeans_model.pkl')
    print("Clustering completed and model saved as 'models/kmeans_model.pkl'.")

    # TASK 2: Classification
    print("Performing classification...")
    # Handle NaN values in the 'cluster' column
    electricity['cluster'] = electricity['cluster'].fillna(-1)  # Assign a default value for unassigned clusters

    X_classification, y_classification = prepare_classification_data(electricity, weather, electricity['cluster'])
    # Remove rows with NaN in the classification labels
    valid_idx = ~y_classification.isna()
    X_classification = X_classification[valid_idx]
    y_classification = y_classification[valid_idx]

    classification_model = train_classifier(X_classification, y_classification)
    joblib.dump(classification_model, 'models/classification_model.pkl')
    print("Classification completed and model saved as 'models/classification_model.pkl'.")

    # TASK 3: Forecasting
    print("Performing forecasting...")
    X_forecasting, y_forecasting = prepare_forecasting_data(electricity, weather, socioeconomic)
    forecasting_model, X_test, y_test = train_forecasting_model(X_forecasting, y_forecasting)
    joblib.dump(forecasting_model, 'models/forecasting_model.pkl')
    print("Forecasting completed and model saved as 'models/forecasting_model.pkl'.")

    print("Execution completed.")
