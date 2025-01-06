import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Function to prepare classification data
def prepare_classification_data(electricity, weather, clusters):
    # Assign cluster labels to the consumption data
    electricity['cluster'] = clusters

    # Ensure the time columns have the same format
    electricity['time'] = pd.to_datetime(electricity['time']).dt.tz_localize(None)
    weather['time'] = pd.to_datetime(weather['time']).dt.tz_localize(None)

    # Merge with weather data
    data = pd.merge(electricity, weather, on=['postalcode', 'time'], how='left')

    # Select relevant features and the target label
    features = ['airtemperature', 'relativehumidity', 'ghi', 'sunelevation', 'sunazimuth', 'windspeed']
    X = data[features]
    y = data['cluster']

    # Handle missing values
    X = X.fillna(X.mean())

    return X, y

# Function to train the classification model
def train_classifier(X, y):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return classifier

if __name__ == "__main__":
    # Load data (this should integrate with the main data loading module)
    electricity = pd.read_parquet('all_data/electricity_consumption.parquet')
    weather = pd.read_parquet('all_data/weather.parquet')

    # Placeholder for clusters (simulation for testing)
    import numpy as np
    np.random.seed(42)
    clusters = np.random.randint(0, 3, size=len(electricity))  # Replace with actual clustering results

    # Prepare the data
    X, y = prepare_classification_data(electricity, weather, clusters)

    # Train the model
    classifier = train_classifier(X, y)
