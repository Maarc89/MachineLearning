import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Function to prepare data for forecasting
def prepare_forecasting_data(electricity, weather, socioeconomic):
    # Normalize the time columns
    electricity['time'] = pd.to_datetime(electricity['time']).dt.tz_localize(None)
    weather['time'] = pd.to_datetime(weather['time'])

    # Extract the year from the 'time' column in electricity
    electricity['year'] = electricity['time'].dt.year

    # Merge datasets by 'postalcode' and 'time'
    data = pd.merge(electricity, weather, on=['postalcode', 'time'], how='left')

    # Merge socioeconomic data by 'postalcode' and 'year'
    data = pd.merge(data, socioeconomic, on=['postalcode', 'year'], how='left')

    # Select relevant features
    features = ['airtemperature', 'relativehumidity', 'ghi', 'sunelevation',
                'windspeed', 'population', 'incomesperhousehold']
    target = 'consumption'

    # Create feature and target datasets
    X = data[features]
    y = data[target]

    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(0)

    return X, y

# Function to train and evaluate forecasting model
def train_forecasting_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    mae_scores = []
    rmse_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

    print("Forecasting model results:")
    print(f"Average MAE: {np.mean(mae_scores):.2f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f}")

    return model, X_test, y_test

if __name__ == "__main__":
    # Load data (this should integrate with the main data loading module)
    electricity = pd.read_parquet('all_data/electricity_consumption.parquet')
    weather = pd.read_parquet('all_data/weather.parquet')
    socioeconomic = pd.read_parquet('all_data/socioeconomic.parquet')

    # Prepare the data
    X, y = prepare_forecasting_data(electricity, weather, socioeconomic)

    # Train the model
    forecasting_model, X_test, y_test = train_forecasting_model(X, y)

    # Save the trained model
    joblib.dump(forecasting_model, 'forecasting_model.pkl')
    print("Model saved as 'forecasting_model.pkl'.")

    # Make predictions with the test set
    y_pred = forecasting_model.predict(X_test)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual values')
    plt.plot(y_pred, label='Predictions')
    plt.legend()
    plt.title('Predictions vs. Actual Values')
    plt.show()
