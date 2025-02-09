# predictive_model.py - Train the Predictive Maintenance Model (Random Forest)

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simulate synthetic sensor data
def generate_synthetic_data(num_samples=1000):
    """
    Function to generate synthetic data for training.
    In production, this will come from real sensor data sources.
    """
    timestamps = pd.date_range('2023-01-01', periods=num_samples, freq='T')
    vibration = np.random.normal(0.5, 0.1, num_samples)  # Vibration data
    temperature = np.random.normal(75, 5, num_samples)  # Temperature data
    pressure = np.random.normal(30, 3, num_samples)  # Pressure data
    failure = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])  # Random failures (5%)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'vibration': vibration,
        'temperature': temperature,
        'pressure': pressure,
        'failure': failure
    })

    # Feature engineering: Compute rolling mean and standard deviation for vibration
    data['mean_vibration'] = data['vibration'].rolling(window=10).mean()
    data['std_vibration'] = data['vibration'].rolling(window=10).std()

    # Compute temperature and pressure differences
    data['temp_difference'] = data['temperature'].diff()
    data['pressure_diff'] = data['pressure'].diff()

    data.dropna(inplace=True)  # Drop rows with NaN values (after rolling window)
    
    return data

# Generate synthetic data
data = generate_synthetic_data()

# Prepare the feature set (X) and target (y)
X = data[['mean_vibration', 'std_vibration', 'temp_difference', 'pressure_diff']]
y = data['failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model as a pickle file
joblib.dump(model, 'predictive_model.pkl')

print("Model trained and saved as 'predictive_model.pkl'.")
