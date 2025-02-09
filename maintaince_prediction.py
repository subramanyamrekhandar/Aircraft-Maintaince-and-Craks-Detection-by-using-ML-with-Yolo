# predictive_maintenance_streamlit.py - Standalone Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle

# Set up the page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Generate synthetic data to simulate a pre-trained model (if model file doesn't exist)
def train_and_save_model():
    X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    with open('predictive_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Check if model exists, else create and save one
try:
    with open('predictive_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    train_and_save_model()
    with open('predictive_model.pkl', 'rb') as f:
        model = pickle.load(f)

# Generate synthetic sensor data
def generate_fake_data():
    """
    Simulate sensor data for vibration, temperature, and pressure.
    """
    return {
        'vibration': random.uniform(0.4, 0.8),
        'temperature': random.uniform(50, 100),
        'pressure': random.uniform(25, 35)
    }

# Predict maintenance needs
def predict_maintenance(mean_vibration, std_vibration, temp_difference, pressure_diff):
    """
    Predict if maintenance is required based on sensor data.
    """
    features = np.array([mean_vibration, std_vibration, temp_difference, pressure_diff]).reshape(1, -1)
    prediction = model.predict(features)
    return bool(prediction[0])

# Streamlit UI Layout
st.title("Predictive Maintenance Dashboard")
st.write("Real-Time Monitoring of Aircraft Sensors with Predictive Maintenance Insights")

# Real-time sensor data display
with st.container():
    st.subheader("Real-Time Sensor Data")
    col1, col2, col3 = st.columns(3)
    vibration_display = col1.metric("Vibration (g)", "Fetching...")
    temperature_display = col2.metric("Temperature (°C)", "Fetching...")
    pressure_display = col3.metric("Pressure (psi)", "Fetching...")

# Real-time graph
st.subheader("Real-Time Sensor Data Graph")
graph_placeholder = st.empty()

# Maintenance prediction status
st.subheader("Maintenance Prediction")
prediction_placeholder = st.empty()

# Main loop for real-time updates
sensor_data_history = {'time': [], 'vibration': [], 'temperature': [], 'pressure': []}

# Loop for real-time updates
for _ in range(100):  # Limit iterations to prevent infinite loop in Streamlit
    # Generate sensor data
    sensor_data = generate_fake_data()
    vibration = sensor_data['vibration']
    temperature = sensor_data['temperature']
    pressure = sensor_data['pressure']

    # Update metrics in the UI
    vibration_display.metric("Vibration (g)", f"{vibration:.2f}")
    temperature_display.metric("Temperature (°C)", f"{temperature:.2f}")
    pressure_display.metric("Pressure (psi)", f"{pressure:.2f}")

    # Prepare prediction input
    prediction_data = {
        'mean_vibration': vibration,
        'std_vibration': random.uniform(0.05, 0.1),  # Simulated std deviation
        'temp_difference': random.uniform(0.5, 1.5),  # Simulated temperature difference
        'pressure_diff': random.uniform(0.5, 1.0)  # Simulated pressure difference
    }

    # Predict maintenance need
    maintenance_needed = predict_maintenance(
        prediction_data['mean_vibration'],
        prediction_data['std_vibration'],
        prediction_data['temp_difference'],
        prediction_data['pressure_diff']
    )

    # Update prediction status in UI
    if maintenance_needed:
        prediction_placeholder.error("🚨 Maintenance Needed! ⚠️")
    else:
        prediction_placeholder.success("✅ No Maintenance Needed.")

    # Append to sensor data history
    current_time = time.time()
    sensor_data_history['time'].append(current_time)
    sensor_data_history['vibration'].append(vibration)
    sensor_data_history['temperature'].append(temperature)
    sensor_data_history['pressure'].append(pressure)

    # Update graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['vibration'], mode='lines+markers', name='Vibration'))
    fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['temperature'], mode='lines+markers', name='Temperature'))
    fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['pressure'], mode='lines+markers', name='Pressure'))
    graph_placeholder.plotly_chart(fig, use_container_width=True)

    # Pause for 1 second before the next update
    time.sleep(1)
