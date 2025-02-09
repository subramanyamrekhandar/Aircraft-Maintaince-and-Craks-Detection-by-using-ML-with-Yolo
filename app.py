import streamlit as st
from streamlit_option_menu import option_menu
import os
import time
from PIL import Image
from ultralytics import YOLO
import pickle
import random
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Load model for Crack Detection
MODEL_PATH = "Model/best.pt"  # Adjust this to your model path
CLASS_FILE = "items.txt"  # Adjust this to your class file

with open(CLASS_FILE, "r") as f:
    class_list = [line.strip() for line in f if line.strip()]

# Load YOLO model
crack_model = YOLO(MODEL_PATH)

# Load model for Maintenance Prediction
try:
    with open('predictive_model.pkl', 'rb') as f:
        maintenance_model = pickle.load(f)
except FileNotFoundError:
    st.error("Predictive Maintenance Model file not found. Please train the model.")

# Sidebar menu for navigation
with st.sidebar:
    selected_pipeline = option_menu(
        menu_title="Select Pipeline",
        options=["Crack Detection", "Maintenance Prediction"],
        icons=["search", "tools"],
        menu_icon="cast",
        default_index=0,
    )

# Crack Detection Logic
def detect_cracks(image_path):
    results = crack_model(image_path)
    annotated_image = results[0].plot()  # Annotated image as numpy array
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls.item())
        class_name = results[0].names[class_id]
        if class_name in class_list:
            detections.append({
                "Class": class_name,
                "Confidence": round(float(box.conf.item()), 2),
                "X_min": round(float(box.xyxy[0][0].item()), 2),
                "Y_min": round(float(box.xyxy[0][1].item()), 2),
                "X_max": round(float(box.xyxy[0][2].item()), 2),
                "Y_max": round(float(box.xyxy[0][3].item()), 2),
            })

    return annotated_image, detections

# Maintenance Prediction Logic
def predict_maintenance(mean_vibration, std_vibration, temp_difference, pressure_diff):
    """
    Predict if maintenance is required based on sensor data.
    """
    features = np.array([mean_vibration, std_vibration, temp_difference, pressure_diff]).reshape(1, -1)
    prediction = maintenance_model.predict(features)
    return bool(prediction[0])

# -------------------- CRACK DETECTION PIPELINE --------------------
if selected_pipeline == "Crack Detection":
    st.title("Aircraft Crack Detection")
    st.write("Upload an image to detect cracks in the aircraft.")

    upload = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if upload:
        image_path = os.path.join("uploads", upload.name)
        os.makedirs("uploads", exist_ok=True)

        with open(image_path, "wb") as f:
            f.write(upload.getbuffer())

        img = Image.open(image_path)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        annotated_image, detections = detect_cracks(image_path)

        st.image(annotated_image, caption="Detected Image", use_column_width=True)
        if detections:
            st.write("### Detected Objects")
            st.dataframe(detections)
        else:
            st.write("No cracks detected.")

# -------------------- MAINTENANCE PREDICTION PIPELINE --------------------
elif selected_pipeline == "Maintenance Prediction":
    st.title("Predictive Maintenance Dashboard")
    st.write("Real-Time Monitoring of Aircraft Sensors with Predictive Maintenance Insights")

    # Live sensor data metrics
    with st.container():
        st.subheader("Real-Time Sensor Data")
        col1, col2, col3 = st.columns(3)
        vibration_display = col1.metric("Vibration (g)", "Fetching...")
        temperature_display = col2.metric("Temperature (°C)", "Fetching...")
        pressure_display = col3.metric("Pressure (psi)", "Fetching...")

    # Placeholder for real-time graph
    st.subheader("Real-Time Sensor Data Graph")
    graph_placeholder = st.empty()

    # Store sensor data history
    sensor_data_history = {'time': [], 'vibration': [], 'temperature': [], 'pressure': []}

    # **Run the graph for a fixed duration (e.g., 30 seconds)**
    run_time = 30  # **Set duration (seconds)**
    start_time = time.time()

    while time.time() - start_time < run_time:
        vibration = random.uniform(0.4, 0.8)
        temperature = random.uniform(50, 100)
        pressure = random.uniform(25, 35)

        # Update sensor metrics first
        vibration_display.metric("Vibration (g)", f"{vibration:.2f}")
        temperature_display.metric("Temperature (°C)", f"{temperature:.2f}")
        pressure_display.metric("Pressure (psi)", f"{pressure:.2f}")

        # Store data for graph
        current_time = time.time()
        sensor_data_history['time'].append(current_time)
        sensor_data_history['vibration'].append(vibration)
        sensor_data_history['temperature'].append(temperature)
        sensor_data_history['pressure'].append(pressure)

        # Update the graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['vibration'], mode='lines+markers', name='Vibration'))
        fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['temperature'], mode='lines+markers', name='Temperature'))
        fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['pressure'], mode='lines+markers', name='Pressure'))
        graph_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(1)  # Wait before next update

    # **After graph stops, display the maintenance prediction**
    st.subheader("Maintenance Prediction")

    prediction_data = {
        'mean_vibration': random.uniform(0.4, 0.8),
        'std_vibration': random.uniform(0.05, 0.1),
        'temp_difference': random.uniform(0.5, 1.5),
        'pressure_diff': random.uniform(0.5, 1.0)
    }

    maintenance_needed = predict_maintenance(
        prediction_data['mean_vibration'],
        prediction_data['std_vibration'],
        prediction_data['temp_difference'],
        prediction_data['pressure_diff']
    )

    # Show maintenance message after the graph stops
    if maintenance_needed:
        st.error("🚨 Maintenance Needed! ⚠️")
    else:
        st.success("✅ No Maintenance Needed.")

# Sidebar Contact Info
st.sidebar.title("Contact")
st.sidebar.info(
    """
    Created by [Subramanyam Rekhandar](https://www.linkedin.com/in/subramanyamrekhandar/).
    """
)
