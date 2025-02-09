import streamlit as st
from PIL import Image
import numpy as np
import os
# import cv2
from ultralytics import YOLO
from streamlit_option_menu import option_menu

# Load YOLO model
MODEL_PATH = "Model/best.pt"  # Adjust this to your model path
CLASS_FILE = "items.txt"  # Adjust this to your class file
with open(CLASS_FILE, "r") as f:
    class_list = [line.strip() for line in f if line.strip()]
model = YOLO(MODEL_PATH)

# Sidebar menu for navigation
# with st.sidebar:
#     selected = st.radio(
#         "Navigation", ["Home", "About", "Information"]
#     )

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "About", "Information"],
        icons=["house", "info-circle", "book"],
        menu_icon="cast",
        default_index=0,
    )

# Function to detect accidents using YOLO
def detect_accident(image_path):
    results = model(image_path)
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

# Home Page
if selected == "Home":
    st.title("Aircraft Crack Detection using YOLOv8 ")
    st.write("Upload an image to detect Aircraft Crack .")

    # Upload image
    upload = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if upload:
        image_path = os.path.join("uploads", upload.name)
        os.makedirs("uploads", exist_ok=True)

        # Save uploaded image
        with open(image_path, "wb") as f:
            f.write(upload.getbuffer())
        img = Image.open(image_path)

        # Display uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Detect accidents
        annotated_image, detections = detect_accident(image_path)

        # Display results
        st.image(annotated_image, caption="Detected Image", use_column_width=True)
        if detections:
            st.write("### Detected Objects")
            st.dataframe(detections)
        else:
            st.write("No Aircraft Crack detected.")

# About Page
elif selected == "About":
    st.title("About")
    st.write("""
    This application detectsAircraft Crack Detection using YOLOv8. 
    Upload an image, and the system will highlight any food-related objects it detects.
    """)

# Information Page
elif selected == "Information":
    st.title("Information")
    st.write("### Supported Classes")
    st.write(", ".join(class_list))
    st.write("### YOLO Model")
    st.write("The YOLO model used in this application is trained on custom data.")
    st.write("### Source Code")
    st.write("You can find the source code on [GitHub](https://github.com/subramanyamrekhandar/Road-Accident-Detection-By-using-AI.git).")

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Created by [Subramanyam Rekhandar](https://www.linkedin.com/in/subramanyamrekhandar/).
    """
)
