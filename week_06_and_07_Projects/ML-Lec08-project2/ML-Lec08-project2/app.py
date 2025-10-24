import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import hog

model = joblib.load("best_model.pkl")

classes = {0: "Car", 1: "Cricket Ball", 2: "Ice Cream"}

# preprocess the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100, 100))
    features, _ = hog(image, visualize=True)  # Extract HOG features
    return features.reshape(1, -1)

# Streamlit UI
st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="centered")
st.markdown("""
    <h2 style='text-align: center; color: #3366ff;'>🖼️ Image Classification App</h2>
    <p style='text-align: center;'>Upload an image and click the button to predict its class.</p>
    <hr>
""", unsafe_allow_html=True)

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Predict Class"):
        # Preprocess and classify the image
        features = preprocess_image(image)
        
        # Check feature shape before prediction
        if features.shape[1] != 8100:
            st.error(f"Feature shape mismatch! Expected (1, 8100), got {features.shape}")
        else:
            prediction = model.predict(features)
            predicted_class = classes[prediction[0]]
            
            st.success(f"🎯 Prediction: **{predicted_class}**")
