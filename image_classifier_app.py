import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

# Title of the App
st.title("Image Classifier! 📸")
st.write("Upload an image, and the app will classify it using a pre-trained MobileNetV2 model.")

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Error Handling: No file uploaded
if uploaded_file is None:
    st.warning("Please upload an image to classify.")
else:
    try:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")

        # Preprocess the image
        image = image.resize((224, 224))  # MobileNetV2 input size
        image_array = np.array(image)  # Convert to numpy array
        if image_array.shape[-1] == 4:  # Convert RGBA to RGB
            image_array = image_array[..., :3]
        image_array = preprocess_input(image_array)  # Preprocess for MobileNetV2
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image_array)

        # Decode predictions and display them
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Debugging: Check the decoded predictions output
        st.write(f"Decoded predictions: {decoded_predictions}")  # This shows the structure of predictions

        # Display predictions
        st.write("### Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}. **{label}** - Confidence: {score:.2f}")

    except Exception as e:
        st.error(f"Error processing the image: {e}")