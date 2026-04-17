import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

st.title("Image Classifier")
st.write("Upload an image, and the app will classify it.")

model = MobileNetV2(weights="imagenet")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.warning("Please upload an image to classify.")
else:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")

        image = image.resize((224, 224))  
        image_array = np.array(image)
        if image_array.shape[-1] == 4: 
            image_array = image_array[..., :3]
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)  


        predictions = model.predict(image_array)


        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
     
        st.write(f"Decoded predictions: {decoded_predictions}")  # 

       
        st.write("### Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}. **{label}** - Confidence: {score:.2f}")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
