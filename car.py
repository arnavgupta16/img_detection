import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt


# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Streamlit app
st.title("Image Detection App")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for model input
    image = Image.open(uploaded_image)
    image = image.resize((224, 224))  # Resize image to match input size of ResNet50
    image = np.array(image)
    image = preprocess_input(image)
    
    # Make prediction using the model
    prediction = model.predict(np.expand_dims(image, axis=0))
    
    # Decode prediction into human-readable labels
    decoded_predictions = decode_predictions(prediction, top=3)[0]
    
    # Display prediction results
    st.subheader("Prediction:")
    for _, label, score in decoded_predictions:
        st.write(f"{label}: {score:.2f}")

   