

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('model.h5')  # Replace with the path to your trained model

# Streamlit App
st.title('Digit Recognition')
# Upload Image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    try:
        image = plt.imread(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        if image.shape[-1] == 3:  # Check if the image is RGB
            image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, (28, 28))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Ensure image has the correct shape (28x28x1)
        if image.shape == (28, 28, 1):
            image = tf.expand_dims(image, axis=0)  # Add batch dimension
        else:
            st.error("Image does not have the correct shape (28x28x1). Please upload a valid image.")
        
        # Make a prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)

        st.write(f"Model predicts the digit as: {predicted_class}")
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")

# Display model summary
st.subheader("Model Summary")
st.text(model.summary())
