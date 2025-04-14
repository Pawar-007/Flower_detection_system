import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling


st.header("Image Classification with Keras")

flower_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Load the model with custom layers
model = load_model('Flower_Recog_model.h5', custom_objects={
    'RandomFlip': RandomFlip,
    'RandomRotation': RandomRotation,
    'RandomZoom': RandomZoom,
    'Rescaling': Rescaling
})

def classify_image(image_path):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)  # Adds batch dimension

        prediction = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(prediction[0])
        outcome = f"The image belongs to {flower_names[np.argmax(result)]} with a confidence of {np.max(result) * 100:.2f}%"
        return outcome
    except Exception as e:
        return f"Error in classification: {str(e)}"

# Temp directory handling
os.makedirs("tempDir", exist_ok=True)

# File upload handling
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_path = os.path.join("tempDir", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display image
    st.image(uploaded_file, width=300)

    # Classify and display result
    result = classify_image(temp_path)
    st.markdown(result)

    # Clean up temporary file
    os.remove(temp_path)
