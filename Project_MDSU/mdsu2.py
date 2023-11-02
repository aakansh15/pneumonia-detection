import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load the Keras model
model = tf.keras.models.load_model('./models/model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image)
    if image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, axis=0)
    return image

# Create the Streamlit web app
st.title('Pneumonia Detection Web App')

# Add drag and drop functionality
uploaded_image = st.file_uploader('Upload a chest X-ray image (Drag and Drop)', type=['jpg', 'png', 'jpeg'], key="fileUploader")

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write('')

    # Preprocess the uploaded image
    processed_image = preprocess_image(uploaded_image)

    # Make predictions
    prediction = model.predict(processed_image)

    # Display the result
    if prediction[0][0] > 0.6555:
        st.write('Prediction: Pneumonia Positive')
    else:
        st.write('Prediction: Pneumonia Negative')