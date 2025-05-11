import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mnist_cnn_model.h5')

st.title("Digit Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_array = np.array(image).astype('float32') / 255
    img_array = img_array.reshape(1, 28, 28, 1)
    
    pred = model.predict(img_array)
    label = np.argmax(pred)
    st.write(f"Prediction: {label}")