import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model once and cache it
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("models/mnistCNN.h5")

model = load_my_model()

# Streamlit UI
st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9), and the model will predict it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(img, caption="Uploaded Image", use_column_width=False, width=150)

    # Preprocess image
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28, 1) / 255.0  # normalize

    # Prediction
    pred = model.predict(im2arr)
    num = np.argmax(pred, axis=1)[0]

    # Show result
    st.markdown(f"### 🔢 Predicted Digit: **{num}**")
