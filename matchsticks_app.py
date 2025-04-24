import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("matchstick_model/matchstick_classifier.h5")

# Class labels
class_names = ['Defective', 'Non-defective']

# Page settings
st.set_page_config(page_title="Matchstick Classifier", layout="centered")

# Title
st.title("🧪 Matchstick Defect Classifier")
st.write("Upload an image of a matchstick to check if it's defective.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    class_idx = int(prediction[0] > 0.5)
    confidence = float(prediction[0])

    # Result
    st.markdown("### 🔍 Prediction:")
    st.success(f"**{class_names[class_idx]}** with confidence: `{confidence:.2f}`")
