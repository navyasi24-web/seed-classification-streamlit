import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os
import matplotlib.pyplot as plt

# ======================== MODEL DOWNLOAD ========================
def download_model_from_drive(file_id, output_path, name):
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {name} model from Google Drive..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            st.success(f"{name} model downloaded!")

# Google Drive file IDs
cnn_file_id = "1YLRXJyc8jL7vDGIQ_HslUD6IeoRxUqoH"
mobilenet_file_id = "1mrligU_-I4nmi4I-4VUXc3dvj-jxvSEg"

# File paths
cnn_model_path = "seed_shape_model.h5"
mobilenet_model_path = "Mobilenetv2_seed_final.h5"

# Download models
download_model_from_drive(cnn_file_id, cnn_model_path, "CNN")
download_model_from_drive(mobilenet_file_id, mobilenet_model_path, "MobileNetV2")

# ======================== LOAD MODELS ========================
with st.spinner("Loading models..."):
    cnn_model = load_model(cnn_model_path)
    mobilenet_model = load_model(mobilenet_model_path)
    st.success("Models loaded successfully!")

# ======================== UI ========================
class_names = ['intact', 'broken', 'spotted', 'immature', 'skin-damaged']

st.title("🌱 Seed Shape Classification App")
st.write("Upload a seed image and choose a model to classify it.")

model_option = st.radio("Select Model", ['MobileNetV2', 'Custom CNN'])
uploaded_file = st.file_uploader("Upload a seed image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    if model_option == 'MobileNetV2':
        img_resized = image_data.resize((224, 224))
        model = mobilenet_model
    else:
        img_resized = image_data.resize((227, 227))
        model = cnn_model

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    top_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🧠 Prediction: `{top_class}` ({confidence:.2f}% confidence)")

    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction * 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    st.subheader("Raw Scores")
    for i, prob in enumerate(prediction):
        st.write(f"**{class_names[i]}**: {prob*100:.2f}%")
