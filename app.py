import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os
import matplotlib.pyplot as plt

# ============ 1. Download CNN model from Google Drive ===============
def download_model_from_drive(model_url, output_path):
    if not os.path.exists(output_path):
        with st.spinner("Downloading CNN model from Google Drive..."):
            gdown.download(model_url, output_path, quiet=False)
            st.success("CNN model downloaded!")

cnn_model_url = "https://drive.google.com/uc?id=1YLRXJyc8jL7vDGIQ_HslUD6IeoRxUqoH"  # 🔁 Replace with real ID
cnn_model_path = "seed_shape_model.h5"
download_model_from_drive(cnn_model_url, cnn_model_path)

# ============ 2. Load Both Models =================
with st.spinner("Loading models..."):
    cnn_model = load_model(cnn_model_path)
    mobilenet_model = load_model("Mobilenet_tuned.h5")  # Should be <100MB
    st.success("Both models loaded successfully!")

# ============ 3. Define Classes ====================
class_names = ['intact', 'broken', 'spotted', 'immature', 'skin-damaged']

# ============ 4. Streamlit UI =====================
st.title("Seed Shape Classification App 🌱")
st.write("Choose a model and upload an image for prediction")

model_option = st.radio("Select Model", ['MobileNetV2', 'Custom CNN'])

uploaded_file = st.file_uploader("Upload a Seed Image", type=["jpg", "png", "jpeg"])

# ============ 5. Prediction =======================
if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Resize based on model
    if model_option == 'MobileNetV2':
        img_resized = image_data.resize((224, 224))
    else:
        img_resized = image_data.resize((227, 227))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if model_option == 'MobileNetV2':
        prediction = mobilenet_model.predict(img_array)[0]
    else:
        prediction = cnn_model.predict(img_array)[0]

    # Display prediction
    top_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🧠 Prediction: `{top_class}` ({confidence:.2f}%)")

    # Show bar chart
    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction * 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # Print all class probabilities
    st.subheader("Raw Scores")
    for i, prob in enumerate(prediction):
        st.write(f"**{class_names[i]}**: {prob*100:.2f}%")
