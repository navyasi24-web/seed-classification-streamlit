import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load both models
cnn_model = load_model('seed_shape_model.h5')
mobilenet_model = load_model('Mobilenet_tuned.h5')

# Define class names
class_names = ['intact', 'broken', 'spotted', 'immature', 'skin-damaged']

# Streamlit UI
st.title("🌱 Seed Shape Classifier (CNN vs MobileNetV2)")

model_option = st.selectbox("Select Model", ['Custom CNN', 'MobileNetV2'])

uploaded_file = st.file_uploader("Upload a seed image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_resized = image_data.resize((224, 224))  # Match MobileNetV2 input size
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Choose model
    if model_option == 'MobileNetV2':
        prediction = mobilenet_model.predict(img_array)[0]
    else:
        # Resize for CNN if needed
        img_resized_cnn = image_data.resize((227, 227))  # Match your CNN input
        img_array_cnn = np.array(img_resized_cnn) / 255.0
        img_array_cnn = np.expand_dims(img_array_cnn, axis=0)
        prediction = cnn_model.predict(img_array_cnn)[0]

    top_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🧠 Prediction: `{top_class}` with {confidence*100:.2f}% confidence")

    st.bar_chart(prediction)
    for i, prob in enumerate(prediction):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")
