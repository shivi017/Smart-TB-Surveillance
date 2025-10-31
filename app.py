# --- Tuberculosis Detection App ---

import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array


# --- Page Config ---
st.set_page_config(page_title="TB Detection", layout="centered", page_icon="🩻")


# --- Constants ---
MODEL_PATH = "best_resnet50_tuned.h5"
IMG_SIZE = (224, 224)


# --- Load model ---
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


# --- Styling ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 2.8em;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.2em;
        color: #FAFAFA;
    }
    .subtitle {
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 2em;
        color: #B0B0B0;
    }
    .result-card {
        background-color: #1E1E1E;
        padding: 1.5em;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        margin-top: 2em;
    }
    .result-label {
        font-size: 1.8em;
        font-weight: bold;
        color: #4FC3F7;
    }
    .positive {
        color: #EF5350;
    }
    .negative {
        color: #66BB6A;
    }
    .confidence {
        margin-top: 0.5em;
        font-size: 1.1em;
        color: #CCCCCC;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #888888;
        margin-top: 3em;
    }
    </style>
""", unsafe_allow_html=True)


# --- Title ---
st.markdown('<div class="title">🩺 Tuberculosis Detection from Chest X-rays</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a chest X-ray image (JPG or PNG) to check for Tuberculosis</div>', unsafe_allow_html=True)


# --- Image Preprocessing ---
def preprocess_image(uploaded_file):
    img = load_img(uploaded_file, target_size=IMG_SIZE)
    img = img.convert("RGB")
    img_array = img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# --- File uploader ---
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])


if uploaded_file:
    # Display image using raw bytes (Streamlit supports this)
    st.image(uploaded_file, caption="📷 Uploaded X-ray", use_column_width=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔎 Analyze X-ray"):
            with st.spinner("Analyzing with deep learning model..."):
                processed_img = preprocess_image(uploaded_file)
                pred = model.predict(processed_img, verbose=0)[0][0]

                if pred > 0.5:
                    label = "🧬 Tuberculosis Detected"
                    confidence = pred
                else:
                    label = "✅ Normal"
                    confidence = 1 - pred

            st.markdown("---")
            st.subheader("🧠 Prediction Result")
            st.success(label)
            st.markdown(f"**Confidence Level:** `{confidence:.2%}`")
            st.progress(float(confidence))

    

else:
    st.info("Please upload a chest X-ray to begin analysis.")

# --- Footer ---
st.markdown('<div class="footer">Model: ResNet50 </div>', unsafe_allow_html=True)
