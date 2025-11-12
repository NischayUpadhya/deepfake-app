import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import tempfile
import os

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('deepfake_detector.h5')
    return model

model = load_model()

st.title("🕵️ Deepfake Detection System")
st.write("Upload an image or video to check if it's **Real or Fake**")

# Upload input
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if "image" in file_type:
        # -----------------------------
        # Image Handling
        # -----------------------------
        img = Image.open(uploaded_file).resize((224, 224))
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        pred = model.predict(img_array)[0][0]
        
        if pred > 0.5:
            st.error("🚫 Deepfake Detected (Fake Image)")
        else:
            st.success("✅ Real Image Detected")

    elif "video" in file_type:
        # -----------------------------
        # Video Handling
        # -----------------------------
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.video(tfile.name)

        st.write("⏳ Processing video frames...")

        fake_count, real_count = 0, 0
        total_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            # Sample every 10th frame for speed
            if total_frames % 10 != 0:
                continue

            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            img_array = np.expand_dims(frame, axis=0) / 255.0

            pred = model.predict(img_array)[0][0]
            if pred > 0.5:
                fake_count += 1
            else:
                real_count += 1

        cap.release()
        os.remove(tfile.name)

        # Final decision
        if fake_count > real_count:
            st.error(f"🚫 Deepfake Detected ({fake_count} fake frames / {total_frames} total)")
        else:
            st.success(f"✅ Real Video Detected ({real_count} real frames / {total_frames} total)")
