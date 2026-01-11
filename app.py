import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLOv10 Food Detector", layout="centered")

st.title("🍔 Food Detection - YOLOv10b")

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting..."):
        results = model(img_array)

    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_container_width=True)
