import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ================== CONFIG ==================
st.set_page_config(
    page_title="YOLOv10b Food Detection",
    layout="centered"
)

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")  # weight của bạn

model = load_model()

# ================== UI ==================
st.title("🍔 Food Detection - YOLOv10b")
st.write("Upload ảnh để YOLOv10b nhận diện")

uploaded_file = st.file_uploader(
    "Chọn ảnh",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    if st.button("🔍 Detect"):
        with st.spinner("YOLOv10b đang chạy..."):
            img_np = np.array(image)

            results = model(
                img_np,
                conf=0.25,
                imgsz=640,
                device="cpu"   # đổi thành 0 nếu có GPU
            )[0]

            annotated_img = results.plot()

            st.image(
                annotated_img,
                caption="Kết quả YOLOv10b",
                use_container_width=True
            )

            st.success("Hoàn tất 🎉")
