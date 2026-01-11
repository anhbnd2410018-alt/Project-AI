import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64
import os

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Ngon Luôn - AI Food Detector",
    page_icon="🍲"
)

# --- 2. HÀM XỬ LÝ ẢNH BANNER ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- 3. CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .banner-container {
        width: 100%;
        margin-bottom: 20px;
    }
    .banner-img {
        width: 100%;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        display: block;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🏠 Home") 
    st.markdown("---")
    st.subheader("1. Input")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh", type=['jpg', 'jpeg', 'png'])
    st.markdown("---")
    st.subheader("2. Settings")
    conf_threshold = st.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.25)
    st.caption("Điều chỉnh độ nhạy của AI.")

# --- 5. GIAO DIỆN CHÍNH ---

# === ĐÃ ĐỔI TÊN FILE TẠI ĐÂY ===
banner_file = 'welcome.png' 

if os.path.exists(banner_file):
    bin_str = get_base64_of_bin_file(banner_file)
    st.markdown(
        f'<div class="banner-container"><img src="data:image/png;base64,{bin_str}" class="banner-img"></div>',
        unsafe_allow_html=True
    )
else:
    st.error(f"⚠️ Chưa tìm thấy file '{banner_file}'. Hãy copy ảnh vào cùng thư mục với file app.py nhé!")

st.write("") 

# --- 6. LOGIC AI ---
model_path = 'model/best.pt'
try:
    model = YOLO(model_path)
except Exception:
    st.error(f"⚠️ Không tìm thấy file model tại {model_path}")
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📸 Ảnh gốc")
        st.image(image, use_container_width=True)
        analyze_button = st.button('🚀 Phân tích ngay', type="primary", use_container_width=True)

    if analyze_button:
        with col2:
            st.write("### 🧠 Kết quả AI")
            with st.spinner('Đang soi món ăn...'):
                results = model(image, conf=conf_threshold)
                res_plotted = results[0].plot()
                st.image(res_plotted, use_container_width=True)
                
                detected_items = []
                for box in results[0].boxes:
                    item_name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    detected_items.append(f"- **{item_name}** ({conf:.1%})")
                
                if detected_items:
                    st.success("Đã nhận diện xong!")
                    with st.expander("📝 Xem danh sách"):
                        st.markdown("\n".join(detected_items))
                else:
                    st.warning("Không tìm thấy món nào.")
else:
    st.info("👈 Hãy upload ảnh bên tay trái để bắt đầu.")