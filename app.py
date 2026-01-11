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

# --- 2. CÁC HÀM CACHE (GIÚP WEB CHẠY NHANH) ---

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        return None

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
    /* Chỉnh font cho tiêu đề Sidebar đẹp hơn */
    [data-testid="stSidebar"] h1 {
        font-family: 'Helvetica', sans-serif;
        color: #FF4B4B;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR (ĐÃ SỬA THEO YÊU CẦU) ---
with st.sidebar:
    # Thêm logo nhỏ ở trên cùng (nếu muốn)
    st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", icon_image=None)
    
    # === THAY ĐỔI 1: Đổi tên tiêu đề ===
    st.title("🍜 Food Việt Nam")
    
    # === THAY ĐỔI 2: Chỉnh nút bấm cho khớp và gọn ===
    # label_visibility="collapsed" sẽ ẩn dòng chữ "Chọn mục" thừa thãi đi
    page = st.radio(
        "Menu", 
        ["🏠 Home", "ℹ️ About"], 
        index=0,
        label_visibility="collapsed" 
    )
    
    st.markdown("---")

    uploaded_file = None
    if page == "🏠 Home":
        st.subheader("📥 Input")
        uploaded_file = st.file_uploader("Upload ảnh tại đây", type=['jpg', 'jpeg', 'png'])

# --- 5. LOGIC CHUYỂN TRANG ---

# === TRANG HOME ===
if page == "🏠 Home":
    # Hiện Banner
    banner_file = 'welcome.png' 
    if os.path.exists(banner_file):
        bin_str = get_base64_of_bin_file(banner_file)
        st.markdown(f'<div class="banner-container"><img src="data:image/png;base64,{bin_str}" class="banner-img"></div>', unsafe_allow_html=True)
    
    st.write("") 

    # Load Model
    model_path = 'model/best.pt'
    model = load_model(model_path)

    if model is None:
        st.error(f"⚠️ Không tìm thấy file model tại {model_path}")
        st.stop()

    # Logic xử lý ảnh
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
                    results = model(image, conf=0.25)
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
        st.info("👈 Mời bạn upload ảnh ở thanh bên trái.")

# === TRANG ABOUT ===
elif page == "ℹ️ About":
    st.title("ℹ️ Giới thiệu")
    
    st.markdown("""
    ### 🌟 Dự án Food Việt Nam
    
    Chào mừng bạn đến với **Food Việt Nam** - công cụ hỗ trợ nhận diện món ăn sử dụng trí tuệ nhân tạo.
    
    #### 🎯 Mục tiêu
    Giúp người dùng dễ dàng nhận biết tên các món ăn thông qua hình ảnh.
    
    #### 🛠 Công nghệ sử dụng
    * **Mô hình AI:** YOLOv10
    * **Dataset:** VietFood
    * **Framework:** Streamlit & Python
    
    #### 👨‍💻 Team phát triển
    * **Nhóm:** Group AI
    """)