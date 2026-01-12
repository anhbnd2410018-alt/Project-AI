import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64
import os
import tempfile

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Food VN - AI Detector",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CÁC HÀM CACHE & HỖ TRỢ ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception:
        return None

# --- 3. CSS (TRANG ĐIỂM CHO WEB) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .banner-container {
        width: 100%;
        margin-bottom: 20px;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .banner-img {
        width: 100%;
        display: block;
    }

    .image-card {
        background-color: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #f0f2f6;
    }
    
    .card-title {
        color: #333;
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }

    /* Nút bấm Gradient */
    div.stButton > button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9068 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: left; color: #FF4B4B;'>Food Việt Nam</h1>", unsafe_allow_html=True)
    
    page = st.radio("Menu", ["🏠 Home", "ℹ️ About"], index=0, label_visibility="collapsed")
    st.markdown("---")
    
    # Biến lưu dữ liệu upload
    source_img = None
    source_vid = None
    media_type = None

    if page == "🏠 Home":
        st.subheader("📥 Dữ liệu đầu vào")
        
        # TẠO 2 TAB: ẢNH & VIDEO
        tab1, tab2 = st.tabs(["🖼️ Ảnh", "🎥 Video"])
        
        with tab1:
            source_img = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg', 'png'], key="img_uploader")
            if source_img: media_type = "image"
                
        with tab2:
            source_vid = st.file_uploader("Tải video lên", type=['mp4', 'avi', 'mov'], key="vid_uploader")
            if source_vid: media_type = "video"

# --- 5. LOGIC CHÍNH ---

if page == "🏠 Home":
    # 1. Hiện Banner
    banner_file = 'welcome.png' 
    if os.path.exists(banner_file):
        bin_str = get_base64_of_bin_file(banner_file)
        st.markdown(f'<div class="banner-container"><img src="data:image/png;base64,{bin_str}" class="banner-img"></div>', unsafe_allow_html=True)
    
    # 2. Load Model
    model_path = 'model/best.pt'
    model = load_model(model_path)
    
    if not model:
        st.error(f"⚠️ LỖI: Không tìm thấy file model tại '{model_path}'. Hãy kiểm tra lại thư mục!")
        st.stop()

    # 3. Xử lý A - NẾU LÀ ẢNH
    if media_type == "image" and source_img:
        col1, col2 = st.columns([1, 1], gap="large") 
        image = Image.open(source_img)

        with col1:
            st.markdown('<div class="image-card"><div class="card-title">📸 Ảnh gốc</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            analyze_btn = st.button("🚀 Phân tích ngay")

        with col2:
            st.markdown('<div class="image-card"><div class="card-title">✨ Kết quả AI</div>', unsafe_allow_html=True)
            if analyze_btn:
                with st.spinner('Đang nhận diện...'):
                    results = model(image, conf=0.25)
                    res_plotted = results[0].plot()
                    st.image(res_plotted, use_container_width=True)
                    
                    # Hiện tên món ăn
                    detected = []
                    for box in results[0].boxes:
                        name = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        detected.append((name, conf))
                    
                    if detected:
                        st.success(f"Tìm thấy {len(detected)} món!")
                        html_tags = ""
                        for name, conf in detected:
                            html_tags += f'<span style="background-color: #e8f5e9; color: #2e7d32; padding: 5px 10px; border-radius: 15px; margin: 5px; font-weight: bold; display: inline-block;">{name} ({conf:.0%})</span>'
                        st.markdown(html_tags, unsafe_allow_html=True)
                    else:
                        st.warning("Không tìm thấy món nào.")
            else:
                st.info("👈 Bấm nút để xem kết quả")
            st.markdown('</div>', unsafe_allow_html=True)

    # 4. Xử lý B - NẾU LÀ VIDEO
    elif media_type == "video" and source_vid:
        st.markdown('<div class="image-card"><div class="card-title">🎥 Phân tích Video (Real-time)</div>', unsafe_allow_html=True)
        
        if st.button("▶️ Bắt đầu chạy Video"):
            # Lưu video tạm thời
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(source_vid.read())
            
            vf = cv2.VideoCapture(tfile.name)
            stframe = st.empty() # Khung hình trống để chiếu video
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                # Resize video nếu quá to để chạy nhanh hơn
                frame = cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1])))

                # AI xử lý
                results = model(frame, conf=0.25)
                res_plotted = results[0].plot()
                
                # Đổi màu BGR -> RGB để hiển thị đúng
                res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                stframe.image(res_plotted, caption='Đang chạy...', use_container_width=True)

            vf.release()
            st.success("Đã xong video!")
        else:
             st.info("Bấm nút trên để AI bắt đầu quét video.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Hãy chọn Ảnh hoặc Video ở menu bên trái để bắt đầu.")

elif page == "ℹ️ About":
    st.title("ℹ️ Giới thiệu")
    st.markdown("""
    <div class="image-card" style="text-align: left;">
        <h3>🍜 Food Việt Nam Project</h3>
        <p>Ứng dụng AI nhận diện món ăn Việt Nam.</p>
        <ul>
            <li><b>Công nghệ:</b> YOLOv10 & Streamlit</li>
            <li><b>Tính năng:</b> Hỗ trợ cả Ảnh và Video</li>
            <li><b>Tác giả:</b> Group 8</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)