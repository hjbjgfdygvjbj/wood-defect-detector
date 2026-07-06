import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import gdown

st.set_page_config(page_title="Wood Defect Detector", layout="centered")

st.title("🪵 Phát hiện lỗi bề mặt gỗ")
st.write("Upload ảnh gỗ để phát hiện các lỗi: Crack, Dead Knot, Live Knot, Knot with Crack")

@st.cache_resource
def load_model():
    model_path = "best_v11m.pt"
    if not os.path.exists(model_path):
        with st.spinner("Đang tải model..."):
            gdown.download(
                f"https://drive.google.com/uc?id=1qoBP_0Q6ggUPmO1wtGvVcSPK9N5uR0Vr",
                model_path,
                quiet=False
            )
    return YOLO(model_path)

model = load_model()

uploaded_file = st.file_uploader("Chọn ảnh gỗ", type=["jpg", "jpeg", "png"])

conf = st.slider("Ngưỡng confidence", 0.1, 0.9, 0.25, 0.05)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    with st.spinner("Đang phân tích..."):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            results = model.predict(tmp.name, conf=conf)
            os.unlink(tmp.name)

        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Kết quả phát hiện", use_column_width=True)

        boxes = results[0].boxes
        if len(boxes) == 0:
            st.warning("Không phát hiện lỗi nào.")
        else:
            st.success(f"Phát hiện được {len(boxes)} lỗi:")
            names = model.names
            for box in boxes:
                cls = int(box.cls)
                conf_score = float(box.conf)
                st.write(f"- **{names[cls]}**: {conf_score:.0%}")