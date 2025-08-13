import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import io
import cv2
import os

st.set_page_config(page_title="YOLO Inference (Streamlit)", layout="wide")

st.title("YOLO Inference (Ultralytics) – Streamlit")

# -----------------------------
# Sidebar: settings
# -----------------------------
st.sidebar.header("Settings")
model_path = "yolo11n_sz640_lr0.001_mos0.0.pt"
imgsz = st.sidebar.slider("imgsz (inference size)", min_value=320, max_value=1280, step=32, value=640)
conf = st.sidebar.slider("conf (confidence threshold)", min_value=0.05, max_value=0.95, step=0.05, value=0.5)
device_choice = st.sidebar.selectbox("device", options=["cpu", "cuda:0"], index=0)
save_annot = st.sidebar.checkbox("Enable download of annotated image", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tips: \n- Streamlit 환경에선 `.show()` 대신 `result.plot()` + `st.image()` 사용\n- GPU가 없으면 `device='cpu'`")

# -----------------------------
# Model loader (cache)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = YOLO(path)
    return model

model = None
load_error = None
try:
    model = load_model(model_path)
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(load_error)
    st.stop()

# 클래스 이름 보여주기
with st.expander("Model classes (model.names)"):
    st.write(model.names)

# -----------------------------
# Input image
# -----------------------------
tab1, tab2 = st.tabs(["Upload Image", "Use Local Path"])

with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
with tab2:
    local_path = st.text_input("Local image path", value="mytest_dataset/Corn_Common_Rust (47).jpg")
    use_local = st.button("Load from local path")

input_image = None
image_name = None

if uploaded:
    image_name = uploaded.name
    input_image = Image.open(uploaded).convert("RGB")
elif use_local and os.path.exists(local_path):
    image_name = os.path.basename(local_path)
    input_image = Image.open(local_path).convert("RGB")

if input_image is None:
    st.info("이미지를 업로드하거나 로컬 경로를 입력한 뒤 실행하세요.")
    st.stop()

st.image(input_image, caption="Input", use_container_width=True)

# -----------------------------
# Inference
# -----------------------------
run = st.button("Run inference")

if run:
    with st.spinner("Running YOLO..."):
        # PIL -> numpy (RGB)
        np_img = np.array(input_image)

        # predict
        results = model.predict(
            source=np_img,     # numpy/PIL 바로 가능
            imgsz=imgsz,
            conf=conf,
            device=device_choice,
            verbose=False
        )

    # 여러 장 처리 가능하지만 여기선 첫 결과만 표시
    result = results[0]

    # -------------------------
    # Boxes -> DataFrame
    # -------------------------
    boxes = result.boxes
    names = result.names  # dict: class_id -> name

    if boxes is not None and len(boxes) > 0:
        # xyxy: (N,4), conf: (N,), cls: (N,)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        rows = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            cls_id = clss[i]
            cls_name = names.get(cls_id, str(cls_id))
            rows.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": float(confs[i]),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "w": float(x2 - x1), "h": float(y2 - y1)
            })
        df = pd.DataFrame(rows)
        st.subheader("Detections")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No detections.")

    # -------------------------
    # Annotated image
    # -------------------------
    plotted_bgr = result.plot()  # BGR (OpenCV)
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    st.image(plotted_rgb, caption="Annotated", use_container_width=True)

    if save_annot:
        # JPEG encode
        success, buffer = cv2.imencode(".jpg", plotted_bgr)
        if success:
            btn = st.download_button(
                label="Download annotated image",
                data=io.BytesIO(buffer.tobytes()),
                file_name=f"annotated_{image_name or 'result'}.jpg",
                mime="image/jpeg"
            )
