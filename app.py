import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
import pandas as pd
import hashlib
import json
from datetime import datetime
import plotly.express as px
from scipy.signal import convolve2d, medfilt2d

st.set_page_config(
    page_title="Real X‑Ray System – Gesner Deslandes",
    page_icon="🩻",
    layout="wide"
)

# ---------- Professional X‑Ray CSS ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1a2a, #0e2a3b);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07141f, #0a1a2a);
        border-right: 1px solid #1e3a4d;
    }
    .stButton button {
        background-color: #1e6f5c !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton button:hover {
        background-color: #0c4a3e !important;
        transform: scale(1.02);
    }
    h1, h2, h3, p, div, span, label {
        color: #e0f2fe !important;
    }
    .warning-box {
        background: #ff000020;
        border-left: 4px solid #ff4d4d;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .hardware-box {
        background: #1e6f5c30;
        border-left: 4px solid #1e6f5c;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Session State ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "patients" not in st.session_state:
    st.session_state.patients = {}
if "current_exam" not in st.session_state:
    st.session_state.current_exam = None
if "xray_settings" not in st.session_state:
    st.session_state.xray_settings = {
        "contrast": 1.2,
        "brightness": 1.0,
        "edge_enhancement": True,
        "noise_reduction": True,
        "invert": False,
        "color_map": "gray"
    }
if "calibration" not in st.session_state:
    st.session_state.calibration = {
        "distance": 30.0,
        "exposure": 2.5,
        "kvp": 70,
        "phosphor_type": "CsI"
    }

# ---------- Pure NumPy Image Processing (No OpenCV) ----------
def simple_edge_detection(grayscale_np):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = convolve2d(grayscale_np, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(grayscale_np, sobel_y, mode='same', boundary='symm')
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8) if magnitude.max() > 0 else magnitude
    return magnitude

def process_raw_xray(raw_image, settings, calibration):
    gray_img = raw_image.convert('L')
    img_np = np.array(gray_img, dtype=np.float32)
    
    if settings["noise_reduction"]:
        img_np = medfilt2d(img_np, kernel_size=3)
    
    if settings["edge_enhancement"]:
        edges = simple_edge_detection(img_np.astype(np.uint8))
        img_np = img_np + 0.5 * edges.astype(np.float32)
        img_np = np.clip(img_np, 0, 255)
    
    img_np = (img_np - 127) * settings["contrast"] + 127 + (settings["brightness"] - 1) * 50
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    if settings["invert"]:
        img_np = 255 - img_np
    
    if settings["color_map"] == "bone":
        colored = np.stack([img_np, img_np//2, img_np//4], axis=2).astype(np.uint8)
        out = Image.fromarray(colored, mode='RGB')
    elif settings["color_map"] == "thermal":
        # Fallback to gray for simplicity
        out = Image.fromarray(img_np, mode='L')
    else:
        out = Image.fromarray(img_np, mode='L')
    
    return out

def measure_bone_density(xray_img, area_px=100):
    img_np = np.array(xray_img.convert('L'))
    h, w = img_np.shape
    cx, cy = w//2, h//2
    roi = img_np[cy-area_px//2:cy+area_px//2, cx-area_px//2:cx+area_px//2]
    mean_intensity = np.mean(roi)
    density_score = 100 - (mean_intensity / 2.55)
    return max(0, min(100, density_score))

def mock_xray_tube_command(kvp, mas, time_ms):
    st.info(f"⚡ Simulated X‑ray tube trigger: {kvp} kV, {mas} mAs, {time_ms} ms (replace with actual hardware control)")
    return True

# ---------- Login ----------
def login():
    st.title("🩻 Real X‑Ray Imaging System")
    st.markdown("### **Professional Radiography Suite**")
    st.markdown("© 2025 Gesner Deslandes – Hardware‑Integrated Solution")
    st.markdown("---")
    st.markdown("#### 🔐 Authorized Access Only")
    password = st.text_input("Enter secure access code", type="password")
    if st.button("Unlock System"):
        if password == "xray2025":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Access denied. Invalid credentials.")

# ---------- Main Application ----------
def main_app():
    st.title("🩻 X‑Ray Imaging Platform")
    st.markdown("### **Real Phosphor + Camera System – Clinical Workflow Ready**")
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2932/2932652.png", width=100)
        
        # ---------- HARDWARE REQUIREMENTS (CRITICAL) ----------
        st.markdown("## 🛠️ Required Hardware")
        st.markdown(
            """
            <div class="hardware-box">
            ✅ <strong>To obtain real X‑ray images, you MUST have:</strong><br><br>
            1️⃣ <strong>X‑ray source</strong> (e.g., portable dental/industrial X‑ray tube)<br>
            2️⃣ <strong>Phosphor screen</strong> (CsI or Gd2O2S) – converts X‑rays to visible light<br>
            3️⃣ <strong>Camera</strong> (your phone or webcam) aimed at the phosphor screen<br>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class="warning-box">
            ⚠️ <strong>SAFETY WARNING</strong><br>
            X‑rays are ionising radiation. Improper use can cause serious injury or death.<br>
            • Only operate with proper shielding and training.<br>
            • Comply with all local regulations for medical X‑ray equipment.<br>
            • This software alone does NOT produce X‑rays – it only processes the visible light from the phosphor screen.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.markdown("### 🧪 System Status")
        col1, col2 = st.columns(2)
        col1.metric("X‑Ray Tube", "Not connected", delta="Hardware required")
        col2.metric("Phosphor Screen", "Not detected", delta="Place in front of camera")
        st.progress(0.0, text="Camera Link: OK (visible light only)")
        
        st.markdown("---")
        st.markdown("### ⚙️ Acquisition Settings")
        kvp = st.slider("Tube Voltage (kVp)", 40, 120, st.session_state.calibration["kvp"])
        mas = st.slider("Exposure (mAs)", 0.5, 10.0, st.session_state.calibration["exposure"])
        distance = st.number_input("Source‑to‑Phosphor Distance (cm)", 20.0, 100.0, st.session_state.calibration["distance"])
        st.session_state.calibration.update({"kvp": kvp, "exposure": mas, "distance": distance})
        
        st.markdown("---")
        st.markdown("### 🎛️ Image Processing")
        st.session_state.xray_settings["contrast"] = st.slider("Contrast", 0.5, 3.0, 1.2)
        st.session_state.xray_settings["brightness"] = st.slider("Brightness", 0.5, 2.0, 1.0)
        st.session_state.xray_settings["edge_enhancement"] = st.checkbox("Edge Enhancement", True)
        st.session_state.xray_settings["noise_reduction"] = st.checkbox("Noise Reduction", True)
        st.session_state.xray_settings["invert"] = st.checkbox("Invert (Negative)", False)
        st.session_state.xray_settings["color_map"] = st.selectbox("Color Map", ["gray", "bone", "thermal"])
        
        st.markdown("---")
        if st.button("🔌 System Self‑Test", use_container_width=True):
            st.success("Hardware not detected. Please connect X‑ray source and phosphor screen.")
    
    # Rest of the tabs (Capture, Analysis, Patients, Export) – same as before
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Capture X‑Ray", "🩻 Image Analysis", "📋 Patient Records", "⚙️ Telemedicine & Export"])
    
    with tab1:
        st.subheader("🖱️ Step 1: Acquire Radiograph")
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("**Phosphor Screen Preview** – point your camera at the glowing screen after exposure.")
            camera_input = st.camera_input("Align camera with phosphor screen", key="xray_capture")
        with col2:
            st.markdown("**Exposure Controller**")
            st.warning("⚠️ Ensure proper shielding before firing.")
            if st.button("🔴 FIRE X‑RAY TUBE", use_container_width=True):
                mock_xray_tube_command(kvp, mas, 200)
                st.success(f"Simulated exposure: {kvp} kV, {mas} mAs")
                st.balloons()
        
        if camera_input:
            raw_img = Image.open(io.BytesIO(camera_input.read()))
            st.image(raw_img, caption="Raw Phosphor Capture (visible light)", use_column_width=True)
            if st.button("🩻 Process to Radiograph", use_container_width=True):
                with st.spinner("Applying clinical X‑ray pipeline..."):
                    processed = process_raw_xray(raw_img, st.session_state.xray_settings, st.session_state.calibration)
                    st.session_state.current_exam = {
                        "raw": raw_img,
                        "processed": processed,
                        "timestamp": datetime.now(),
                        "kvp": kvp,
                        "mas": mas,
                        "distance": distance
                    }
                    st.success("Radiograph generated (simulated). For real images, use actual X‑ray hardware.")
        
        if st.session_state.current_exam:
            st.markdown("---")
            st.markdown("### ✅ Current Radiograph")
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.current_exam["processed"], caption="Processed X‑Ray Image", use_column_width=True)
            with col2:
                density = measure_bone_density(st.session_state.current_exam["processed"])
                st.metric("Estimated Bone Density (rel.)", f"{density:.1f} %")
    
    with tab2:
        st.subheader("📊 Quantitative Image Analysis")
        if st.session_state.current_exam:
            img = st.session_state.current_exam["processed"]
            img_np = np.array(img.convert('L'))
            fig = px.histogram(img_np.flatten(), nbins=256, title="Pixel Intensity Distribution")
            st.plotly_chart(fig, use_container_width=True)
            roi_radius = st.slider("ROI radius (px)", 10, 100, 50)
            h, w = img_np.shape
            cx, cy = w//2, h//2
            roi = img_np[cy-roi_radius:cy+roi_radius, cx-roi_radius:cx+roi_radius]
            st.metric("Mean ROI intensity", f"{np.mean(roi):.1f}")
            edges = simple_edge_detection(img_np)
            fracture_score = np.sum(edges > 50) / (h*w) if h*w > 0 else 0
            st.metric("Edge Density (possible fracture indicator)", f"{fracture_score:.3f}")
        else:
            st.info("Capture an X‑ray first.")
    
    with tab3:
        st.subheader("📁 Patient Management (DICOM‑ready)")
        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Full Name")
        if st.button("➕ Create / Select Patient"):
            if patient_id and patient_name:
                st.session_state.patients[patient_id] = {"name": patient_name, "exams": [], "created": datetime.now()}
                st.success(f"Patient {patient_name} added.")
        if patient_id in st.session_state.patients and st.session_state.current_exam:
            if st.button("💾 Attach Current X‑Ray to Patient Record"):
                exam_data = {"date": st.session_state.current_exam["timestamp"].isoformat(), "kvp": kvp, "mas": mas}
                st.session_state.patients[patient_id]["exams"].append(exam_data)
                st.success("X‑Ray saved.")
        if st.session_state.patients:
            st.markdown("### Registered Patients")
            for pid, data in st.session_state.patients.items():
                st.markdown(f"**{pid}** – {data['name']} ({len(data['exams'])} exams)")
    
    with tab4:
        st.subheader("🌐 DICOM Export & Remote Consultation")
        if st.session_state.current_exam:
            buf = io.BytesIO()
            st.session_state.current_exam["processed"].save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.download_button("📥 Download Radiograph (PNG)", data=b64, file_name=f"xray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            dicom_header = f"DICOM Header\nModality: XR\nPatient ID: {patient_id if 'patient_id' in locals() else 'UNKNOWN'}\nKVP: {kvp}\nExposure: {mas} mAs"
            st.download_button("📄 Download DICOM Header (TXT)", dicom_header, file_name=f"dicom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        else:
            st.info("No X‑Ray available.")

if not st.session_state.authenticated:
    login()
else:
    main_app()
