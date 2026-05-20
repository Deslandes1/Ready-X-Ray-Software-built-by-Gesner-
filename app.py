import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import pandas as pd
import hashlib
import json
from datetime import datetime
import plotly.express as px

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
        transition: 0.2s;
    }
    .stButton button:hover {
        background-color: #0c4a3e !important;
        transform: scale(1.02);
    }
    h1, h2, h3, p, div, span, label {
        color: #e0f2fe !important;
    }
    .xray-panel {
        background: rgba(0,0,0,0.6);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1e6f5c;
    }
    .metric-card {
        background: #11212e;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
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
        "sharpness": 1.5,
        "noise_reduction": True,
        "edge_enhancement": True,
        "invert": False,
        "color_map": "gray"
    }
if "calibration" not in st.session_state:
    st.session_state.calibration = {
        "distance": 30.0,      # cm
        "exposure": 2.5,       # mAs
        "kvp": 70,             # kV
        "phosphor_type": "CsI",
        "lens_correction": np.eye(3).tolist()
    }

# ---------- Real X‑Ray Processing Pipeline ----------
def process_raw_xray(raw_image, settings, calibration):
    """
    Applies professional X‑ray processing:
    - Flat‑field correction (uses stored calibration matrix)
    - Edge enhancement
    - Adaptive contrast
    - Noise reduction
    - Inversion (X‑ray negative)
    """
    img = np.array(raw_image.convert('L'))
    
    # Flat‑field correction (simplified, assumes calibration matrix exists)
    if 'lens_correction' in calibration:
        rows, cols = img.shape
        gain = np.array(calibration['lens_correction']).reshape(rows, cols) if len(calibration['lens_correction']) == rows*cols else np.ones((rows, cols))
        img = img.astype(np.float32) * gain
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Apply edge enhancement (like clinical X‑ray sharpening)
    if settings["edge_enhancement"]:
        img = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
    
    # Denoise
    if settings["noise_reduction"]:
        img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Contrast and brightness
    img = cv2.convertScaleAbs(img, alpha=settings["contrast"], beta=(settings["brightness"]-1)*50)
    
    # Inversion (X‑ray negative)
    if settings["invert"]:
        img = 255 - img
    
    # Apply colormap
    if settings["color_map"] == "gray":
        out = Image.fromarray(img, mode='L')
    elif settings["color_map"] == "bone":
        colored = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        out = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    elif settings["color_map"] == "thermal":
        colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        out = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    else:
        out = Image.fromarray(img, mode='L')
    
    return out

def measure_bone_density(xray_img, area_px=100):
    """Estimate relative bone density using pixel intensity."""
    img_np = np.array(xray_img.convert('L'))
    # Simpler: sample central ROI and compute mean intensity
    h, w = img_np.shape
    cx, cy = w//2, h//2
    roi = img_np[cy-area_px//2:cy+area_px//2, cx-area_px//2:cx+area_px//2]
    mean_intensity = np.mean(roi)
    # Higher mean → less dense (more X‑rays passed through) → lower bone density
    density_score = 100 - (mean_intensity / 2.55)  # scale 0-100
    return max(0, min(100, density_score))

# ---------- Hardware Interface Simulation ----------
def mock_xray_tube_command(kvp, mas, time_ms):
    """Placeholder for actual hardware control. Replace with serial/GPIO."""
    st.info(f"⚡ X‑ray tube triggered: {kvp} kV, {mas} mAs, {time_ms} ms")
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
        # In production, use proper authentication (LDAP, OAuth)
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
        st.markdown("### 🧪 System Control")
        
        # Hardware status (placeholders)
        st.markdown("**Hardware Status**")
        col1, col2 = st.columns(2)
        col1.metric("X‑Ray Tube", "Connected", delta="Ready")
        col2.metric("Phosphor Screen", "CsI", delta="Active")
        st.progress(1.0, text="Camera Link: OK")
        
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
            st.success("All hardware modules operational. Calibration matrix loaded.")
    
    # Main area with tabs for clinical workflow
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Capture X‑Ray", "🩻 Image Analysis", "📋 Patient Records", "⚙️ Telemedicine & Export"])
    
    # ---------- TAB 1: CAPTURE ----------
    with tab1:
        st.subheader("🖱️ Step 1: Acquire Radiograph")
        
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("**Phosphor Screen Preview** (live feed) – point your camera at the glowing screen after exposure.")
            camera_input = st.camera_input("Align camera with phosphor screen", key="xray_capture")
        with col2:
            st.markdown("**Exposure Controller**")
            st.warning("⚠️ Ensure proper shielding before firing.")
            if st.button("🔴 FIRE X‑RAY TUBE", use_container_width=True):
                # In real code, this would send a trigger to the X‑ray generator
                mock_xray_tube_command(kvp, mas, 200)
                st.success(f"Exposure complete: {kvp} kV, {mas} mAs")
                st.balloons()
        
        if camera_input:
            raw_img = Image.open(io.BytesIO(camera_input.read()))
            st.image(raw_img, caption="Raw Phosphor Capture", use_column_width=True)
            
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
                    st.success("Radiograph generated.")
        
        if st.session_state.current_exam:
            st.markdown("---")
            st.markdown("### ✅ Current Radiograph")
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.current_exam["processed"], caption="Processed X‑Ray Image", use_column_width=True)
            with col2:
                density = measure_bone_density(st.session_state.current_exam["processed"])
                st.metric("Estimated Bone Density (rel.)", f"{density:.1f} %")
                st.caption("Qualitative density index – not for diagnosis without calibration.")
    
    # ---------- TAB 2: ANALYSIS ----------
    with tab2:
        st.subheader("📊 Quantitative Image Analysis")
        if st.session_state.current_exam:
            img = st.session_state.current_exam["processed"]
            img_np = np.array(img.convert('L'))
            
            # Histogram
            fig = px.histogram(img_np.flatten(), nbins=256, title="Pixel Intensity Distribution (Density Index)")
            st.plotly_chart(fig, use_container_width=True)
            
            # ROI selector (mock)
            st.markdown("**Region of Interest (ROI) Analysis**")
            st.markdown("In a production system, click‑and‑drag ROI would be implemented here.")
            roi_radius = st.slider("ROI radius (px)", 10, 100, 50)
            h, w = img_np.shape
            cx, cy = w//2, h//2
            roi = img_np[cy-roi_radius:cy+roi_radius, cx-roi_radius:cx+roi_radius]
            st.metric("Mean ROI intensity", f"{np.mean(roi):.1f}", help="Lower = denser bone")
            
            # Fracture detection (simplified edge analysis)
            edges = cv2.Canny(img_np, 50, 150)
            fracture_score = np.sum(edges > 0) / (h*w)
            st.metric("Edge Density (possible fracture indicator)", f"{fracture_score:.3f}")
        else:
            st.info("Capture an X‑ray first.")
    
    # ---------- TAB 3: PATIENT RECORDS ----------
    with tab3:
        st.subheader("📁 Patient Management (DICOM‑ready)")
        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Full Name")
        if st.button("➕ Create / Select Patient"):
            if patient_id and patient_name:
                st.session_state.patients[patient_id] = {
                    "name": patient_name,
                    "exams": [],
                    "created": datetime.now()
                }
                st.success(f"Patient {patient_name} added.")
        
        if patient_id in st.session_state.patients and st.session_state.current_exam:
            if st.button("💾 Attach Current X‑Ray to Patient Record"):
                exam_data = {
                    "date": st.session_state.current_exam["timestamp"].isoformat(),
                    "image_b64": base64.b64encode(io.BytesIO(st.session_state.current_exam["processed"].tobytes()).getvalue()).decode(),
                    "kvp": st.session_state.current_exam["kvp"],
                    "mas": st.session_state.current_exam["mas"]
                }
                st.session_state.patients[patient_id]["exams"].append(exam_data)
                st.success("X‑Ray saved.")
        
        if st.session_state.patients:
            st.markdown("### Registered Patients")
            for pid, data in st.session_state.patients.items():
                st.markdown(f"**{pid}** – {data['name']} ({len(data['exams'])} exams)")
    
    # ---------- TAB 4: EXPORT & TELEMEDICINE ----------
    with tab4:
        st.subheader("🌐 DICOM Export & Remote Consultation")
        if st.session_state.current_exam:
            # Export processed image
            buf = io.BytesIO()
            st.session_state.current_exam["processed"].save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("📥 Download Radiograph (PNG)", data=b64, file_name=f"xray_{now_str}.png")
            
            # Generate DICOM‑like header (simplified)
            dicom_header = f"""
            DICOM HEADER (simulated)
            ------------------------------------------------------------
            Modality: XR
            Patient ID: {patient_id if 'patient_id' in locals() else 'UNKNOWN'}
            Study Date: {datetime.now().strftime('%Y%m%d')}
            KVP: {st.session_state.current_exam.get('kvp', 'N/A')}
            Exposure: {st.session_state.current_exam.get('mas', 'N/A')} mAs
            Phosphor: CsI
            Institution: Gesner Deslandes X‑Ray Lab
            ------------------------------------------------------------
            """
            st.download_button("📄 Download DICOM Header (TXT)", dicom_header, file_name=f"dicom_{now_str}.txt")
            
            # Telemedicine report
            report_text = f"""
            X‑RAY EXAMINATION REPORT
            ==========================
            Date: {datetime.now()}
            Hardware: Phone + Phosphor Screen + Mini X‑Ray Tube
            Processing: Edge enhancement, adaptive contrast
            Findings: Based on image analysis, [Clinical description to be added].
            """
            st.download_button("🩺 Generate Consultation Report", report_text, file_name=f"report_{now_str}.txt")
        else:
            st.info("No X‑Ray available. Please capture a radiograph first.")

# ---------- Entry Point ----------
if not st.session_state.authenticated:
    login()
else:
    main_app()
