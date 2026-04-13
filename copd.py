import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import joblib

from config import CNN_MODEL_PATH, ANN_MODEL_PATH, SCALER_PATH, SPIROMETRY_CSV
from utils import predict_copd, fev1_fvc_status, bmi_category

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COPD Prediction System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: rgba(10, 25, 35, 0.97);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #cfd8dc !important; }

/* tabs */
[data-testid="stTabs"] button {
    color: #90a4ae !important;
    font-weight: 600;
    font-size: 0.95rem;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #80cbc4 !important;
    border-bottom: 2px solid #80cbc4 !important;
    background: rgba(128,203,196,0.08) !important;
}

/* cards */
.card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 16px;
    padding: 26px 30px;
    margin-bottom: 18px;
    backdrop-filter: blur(6px);
}
.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #80cbc4;
    margin-bottom: 16px;
    text-transform: uppercase;
}

/* hero */
.hero { text-align: center; padding: 32px 0 20px; }
.hero h1 {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(90deg, #80cbc4, #4dd0e1, #80deea);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero p { color: #90a4ae; font-size: 1rem; }

/* result */
.result-normal {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border: 1px solid #43a047; border-radius: 16px;
    padding: 28px; text-align: center;
}
.result-copd {
    background: linear-gradient(135deg, #7f0000, #b71c1c);
    border: 1px solid #ef5350; border-radius: 16px;
    padding: 28px; text-align: center;
}
.result-title { font-size: 1.9rem; font-weight: 800; color: #fff; margin-bottom: 6px; }
.result-sub   { font-size: 0.95rem; color: rgba(255,255,255,0.82); }

/* severity badge */
.badge {
    display: inline-block; padding: 4px 14px;
    border-radius: 20px; font-size: 0.78rem; font-weight: 700;
    margin-top: 10px; letter-spacing: 0.05em;
}
.badge-mild     { background: #f9a825; color: #000; }
.badge-moderate { background: #ef6c00; color: #fff; }
.badge-severe   { background: #b71c1c; color: #fff; }
.badge-normal   { background: #2e7d32; color: #fff; }

/* metric tile */
.metric-tile {
    background: rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px 18px; text-align: center;
}
.metric-tile .val { font-size: 1.45rem; font-weight: 700; color: #80cbc4; }
.metric-tile .lbl { font-size: 0.72rem; color: #90a4ae; margin-top: 2px; }

/* progress bar */
.prog-wrap { margin-top: 12px; }
.prog-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #90a4ae; margin-bottom: 3px; }
.prog-bar-bg { background: rgba(255,255,255,0.09); border-radius: 8px; height: 9px; }
.prog-bar-fill { height: 9px; border-radius: 8px; }

/* inputs */
label { color: #b0bec5 !important; font-size: 0.86rem !important; }

/* button */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00838f, #006064) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 12px 36px !important;
    font-size: 1rem !important; font-weight: 700 !important;
    letter-spacing: 0.05em !important; width: 100%;
}
[data-testid="stButton"] > button:hover { opacity: 0.87 !important; }

/* file uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(128,203,196,0.35) !important;
    border-radius: 12px !important;
}

hr { border-color: rgba(255,255,255,0.07) !important; }

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
}

/* about section */
.about-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(128,203,196,0.2);
    border-radius: 14px; padding: 22px 26px; margin-bottom: 16px;
}
.about-card h3 { color: #80cbc4; margin-bottom: 10px; font-size: 1rem; }
.about-card p, .about-card li { color: #b0bec5; font-size: 0.9rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    cnn    = load_model(CNN_MODEL_PATH)
    ann    = load_model(ANN_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return cnn, ann, scaler

cnn_model, ann_model, scaler = load_models()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 COPD AI System")
    st.markdown("---")
    st.markdown("""
**Dual-Model Fusion AI**

| Model | Input | Accuracy |
|-------|-------|----------|
| CNN | Lung Image | 98.9% |
| ANN | Clinical Data | ~95% |
| Fusion | Both | — |
""")
    st.markdown("---")
    st.markdown("""
**Dataset Summary**
- 🖼️ 4,721 training images
- 🖼️ 550 test images
- 📊 1,000 spirometry records
- 2 classes: Normal / Emphysema
""")
    st.markdown("---")
    st.markdown("""
<small style='color:#546e7a'>
⚠️ For research & educational use only.<br>
Not a substitute for clinical diagnosis.
</small>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🫁 COPD Prediction System</h1>
  <p>AI-powered early detection · Lung Imaging + Spirometry Fusion</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_stats, tab_about = st.tabs(["🔍 Predict", "📊 Dataset Stats", "ℹ️ About"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Image upload ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="card"><div class="card-title">📷 Lung Image (CT / X-Ray)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload lung image", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Uploaded image", use_container_width=True)
        else:
            st.markdown("""
<div style='text-align:center;padding:40px 0;color:#546e7a;'>
  <div style='font-size:3rem'>🫁</div>
  <div style='font-size:0.85rem;margin-top:8px'>Upload a CT or X-ray image</div>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Patient details ───────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="card"><div class="card-title">🩺 Patient Details</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120,
                                  value=None, placeholder="e.g. 55", step=1)
        with c2:
            sex = st.selectbox("Sex", ["Select", "Female", "Male"])

        c3, c4 = st.columns(2)
        with c3:
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0,
                                     value=None, placeholder="e.g. 170", step=0.1, format="%.1f")
        with c4:
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0,
                                     value=None, placeholder="e.g. 70", step=0.1, format="%.1f")

        # Auto-calculate BMI
        auto_bmi = None
        if height and weight and height > 0:
            auto_bmi = round(weight / ((height / 100) ** 2), 1)

        c5, c6 = st.columns(2)
        with c5:
            bmi = st.number_input(
                "BMI" + (" (auto-calculated)" if auto_bmi else ""),
                min_value=5.0, max_value=80.0,
                value=float(auto_bmi) if auto_bmi else None,
                placeholder="e.g. 24.2", step=0.1, format="%.1f"
            )
        with c6:
            smoking = st.selectbox("Smoking Status", ["Select", "Never", "Former", "Current"])

        c7, c8 = st.columns(2)
        with c7:
            fev1 = st.number_input("FEV1 (L)", min_value=0.1, max_value=10.0,
                                   value=None, placeholder="e.g. 2.5",
                                   step=0.01, format="%.2f",
                                   help="Forced Expiratory Volume in 1 second")
        with c8:
            fvc = st.number_input("FVC (L)", min_value=0.1, max_value=10.0,
                                  value=None, placeholder="e.g. 3.2",
                                  step=0.01, format="%.2f",
                                  help="Forced Vital Capacity")

        # Live FEV1/FVC preview
        if fev1 and fvc and fvc > 0:
            ratio = fev1 / fvc
            color = "#ef5350" if ratio < 0.7 else "#66bb6a"
            st.markdown(f"""
<div style='margin-top:8px;padding:8px 14px;background:rgba(255,255,255,0.05);
border-radius:8px;font-size:0.82rem;color:{color}'>
  FEV1/FVC = <strong>{ratio:.2f}</strong>
  {'  ⚠️ Below 0.70 — possible obstruction' if ratio < 0.7 else '  ✅ Within normal range'}
</div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button("🔍  Run Prediction", use_container_width=True)

    # ── Prediction logic ──────────────────────────────────────────────────────
    if predict_clicked:
        errors = []
        if uploaded_file is None: errors.append("Upload a lung image.")
        if age is None:           errors.append("Enter Age.")
        if sex == "Select":       errors.append("Select Sex.")
        if height is None:        errors.append("Enter Height.")
        if weight is None:        errors.append("Enter Weight.")
        if bmi is None:           errors.append("Enter BMI.")
        if smoking == "Select":   errors.append("Select Smoking Status.")
        if fev1 is None:          errors.append("Enter FEV1.")
        if fvc is None:           errors.append("Enter FVC.")

        if errors:
            st.error("⚠️ Please fix: " + " · ".join(errors))
        else:
            with st.spinner("Analysing image and clinical data…"):
                result = predict_copd(
                    cnn_model, ann_model, scaler,
                    uploaded_file, age, sex, height, weight, bmi, smoking, fev1, fvc
                )

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            res_col, detail_col = st.columns([1, 1], gap="large")

            # ── Result card ───────────────────────────────────────────────────
            with res_col:
                prob = result["final"]

                # GOLD-inspired severity
                if not result["is_copd"]:
                    severity_badge = '<span class="badge badge-normal">NORMAL</span>'
                elif prob < 0.65:
                    severity_badge = '<span class="badge badge-mild">MILD RISK</span>'
                elif prob < 0.80:
                    severity_badge = '<span class="badge badge-moderate">MODERATE RISK</span>'
                else:
                    severity_badge = '<span class="badge badge-severe">HIGH RISK</span>'

                if result["is_copd"]:
                    st.markdown(f"""
<div class="result-copd">
  <div class="result-title">⚠️ COPD Detected</div>
  <div class="result-sub">Risk probability: <strong>{prob*100:.1f}%</strong></div>
  {severity_badge}
  <div class="result-sub" style="margin-top:12px;font-size:0.82rem">
    Please consult a pulmonologist immediately.
  </div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="result-normal">
  <div class="result-title">✅ Normal</div>
  <div class="result-sub">COPD probability: <strong>{prob*100:.1f}%</strong></div>
  {severity_badge}
  <div class="result-sub" style="margin-top:12px;font-size:0.82rem">
    No significant COPD indicators detected.
  </div>
</div>""", unsafe_allow_html=True)

            # ── Model breakdown ───────────────────────────────────────────────
            with detail_col:
                st.markdown('<div class="card"><div class="card-title">Model Breakdown</div>', unsafe_allow_html=True)

                def prog(label, value, color):
                    pct = value * 100
                    return f"""
<div class="prog-wrap">
  <div class="prog-label"><span>{label}</span><span>{pct:.1f}%</span></div>
  <div class="prog-bar-bg">
    <div class="prog-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
  </div>
</div>"""

                st.markdown(
                    prog("🖼️ CNN (Image) — COPD risk",    result["cnn_prob"],  "#ef5350") +
                    prog("🩺 ANN (Clinical) — COPD risk", result["ann_prob"],  "#ff7043") +
                    prog("⚡ Fusion — Final COPD risk",   result["final"],     "#ffa726"),
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Clinical insights ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🔬 Clinical Insights")

            ratio, ratio_status = fev1_fvc_status(fev1, fvc)
            bmi_cat = bmi_category(bmi)

            i1, i2, i3, i4, i5 = st.columns(5)
            ratio_color = "#ef5350" if ratio < 0.7 else "#66bb6a"
            ratio_icon  = "⚠️" if ratio < 0.7 else "✅"

            with i1:
                st.markdown(f"""
<div class="metric-tile">
  <div class="val" style="color:{ratio_color}">{ratio:.2f}</div>
  <div class="lbl">FEV1/FVC Ratio<br><small>{ratio_icon} {ratio_status}</small></div>
</div>""", unsafe_allow_html=True)
            with i2:
                st.markdown(f"""
<div class="metric-tile">
  <div class="val">{bmi:.1f}</div>
  <div class="lbl">BMI<br><small>{bmi_cat}</small></div>
</div>""", unsafe_allow_html=True)
            with i3:
                st.markdown(f"""
<div class="metric-tile">
  <div class="val">{fev1:.2f} L</div>
  <div class="lbl">FEV1<br><small>Expiratory Vol.</small></div>
</div>""", unsafe_allow_html=True)
            with i4:
                st.markdown(f"""
<div class="metric-tile">
  <div class="val">{fvc:.2f} L</div>
  <div class="lbl">FVC<br><small>Vital Capacity</small></div>
</div>""", unsafe_allow_html=True)
            with i5:
                st.markdown(f"""
<div class="metric-tile">
  <div class="val">{age}</div>
  <div class="lbl">Age<br><small>{smoking} smoker</small></div>
</div>""", unsafe_allow_html=True)

            # ── Debug expander ────────────────────────────────────────────────
            with st.expander("🔧 Raw Model Outputs"):
                d1, d2, d3 = st.columns(3)
                d1.metric("CNN Raw Output",  f"{result['cnn_raw']:.4f}")
                d2.metric("ANN Raw Output",  f"{result['ann_raw']:.4f}")
                d3.metric("Fusion Score",    f"{result['final']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("### 📊 Dataset Overview")

    @st.cache_data
    def load_spirometry():
        return pd.read_csv(SPIROMETRY_CSV)

    try:
        df = load_spirometry()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
<div class="metric-tile">
  <div class="val">{len(df)}</div>
  <div class="lbl">Total Records</div>
</div>""", unsafe_allow_html=True)
        with s2:
            copd_count = int(df["COPD_Label"].sum()) if "COPD_Label" in df.columns else 0
            st.markdown(f"""
<div class="metric-tile">
  <div class="val" style="color:#ef5350">{copd_count}</div>
  <div class="lbl">COPD Cases</div>
</div>""", unsafe_allow_html=True)
        with s3:
            normal_count = len(df) - copd_count
            st.markdown(f"""
<div class="metric-tile">
  <div class="val" style="color:#66bb6a">{normal_count}</div>
  <div class="lbl">Normal Cases</div>
</div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
<div class="metric-tile">
  <div class="val">8</div>
  <div class="lbl">Features</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Spirometry Data Sample**")
            st.dataframe(df.head(10), use_container_width=True, height=300)
        with col_b:
            st.markdown("**Feature Statistics**")
            st.dataframe(df.describe().round(2), use_container_width=True, height=300)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Image Dataset**")
        img_data = {
            "Split": ["Train", "Train", "Test", "Test"],
            "Class": ["Normal", "Emphysema", "Normal", "Emphysema"],
            "Count": [2671, 2050, 300, 250],
        }
        img_df = pd.DataFrame(img_data)
        st.dataframe(img_df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.warning("spirometry_dataset.csv not found in data/")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ About This System")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
<div class="about-card">
<h3>🤖 How It Works</h3>
<p>This system uses a <strong>dual-model fusion</strong> approach:</p>
<ul>
  <li><strong>CNN</strong> — A Convolutional Neural Network analyses lung CT/X-ray images for emphysema patterns (98.9% accuracy)</li>
  <li><strong>ANN</strong> — An Artificial Neural Network evaluates 8 clinical spirometry features for COPD risk (~95% accuracy)</li>
  <li><strong>Fusion</strong> — Both predictions are combined with equal weighting (50/50) for the final result</li>
</ul>
</div>

<div class="about-card">
<h3>🔬 Clinical Background</h3>
<p>COPD (Chronic Obstructive Pulmonary Disease) is a chronic inflammatory lung disease causing obstructed airflow. Key indicators include:</p>
<ul>
  <li><strong>FEV1/FVC &lt; 0.70</strong> — GOLD criteria for airflow obstruction</li>
  <li><strong>Smoking history</strong> — Primary risk factor</li>
  <li><strong>Age &gt; 40</strong> — Increased susceptibility</li>
  <li><strong>Emphysema patterns</strong> on imaging</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with a2:
        st.markdown("""
<div class="about-card">
<h3>📐 Model Architecture</h3>
<p><strong>CNN (Image Model)</strong></p>
<ul>
  <li>Input: 128×128 RGB images</li>
  <li>Conv2D(32) → MaxPool → Conv2D(64) → MaxPool</li>
  <li>Flatten → Dense(128) → Dropout(0.5) → Sigmoid</li>
  <li>Trained on 4,721 images · Tested on 550</li>
</ul>
<p><strong>ANN (Clinical Model)</strong></p>
<ul>
  <li>Input: 8 standardised features</li>
  <li>Dense(8, ReLU) → Dropout(0.3) → Dense(4, ReLU) → Sigmoid</li>
  <li>Trained on 800 records · Tested on 200</li>
</ul>
</div>

<div class="about-card">
<h3>📁 Project Structure</h3>
<pre style="color:#80cbc4;font-size:0.78rem;line-height:1.6">
COPD-Prediction-System/
├── app/copd.py        ← This app
├── config.py          ← All paths & settings
├── utils.py           ← Prediction logic
├── models/            ← Trained .h5 & .pkl
├── data/
│   ├── images/        ← CT/X-ray dataset
│   └── spirometry_dataset.csv
├── notebooks/         ← Training notebooks
├── docs/              ← Report & slides
├── research_papers/   ← Reference PDFs
└── requirements.txt
</pre>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="about-card" style="border-color:rgba(239,83,80,0.3)">
<h3>⚠️ Disclaimer</h3>
<p>This tool is intended for <strong>research and educational purposes only</strong>.
It is not a certified medical device and must not be used as a substitute for professional
clinical diagnosis. Always consult a qualified pulmonologist for medical decisions.</p>
</div>
""", unsafe_allow_html=True)
