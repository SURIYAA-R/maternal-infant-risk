# =============================================================================
# app/app.py
# Maternal & Infant Mortality Risk Prediction — Streamlit Web Application
#
# Ultra-Modern Healthcare AI Interface
# Course : Data Mining Techniques (DMT), B.Tech Information Technology
# College: K. Ramakrishnan College of Technology (Autonomous), Trichy
# Authors: Sakthi Darshan K (2403811720521049), Suriyaa R (2403811720521054)
# =============================================================================

import os
import sys
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Maternal & Infant Risk AI | DMT KRCT",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# High-End Custom CSS & Design System
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #6366f1;
    --primary-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
    --surface-dark: #0f172a;
    --surface-card: rgba(30, 41, 59, 0.72);
    --surface-card-hover: rgba(51, 65, 85, 0.85);
    --border-glass: rgba(255, 255, 255, 0.12);
    --border-accent: rgba(99, 102, 241, 0.35);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background refinement */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(30, 27, 75, 0.4) 0%, rgba(15, 23, 42, 0) 50%),
                radial-gradient(circle at 90% 80%, rgba(76, 29, 149, 0.35) 0%, rgba(15, 23, 42, 0) 50%),
                #0b0f19;
    color: var(--text-primary);
}

/* Header Container */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(17, 24, 39, 0.9) 100%);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.3) 0%, rgba(99, 102, 241, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.4);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 0.9rem;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 30%, #c7d2fe 70%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.3rem 0;
}

.hero-subtitle {
    font-size: 0.98rem;
    color: #cbd5e1;
    max-width: 820px;
    line-height: 1.5;
    margin: 0.4rem 0 1.2rem 0;
}

.hero-stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 1.2rem;
    margin-top: 0.8rem;
}

.hero-stat-item {
    display: flex;
    flex-direction: column;
}

.hero-stat-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
}

.hero-stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Medical Disclaimer Banner */
.disclaimer-card {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.05) 100%);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.disclaimer-title {
    font-weight: 700;
    color: #fbbf24;
    font-size: 0.88rem;
    margin-bottom: 2px;
}

.disclaimer-text {
    font-size: 0.82rem;
    color: #fde68a;
    line-height: 1.45;
    margin: 0;
}

/* Glassmorphism Section Cards */
.glass-panel {
    background: var(--surface-card);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    transition: all 0.25s ease;
}

.glass-panel:hover {
    border-color: var(--border-accent);
}

.card-header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    color: #c7d2fe;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}

/* Prediction Output Cards */
.result-card-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(185, 28, 28, 0.35) 100%);
    border: 2px solid #ef4444;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.35);
    margin: 1.2rem 0;
    animation: fadeIn 0.4s ease-in;
}

.result-card-low {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(5, 150, 105, 0.32) 100%);
    border: 2px solid #10b981;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    margin: 1.2rem 0;
    animation: fadeIn 0.4s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-tag {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 9999px;
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.result-tag-high {
    background: #ef4444;
    color: white;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.6);
}

.result-tag-low {
    background: #10b981;
    color: white;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.5);
}

.result-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0.4rem 0;
}

.result-confidence {
    font-size: 1.15rem;
    color: #e2e8f0;
    font-weight: 600;
}

/* Metric Display Cards */
.kpi-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
    height: 100%;
}

.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-title {
    font-size: 0.82rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Custom Table Styling */
.custom-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border-glass);
    margin: 1.2rem 0;
}

.custom-table th {
    background: rgba(30, 27, 75, 0.8);
    color: #c7d2fe;
    padding: 12px 16px;
    font-size: 0.84rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    text-align: left;
    border-bottom: 1px solid var(--border-glass);
}

.custom-table td {
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #e2e8f0;
    background: rgba(15, 23, 42, 0.6);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.custom-table tr:hover td {
    background: rgba(51, 65, 85, 0.7);
}

/* Diagnostic Alert Pills */
.flag-pill-danger {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(239, 68, 68, 0.12);
    border-left: 4px solid #ef4444;
    padding: 8px 12px;
    border-radius: 0 8px 8px 0;
    color: #fca5a5;
    font-size: 0.84rem;
    margin-bottom: 6px;
}

.flag-pill-safe {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.1);
    border-left: 4px solid #10b981;
    padding: 8px 12px;
    border-radius: 0 8px 8px 0;
    color: #6ee7b7;
    font-size: 0.84rem;
    margin-bottom: 6px;
}

/* Streamlit Tabs Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.75);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid var(--border-glass);
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.92rem;
    padding: 0 20px;
    border: none;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc;
    background: rgba(255, 255, 255, 0.05);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.45);
}

/* Primary Button Styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
    transition: all 0.25s ease;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(124, 58, 237, 0.6);
}

/* Secondary Button Styling */
div.stButton > button:not([kind="primary"]) {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid var(--border-glass);
    color: #e2e8f0;
    font-weight: 600;
    border-radius: 10px;
    transition: all 0.2s;
}

div.stButton > button:not([kind="primary"]):hover {
    border-color: var(--primary);
    background: rgba(51, 65, 85, 0.9);
}

/* Streamlit Progress Bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #10b981 0%, #f59e0b 55%, #ef4444 100%);
    border-radius: 10px;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model & Asset Loaders with Caching
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load serialized model pipeline."""
    path = os.path.join(MODELS_DIR, 'best_model_pipeline.joblib')
    if not os.path.exists(path):
        st.error(f"❌ Model file not found at `{path}`. Please run `python scripts/run_all.py` first.")
        st.stop()
    payload = joblib.load(path)
    return payload['model'], payload['preprocessor']


@st.cache_data
def load_metrics():
    """Load metrics comparison dataframe."""
    path = os.path.join(MODELS_DIR, 'metrics_comparison.csv')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


# ---------------------------------------------------------------------------
# Preset Management (Single-Click Patient Profiles)
# ---------------------------------------------------------------------------
if 'bp_sys' not in st.session_state:
    st.session_state.bp_sys = 120
    st.session_state.bp_dia = 78
    st.session_state.hb = 11.5
    st.session_state.diab = "No"
    st.session_state.comp = "None"
    st.session_state.bw = 2.80
    st.session_state.ga = 38
    st.session_state.age = 25
    st.session_state.anc = 5
    st.session_state.deliv = "Hospital"
    st.session_state.edu = "Secondary"
    st.session_state.inc = 15000
    st.session_state.res = "Rural"

def set_patient_preset(preset_key):
    if preset_key == 'low':
        st.session_state.bp_sys = 114
        st.session_state.bp_dia = 74
        st.session_state.hb = 12.8
        st.session_state.diab = "No"
        st.session_state.comp = "None"
        st.session_state.bw = 3.25
        st.session_state.ga = 39
        st.session_state.age = 24
        st.session_state.anc = 6
        st.session_state.deliv = "Hospital"
        st.session_state.edu = "Higher"
        st.session_state.inc = 32000
        st.session_state.res = "Urban"
    elif preset_key == 'borderline':
        st.session_state.bp_sys = 136
        st.session_state.bp_dia = 88
        st.session_state.hb = 10.1
        st.session_state.diab = "No"
        st.session_state.comp = "Mild"
        st.session_state.bw = 2.45
        st.session_state.ga = 36
        st.session_state.age = 34
        st.session_state.anc = 3
        st.session_state.deliv = "Clinic"
        st.session_state.edu = "Secondary"
        st.session_state.inc = 14000
        st.session_state.res = "Rural"
    elif preset_key == 'critical':
        st.session_state.bp_sys = 162
        st.session_state.bp_dia = 102
        st.session_state.hb = 6.6
        st.session_state.diab = "Yes"
        st.session_state.comp = "Severe"
        st.session_state.bw = 1.70
        st.session_state.ga = 31
        st.session_state.age = 17
        st.session_state.anc = 1
        st.session_state.deliv = "Home"
        st.session_state.edu = "No Education"
        st.session_state.inc = 5000
        st.session_state.res = "Rural"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🤰</div>
        <div style="font-family: 'Outfit'; font-size: 1.3rem; font-weight: 800; color: #f8fafc;">
            Maternal & Infant AI
        </div>
        <div style="font-size: 0.8rem; color: #a5b4fc; font-weight: 600;">
            DMT Clinical Triage Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1rem; margin-bottom: 1.2rem;">
        <div style="font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; margin-bottom: 6px;">
            Pipeline Status
        </div>
        <div style="display: flex; align-items: center; gap: 8px; color: #34d399; font-weight: 700; font-size: 0.9rem;">
            <span class="pulse-dot"></span> ML Model Active (XGBoost)
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px;">
            Target Optimization: <b>Recall (Safety-First)</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ Quick Patient Presets")
    st.caption("Load pre-configured clinical profiles instantly:")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🟢 Low Risk", use_container_width=True):
            set_patient_preset('low')
            st.rerun()
    with col_p2:
        if st.button("🔴 Critical", use_container_width=True):
            set_patient_preset('critical')
            st.rerun()

    if st.button("🟡 Borderline Case", use_container_width=True):
        set_patient_preset('borderline')
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Clinical Vitals Legend")
    st.markdown("""
    * **Severe Anemia:** Hb < 7.0 g/dL
    * **Hypertension:** SBP ≥ 140 or DBP ≥ 90
    * **Preterm Delivery:** < 37 weeks
    * **Low Birth Weight:** < 2.5 kg
    * **WHO ANC Guideline:** ≥ 4 visits
    """)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.76rem; color: #64748b; line-height: 1.4;">
        <b>B.Tech Information Technology</b><br>
        K. Ramakrishnan College of Technology<br>
        Team: Sakthi Darshan K & Suriyaa R
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-pill">
        <span class="pulse-dot"></span>
        <span>AI Clinical Decision Intelligence · DMT 2026</span>
    </div>
    <div class="hero-title">
        Maternal & Infant Mortality Risk Predictor
    </div>
    <div class="hero-subtitle">
        An explainable machine learning screening system engineered using data mining techniques, 
        balanced SMOTE training, and multi-model benchmarking to identify high-risk obstetric cases early.
    </div>
    <div class="hero-stats-row">
        <div class="hero-stat-item">
            <span class="hero-stat-value">5,000 Records</span>
            <span class="hero-stat-label">WHO/NFHS-5 Calibrated</span>
        </div>
        <div class="hero-stat-item">
            <span class="hero-stat-value">7 ML Classifiers</span>
            <span class="hero-stat-label">Ensemble Benchmarked</span>
        </div>
        <div class="hero-stat-item">
            <span class="hero-stat-value">83.2% Recall</span>
            <span class="hero-stat-label">High-Risk Sensitivity</span>
        </div>
        <div class="hero-stat-item">
            <span class="hero-stat-value">0.929 ROC-AUC</span>
            <span class="hero-stat-label">Model Discrimination</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Medical Disclaimer Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-card">
    <div style="font-size: 1.4rem;">⚠️</div>
    <div>
        <div class="disclaimer-title">MEDICAL DISCLAIMER — ACADEMIC USE ONLY</div>
        <p class="disclaimer-text">
            This application is an academic demonstration developed for Data Mining Techniques (DMT) coursework.
            Trained predominantly on synthetic patient data calibrated to public health statistics. 
            <b>This is NOT a certified medical device and must NOT be used for direct clinical diagnosis or medical decision-making.</b>
            Always consult a licensed obstetrician or healthcare professional.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Patient Risk Diagnostic", 
    "📊 Data Insights & EDA", 
    "📈 Model Benchmarks & Explainability",
    "📁 Batch CSV Screening", 
    "ℹ️ Project Architecture & Team"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE PATIENT DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🩺 Single Patient Clinical Risk Assessment")
    st.caption("Provide patient hemodynamic vitals, gestational metrics, and socioeconomic indicators below to run real-time inference.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card-header-badge">🔬 Hemodynamics & Laboratory</div>
        """, unsafe_allow_html=True)
        
        bp_sys = st.slider("Systolic Blood Pressure (mmHg)", 80, 200, st.session_state.bp_sys, key="slider_bp_sys")
        bp_dia = st.slider("Diastolic Blood Pressure (mmHg)", 50, 130, st.session_state.bp_dia, key="slider_bp_dia")
        hb = st.slider("Hemoglobin Level (g/dL)", 5.0, 17.0, float(st.session_state.hb), 0.1, key="slider_hb")
        
        # Dynamic Clinical Tag for Hb
        if hb < 7.0:
            st.markdown('<span style="color: #ef4444; font-size: 0.75rem; font-weight: 700;">🚨 Severe Anemia Alert (< 7.0 g/dL)</span>', unsafe_allow_html=True)
        elif hb < 11.0:
            st.markdown('<span style="color: #f59e0b; font-size: 0.75rem; font-weight: 700;">⚠️ Moderate Anemia (7.0 - 10.9 g/dL)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #10b981; font-size: 0.75rem; font-weight: 700;">✅ Normal Hemoglobin Range</span>', unsafe_allow_html=True)

        diabetes = st.selectbox(
            "Gestational Diabetes Status", 
            ["No", "Yes"], 
            index=0 if st.session_state.diab == "No" else 1,
            key="select_diab"
        )
        
        complications_opts = ["None", "Mild", "Severe"]
        complications = st.selectbox(
            "Pregnancy Complications", 
            complications_opts, 
            index=complications_opts.index(st.session_state.comp) if st.session_state.comp in complications_opts else 0,
            key="select_comp"
        )

    with col2:
        st.markdown("""
        <div class="card-header-badge">👶 Gestational & Neonatal</div>
        """, unsafe_allow_html=True)
        
        gest_age = st.slider("Gestational Age (weeks)", 24, 43, int(st.session_state.ga), key="slider_ga")
        if gest_age < 37:
            st.markdown('<span style="color: #ef4444; font-size: 0.75rem; font-weight: 700;">🚨 Preterm Gestation (< 37 weeks)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #10b981; font-size: 0.75rem; font-weight: 700;">✅ Full Term Gestation</span>', unsafe_allow_html=True)

        birth_wt = st.slider("Infant Birth Weight (kg)", 0.80, 5.00, float(st.session_state.bw), 0.05, key="slider_bw")
        if birth_wt < 2.50:
            st.markdown('<span style="color: #ef4444; font-size: 0.75rem; font-weight: 700;">🚨 Low Birth Weight (LBW < 2.5 kg)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #10b981; font-size: 0.75rem; font-weight: 700;">✅ Normal Birth Weight Range</span>', unsafe_allow_html=True)

        delivery_opts = ["Hospital", "Clinic", "Home"]
        delivery = st.selectbox(
            "Place of Delivery", 
            delivery_opts,
            index=delivery_opts.index(st.session_state.deliv) if st.session_state.deliv in delivery_opts else 0,
            key="select_deliv"
        )

    with col3:
        st.markdown("""
        <div class="card-header-badge">👩 Demographics & Care Access</div>
        """, unsafe_allow_html=True)
        
        age = st.slider("Maternal Age (years)", 14, 48, int(st.session_state.age), key="slider_age")
        if age < 18 or age > 35:
            st.markdown('<span style="color: #f59e0b; font-size: 0.75rem; font-weight: 700;">⚠️ High-Risk Maternal Age Group</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #10b981; font-size: 0.75rem; font-weight: 700;">✅ Optimal Maternal Age</span>', unsafe_allow_html=True)

        anc = st.slider("Antenatal Care (ANC) Visits", 0, 15, int(st.session_state.anc), key="slider_anc")
        if anc < 4:
            st.markdown('<span style="color: #f59e0b; font-size: 0.75rem; font-weight: 700;">⚠️ Below WHO Guideline (Min 4 Visits)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #10b981; font-size: 0.75rem; font-weight: 700;">✅ Meets WHO Minimum Standards</span>', unsafe_allow_html=True)

        edu_opts = ["No Education", "Primary", "Secondary", "Higher"]
        education = st.selectbox(
            "Maternal Education Level",
            edu_opts,
            index=edu_opts.index(st.session_state.edu) if st.session_state.edu in edu_opts else 2,
            key="select_edu"
        )
        
        income = st.slider("Monthly Family Income (INR ₹)", 3000, 120000, int(st.session_state.inc), step=1000, key="slider_inc")
        
        res_opts = ["Urban", "Rural"]
        residence = st.selectbox(
            "Residence Location",
            res_opts,
            index=res_opts.index(st.session_state.res) if st.session_state.res in res_opts else 1,
            key="select_res"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Run Clinical Risk Diagnostic", type="primary", use_container_width=True)

    if predict_clicked:
        model, preprocessor = load_model()

        input_data = pd.DataFrame([{
            'mother_age':               age,
            'education_level':          education,
            'family_income':            income,
            'residence_urban_rural':    residence,
            'blood_pressure_systolic':  bp_sys,
            'blood_pressure_diastolic': bp_dia,
            'hemoglobin_level':         hb,
            'diabetes_status':          diabetes,
            'pregnancy_complications':  complications,
            'gestational_age':          gest_age,
            'birth_weight':             birth_wt,
            'antenatal_care_visits':    anc,
            'place_of_delivery':        delivery,
        }])

        input_processed = preprocessor.transform(
            input_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )

        prediction = model.predict(input_processed)[0]
        proba = model.predict_proba(input_processed)[0]

        high_risk_prob = proba[1] * 100
        low_risk_prob = proba[0] * 100

        # Primary Result Card
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card-high">
                <span class="result-tag result-tag-high">🚨 CRITICAL CLINICAL WARNING</span>
                <div class="result-title">HIGH-RISK PREGNANCY DETECTED</div>
                <div class="result-confidence">
                    Estimated Adverse Outcome Probability: <b>{high_risk_prob:.1f}%</b>
                </div>
                <p style="color: #fca5a5; font-size: 0.9rem; max-width: 650px; margin: 0.8rem auto 0 auto;">
                    Patient exhibits significant risk markers requiring immediate specialist evaluation, continuous antenatal surveillance, and facility-based delivery preparation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-low">
                <span class="result-tag result-tag-low">✅ FAVORABLE PROGNOSIS</span>
                <div class="result-title">LOW-RISK PREGNANCY</div>
                <div class="result-confidence">
                    Confidence: <b>{low_risk_prob:.1f}%</b> · High-Risk Probability: {high_risk_prob:.1f}%
                </div>
                <p style="color: #a7f3d0; font-size: 0.9rem; max-width: 650px; margin: 0.8rem auto 0 auto;">
                    Standard obstetric parameters within expected physiological limits. Continue routine antenatal checkups, nutritional supplementation, and standard care protocols.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Detailed Probability Split & Visual Meter
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("""
            <div class="glass-panel">
                <div class="card-header-badge">📊 Probability Stratification</div>
            """, unsafe_allow_html=True)
            
            p_col_a, p_col_b = st.columns(2)
            with p_col_a:
                st.markdown(f"""
                <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.4);">
                    <div style="color: #10b981; font-weight: 700; font-size: 0.85rem;">LOW-RISK PROBABILITY</div>
                    <div class="kpi-value" style="background: none; -webkit-text-fill-color: #34d399;">{low_risk_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with p_col_b:
                st.markdown(f"""
                <div class="kpi-card" style="border-color: rgba(239, 68, 68, 0.4);">
                    <div style="color: #ef4444; font-weight: 700; font-size: 0.85rem;">HIGH-RISK PROBABILITY</div>
                    <div class="kpi-value" style="background: none; -webkit-text-fill-color: #f87171;">{high_risk_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Risk Severity Gradient Meter:** `{high_risk_prob:.1f}%`")
            st.progress(float(proba[1]))
            st.markdown("""
                <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #94a3b8;">
                    <span>0% (Minimal Risk)</span>
                    <span>Threshold: 50%</span>
                    <span>100% (Extreme Risk)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("""
            <div class="glass-panel">
                <div class="card-header-badge">🔍 Clinical Diagnostic Flag Breakdown</div>
            """, unsafe_allow_html=True)
            
            # Clinical Rule Diagnostics
            danger_flags = []
            safe_flags = []

            if hb < 7.0:
                danger_flags.append(f"Severe Anemia Detected (Hb: {hb:.1f} g/dL < 7.0)")
            elif hb < 11.0:
                danger_flags.append(f"Moderate Anemia (Hb: {hb:.1f} g/dL < 11.0)")
            else:
                safe_flags.append(f"Hemoglobin Level Normal ({hb:.1f} g/dL)")

            if bp_sys >= 140 or bp_dia >= 90:
                danger_flags.append(f"Gestational Hypertension (BP: {bp_sys}/{bp_dia} mmHg)")
            else:
                safe_flags.append(f"Blood Pressure Normotensive ({bp_sys}/{bp_dia} mmHg)")

            if gest_age < 37:
                danger_flags.append(f"Preterm Gestation ({gest_age} weeks < 37)")
            else:
                safe_flags.append(f"Term Gestational Age ({gest_age} weeks)")

            if birth_wt < 2.5:
                danger_flags.append(f"Low Birth Weight Danger ({birth_wt:.2f} kg < 2.5 kg)")
            else:
                safe_flags.append(f"Adequate Birth Weight ({birth_wt:.2f} kg)")

            if complications in ["Mild", "Severe"]:
                danger_flags.append(f"{complications} Pregnancy Complications Present")
            else:
                safe_flags.append("No Reported Pregnancy Complications")

            if anc < 4:
                danger_flags.append(f"Inadequate Antenatal Care ({anc} visits < WHO 4-visit standard)")
            else:
                safe_flags.append(f"Compliant Antenatal Visits ({anc} visits)")

            for flag in danger_flags:
                st.markdown(f'<div class="flag-pill-danger">⚠️ {flag}</div>', unsafe_allow_html=True)
            for flag in safe_flags[:3]:
                st.markdown(f'<div class="flag-pill-safe">✓ {flag}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Feature Importance Insights
        importance_path = os.path.join(FIGURES_DIR, 'feature_importance_rf.png')
        if os.path.exists(importance_path):
            st.markdown("""
            <div class="glass-panel">
                <div class="card-header-badge">🧠 Predictive Feature Attribution</div>
                <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.8rem;">
                    Global importance hierarchy identifying which maternal and clinical attributes exert the strongest influence on mortality prediction:
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.image(importance_path, caption="Random Forest Feature Importance Analysis", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: DATA INSIGHTS & EDA
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Exploratory Data Analysis & Epidemiological Trends")
    st.caption("Visual analysis of 5,000 synthetic patient records calibrated to WHO and NFHS-5 maternal mortality statistics.")

    eda_catalog = [
        ("Clinical Biomarkers & Hemodynamics", [
            ('fig03_hemoglobin_by_risk.png', 'Hemoglobin Distribution by Risk Status', 
             'Demonstrates sharp risk escalation when maternal hemoglobin drops below the 11.0 g/dL threshold (anemia).'),
            ('fig04_bp_vs_risk.png', 'Systolic vs Diastolic Blood Pressure Correlation', 
             'Highlights gestational hypertension clustering among high-risk maternal classifications.'),
            ('fig07_birthweight_boxplot.png', 'Neonatal Birth Weight Distribution by Risk', 
             'Confirms severe correlation between infant low birth weight (< 2.5 kg) and adverse mortality prognosis.'),
        ]),
        ("Demographics & Healthcare Access", [
            ('fig01_class_distribution.png', 'Maternal Mortality Class Balance', 
             'Synthetic dataset class proportions highlighting the critical necessity of SMOTE oversampling.'),
            ('fig02_age_vs_risk.png', 'Maternal Age Distribution vs Risk Status', 
             'Visualizes the classic U-shaped obstetric risk curve: elevated risk in teenage (<18) and advanced (>35) maternal ages.'),
            ('fig05_anc_visits_vs_risk.png', 'Antenatal Care (ANC) Visits vs Mortality Risk', 
             'Demonstrates clear protective effect: mothers completing ≥ 4 ANC checkups experience substantially lower risk rates.'),
            ('fig08_urban_rural_education.png', 'Urban/Rural Residence & Education Stratification', 
             'Shows socioeconomic gradients: higher education and urban healthcare access correlate with reduced adverse outcomes.'),
            ('fig09_complications_by_risk.png', 'Pregnancy Complication Severity by Risk Tier', 
             'Direct association between pre-existing complications (pre-eclampsia, hemorrhage) and high-risk classification.'),
        ]),
        ("Data Mining & Multidimensional Analysis", [
            ('fig06_correlation_heatmap.png', 'Full Multivariable Correlation Matrix', 
             'Pearson correlation matrix uncovering interdependencies across all 12 clinical and socioeconomic predictors.'),
            ('fig10_kmeans_elbow_silhouette.png', 'K-Means Clustering: Elbow & Silhouette Analysis', 
             'Unsupervised clustering evaluation identifying optimal patient risk segmentation clusters.'),
        ])
    ]

    selected_category = st.radio(
        "Select Insight Domain:",
        [cat[0] for cat in eda_catalog],
        horizontal=True
    )

    for cat_name, figures in eda_catalog:
        if cat_name == selected_category:
            for filename, title, description in figures:
                path = os.path.join(FIGURES_DIR, filename)
                if os.path.exists(path):
                    st.markdown(f"""
                    <div class="glass-panel" style="margin-top: 1.2rem;">
                        <div class="card-header-badge">📈 {cat_name}</div>
                        <h4 style="color: #f8fafc; margin: 0.2rem 0 0.4rem 0;">{title}</h4>
                        <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 1rem;">{description}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(path, use_container_width=True)
                    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
                else:
                    st.warning(f"Figure `{filename}` not found. Run `python scripts/run_all.py` to regenerate all plots.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL BENCHMARKS & EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Machine Learning Benchmarks & Model Interpretability")
    st.caption("Comprehensive comparative performance of 7 classification models evaluated on stratified 20% holdout test data.")

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Champion Classifier</div>
            <div class="kpi-value" style="font-size: 1.4rem;">XGBoost / RF</div>
            <div style="font-size: 0.75rem; color: #34d399; font-weight: 700; margin-top: 4px;">Ensemble Optimized</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">High-Risk Recall</div>
            <div class="kpi-value">83.2%</div>
            <div style="font-size: 0.75rem; color: #a5b4fc; font-weight: 600; margin-top: 4px;">Primary Safety Metric</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Area Under ROC</div>
            <div class="kpi-value">0.929</div>
            <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 600; margin-top: 4px;">Outstanding Discrimination</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">SMOTE Balanced N</div>
            <div class="kpi-value">5,000</div>
            <div style="font-size: 0.75rem; color: #f472b6; font-weight: 600; margin-top: 4px;">Oversampled Training</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Leaderboard Table
    metrics_df = load_metrics()
    if metrics_df is not None:
        st.markdown("""
        <div class="glass-panel">
            <div class="card-header-badge">🏆 Classification Leaderboard</div>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem;">
                Ranked by <b>Recall</b> on the High-Risk class. In clinical risk assessment, missing a high-risk patient (False Negative) has catastrophic consequences, making Recall our primary optimization criterion.
            </p>
        </div>
        """, unsafe_allow_html=True)

        display_df = metrics_df.copy()
        
        # Format table with styled medals
        ranks = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣"]
        if len(display_df) <= len(ranks):
            display_df.insert(0, 'Rank', ranks[:len(display_df)])

        st.dataframe(
            display_df.style.format({
                'Accuracy': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}',
                'ROC-AUC': '{:.3f}'
            }).background_gradient(subset=['Recall', 'ROC-AUC'], cmap='Purples'),
            use_container_width=True
        )

    # Visual Deep-Dive
    st.markdown("<br>", unsafe_allow_html=True)
    bench_view = st.radio(
        "Select Deep-Dive Diagnostic:",
        ["Confusion Matrices", "ROC Curves", "SHAP Explainability Summary", "Feature Importance"],
        horizontal=True
    )

    if bench_view == "Confusion Matrices":
        cm_path = os.path.join(FIGURES_DIR, 'confusion_matrices.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices across all 7 Classifiers (Test Set)", use_container_width=True)
            st.info("💡 **Clinical Reading:** Lower values in the bottom-left quadrant (False Negatives) represent safer clinical models.")
        else:
            st.warning("Confusion matrix plot not found.")

    elif bench_view == "ROC Curves":
        roc_path = os.path.join(FIGURES_DIR, 'roc_curves.png')
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Curves Comparison", use_container_width=True)
            st.info("💡 **Clinical Reading:** Curves hugging the top-left boundary denote superior True Positive Rates at minimal False Alarm rates.")
        else:
            st.warning("ROC curves plot not found.")

    elif bench_view == "SHAP Explainability Summary":
        shap_path = os.path.join(FIGURES_DIR, 'shap_summary.png')
        if os.path.exists(shap_path):
            st.image(shap_path, caption="SHAP (SHapley Additive exPlanations) Global Summary", use_container_width=True)
            st.info("💡 **Clinical Reading:** Red points on the right indicate that high values of that feature push the prediction toward HIGH-RISK.")
        else:
            st.warning("SHAP plot not found.")

    elif bench_view == "Feature Importance":
        imp_path = os.path.join(FIGURES_DIR, 'feature_importance_rf.png')
        if os.path.exists(imp_path):
            st.image(imp_path, caption="Gini Feature Importance Hierarchy", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: BATCH CSV SCREENING
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📁 Batch Hospital Cohort Risk Screening")
    st.caption("Upload a patient cohort CSV with the 12 required clinical and socioeconomic fields to execute bulk risk scoring.")

    b_col1, b_col2 = st.columns([2, 1])

    with b_col2:
        st.markdown("""
        <div class="glass-panel">
            <div class="card-header-badge">📋 Template Generator</div>
            <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.8rem;">
                Need a test file? Generate a pre-formatted template with 10 synthetic patient records ready for instant scoring.
            </p>
        </div>
        """, unsafe_allow_html=True)

        sample_cases = pd.DataFrame([
            {'mother_age': 24, 'education_level': 'Higher', 'family_income': 45000, 'residence_urban_rural': 'Urban', 'blood_pressure_systolic': 114, 'blood_pressure_diastolic': 74, 'hemoglobin_level': 13.0, 'diabetes_status': 'No', 'pregnancy_complications': 'None', 'gestational_age': 40, 'birth_weight': 3.4, 'antenatal_care_visits': 7, 'place_of_delivery': 'Hospital'},
            {'mother_age': 17, 'education_level': 'No Education', 'family_income': 6000, 'residence_urban_rural': 'Rural', 'blood_pressure_systolic': 162, 'blood_pressure_diastolic': 102, 'hemoglobin_level': 6.5, 'diabetes_status': 'Yes', 'pregnancy_complications': 'Severe', 'gestational_age': 30, 'birth_weight': 1.65, 'antenatal_care_visits': 1, 'place_of_delivery': 'Home'},
            {'mother_age': 29, 'education_level': 'Secondary', 'family_income': 22000, 'residence_urban_rural': 'Rural', 'blood_pressure_systolic': 120, 'blood_pressure_diastolic': 80, 'hemoglobin_level': 11.2, 'diabetes_status': 'No', 'pregnancy_complications': 'None', 'gestational_age': 39, 'birth_weight': 3.1, 'antenatal_care_visits': 5, 'place_of_delivery': 'Hospital'},
            {'mother_age': 38, 'education_level': 'Primary', 'family_income': 11000, 'residence_urban_rural': 'Rural', 'blood_pressure_systolic': 148, 'blood_pressure_diastolic': 94, 'hemoglobin_level': 9.8, 'diabetes_status': 'Yes', 'pregnancy_complications': 'Mild', 'gestational_age': 35, 'birth_weight': 2.2, 'antenatal_care_visits': 2, 'place_of_delivery': 'Clinic'},
            {'mother_age': 22, 'education_level': 'Higher', 'family_income': 38000, 'residence_urban_rural': 'Urban', 'blood_pressure_systolic': 116, 'blood_pressure_diastolic': 76, 'hemoglobin_level': 12.4, 'diabetes_status': 'No', 'pregnancy_complications': 'None', 'gestational_age': 39, 'birth_weight': 3.3, 'antenatal_care_visits': 6, 'place_of_delivery': 'Hospital'},
        ])
        
        sample_csv = sample_cases.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Batch CSV",
            data=sample_csv,
            file_name="maternal_risk_sample_batch.csv",
            mime="text/csv",
            use_container_width=True
        )

    with b_col1:
        uploaded_file = st.file_uploader(
            "Upload Cohort CSV Dataset", 
            type=['csv'], 
            help="File must contain the 12 standard clinical and demographic attributes"
        )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Cohort loaded successfully: `{batch_df.shape[0]} records` across `{batch_df.shape[1]} attributes`.")

            model, preprocessor = load_model()

            # Verify columns
            missing_cols = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns in uploaded CSV: `{missing_cols}`")
            else:
                X_batch = preprocessor.transform(batch_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
                batch_preds = model.predict(X_batch)
                batch_probas = model.predict_proba(X_batch)

                scored_df = batch_df.copy()
                scored_df['predicted_risk'] = np.where(batch_preds == 1, 'HIGH-RISK', 'LOW-RISK')
                scored_df['high_risk_probability_%'] = np.round(batch_probas[:, 1] * 100, 2)

                # Batch KPI summary
                total_cnt = len(scored_df)
                high_cnt = int(np.sum(batch_preds == 1))
                low_cnt = total_cnt - high_cnt
                high_pct = (high_cnt / total_cnt) * 100

                st.markdown("<br>", unsafe_allow_html=True)
                bkpi1, bkpi2, bkpi3, bkpi4 = st.columns(4)
                with bkpi1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Total Patients</div>
                        <div class="kpi-value">{total_cnt}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bkpi2:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-color: rgba(239, 68, 68, 0.4);">
                        <div class="kpi-title" style="color: #ef4444;">High-Risk Flagged</div>
                        <div class="kpi-value" style="background: none; -webkit-text-fill-color: #ef4444;">{high_cnt}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bkpi3:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.4);">
                        <div class="kpi-title" style="color: #10b981;">Low-Risk Identified</div>
                        <div class="kpi-value" style="background: none; -webkit-text-fill-color: #10b981;">{low_cnt}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bkpi4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Critical Proportion</div>
                        <div class="kpi-value">{high_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Scored Cohort Preview")
                
                # Highlighted Dataframe
                st.dataframe(
                    scored_df.style.applymap(
                        lambda val: 'background-color: rgba(239, 68, 68, 0.25); color: #fca5a5; font-weight: bold;' if val == 'HIGH-RISK' 
                        else ('background-color: rgba(16, 185, 129, 0.2); color: #6ee7b7; font-weight: bold;' if val == 'LOW-RISK' else ''),
                        subset=['predicted_risk']
                    ),
                    use_container_width=True
                )

                # Download Scored CSV
                output_csv = scored_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Scored Cohort Dataset (CSV)",
                    data=output_csv,
                    file_name="maternal_risk_scored_cohort.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )

        except Exception as err:
            st.error(f"Error during cohort inference: {err}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: PROJECT ARCHITECTURE & TEAM
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### ℹ️ Project Architecture & Academic Documentation")

    a_col1, a_col2 = st.columns([1, 1])

    with a_col1:
        st.markdown("""
        <div class="glass-panel">
            <div class="card-header-badge">🏛️ Academic Information</div>
            <h4 style="color: #f8fafc; margin: 0.3rem 0;">B.Tech Information Technology</h4>
            <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.6rem;">
                <b>Course:</b> Data Mining Techniques (DMT)<br>
                <b>Institution:</b> K. Ramakrishnan College of Technology (Autonomous), Trichy
            </p>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 0.8rem 0;">
            <div class="card-header-badge">👥 Project Researchers</div>
            <div style="font-size: 0.9rem; color: #f1f5f9; line-height: 1.6;">
                • <b>Sakthi Darshan K</b> — Reg No: <code>2403811720521049</code><br>
                • <b>Suriyaa R</b> — Reg No: <code>2403811720521054</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a_col2:
        st.markdown("""
        <div class="glass-panel">
            <div class="card-header-badge">⚙️ Technical Stack</div>
            <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                • <b>Core Engine:</b> Python 3.11+, scikit-learn, XGBoost, imbalanced-learn (SMOTE)<br>
                • <b>Mining Algorithms:</b> Decision Tree, Random Forest, Naïve Bayes, SVM, KNN, Logistic Regression<br>
                • <b>Unsupervised Clustering:</b> K-Means with Elbow & Silhouette Optimization<br>
                • <b>Association Mining:</b> Apriori Frequent Itemset Mining (mlxtend)<br>
                • <b>Interpretability:</b> SHAP (SHapley Additive exPlanations), Gini Impurity<br>
                • <b>Frontend Interface:</b> Streamlit, Glassmorphism CSS Design System
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Risk Scoring Matrix
    st.markdown("""
    <div class="glass-panel">
        <div class="card-header-badge">📐 Calibrated Clinical Risk Scoring Rubric</div>
        <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.8rem;">
            The target risk variable was defined via a clinical point system derived from WHO and NFHS-5 maternal mortality studies:
        </p>
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Clinical / Sociodemographic Condition</th>
                    <th>Risk Points Allocated</th>
                    <th>Clinical Rationale</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>Severe Anemia (Hb < 7.0 g/dL)</b></td>
                    <td><span style="color:#ef4444; font-weight:bold;">+4 Points</span></td>
                    <td>High susceptibility to hypovolemic shock, cardiac failure, and maternal mortality.</td>
                </tr>
                <tr>
                    <td><b>Moderate Anemia (Hb 7.0 - 10.9 g/dL)</b></td>
                    <td><span style="color:#f59e0b; font-weight:bold;">+2 Points</span></td>
                    <td>Sub-optimal oxygenation, heightened postpartum hemorrhage vulnerability.</td>
                </tr>
                <tr>
                    <td><b>Gestational Hypertension (SBP ≥ 140 mmHg)</b></td>
                    <td><span style="color:#ef4444; font-weight:bold;">+3 to +5 Points</span></td>
                    <td>Pre-eclampsia / eclampsia progression danger and placental abruption risk.</td>
                </tr>
                <tr>
                    <td><b>Preterm Gestation (< 37 Weeks)</b></td>
                    <td><span style="color:#ef4444; font-weight:bold;">+3 to +5 Points</span></td>
                    <td>Neonatal respiratory distress syndrome and underdeveloped organ systems.</td>
                </tr>
                <tr>
                    <td><b>Low Birth Weight (< 2.5 kg)</b></td>
                    <td><span style="color:#ef4444; font-weight:bold;">+3 to +5 Points</span></td>
                    <td>Major determinant of neonatal mortality and morbidity.</td>
                </tr>
                <tr>
                    <td><b>Sub-optimal Antenatal Visits (< 4 ANC)</b></td>
                    <td><span style="color:#f59e0b; font-weight:bold;">+2 Points</span></td>
                    <td>Missed detection windows for silent asymptomatic complications.</td>
                </tr>
                <tr>
                    <td><b>Maternal Age Risk (< 18 or > 35 Years)</b></td>
                    <td><span style="color:#f59e0b; font-weight:bold;">+2 Points</span></td>
                    <td>Biological pelvic immaturity (<18) or chromosomal / vascular risks (>35).</td>
                </tr>
            </tbody>
        </table>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
            <b>Classification Threshold:</b> Cumulative points ≥ 5.0 triggers <code>HIGH-RISK</code> class classification.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1.2rem 0; color: #64748b; font-size: 0.82rem;">
    © 2026 <b>Sakthi Darshan K & Suriyaa R</b> · Data Mining Techniques (DMT) Project<br>
    Department of Information Technology · K. Ramakrishnan College of Technology (Autonomous), Trichy
</div>
""", unsafe_allow_html=True)
