# =============================================================================
# app/app.py
# Maternal & Infant Mortality Risk Prediction — Streamlit Web Application
#
# Tabs:
#   1. Prediction   — single-patient risk prediction with probability gauge
#   2. Data Insights — EDA charts from reports/figures/
#   3. Model Perf.  — metrics table, confusion matrices, ROC curves
#   4. Batch Predict — upload CSV, download scored CSV
#   5. About        — project info, team, disclaimer
#
# Authors : Sakthi Darshan K, Suriyaa R
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths — works both locally and on Streamlit Cloud
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

# Add project root so src/ is importable
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Medical Disclaimer — shown on EVERY page
# ---------------------------------------------------------------------------
DISCLAIMER = """
⚠️ **MEDICAL DISCLAIMER**

🔬 *Academic demonstration only. Trained on SYNTHETIC data generated for
coursework. This is NOT a medical device and must NOT be used for clinical
decision-making. Always consult a qualified healthcare professional.*
"""

# ---------------------------------------------------------------------------
# Cache model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the serialised model + preprocessor pipeline."""
    path = os.path.join(MODELS_DIR, 'best_model_pipeline.joblib')
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please run notebooks 04 & 05 first.")
        st.stop()
    payload = joblib.load(path)
    return payload['model'], payload['preprocessor']


@st.cache_data
def load_metrics():
    """Load the metrics comparison CSV."""
    path = os.path.join(MODELS_DIR, 'metrics_comparison.csv')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_feature_names():
    """Load feature names used by the preprocessor."""
    path = os.path.join(MODELS_DIR, 'feature_names.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Maternal & Infant Risk Predictor",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #e0e0ff; margin: 0.3rem 0 0 0; }

    /* Risk cards */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.3rem; font-weight: bold; margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.3rem; font-weight: bold; margin: 1rem 0;
    }

    /* Disclaimer banner */
    .disclaimer-box {
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin: 1rem 0; font-size: 0.85rem; color: #856404;
    }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🤰 Maternal & Infant Mortality Risk Predictor</h1>
    <p>Data Mining Techniques — B.Tech IT | K. Ramakrishnan College of Technology</p>
</div>
""", unsafe_allow_html=True)

# Permanent disclaimer
st.markdown(DISCLAIMER)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Prediction", "📊 Data Insights", "📈 Model Performance",
    "📁 Batch Prediction", "ℹ️ About"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Single Patient Risk Prediction")
    st.markdown("Enter the patient's clinical, maternal and socioeconomic "
                "details to predict pregnancy risk level.")

    model, preprocessor = load_model()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔬 Clinical Attributes**")
        bp_sys = st.slider("Systolic BP (mmHg)", 80, 200, 120, key="bp_sys")
        bp_dia = st.slider("Diastolic BP (mmHg)", 50, 130, 78, key="bp_dia")
        hb = st.slider("Hemoglobin (g/dL)", 5.0, 17.0, 11.5, 0.1, key="hb")
        diabetes = st.selectbox("Diabetes Status", ["No", "Yes"], key="diab")
        complications = st.selectbox("Pregnancy Complications",
                                     ["None", "Mild", "Severe"], key="comp")
        birth_wt = st.slider("Birth Weight (kg)", 0.8, 5.0, 2.8, 0.05, key="bw")
        gest_age = st.slider("Gestational Age (weeks)", 24, 43, 38, key="ga")

    with col2:
        st.markdown("**👩 Maternal & Care**")
        age = st.slider("Mother's Age (years)", 14, 48, 25, key="age")
        anc = st.slider("Antenatal Care Visits", 0, 15, 5, key="anc")
        delivery = st.selectbox("Place of Delivery",
                                ["Hospital", "Clinic", "Home"], key="del")

    with col3:
        st.markdown("**🏠 Socioeconomic**")
        education = st.selectbox("Education Level",
                                 ["No Education", "Primary", "Secondary", "Higher"],
                                 key="edu")
        income = st.slider("Family Income (INR/month)", 3000, 120000, 15000,
                           step=1000, key="inc")
        residence = st.selectbox("Residence", ["Urban", "Rural"], key="res")

    st.markdown("---")

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Build a single-row DataFrame matching the training schema
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

        # Preprocess using the saved ColumnTransformer
        from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        input_processed = preprocessor.transform(
            input_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )

        # Predict
        prediction = model.predict(input_processed)[0]
        proba = model.predict_proba(input_processed)[0]

        risk_label = "HIGH-RISK" if prediction == 1 else "LOW-RISK"
        confidence = proba[prediction] * 100

        # Display result
        if prediction == 1:
            st.markdown(f'<div class="risk-high">🚨 {risk_label} — '
                        f'Confidence: {confidence:.1f}%</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ {risk_label} — '
                        f'Confidence: {confidence:.1f}%</div>',
                        unsafe_allow_html=True)

        # Probability gauge
        st.markdown("#### Probability Breakdown")
        col_a, col_b = st.columns(2)
        col_a.metric("LOW-RISK probability", f"{proba[0]*100:.1f}%")
        col_b.metric("HIGH-RISK probability", f"{proba[1]*100:.1f}%")

        # Progress bar as a simple gauge
        st.markdown("**Risk Probability Gauge**")
        st.progress(float(proba[1]))

        # Feature importance panel
        st.markdown("#### 🔍 Why This Prediction?")
        st.markdown("The model's top features (by Random Forest importance) are "
                     "shown below. Features like hemoglobin level, blood pressure, "
                     "birth weight and gestational age are typically the strongest "
                     "predictors of risk.")
        importance_path = os.path.join(FIGURES_DIR, 'feature_importance_rf.png')
        if os.path.exists(importance_path):
            st.image(importance_path, caption="SYNTHETIC — Feature Importance (Random Forest)")
        else:
            st.info("Feature importance plot not yet generated. Run Notebook 05 first.")

    st.markdown(DISCLAIMER)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Exploratory Data Analysis — SYNTHETIC Data")
    st.markdown("These charts were generated from the 5,000-row synthetic dataset. "
                "**Not real clinical data.**")

    eda_figures = [
        ('fig01_class_distribution.png',   'Class Distribution'),
        ('fig02_age_vs_risk.png',          'Mother Age vs Risk'),
        ('fig03_hemoglobin_by_risk.png',   'Hemoglobin Distribution by Risk'),
        ('fig04_bp_vs_risk.png',           'Blood Pressure vs Risk'),
        ('fig05_anc_visits_vs_risk.png',   'ANC Visits vs Risk'),
        ('fig06_correlation_heatmap.png',  'Correlation Heatmap'),
        ('fig07_birthweight_boxplot.png',  'Birth Weight Boxplot'),
        ('fig08_urban_rural_education.png', 'Urban/Rural & Education'),
        ('fig09_complications_by_risk.png', 'Complications by Risk'),
    ]

    for filename, title in eda_figures:
        path = os.path.join(FIGURES_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=f"SYNTHETIC — {title}", use_container_width=True)
            st.markdown("---")
        else:
            st.warning(f"Figure not found: {filename}. Run Notebook 03 first.")

    st.markdown(DISCLAIMER)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Model Performance Comparison — SYNTHETIC Data")

    metrics_df = load_metrics()
    if metrics_df is not None:
        st.markdown("#### Metrics Table (sorted by Recall)")
        st.dataframe(metrics_df.style.format('{:.4f}').background_gradient(
            cmap='RdYlGn', axis=0), use_container_width=True)

        st.markdown("#### Why Recall Matters Most")
        st.info(
            "In maternal risk prediction, a **false negative** (missing a HIGH-RISK "
            "pregnancy) is far more dangerous than a false positive. A missed high-risk "
            "case could mean a preventable maternal or neonatal death. Therefore, we "
            "optimise for **RECALL** — the ability to correctly identify all truly "
            "high-risk pregnancies — even if it means some low-risk patients receive "
            "extra monitoring."
        )
    else:
        st.warning("Metrics file not found. Run Notebook 05 first.")

    # Confusion matrices
    cm_path = os.path.join(FIGURES_DIR, 'confusion_matrices.png')
    if os.path.exists(cm_path):
        st.image(cm_path, caption="SYNTHETIC — Confusion Matrices",
                 use_container_width=True)

    # ROC curves
    roc_path = os.path.join(FIGURES_DIR, 'roc_curves.png')
    if os.path.exists(roc_path):
        st.image(roc_path, caption="SYNTHETIC — ROC Curves",
                 use_container_width=True)

    # SHAP
    shap_path = os.path.join(FIGURES_DIR, 'shap_summary.png')
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SYNTHETIC — SHAP Summary Plot",
                 use_container_width=True)

    st.markdown(DISCLAIMER)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📁 Batch Prediction — Upload CSV")
    st.markdown("Upload a CSV file with the 12 input attributes. "
                "The app will return a scored CSV with predicted risk labels.")

    st.markdown("**Required columns:** `mother_age`, `education_level`, "
                "`family_income`, `residence_urban_rural`, `blood_pressure_systolic`, "
                "`blood_pressure_diastolic`, `hemoglobin_level`, `diabetes_status`, "
                "`pregnancy_complications`, `gestational_age`, `birth_weight`, "
                "`antenatal_care_visits`, `place_of_delivery`")

    uploaded = st.file_uploader("Choose a CSV file", type=['csv'], key='batch')

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write(f"Uploaded: {batch_df.shape[0]} rows × {batch_df.shape[1]} columns")
            st.dataframe(batch_df.head(), use_container_width=True)

            model, preprocessor = load_model()
            from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

            # Preprocess
            X_batch = preprocessor.transform(
                batch_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
            )

            # Predict
            preds = model.predict(X_batch)
            probas = model.predict_proba(X_batch)

            # Add results to dataframe
            result_df = batch_df.copy()
            result_df['predicted_risk'] = np.where(preds == 1, 'HIGH-RISK', 'LOW-RISK')
            result_df['high_risk_probability'] = np.round(probas[:, 1] * 100, 2)

            st.markdown("#### Results")
            st.dataframe(result_df, use_container_width=True)

            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Scored CSV",
                data=csv,
                file_name="batch_risk_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.markdown(DISCLAIMER)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### Maternal & Infant Mortality Risk Prediction Using Data Mining Techniques

    **Course:** Data Mining Techniques (DMT), B.Tech Information Technology
    **Institution:** K. Ramakrishnan College of Technology (Autonomous), Trichy

    **Team:**
    - Sakthi Darshan K (2403811720521049)
    - Suriyaa R (2403811720521054)

    ---

    ### Objective
    Analyse maternal and infant healthcare records to identify key factors
    associated with mortality risk and classify each pregnancy as **HIGH-RISK**
    or **LOW-RISK** using data mining techniques.

    ### Dataset
    - **Real data:** UCI / Kaggle Maternal Health Risk Data Set (~1,014 records)
    - **Synthetic data:** 5,000 records with 12 attributes, generated with
      clinically plausible distributions from WHO and NFHS-5 statistics.

    ### Techniques
    | Category | Algorithms |
    |----------|-----------|
    | Classification | Decision Tree, Random Forest, Naïve Bayes, Logistic Regression, KNN, SVM, XGBoost |
    | Clustering | K-Means (Elbow + Silhouette) |
    | Association | Apriori (mlxtend) |
    | Oversampling | SMOTE (training set only) |
    | Explainability | SHAP, Feature Importance |

    ### Tech Stack
    Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn ·
    mlxtend · Matplotlib · Seaborn · SHAP · Streamlit · joblib

    ---

    ### Risk Score Rule
    The target variable is defined by a documented, weighted risk score:

    | Condition | Points |
    |-----------|--------|
    | Severe anemia (Hb < 7) | 4 |
    | Moderate anemia (Hb 7–10.9) | 2 |
    | Hypertension (SBP ≥ 140) | 3–5 |
    | Gestational diabetes | 2 |
    | Severe complications | 4 |
    | Low birth weight (< 2.5 kg) | 3–5 |
    | Preterm (< 37 weeks) | 3–5 |
    | ANC visits < 4 | 2 |
    | Home delivery | 1 |
    | Mother age < 18 or > 35 | 2 |

    **Threshold:** Score ≥ 5 → HIGH-RISK  
    **Noise:** N(0, 1.5) added for non-trivial classification.
    """)

    st.markdown(DISCLAIMER)

    st.markdown("""
    ---
    ### Synthetic Data Limitation
    The primary dataset used in this project is **synthetically generated** for
    academic purposes. While distributions are informed by published WHO and
    NFHS-5 statistics, the data does NOT represent real patients. Model
    performance metrics reflect synthetic-data patterns and should NOT be
    extrapolated to real clinical populations without validation on genuine
    patient data.
    """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem'>"
    "© 2026 Sakthi Darshan K & Suriyaa R — DMT Project | "
    "K. Ramakrishnan College of Technology, Trichy"
    "</div>",
    unsafe_allow_html=True
)
