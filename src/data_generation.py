# =============================================================================
# src/data_generation.py
# Maternal & Infant Mortality Risk Prediction
# SYNTHETIC DATA GENERATOR — for academic demonstration only
#
# Generates 5,000 records with 12 clinically plausible attributes and a
# rule-based binary risk label (HIGH-RISK / LOW-RISK).
#
# Distributions are informed by published WHO and NFHS-5 (India) statistics.
# A documented, weighted risk score drives the target variable.
# Gaussian noise is injected so the classification task is non-trivial.
#
# Authors : Sakthi Darshan K, Suriyaa R
# Course  : Data Mining Techniques (DMT), B.Tech IT
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SAMPLES = 5000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUTPUT_FILE = 'synthetic_maternal_infant_risk.csv'

# ---------------------------------------------------------------------------
# Helper — bounded normal draw
# ---------------------------------------------------------------------------
def _bounded_normal(mean, std, low, high, size, rng):
    """Draw from a normal distribution and clip to [low, high]."""
    values = rng.normal(mean, std, size)
    return np.clip(values, low, high)


def generate_synthetic_dataset(n=N_SAMPLES, seed=RANDOM_SEED):
    """
    Build the full synthetic dataset with 12 input features + risk_label.

    Returns
    -------
    pd.DataFrame  —  shape (n, 13)
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. MATERNAL & SOCIOECONOMIC ATTRIBUTES
    # ------------------------------------------------------------------

    # Mother's age — skewed toward 20-30 (NFHS-5 median ≈ 23)
    mother_age = _bounded_normal(25, 5.5, 14, 48, n, rng).astype(int)

    # Education level — categorical (NFHS-5 proportions for rural India)
    education_level = rng.choice(
        ['No Education', 'Primary', 'Secondary', 'Higher'],
        size=n,
        p=[0.15, 0.25, 0.40, 0.20]                     # approximate NFHS-5
    )

    # Family income (INR per month) — log-normal, reflecting Indian households
    family_income = np.exp(
        _bounded_normal(np.log(15000), 0.7, np.log(3000), np.log(120000), n, rng)
    ).astype(int)

    # Residence — NFHS-5: ~65 % rural in India
    residence_urban_rural = rng.choice(
        ['Urban', 'Rural'], size=n, p=[0.35, 0.65]
    )

    # ------------------------------------------------------------------
    # 2. CLINICAL ATTRIBUTES
    # ------------------------------------------------------------------

    # Systolic BP (mmHg) — normal ≈ 110–120; hypertension > 140
    blood_pressure_systolic = _bounded_normal(118, 16, 80, 200, n, rng).astype(int)

    # Diastolic BP (mmHg) — typically 60-80; high > 90
    blood_pressure_diastolic = _bounded_normal(76, 11, 50, 130, n, rng).astype(int)

    # Hemoglobin (g/dL) — NFHS-5: mean ≈ 11.8 for pregnant women; anemia < 11
    hemoglobin_level = np.round(
        _bounded_normal(11.5, 1.8, 5.0, 17.0, n, rng), 1
    )

    # Diabetes status — ~8 % gestational diabetes prevalence in India
    diabetes_status = rng.choice(
        ['No', 'Yes'], size=n, p=[0.92, 0.08]
    )

    # Pregnancy complications (pre-eclampsia, placenta previa, etc.)
    pregnancy_complications = rng.choice(
        ['None', 'Mild', 'Severe'], size=n, p=[0.70, 0.20, 0.10]
    )

    # ------------------------------------------------------------------
    # 3. INFANT / DELIVERY ATTRIBUTES
    # ------------------------------------------------------------------

    # Gestational age at delivery (weeks) — mean ≈ 38; preterm < 37
    gestational_age = _bounded_normal(38.5, 2.5, 24, 43, n, rng).astype(int)

    # Birth weight (kg) — mean ≈ 2.8; low < 2.5
    birth_weight = np.round(
        _bounded_normal(2.8, 0.55, 0.8, 5.0, n, rng), 2
    )

    # ANC visits — WHO recommends ≥ 8; Indian median ≈ 5 (NFHS-5)
    antenatal_care_visits = np.clip(
        rng.poisson(5, n), 0, 15
    ).astype(int)

    # Place of delivery — NFHS-5: institutional delivery ~89 % nationally
    place_of_delivery = rng.choice(
        ['Hospital', 'Clinic', 'Home'], size=n, p=[0.60, 0.25, 0.15]
    )

    # ------------------------------------------------------------------
    # Correlations: make attributes less independent
    # ------------------------------------------------------------------
    # Rural women are more likely to deliver at home and have fewer ANC visits
    rural_mask = (residence_urban_rural == 'Rural')
    antenatal_care_visits[rural_mask] = np.clip(
        antenatal_care_visits[rural_mask] - rng.integers(0, 2, rural_mask.sum()),
        0, 15
    )
    # Increase home-delivery probability for rural
    for i in np.where(rural_mask)[0]:
        if rng.random() < 0.12:                         # extra 12 % chance
            place_of_delivery[i] = 'Home'

    # Lower education → slightly lower hemoglobin (nutritional anemia)
    low_edu_mask = np.isin(education_level, ['No Education', 'Primary'])
    hemoglobin_level[low_edu_mask] -= rng.uniform(0, 1.0, low_edu_mask.sum())
    hemoglobin_level = np.clip(np.round(hemoglobin_level, 1), 5.0, 17.0)

    # Higher systolic → higher diastolic (physiological correlation)
    blood_pressure_diastolic = np.clip(
        (blood_pressure_diastolic + 0.3 * (blood_pressure_systolic - 118)).astype(int),
        50, 130
    )

    # Preterm babies tend to weigh less
    preterm_mask = (gestational_age < 37)
    birth_weight[preterm_mask] -= rng.uniform(0.2, 0.8, preterm_mask.sum())
    birth_weight = np.clip(np.round(birth_weight, 2), 0.8, 5.0)

    # ------------------------------------------------------------------
    # 4. RISK SCORE — documented, weighted, rule-based
    # ------------------------------------------------------------------
    #
    # Each condition adds points.  Total ≥ threshold → HIGH-RISK.
    #
    # Condition                          Points   Clinical rationale
    # ─────────────────────────────────  ──────   ──────────────────
    # Severe anemia (Hb < 7)              4       Life-threatening
    # Moderate anemia (Hb 7–10.9)         2       WHO moderate anemia
    # Hypertension (SBP ≥ 140)            3       Pre-eclampsia risk
    # Severe hypertension (SBP ≥ 160)     5       Eclampsia risk
    # Diabetes (GDM)                      2       Macrosomia / complications
    # Severe pregnancy complications      4       Direct obstetric cause
    # Mild pregnancy complications        1       Needs monitoring
    # Low birth weight (< 2.5 kg)         3       Neonatal mortality
    # Very low birth weight (< 1.5 kg)    5       Extreme prematurity
    # Preterm (< 37 weeks)                3       Respiratory distress
    # Very preterm (< 32 weeks)           5       High NICU need
    # ANC visits < 4                      2       WHO minimum standard
    # Home delivery                       1       No emergency backup
    # Maternal age < 18                   2       Adolescent pregnancy
    # Maternal age > 35                   2       Advanced maternal age
    # ─────────────────────────────────
    # Threshold: score ≥ 5 → HIGH-RISK

    score = np.zeros(n, dtype=float)

    # Anemia
    score += np.where(hemoglobin_level < 7,    4, 0)
    score += np.where((hemoglobin_level >= 7) & (hemoglobin_level < 11), 2, 0)

    # Blood pressure
    score += np.where((blood_pressure_systolic >= 140) & (blood_pressure_systolic < 160), 3, 0)
    score += np.where(blood_pressure_systolic >= 160, 5, 0)

    # Diabetes
    score += np.where(diabetes_status == 'Yes', 2, 0)

    # Pregnancy complications
    score += np.where(pregnancy_complications == 'Severe', 4, 0)
    score += np.where(pregnancy_complications == 'Mild',   1, 0)

    # Birth weight
    score += np.where((birth_weight < 2.5) & (birth_weight >= 1.5), 3, 0)
    score += np.where(birth_weight < 1.5, 5, 0)

    # Gestational age
    score += np.where((gestational_age < 37) & (gestational_age >= 32), 3, 0)
    score += np.where(gestational_age < 32, 5, 0)

    # ANC visits
    score += np.where(antenatal_care_visits < 4, 2, 0)

    # Place of delivery
    score += np.where(place_of_delivery == 'Home', 1, 0)

    # Maternal age
    score += np.where(mother_age < 18, 2, 0)
    score += np.where(mother_age > 35, 2, 0)

    # ------------------------------------------------------------------
    # 5. INJECT NOISE — so the task is non-trivial (~5 % label noise)
    # ------------------------------------------------------------------
    noise = rng.normal(0, 1.5, n)                       # Gaussian perturbation
    noisy_score = score + noise

    # Threshold
    RISK_THRESHOLD = 5.0
    risk_label = np.where(noisy_score >= RISK_THRESHOLD, 'HIGH-RISK', 'LOW-RISK')

    # ------------------------------------------------------------------
    # 6. ASSEMBLE DATAFRAME
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        'mother_age':               mother_age,
        'education_level':          education_level,
        'family_income':            family_income,
        'residence_urban_rural':    residence_urban_rural,
        'blood_pressure_systolic':  blood_pressure_systolic,
        'blood_pressure_diastolic': blood_pressure_diastolic,
        'hemoglobin_level':         hemoglobin_level,
        'diabetes_status':          diabetes_status,
        'pregnancy_complications':  pregnancy_complications,
        'gestational_age':          gestational_age,
        'birth_weight':             birth_weight,
        'antenatal_care_visits':    antenatal_care_visits,
        'place_of_delivery':        place_of_delivery,
        'risk_label':               risk_label,
    })

    return df


def save_dataset(df, output_dir=OUTPUT_DIR, filename=OUTPUT_FILE):
    """Write the synthetic dataset to CSV with a SYNTHETIC banner row."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[INFO] SYNTHETIC dataset saved -> {os.path.abspath(filepath)}")
    print(f"       Shape: {df.shape}")
    print(f"       Class balance:\n{df['risk_label'].value_counts().to_string()}")
    return filepath


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("SYNTHETIC DATA GENERATOR")
    print("Academic demonstration only — NOT real clinical data")
    print("=" * 70)

    df = generate_synthetic_dataset()
    save_dataset(df)

    # Quick sanity checks
    print("\n--- Sample rows ---")
    print(df.head(3).to_string())
    print("\n--- Descriptive statistics ---")
    print(df.describe(include='all').to_string())
