# =============================================================================
# src/preprocessing.py
# Maternal & Infant Mortality Risk Prediction
#
# scikit-learn Pipeline + ColumnTransformer for reproducible preprocessing:
#   • Missing-value imputation (median / mode)
#   • Outlier capping via IQR
#   • Label & one-hot encoding for categoricals
#   • StandardScaler for numerics
#   • Stratified 80/20 train-test split
#   • SMOTE applied on TRAINING SET ONLY
#
# Authors : Sakthi Darshan K, Suriyaa R
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    'mother_age', 'family_income',
    'blood_pressure_systolic', 'blood_pressure_diastolic',
    'hemoglobin_level', 'gestational_age', 'birth_weight',
    'antenatal_care_visits',
]

CATEGORICAL_FEATURES = [
    'education_level', 'residence_urban_rural',
    'diabetes_status', 'pregnancy_complications',
    'place_of_delivery',
]

TARGET = 'risk_label'

# ---------------------------------------------------------------------------
# Custom transformer: IQR-based outlier capper
# ---------------------------------------------------------------------------
class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Cap outliers at Q1 - 1.5·IQR and Q3 + 1.5·IQR per column.
    Fitted on the training set so test set sees the same bounds.
    """
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        # Store per-column bounds
        Q1 = np.nanpercentile(X, 25, axis=0)
        Q3 = np.nanpercentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = np.array(X, dtype=float).copy()
        for col in range(X.shape[1]):
            X[:, col] = np.clip(X[:, col], self.lower_[col], self.upper_[col])
        return X


# ---------------------------------------------------------------------------
# Build the preprocessing ColumnTransformer
# ---------------------------------------------------------------------------
def build_preprocessor():
    """
    Returns a fitted-ready ColumnTransformer.

    Numeric path  : Impute (median) → Cap outliers (IQR) → StandardScaler
    Category path : Impute (most frequent) → OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ('imputer',  SimpleImputer(strategy='median')),
        ('capper',   OutlierCapper(factor=1.5)),
        ('scaler',   StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline,      NUMERIC_FEATURES),
            ('cat', categorical_pipeline,  CATEGORICAL_FEATURES),
        ],
        remainder='drop'                   # drop any extra columns
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Full preprocessing routine
# ---------------------------------------------------------------------------
def preprocess_data(df, test_size=0.2, random_state=42, apply_smote=True):
    """
    End-to-end preprocessing:
      1. Encode target (HIGH-RISK=1, LOW-RISK=0)
      2. Stratified train/test split
      3. Fit preprocessor on training data, transform both sets
      4. Optionally apply SMOTE to training set ONLY

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    # --- Encode target ---------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])       # HIGH-RISK=0, LOW-RISK=1
    # We want HIGH-RISK = 1 for positive-class interpretation
    y = 1 - y                              # flip: HIGH-RISK=1, LOW-RISK=0

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # --- Stratified split -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # --- Fit preprocessor on training data only ---------------------------
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    # Feature names after encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_feature_names

    # --- SMOTE on training set only ---------------------------------------
    # WHY only training?  Applying SMOTE before the split would generate
    # synthetic minority samples that may be near-copies of test-set points,
    # causing DATA LEAKAGE and inflated performance estimates.
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
        print(f"[SMOTE] Training set resampled: {np.bincount(y_train)}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


# ---------------------------------------------------------------------------
# Convenience: save processed splits to disk
# ---------------------------------------------------------------------------
def save_processed(X_train, X_test, y_train, y_test, feature_names,
                   output_dir=None):
    """Save train/test arrays as CSVs in data/processed/."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['risk_label'] = y_train

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['risk_label'] = y_test

    train_path = os.path.join(output_dir, 'train_processed.csv')
    test_path  = os.path.join(output_dir, 'test_processed.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"[INFO] Processed training set saved -> {os.path.abspath(train_path)}  "
          f"({train_df.shape})")
    print(f"[INFO] Processed test set saved     -> {os.path.abspath(test_path)}  "
          f"({test_df.shape})")


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        from src.data_generation import generate_synthetic_dataset
    except ImportError:
        from data_generation import generate_synthetic_dataset

    print("=" * 60)
    print("PREPROCESSING SMOKE TEST")
    print("=" * 60)

    df = generate_synthetic_dataset()
    X_tr, X_te, y_tr, y_te, prep, fnames = preprocess_data(df)

    print(f"\nFeatures ({len(fnames)}): {fnames}")
    print(f"X_train shape: {X_tr.shape}   y_train balance: {np.bincount(y_tr)}")
    print(f"X_test  shape: {X_te.shape}   y_test  balance: {np.bincount(y_te)}")

    save_processed(X_tr, X_te, y_tr, y_te, fnames)
