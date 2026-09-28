# =============================================================================
# src/models.py
# Maternal & Infant Mortality Risk Prediction
#
# Training & hyperparameter tuning for:
#   Classification — DT, RF, NB, LR, KNN, SVM, XGBoost  (GridSearchCV 5-fold)
#   Clustering     — K-Means (elbow + silhouette)
#   Association    — Apriori via mlxtend
#
# Authors : Sakthi Darshan K, Suriyaa R
# =============================================================================

import sys
import warnings
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def get_classifiers_and_params():
    """
    Return a dict of {name: (estimator, param_grid)} for GridSearchCV.

    Why each algorithm?
    -------------------
    Decision Tree   — Interpretable; mirrors clinical decision logic
    Random Forest   — Ensemble bagging reduces variance; handles mixed features
    Naïve Bayes     — Fast baseline; works with limited data
    Logistic Reg.   — Probabilistic output; good linear baseline
    KNN             — Non-parametric; captures local patterns
    SVM             — Strong generalisation via margin maximisation
    XGBoost         — State-of-the-art boosting; often top performer
    """
    classifiers = {
        # ---- Decision Tree ------------------------------------------------
        # Interpretable tree structure mirrors clinical decision-making.
        'Decision Tree': (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                'max_depth':        [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
            }
        ),
        # ---- Random Forest ------------------------------------------------
        # Ensemble of decorrelated trees; reduces overfitting.
        'Random Forest': (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                'n_estimators': [100, 200],
                'max_depth':    [5, 10, None],
                'min_samples_split': [2, 5],
            }
        ),
        # ---- Naïve Bayes --------------------------------------------------
        # Fast probabilistic baseline assuming feature independence.
        'Naive Bayes': (
            GaussianNB(),
            {
                'var_smoothing': [1e-9, 1e-8, 1e-7],
            }
        ),
        # ---- Logistic Regression ------------------------------------------
        # Linear model yielding calibrated probabilities.
        'Logistic Regression': (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {
                'C':       [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
            }
        ),
        # ---- K-Nearest Neighbours -----------------------------------------
        # Instance-based; captures non-linear boundaries.
        'KNN': (
            KNeighborsClassifier(),
            {
                'n_neighbors': [3, 5, 7, 11],
                'weights':     ['uniform', 'distance'],
            }
        ),
        # ---- Support Vector Machine ----------------------------------------
        # Maximises margin; effective in high-dimensional space.
        'SVM': (
            SVC(random_state=RANDOM_STATE, probability=True),
            {
                'C':      [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
            }
        ),
        # ---- XGBoost -------------------------------------------------------
        # Gradient boosting; often top-ranked in tabular competitions.
        'XGBoost': (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False,
            ),
            {
                'n_estimators':   [100, 200],
                'max_depth':      [3, 5, 7],
                'learning_rate':  [0.05, 0.1],
            }
        ),
    }
    return classifiers


def train_all_classifiers(X_train, y_train, cv=5, scoring='recall'):
    """
    Train every classifier with GridSearchCV.

    Parameters
    ----------
    scoring : str
        We optimise for RECALL because a false negative (missed high-risk
        pregnancy) is far more dangerous than a false positive.

    Returns
    -------
    dict  —  {name: best_estimator}
    """
    classifiers = get_classifiers_and_params()
    best_models = {}

    for name, (estimator, params) in classifiers.items():
        print(f"\n>>> Training {name} …")
        grid = GridSearchCV(
            estimator, params,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"    Best params : {grid.best_params_}")
        print(f"    Best {scoring}: {grid.best_score_:.4f}")

    return best_models


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTERING — K-Means
# ═══════════════════════════════════════════════════════════════════════════

def run_kmeans_analysis(X, k_range=range(2, 11)):
    """
    Run K-Means for each k in k_range.

    Returns
    -------
    results : list of dicts  —  k, inertia, silhouette
    best_model : KMeans       —  model with highest silhouette score
    """
    results = []
    best_sil = -1
    best_model = None

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
        print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

        if sil > best_sil:
            best_sil = sil
            best_model = km

    return results, best_model


# ═══════════════════════════════════════════════════════════════════════════
# ASSOCIATION RULES — Apriori
# ═══════════════════════════════════════════════════════════════════════════

def bin_for_apriori(df):
    """
    Convert selected numeric columns into categorical bins suitable for
    one-hot encoding and Apriori mining.
    """
    binned = pd.DataFrame()

    # Age groups
    binned['age_group'] = pd.cut(
        df['mother_age'],
        bins=[0, 18, 25, 35, 50],
        labels=['adolescent', 'young_adult', 'adult', 'advanced']
    )

    # Hemoglobin — anemia severity
    binned['anemia'] = pd.cut(
        df['hemoglobin_level'],
        bins=[0, 7, 11, 20],
        labels=['severe_anemia', 'moderate_anemia', 'normal_hb']
    )

    # Blood pressure category
    binned['bp_category'] = pd.cut(
        df['blood_pressure_systolic'],
        bins=[0, 120, 140, 200],
        labels=['normal_bp', 'elevated_bp', 'hypertension']
    )

    # Birth weight
    binned['bw_category'] = pd.cut(
        df['birth_weight'],
        bins=[0, 1.5, 2.5, 5.5],
        labels=['very_low_bw', 'low_bw', 'normal_bw']
    )

    # ANC visits
    binned['anc_category'] = pd.cut(
        df['antenatal_care_visits'],
        bins=[-1, 3, 7, 20],
        labels=['low_anc', 'moderate_anc', 'high_anc']
    )

    # Categorical columns — pass through
    for col in ['diabetes_status', 'pregnancy_complications',
                'place_of_delivery', 'risk_label']:
        binned[col] = df[col]

    return binned


def run_apriori(df, min_support=0.05, min_confidence=0.3, top_n=10):
    """
    Run Apriori on binned attributes and return top rules by lift.

    Parameters
    ----------
    df : pd.DataFrame  — raw synthetic data (before preprocessing)
    """
    binned = bin_for_apriori(df)

    # One-hot encode everything for Apriori
    onehot = pd.get_dummies(binned, dtype=bool)

    # Mine frequent itemsets
    freq_items = apriori(onehot, min_support=min_support, use_colnames=True)
    print(f"[Apriori] Frequent itemsets found: {len(freq_items)}")

    if len(freq_items) == 0:
        print("[Apriori] No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    # Generate rules
    rules = association_rules(freq_items, metric='confidence',
                              min_threshold=min_confidence)

    # Sort by lift and return top N
    rules = rules.sort_values('lift', ascending=False).head(top_n)
    print(f"[Apriori] Top {top_n} rules by lift returned.")
    return rules
