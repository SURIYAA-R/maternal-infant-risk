# =============================================================================
# src/evaluation.py
# Maternal & Infant Mortality Risk Prediction
#
# Model evaluation utilities:
#   • Metrics table (Accuracy, Precision, Recall, F1, ROC-AUC)
#   • Confusion matrices
#   • ROC curves
#   • Feature importance (Random Forest)
#   • SHAP summary
#   • Cross-validation summary
#   • Model saving with joblib
#
# Authors : Sakthi Darshan K, Suriyaa R
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                # non-interactive backend for saving plots
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import joblib

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')

# ---------------------------------------------------------------------------
# Metrics table for all models
# ---------------------------------------------------------------------------
def compute_metrics(models: dict, X_test, y_test):
    """
    Compute classification metrics for every model.

    Returns
    -------
    pd.DataFrame  —  rows = models, columns = metrics
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'Model':     name,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1-Score':  f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC':   roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        }
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index('Model')
    return df.sort_values('Recall', ascending=False)


# ---------------------------------------------------------------------------
# Confusion matrix plots
# ---------------------------------------------------------------------------
def plot_confusion_matrices(models: dict, X_test, y_test, save_dir=FIGURES_DIR):
    """Plot and save a confusion matrix for every model."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(models)
    cols = 3
    rows_needed = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_needed, cols, figsize=(5 * cols, 4 * rows_needed))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['LOW', 'HIGH'], yticklabels=['LOW', 'HIGH'])
        axes[idx].set_title(name, fontsize=10)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('SYNTHETIC — Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return path


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------
def plot_roc_curves(models: dict, X_test, y_test, save_dir=FIGURES_DIR):
    """Plot overlaid ROC curves for all models that support predict_proba."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        if not hasattr(model, 'predict_proba'):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('SYNTHETIC — ROC Curves Comparison')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return path


# ---------------------------------------------------------------------------
# Feature importance (Random Forest)
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, save_dir=FIGURES_DIR, top_n=15):
    """Bar plot of Random Forest feature importances."""
    os.makedirs(save_dir, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices][::-1], color='teal')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title('SYNTHETIC — Random Forest Feature Importance (Top 15)')
    plt.tight_layout()
    path = os.path.join(save_dir, 'feature_importance_rf.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return path


# ---------------------------------------------------------------------------
# SHAP summary
# ---------------------------------------------------------------------------
def plot_shap_summary(model, X_test, feature_names, save_dir=FIGURES_DIR):
    """SHAP beeswarm plot for the given model."""
    import shap
    os.makedirs(save_dir, exist_ok=True)

    # Use TreeExplainer for tree-based models, otherwise KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        # Fallback: sample for speed
        background = shap.sample(X_test, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:200])

    # Handle multi-output (binary) — take class-1 values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test if not isinstance(X_test, np.ndarray)
                      else pd.DataFrame(X_test, columns=feature_names),
                      feature_names=feature_names, show=False)
    plt.title('SYNTHETIC — SHAP Summary Plot', fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, 'shap_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return path


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_models(models: dict, X_train, y_train, cv=5):
    """
    5-fold cross-validation returning mean ± std for each model.
    """
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='recall', n_jobs=-1)
        rows.append({
            'Model':      name,
            'CV_Mean':    scores.mean(),
            'CV_Std':     scores.std(),
            'CV_Scores':  scores.tolist(),
        })
        print(f"  {name:25s}  recall = {scores.mean():.4f} +/- {scores.std():.4f}")
    return pd.DataFrame(rows).set_index('Model')


# ---------------------------------------------------------------------------
# Save winning model
# ---------------------------------------------------------------------------
def save_best_model(model, preprocessor, filepath=None):
    """Serialise the best model + preprocessor pipeline with joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, 'best_model_pipeline.joblib')

    payload = {
        'model':        model,
        'preprocessor': preprocessor,
    }
    joblib.dump(payload, filepath)
    print(f"[SAVED] Best model pipeline -> {os.path.abspath(filepath)}")
    return filepath
