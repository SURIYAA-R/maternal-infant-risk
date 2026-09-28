"""
End-to-End Execution & Testing Script
Runs all phases: Data Generation -> Preprocessing -> EDA Figures -> Model Training -> Evaluation -> Pipeline Serialization -> Smoke Tests.
"""

import os
import sys

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from src.data_generation import generate_synthetic_dataset, save_dataset
from src.preprocessing import preprocess_data, save_processed
from src.models import train_all_classifiers, run_kmeans_analysis, run_apriori
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance,
    plot_shap_summary,
    cross_validate_models,
    save_best_model
)

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def step1_data_generation():
    print("\n" + "=" * 60)
    print("STEP 1: SYNTHETIC DATA GENERATION")
    print("=" * 60)
    df = generate_synthetic_dataset(n=5000, seed=42)
    path = save_dataset(df)
    print(f"Data Generation SUCCESS: {path} ({df.shape})")
    return df


def step2_preprocessing(df):
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        df, test_size=0.2, random_state=42, apply_smote=True
    )
    save_processed(X_train, X_test, y_train, y_test, feature_names)
    print(f"Preprocessing SUCCESS: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor, feature_names


def step3_generate_eda_figures(df):
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING EDA FIGURES (fig01 to fig09)")
    print("=" * 60)
    sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.1)

    # Fig 01: Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['risk_label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
    ax.set_title('SYNTHETIC - Class Distribution of Risk Label', fontweight='bold')
    ax.set_xlabel('Risk Label')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 30, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig01_class_distribution.png'), dpi=150)
    plt.close()
    print("Saved: fig01_class_distribution.png")

    # Fig 02: Age vs Risk
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, color in zip(['LOW-RISK', 'HIGH-RISK'], ['#2ecc71', '#e74c3c']):
        subset = df[df['risk_label'] == label]
        axes[0].hist(subset['mother_age'], bins=20, alpha=0.6, label=label, color=color, edgecolor='black')
    axes[0].set_title('SYNTHETIC - Age Distribution by Risk Level', fontweight='bold')
    axes[0].set_xlabel('Mother Age (years)')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    sns.boxplot(data=df, x='risk_label', y='mother_age', ax=axes[1],
                palette={'LOW-RISK': '#2ecc71', 'HIGH-RISK': '#e74c3c'})
    axes[1].set_title('SYNTHETIC - Age Boxplot by Risk', fontweight='bold')
    axes[1].set_xlabel('Risk Label')
    axes[1].set_ylabel('Mother Age (years)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig02_age_vs_risk.png'), dpi=150)
    plt.close()
    print("Saved: fig02_age_vs_risk.png")

    # Fig 03: Hemoglobin by Risk
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in zip(['LOW-RISK', 'HIGH-RISK'], ['#2ecc71', '#e74c3c']):
        subset = df[df['risk_label'] == label]
        sns.kdeplot(subset['hemoglobin_level'], ax=ax, label=label, color=color, fill=True, alpha=0.3, linewidth=2)
    ax.axvline(x=11, color='orange', linestyle='--', label='Anemia threshold (11 g/dL)')
    ax.axvline(x=7, color='red', linestyle='--', label='Severe anemia (7 g/dL)')
    ax.set_title('SYNTHETIC - Hemoglobin Level Distribution by Risk', fontweight='bold')
    ax.set_xlabel('Hemoglobin (g/dL)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig03_hemoglobin_by_risk.png'), dpi=150)
    plt.close()
    print("Saved: fig03_hemoglobin_by_risk.png")

    # Fig 04: Blood Pressure vs Risk
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.violinplot(data=df, x='risk_label', y='blood_pressure_systolic', ax=axes[0],
                   palette={'LOW-RISK': '#2ecc71', 'HIGH-RISK': '#e74c3c'}, inner='box')
    axes[0].axhline(y=140, color='orange', linestyle='--', alpha=0.7, label='Hypertension (140)')
    axes[0].set_title('SYNTHETIC - Systolic BP by Risk', fontweight='bold')
    axes[0].set_ylabel('Systolic BP (mmHg)')
    axes[0].legend()
    sns.violinplot(data=df, x='risk_label', y='blood_pressure_diastolic', ax=axes[1],
                   palette={'LOW-RISK': '#2ecc71', 'HIGH-RISK': '#e74c3c'}, inner='box')
    axes[1].axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='High diastolic (90)')
    axes[1].set_title('SYNTHETIC - Diastolic BP by Risk', fontweight='bold')
    axes[1].set_ylabel('Diastolic BP (mmHg)')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig04_bp_vs_risk.png'), dpi=150)
    plt.close()
    print("Saved: fig04_bp_vs_risk.png")

    # Fig 05: ANC Visits vs Risk
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='antenatal_care_visits', hue='risk_label', ax=ax,
                  palette={'LOW-RISK': '#2ecc71', 'HIGH-RISK': '#e74c3c'})
    ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, label='WHO min (4 visits)')
    ax.set_title('SYNTHETIC - ANC Visit Count by Risk Level', fontweight='bold')
    ax.set_xlabel('Number of Antenatal Care Visits')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig05_anc_visits_vs_risk.png'), dpi=150)
    plt.close()
    print("Saved: fig05_anc_visits_vs_risk.png")

    # Fig 06: Correlation Heatmap
    df_corr = df.copy()
    df_corr['risk_numeric'] = (df_corr['risk_label'] == 'HIGH-RISK').astype(int)
    numeric_cols = ['mother_age', 'family_income', 'blood_pressure_systolic',
                    'blood_pressure_diastolic', 'hemoglobin_level', 'gestational_age',
                    'birth_weight', 'antenatal_care_visits', 'risk_numeric']
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df_corr[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, linewidths=0.5)
    ax.set_title('SYNTHETIC - Correlation Heatmap (Numeric Features + Risk)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig06_correlation_heatmap.png'), dpi=150)
    plt.close()
    print("Saved: fig06_correlation_heatmap.png")

    # Fig 07: Birth Weight Boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x='risk_label', y='birth_weight', ax=ax,
                palette={'LOW-RISK': '#2ecc71', 'HIGH-RISK': '#e74c3c'})
    ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.7, label='LBW threshold (2.5 kg)')
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='VLBW threshold (1.5 kg)')
    ax.set_title('SYNTHETIC - Birth Weight by Risk Level', fontweight='bold')
    ax.set_ylabel('Birth Weight (kg)')
    ax.set_xlabel('Risk Label')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig07_birthweight_boxplot.png'), dpi=150)
    plt.close()
    print("Saved: fig07_birthweight_boxplot.png")

    # Fig 08: Urban/Rural & Education
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ct1 = pd.crosstab(df['residence_urban_rural'], df['risk_label'], normalize='index') * 100
    ct1.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0].set_title('SYNTHETIC - Risk by Residence', fontweight='bold')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_xlabel('Residence')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(title='Risk')

    edu_order = ['No Education', 'Primary', 'Secondary', 'Higher']
    ct2 = pd.crosstab(df['education_level'], df['risk_label'], normalize='index') * 100
    ct2 = ct2.reindex(edu_order)
    ct2.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[1].set_title('SYNTHETIC - Risk by Education Level', fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_xlabel('Education Level')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15)
    axes[1].legend(title='Risk')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig08_urban_rural_education.png'), dpi=150)
    plt.close()
    print("Saved: fig08_urban_rural_education.png")

    # Fig 09: Complications by Risk
    fig, ax = plt.subplots(figsize=(8, 5))
    ct3 = pd.crosstab(df['pregnancy_complications'], df['risk_label'])
    ct3.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    ax.set_title('SYNTHETIC - Pregnancy Complications by Risk Level', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xlabel('Complication Severity')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Risk')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig09_complications_by_risk.png'), dpi=150)
    plt.close()
    print("Saved: fig09_complications_by_risk.png")


def step4_train_models(X_train, y_train, df):
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING (7 CLASSIFIERS, K-MEANS, APRIORI)")
    print("=" * 60)
    print("Training 7 classifiers with GridSearchCV 5-fold (recall-optimised)...")
    best_models = train_all_classifiers(X_train, y_train)

    print("\nRunning K-Means clustering analysis...")
    km_results, best_km = run_kmeans_analysis(X_train, k_range=range(2, 9))
    km_df = pd.DataFrame(km_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(km_df['k'], km_df['inertia'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('SYNTHETIC - K-Means Elbow Method (Inertia)', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(km_df['k'], km_df['silhouette'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_title('SYNTHETIC - Silhouette Score vs k', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True, alpha=0.3)
    best_k = km_df.loc[km_df['silhouette'].idxmax(), 'k']
    axes[1].axvline(x=best_k, color='green', linestyle='--', label=f'Best k={best_k}')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig10_kmeans_elbow_silhouette.png'), dpi=150)
    plt.close()
    print(f"Saved: fig10_kmeans_elbow_silhouette.png (Best k={best_k})")

    print("\nRunning Apriori association rules...")
    rules = run_apriori(df, min_support=0.03, min_confidence=0.25, top_n=10)
    if len(rules) > 0:
        print(f"Apriori produced {len(rules)} rules.")
    return best_models


def step5_evaluate_and_save(best_models, X_train, X_test, y_train, y_test, preprocessor, feature_names):
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATION, SHAP & WINNING MODEL SELECTION")
    print("=" * 60)
    
    # 1. Metrics Comparison Table
    metrics_df = compute_metrics(best_models, X_test, y_test)
    metrics_path = os.path.join(MODELS_DIR, 'metrics_comparison.csv')
    metrics_df.to_csv(metrics_path)
    print("\n--- MODEL PERFORMANCE COMPARISON (Sorted by Recall) ---")
    print(metrics_df.to_string())
    print(f"Metrics table saved -> {metrics_path}")

    # 2. Confusion Matrices Plot
    plot_confusion_matrices(best_models, X_test, y_test, save_dir=FIGURES_DIR)

    # 3. Overlaid ROC Curves
    plot_roc_curves(best_models, X_test, y_test, save_dir=FIGURES_DIR)

    # 4. Feature Importance for Random Forest
    if 'Random Forest' in best_models:
        plot_feature_importance(best_models['Random Forest'], feature_names, save_dir=FIGURES_DIR)

    # 5. Winning Model Selection (Primary criterion: RECALL, Secondary: ROC-AUC / F1)
    winning_model_name = metrics_df.index[0]
    winning_model = best_models[winning_model_name]
    winning_recall = metrics_df.loc[winning_model_name, 'Recall']
    winning_auc = metrics_df.loc[winning_model_name, 'ROC-AUC']
    print(f"\nWinning Model Selected: {winning_model_name} (Recall={winning_recall:.4f}, ROC-AUC={winning_auc:.4f})")

    # 6. SHAP Summary Plot on winning model (or Random Forest)
    shap_model = winning_model if hasattr(winning_model, 'feature_importances_') else best_models.get('Random Forest', winning_model)
    plot_shap_summary(shap_model, X_test, feature_names, save_dir=FIGURES_DIR)

    # 7. Save Models
    save_best_model(winning_model, preprocessor, filepath=os.path.join(MODELS_DIR, 'best_model_pipeline.joblib'))

    cache_path = os.path.join(MODELS_DIR, 'all_trained_models.joblib')
    joblib.dump({
        'models': best_models,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }, cache_path)
    print(f"All models cache saved -> {cache_path}")


def step6_smoke_test_pipeline():
    print("\n" + "=" * 60)
    print("STEP 6: SMOKE TEST THE SAVED MODEL & PREPROCESSOR")
    print("=" * 60)
    model_path = os.path.join(MODELS_DIR, 'best_model_pipeline.joblib')
    payload = joblib.load(model_path)
    model = payload['model']
    preprocessor = payload['preprocessor']

    # Test single-record prediction matching raw schema
    sample_patient = pd.DataFrame([{
        'mother_age': 28,
        'education_level': 'Secondary',
        'family_income': 18000,
        'residence_urban_rural': 'Rural',
        'blood_pressure_systolic': 145,
        'blood_pressure_diastolic': 95,
        'hemoglobin_level': 8.5,
        'diabetes_status': 'Yes',
        'pregnancy_complications': 'Mild',
        'gestational_age': 34,
        'birth_weight': 2.1,
        'antenatal_care_visits': 2,
        'place_of_delivery': 'Clinic'
    }])

    X_transformed = preprocessor.transform(sample_patient)
    pred = model.predict(X_transformed)[0]
    prob = model.predict_proba(X_transformed)[0][1] if hasattr(model, 'predict_proba') else None
    
    label = 'HIGH-RISK' if pred == 1 else 'LOW-RISK'
    print(f"Sample Patient Prediction: {label} (Probability of High Risk: {prob:.4f} if prob else N/A)")
    print("Pipeline scoring: PASSED!")


def step7_verify_app_syntax():
    print("\n" + "=" * 60)
    print("STEP 7: VERIFY STREAMLIT APP INTEGRATION")
    print("=" * 60)
    app_file = os.path.join(PROJECT_ROOT, 'app', 'app.py')
    with open(app_file, 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, app_file, 'exec')
    print("Streamlit app code compilation: PASSED (syntax clean)!")


if __name__ == '__main__':
    print("STARTING END-TO-END PROJECT PIPELINE...")
    df = step1_data_generation()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = step2_preprocessing(df)
    step3_generate_eda_figures(df)
    best_models = step4_train_models(X_train, y_train, df)
    step5_evaluate_and_save(best_models, X_train, X_test, y_train, y_test, preprocessor, feature_names)
    step6_smoke_test_pipeline()
    step7_verify_app_syntax()
    print("\n" + "=" * 60)
    print("ALL PHASES (1 TO 9) EXECUTED AND TESTED SUCCESSFULLY!")
    print("=" * 60)
