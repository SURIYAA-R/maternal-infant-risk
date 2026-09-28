# Maternal & Infant Mortality Risk Prediction
## Project Report

### Data Mining Techniques (DMT) — B.Tech Information Technology
### K. Ramakrishnan College of Technology (Autonomous), Trichy

**Team:**
- Sakthi Darshan K (2403811720521049)
- Suriyaa R (2403811720521054)

---

> ⚠️ **This project uses SYNTHETIC data generated for academic demonstration.
> Results must NOT be interpreted as real clinical findings.**

---

## Abstract

Maternal and infant mortality remain critical public health challenges globally, with India accounting for a significant share of preventable deaths. Early identification of high-risk pregnancies can enable timely medical interventions that save lives. This project applies data mining techniques — including classification, clustering, and association rule mining — to predict whether a pregnancy is HIGH-RISK or LOW-RISK based on 12 clinical, maternal-care, and socioeconomic attributes. Seven classification algorithms (Decision Tree, Random Forest, Naïve Bayes, Logistic Regression, KNN, SVM, and XGBoost) are trained and evaluated using GridSearchCV with 5-fold cross-validation. SMOTE is used to address class imbalance on the training set only. The best model is selected based on RECALL (sensitivity) for the HIGH-RISK class, since a false negative — missing a genuinely at-risk pregnancy — carries far greater clinical consequence than a false positive. The trained model is deployed as an interactive Streamlit web application that allows healthcare workers to input patient details and receive an instant risk prediction with probability and explainability. The system is trained primarily on a 5,000-record synthetic dataset with clinically plausible distributions informed by WHO and NFHS-5 statistics, alongside the real UCI Maternal Health Risk dataset for benchmarking.

**Keywords:** Maternal mortality, infant mortality, risk prediction, data mining, classification, Random Forest, XGBoost, SMOTE, SHAP, Streamlit.

---

## 1. Introduction

### 1.1 Background

According to the WHO, approximately 295,000 women died during and following pregnancy and childbirth in 2017, with most deaths occurring in low-resource settings. India, despite significant progress, still accounts for roughly 12% of global maternal deaths (NFHS-5, 2019–21). Infant mortality in India stands at 35.2 per 1,000 live births (SRS 2020), with neonatal deaths constituting the majority.

Key risk factors include anaemia, hypertensive disorders (pre-eclampsia/eclampsia), gestational diabetes, inadequate antenatal care, home delivery without skilled attendance, adolescent or advanced maternal age, preterm birth, and low birth weight. Many of these factors are identifiable early through routine screening, yet resource-constrained health systems often lack the data infrastructure to flag at-risk patients systematically.

### 1.2 Motivation

Data mining techniques offer the ability to extract patterns from healthcare records and build predictive models that can assist frontline health workers in identifying high-risk pregnancies — even in settings where specialist obstetricians are unavailable. An automated risk scoring tool, deployed as a simple web application, could:

- Reduce diagnostic delay by providing instant risk assessments.
- Enable targeted referral of high-risk patients to tertiary care.
- Support policy-makers in allocating maternal health resources.

### 1.3 Objective

1. Generate a comprehensive synthetic dataset with 12 clinically relevant attributes and a documented rule-based risk label.
2. Perform exploratory data analysis to identify key risk factors.
3. Train and compare multiple classification algorithms to predict pregnancy risk.
4. Apply clustering and association rule mining for additional insights.
5. Deploy the best model as a publicly accessible Streamlit web application.

---

## 2. Literature Survey

1. **Mohan, S., Thirumalai, C., & Srivastava, G. (2019).** "Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques," *IEEE Access*, vol. 7, pp. 81542–81554. — Demonstrated that ensemble methods (Random Forest, XGBoost) outperform individual classifiers on clinical datasets; motivated our use of multiple algorithms with GridSearchCV.

2. **Ahmed, M., Kashem, M.A., Rahman, M., & Khatun, S. (2020).** "Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT)," *Applied Sciences*, vol. 10, no. 22, p. 7914. — Identified blood pressure, haemoglobin, age, and ANC visits as top maternal risk factors; guided our feature selection.

3. **Jhee, J.H., et al. (2019).** "Prediction model development of late-onset preeclampsia using machine learning-based methods," *PLOS ONE*, vol. 14, no. 8. — Applied logistic regression and gradient boosting to predict pre-eclampsia from clinical features; validated the relevance of SBP, DBP, and gestational age.

4. **Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002).** "SMOTE: Synthetic Minority Over-sampling Technique," *JAIR*, vol. 16, pp. 321–357. — Foundational paper for SMOTE; we apply it on the training set only to avoid data leakage.

5. **Lundberg, S.M., & Lee, S.I. (2017).** "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, pp. 4765–4774. — Introduced SHAP values for model explainability; we use SHAP to explain individual predictions in the web application.

6. **Alkema, L., et al. (2016).** "Global, regional, and national levels and trends in maternal mortality between 1990 and 2015," *The Lancet*, vol. 387, no. 10017, pp. 462–474. — Provided epidemiological context for maternal mortality trends that informed our synthetic data distributions.

---

## 3. Existing System vs Proposed System

| Aspect | Existing Systems | Proposed System |
|--------|-----------------|----------------|
| Risk assessment | Manual clinical scoring (e.g., MEWS) | Automated ML-based prediction |
| Features used | Limited vital signs | 12 attributes (clinical + socioeconomic) |
| Explainability | Opaque rule-of-thumb | SHAP values + feature importance |
| Accessibility | Hospital-only, paper-based | Web application accessible on any device |
| Algorithms | Single algorithm or none | 7 classifiers compared with cross-validation |
| Class imbalance | Often ignored | SMOTE on training set only |
| Deployment | Not deployed | Streamlit Cloud with public URL |

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  data/raw/                        data/processed/                │
│  ├── synthetic_maternal_infant_risk.csv    ├── train_processed.csv│
│  └── uci_maternal_health_risk.csv         └── test_processed.csv │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                            │
│  src/preprocessing.py                                            │
│  • Missing value imputation (median / mode)                      │
│  • IQR-based outlier capping                                     │
│  • Label / One-Hot encoding                                      │
│  • StandardScaler                                                │
│  • Stratified 80/20 split                                        │
│  • SMOTE (training set only)                                     │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                 │
│  src/models.py                                                   │
│  Classification: DT, RF, NB, LR, KNN, SVM, XGBoost             │
│  Clustering:     K-Means (elbow + silhouette)                   │
│  Association:    Apriori (mlxtend)                               │
│  Tuning:         GridSearchCV (5-fold, scoring=recall)          │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                               │
│  src/evaluation.py                                               │
│  • Metrics: Accuracy, Precision, Recall, F1, ROC-AUC            │
│  • Confusion matrices, ROC curves                                │
│  • Feature importance, SHAP summary                              │
│  • 5-fold cross-validation                                       │
│  • Model selection (best recall) → joblib serialisation          │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT LAYER                                │
│  app/app.py (Streamlit)                                          │
│  Tabs: Prediction | Data Insights | Model Perf | Batch | About  │
│  Deployment: Streamlit Community Cloud / Docker                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Module Description

### 5.1 Data Generation (`src/data_generation.py`)
Generates 5,000 synthetic records with 12 attributes using clinically plausible distributions from WHO and NFHS-5 data. The target variable is derived from a documented, weighted risk score with 15 conditions and a threshold of ≥ 5 points. Gaussian noise N(0, 1.5) is injected for non-trivial classification.

### 5.2 Preprocessing (`src/preprocessing.py`)
Implements a scikit-learn Pipeline with ColumnTransformer:
- **Numeric path:** SimpleImputer (median) → OutlierCapper (IQR × 1.5) → StandardScaler
- **Categorical path:** SimpleImputer (most frequent) → OneHotEncoder
- Stratified 80/20 train-test split preserves class proportions
- SMOTE applied on training set only to prevent data leakage

### 5.3 Models (`src/models.py`)
Defines 7 classifiers with hyperparameter grids for GridSearchCV, K-Means clustering analysis, and Apriori association rule mining with clinical binning.

### 5.4 Evaluation (`src/evaluation.py`)
Computes classification metrics, generates confusion matrices, ROC curves, feature importance plots, SHAP summaries, and cross-validation results. Saves the winning model pipeline with joblib.

### 5.5 Web Application (`app/app.py`)
Streamlit application with 5 tabs for single-patient prediction (with probability gauge and feature importance), EDA visualisation, model performance display, batch CSV prediction, and project information. Medical disclaimer displayed on every screen.

---

## 6. Implementation

### 6.1 Technology Stack
- **Language:** Python 3.11
- **Data:** pandas, NumPy
- **ML:** scikit-learn, XGBoost, imbalanced-learn, mlxtend
- **Explainability:** SHAP
- **Visualisation:** Matplotlib, Seaborn
- **Serialisation:** joblib
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud, Docker
- **Version Control:** Git, GitHub

### 6.2 Dataset Preparation
The synthetic dataset contains 5,000 records with the following class balance (approximate):
- LOW-RISK: ~60–65%
- HIGH-RISK: ~35–40%

After SMOTE on the training set, both classes are balanced 50/50 for model training.

### 6.3 Model Training
All 7 classifiers are tuned using GridSearchCV with 5-fold cross-validation, optimising for recall on the HIGH-RISK class. The rationale is that false negatives (missing a high-risk pregnancy) are clinically far more dangerous than false positives.

---

## 7. Results & Discussion

> **Note:** All metrics below are from the SYNTHETIC dataset and should NOT be interpreted as real clinical performance.

The metrics comparison table (Accuracy, Precision, Recall, F1-Score, ROC-AUC) for all 7 models is generated in Notebook 05 and displayed in the Streamlit application. Key findings:

- **Random Forest** and **XGBoost** typically achieve the highest recall and F1-scores, benefiting from their ensemble nature and ability to capture non-linear interactions.
- **Logistic Regression** provides a strong linear baseline with well-calibrated probabilities.
- **Naïve Bayes** is the fastest to train but assumes feature independence, which reduces performance when features are correlated (e.g., SBP and DBP).
- **SVM** with RBF kernel performs well but is computationally expensive.
- **KNN** captures local patterns but is sensitive to the choice of k and feature scaling.
- **Decision Tree** is the most interpretable but prone to overfitting without depth constraints.

The model with the highest RECALL on the HIGH-RISK class is selected as the deployment model, consistent with the clinical priority of minimising false negatives.

### Feature Importance
Random Forest feature importance and SHAP analysis consistently identify the following as top predictors:
1. Hemoglobin level (anaemia severity)
2. Blood pressure (systolic)
3. Birth weight
4. Gestational age
5. Pregnancy complications
6. Antenatal care visits

These align with established clinical evidence and WHO guidelines.

---

## 8. Conclusion

This project demonstrates the application of data mining techniques to maternal and infant risk prediction. Seven classification algorithms were systematically trained, tuned, and compared, with the best model deployed as a publicly accessible Streamlit web application. The use of SMOTE (training-set-only), cross-validation, SHAP explainability, and recall-focused model selection follows best practices in responsible ML for healthcare.

### Limitations
1. The primary dataset is **synthetic** — performance metrics do not reflect real clinical accuracy.
2. The risk score rule, while informed by published statistics, is a simplification of complex obstetric risk.
3. The model has not been validated on diverse, real-world clinical populations.
4. Social determinants of health (access to transport, cultural practices) are not fully captured.

---

## 9. Future Enhancement

1. **Real data validation:** Partner with hospitals to train and validate on genuine patient records.
2. **Multi-class prediction:** Extend to LOW / MEDIUM / HIGH risk levels.
3. **Temporal modelling:** Use longitudinal ANC visit data (time series) for dynamic risk tracking.
4. **Mobile app:** Deploy as an Android/iOS app for community health workers (ASHA workers).
5. **Federated learning:** Train across multiple hospital systems without sharing patient data.
6. **Integration with EHR:** Connect to electronic health record systems for automated data ingestion.
7. **Multilingual interface:** Support Tamil, Hindi, and other regional languages.

---

## 10. References (IEEE Format)

[1] S. Mohan, C. Thirumalai, and G. Srivastava, "Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques," *IEEE Access*, vol. 7, pp. 81542–81554, 2019.

[2] M. Ahmed, M. A. Kashem, M. Rahman, and S. Khatun, "Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT)," *Applied Sciences*, vol. 10, no. 22, p. 7914, 2020.

[3] J. H. Jhee *et al.*, "Prediction model development of late-onset preeclampsia using machine learning-based methods," *PLOS ONE*, vol. 14, no. 8, 2019.

[4] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *J. Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[5] S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model Predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 4765–4774, 2017.

[6] L. Alkema *et al.*, "Global, regional, and national levels and trends in maternal mortality between 1990 and 2015," *The Lancet*, vol. 387, no. 10017, pp. 462–474, 2016.

[7] World Health Organization, "WHO recommendations on antenatal care for a positive pregnancy experience," WHO, 2016.

[8] International Institute for Population Sciences (IIPS), "National Family Health Survey (NFHS-5), 2019–21: India," Mumbai: IIPS, 2021.

---

## Appendix — Source Code

The complete source code is available in the GitHub repository. Key files:

- `src/data_generation.py` — Synthetic data generator
- `src/preprocessing.py` — Preprocessing pipeline
- `src/models.py` — Model training and tuning
- `src/evaluation.py` — Evaluation and model selection
- `app/app.py` — Streamlit web application
- `notebooks/01–05` — Jupyter notebooks for each pipeline stage
