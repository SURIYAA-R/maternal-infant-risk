# Viva Question Bank — 25 Questions with Answers

## Maternal & Infant Mortality Risk Prediction Using Data Mining Techniques
### Data Mining Techniques (DMT), B.Tech IT
### K. Ramakrishnan College of Technology (Autonomous), Trichy

---

### Q1. What is the objective of your project?

**Answer:** The objective is to analyse maternal and infant healthcare records to identify key risk factors associated with mortality and classify each pregnancy as HIGH-RISK or LOW-RISK using data mining techniques. We deploy the best-performing model as an interactive Streamlit web application for healthcare workers.

---

### Q2. Why did you use synthetic data instead of real patient data?

**Answer:** Real patient data is difficult to obtain due to privacy regulations (e.g., HIPAA, Indian IT Act) and ethical review requirements. We generated a 5,000-row synthetic dataset with clinically plausible distributions drawn from published WHO and NFHS-5 statistics. This allows us to demonstrate the full data mining pipeline while clearly labelling all outputs as synthetic. We also used the real UCI Maternal Health Risk dataset (~1,014 records) for benchmarking.

---

### Q3. Explain the risk-score rule used to define the target variable.

**Answer:** The target is derived from a weighted, rule-based score. Each clinical condition (e.g., severe anemia = 4 points, hypertension = 3–5 points, low birth weight = 3–5 points, preterm = 3–5 points, <4 ANC visits = 2 points) contributes points. If the total score ≥ 5, the pregnancy is labelled HIGH-RISK. We inject Gaussian noise N(0, 1.5) to ensure the classification task is non-trivial and accuracy does not reach 100%.

---

### Q4. Why did you choose Random Forest? What are its advantages?

**Answer:** Random Forest is an ensemble method that builds many decorrelated decision trees and aggregates their predictions (bagging). Advantages:
- Reduces overfitting compared to a single decision tree
- Handles both numeric and categorical features
- Provides feature importance scores for interpretability
- Robust to outliers and missing values
- Performs well on tabular data without extensive hyperparameter tuning

---

### Q5. Why did you include XGBoost? How does it differ from Random Forest?

**Answer:** XGBoost uses gradient boosting — it builds trees sequentially, with each tree correcting errors from the previous one (boosting), unlike Random Forest's parallel bagging. XGBoost often achieves higher accuracy on structured/tabular data and has built-in regularisation (L1/L2) to prevent overfitting. It is frequently the top performer in machine learning competitions.

---

### Q6. Explain the Decision Tree algorithm and why it suits this problem.

**Answer:** A Decision Tree recursively splits the data on feature values that maximise information gain (or Gini impurity reduction). It is ideal for clinical risk prediction because: (a) the tree structure directly mirrors clinical decision logic ("if SBP > 140 AND Hb < 11, then HIGH-RISK"), making it interpretable for healthcare workers; (b) it handles mixed feature types without scaling; (c) it produces rule-based explanations. However, single trees are prone to overfitting, which is why we also use ensemble methods.

---

### Q7. Why is Naïve Bayes included despite its independence assumption?

**Answer:** Naïve Bayes assumes features are conditionally independent given the class, which is violated here (e.g., SBP and DBP are correlated). However, it is included as a fast probabilistic baseline that works well even with limited data. It provides a lower bound on performance and helps us assess whether more complex models provide meaningful improvement. Despite the independence assumption, Naïve Bayes often produces competitive results in practice.

---

### Q8. How does KNN work, and what is its limitation in this context?

**Answer:** K-Nearest Neighbours classifies a new sample based on the majority class among its k nearest training neighbours (by Euclidean distance). It captures non-linear decision boundaries without any parametric assumptions. Limitations: (a) it is sensitive to feature scaling (which we address with StandardScaler); (b) it is computationally expensive at prediction time for large datasets; (c) it suffers from the curse of dimensionality with many features.

---

### Q9. Why did you use SVM? When does it struggle?

**Answer:** SVM (Support Vector Machine) finds the hyperplane that maximises the margin between classes. With the RBF kernel, it can model non-linear boundaries. It is effective in high-dimensional spaces and robust against overfitting (via the C parameter). It struggles with: (a) large datasets (slow training); (b) noisy data with overlapping classes; (c) probability calibration (less native than logistic regression).

---

### Q10. What is SMOTE and why is it needed here?

**Answer:** SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic minority-class samples by interpolating between existing minority neighbours. It is needed because our dataset has class imbalance (more LOW-RISK than HIGH-RISK). Without SMOTE, classifiers tend to predict the majority class, achieving high accuracy but poor recall on the minority (HIGH-RISK) class.

---

### Q11. Why must SMOTE be applied AFTER the train-test split? What is data leakage?

**Answer:** If SMOTE is applied before the split, synthetic minority samples may be generated from (or near) test-set points. The model then "sees" information from the test set during training — this is data leakage. It leads to inflated performance estimates that do not generalise to real-world data. The correct approach is: split first, apply SMOTE on the training fold only, and keep the test set untouched.

---

### Q12. What is the difference between Precision and Recall?

**Answer:**
- **Precision** = TP / (TP + FP) — "Of all patients we predicted as HIGH-RISK, how many truly are?"
- **Recall** = TP / (TP + FN) — "Of all truly HIGH-RISK patients, how many did we correctly identify?"

In maternal health, recall is more critical because a false negative (FN = missed high-risk) can result in a preventable death, while a false positive (FP = false alarm) only causes extra monitoring.

---

### Q13. Why did you optimise for Recall instead of Accuracy?

**Answer:** Accuracy can be misleading with imbalanced data — a model predicting all cases as LOW-RISK would achieve ~65% accuracy but 0% recall on HIGH-RISK. In clinical screening, the cost of a false negative (missing a high-risk pregnancy → maternal/neonatal death) is catastrophically higher than a false positive (extra monitoring → minor inconvenience). Therefore, we optimise for recall to catch as many true high-risk cases as possible.

---

### Q14. What is ROC-AUC and why is it useful?

**Answer:** ROC-AUC (Receiver Operating Characteristic — Area Under Curve) measures a model's ability to discriminate between classes across all threshold settings. AUC = 1.0 means perfect discrimination; AUC = 0.5 means random guessing. It is useful because: (a) it is threshold-independent; (b) it is robust to class imbalance; (c) it allows comparison of models on a single summary metric.

---

### Q15. Explain the K-Means clustering analysis. What did the elbow method show?

**Answer:** K-Means partitions data into k clusters by minimising within-cluster sum of squares (inertia). The elbow method plots inertia vs. k — the "elbow" point where adding more clusters provides diminishing returns suggests the optimal k. We also used the silhouette score, which measures how well each point fits its assigned cluster vs. neighbouring clusters. The analysis reveals natural groupings in the data that may correspond to risk strata.

---

### Q16. What is Apriori? What do support, confidence, and lift mean?

**Answer:** Apriori is an association rule mining algorithm that finds frequent co-occurring itemsets in transactional data.
- **Support** = fraction of records containing the itemset (how common is it?)
- **Confidence** = P(consequent | antecedent) — if the antecedent is present, how often does the consequent occur?
- **Lift** = confidence / P(consequent) — how much more likely is the consequent when the antecedent is present, compared to baseline? Lift > 1 indicates a positive association.

We binned continuous variables into clinical categories (e.g., anemia severity, BP category) before mining.

---

### Q17. What is SHAP? How does it explain predictions?

**Answer:** SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for a specific prediction, based on cooperative game theory (Shapley values). For each prediction, SHAP shows which features pushed the prediction toward HIGH-RISK (positive SHAP value) or LOW-RISK (negative SHAP value). This provides local, instance-level explainability — crucial for clinical trust and accountability.

---

### Q18. What preprocessing steps did you apply and why?

**Answer:**
1. **Missing-value imputation** (median for numeric, mode for categorical) — handles incomplete records
2. **IQR-based outlier capping** — prevents extreme values from distorting the model
3. **OneHotEncoder** for categorical features — converts nominal categories to numeric
4. **StandardScaler** for numeric features — normalises feature ranges (critical for KNN, SVM, logistic regression)
5. **ColumnTransformer** — applies different pipelines to numeric vs. categorical columns in a single, reproducible step

---

### Q19. What is cross-validation and why did you use 5-fold?

**Answer:** Cross-validation splits the training data into k folds, trains on (k-1) folds and validates on the held-out fold, rotating through all folds. It provides a more robust estimate of model performance than a single train-test split by averaging over k evaluations. We use 5-fold as a standard compromise between computational cost and variance reduction.

---

### Q20. How do you prevent overfitting in your models?

**Answer:**
1. **Train-test split** — hold out 20% data for final evaluation
2. **Cross-validation** — reduces optimistic bias from a single split
3. **Regularisation** — L2 in Logistic Regression, max_depth in trees, C in SVM, L1/L2 in XGBoost
4. **GridSearchCV** — selects hyperparameters that generalise (via CV score, not training score)
5. **Ensemble methods** — Random Forest bagging reduces single-tree overfitting
6. **SMOTE after split** — prevents data leakage that would artificially inflate scores

---

### Q21. What are the ethical limitations of this system?

**Answer:**
1. **Not a medical device** — it should never replace clinical judgement
2. **Trained on synthetic data** — performance does not reflect real clinical accuracy
3. **Bias risk** — if real data has sampling bias (e.g., under-representation of tribal populations), the model would inherit those biases
4. **Consent and privacy** — deploying on real patient data requires IRB approval and anonymisation
5. **Liability** — misclassification could harm patients; the system should only be used as a screening aid alongside clinical expertise
6. **Digital divide** — web-app deployment assumes internet access, which may be limited in rural areas

---

### Q22. Why did you choose Streamlit for the web application?

**Answer:** Streamlit is ideal for ML demos because: (a) it requires only Python — no frontend (HTML/JS) expertise needed; (b) it provides built-in widgets for sliders, file uploaders, and data display; (c) Streamlit Community Cloud offers free one-click deployment; (d) `@st.cache_resource` efficiently caches model loading; (e) it integrates seamlessly with pandas, matplotlib, and scikit-learn.

---

### Q23. How does the batch prediction feature work?

**Answer:** The user uploads a CSV file with the 12 input attributes. The app passes each row through the same preprocessing pipeline (ColumnTransformer) and trained model. It adds two new columns — `predicted_risk` (HIGH-RISK/LOW-RISK) and `high_risk_probability` (%) — and provides a download button for the scored CSV. This allows healthcare administrators to screen an entire patient cohort at once.

---

### Q24. What is the F1-Score and when is it useful?

**Answer:** F1-Score is the harmonic mean of precision and recall: F1 = 2 × (Precision × Recall) / (Precision + Recall). It is useful when you want a single metric that balances both false positives and false negatives. It is especially valuable when classes are imbalanced and accuracy is misleading. However, in our case, we prioritise recall over F1 because of the asymmetric cost of errors.

---

### Q25. If you could improve this project, what would you do differently?

**Answer:**
1. **Use real patient data** from partner hospitals, with proper ethical approvals
2. **Extend to multi-class** (LOW / MEDIUM / HIGH risk) for more nuanced triage
3. **Add temporal features** — track changes across ANC visits over time
4. **Deploy as a mobile app** for community health workers (ASHA workers) in rural India
5. **Use federated learning** to train across multiple hospitals without sharing raw data
6. **Include LIME** alongside SHAP for complementary local explanations
7. **Conduct a fairness audit** to ensure the model does not discriminate against specific demographic subgroups
