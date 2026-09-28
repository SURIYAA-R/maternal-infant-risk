# Maternal & Infant Mortality Risk Prediction

> **Data Mining Techniques (DMT) — B.Tech Information Technology**
> K. Ramakrishnan College of Technology (Autonomous), Trichy
>
> **Team:** Sakthi Darshan K (2403811720521049) · Suriyaa R (2403811720521054)

---

## ⚠️ Medical Disclaimer

> **Academic demonstration only. The model is trained predominantly on synthetic
> data generated for coursework. This is NOT a medical device and must NOT be
> used for clinical decision-making. Always consult a qualified healthcare
> professional.**

---

## Problem Statement

Maternal and infant mortality remain critical public health concerns in India
and worldwide. Early identification of high-risk pregnancies can enable timely
medical interventions. This project applies data mining techniques —
classification, clustering and association rule mining — to predict whether a
pregnancy is **HIGH-RISK** or **LOW-RISK** based on 12 clinical, maternal-care
and socioeconomic attributes.

## Dataset

| Source | Records | Role |
|--------|---------|------|
| [UCI / Kaggle Maternal Health Risk Data Set](https://archive.ics.uci.edu/dataset/863/maternal+health+risk) | ~1 014 | Real-data anchor for benchmarking |
| Synthetic dataset (`src/data_generation.py`) | 5 000 | Extended attributes, academic demo |

The synthetic data is generated with clinically plausible distributions drawn
from published WHO and NFHS-5 statistics. **All synthetic outputs are clearly
labelled as such.**

## Architecture

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│  Raw Data    │───▶│ Preprocess  │───▶│  Model Train │───▶│ Evaluation │
│  (CSV)       │    │  Pipeline   │    │  & Tune      │    │  & Select  │
└──────────────┘    └─────────────┘    └──────────────┘    └─────┬──────┘
                                                                 │
                                                                 ▼
                                                          ┌────────────┐
                                                          │ Streamlit  │
                                                          │ Web App    │
                                                          └────────────┘
```

## Repository Structure

```
maternal-infant-risk/
├── app/                  # Streamlit web application
│   └── app.py
├── data/
│   ├── raw/              # Original / generated datasets
│   └── processed/        # Cleaned, encoded, split datasets
├── models/               # Serialised model + pipeline (.joblib)
├── notebooks/            # Jupyter notebooks (01–05)
├── reports/
│   └── figures/          # Saved EDA & evaluation plots
├── src/                  # Reusable Python modules
│   ├── __init__.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── Dockerfile            # Alternative deployment
├── requirements.txt      # Pinned dependencies
├── .gitignore
└── README.md             # ← you are here
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/maternal-infant-risk.git
cd maternal-infant-risk

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data & run notebooks
python src/data_generation.py
jupyter notebook

# 5. Launch the Streamlit app locally
streamlit run app/app.py
```

## Live Demo

> 🔗 **[Streamlit Cloud URL]** — *(will be added after deployment in Phase 8)*

## Techniques Used

| Category | Algorithms |
|----------|-----------|
| Classification | Decision Tree · Random Forest · Naïve Bayes · Logistic Regression · KNN · SVM · XGBoost |
| Clustering | K-Means (elbow + silhouette) |
| Association Rules | Apriori (mlxtend) |
| Oversampling | SMOTE (training set only) |
| Explainability | SHAP, feature importance |

## Results

> *(Metrics table will be inserted after Phase 6.)*

## Screenshots

> *(Screenshots of the Streamlit app will be inserted after Phase 7.)*

## References

> *(IEEE-format references will be added in the final report — Phase 9.)*

## License

This project is submitted as academic coursework and is shared for educational
purposes only.
