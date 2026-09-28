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

### Synthetic Data Limitation

The primary dataset is **synthetically generated** for academic purposes. While
distributions are informed by published statistics, the data does NOT represent
real patients. Model performance metrics should NOT be extrapolated to real
clinical populations without validation on genuine patient data.

## Architecture

```
┌──────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│  Raw Data (CSV)  │───▶│ Preprocessing│───▶│ Model Train  │───▶│ Evaluation │
│  Synthetic +     │    │ Pipeline     │    │ & Tune       │    │ & Select   │
│  UCI Dataset     │    │ (IQR, SMOTE) │    │ (GridSearchCV│    │ (Recall)   │
└──────────────────┘    └──────────────┘    └──────────────┘    └──────┬─────┘
                                                                       │
                                                                       ▼
                                                               ┌────────────┐
                                                               │ Streamlit  │
                                                               │ Web App    │
                                                               │ (5 tabs)   │
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
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_building.ipynb
│   └── 05_evaluation.ipynb
├── reports/
│   ├── figures/          # Saved EDA & evaluation plots
│   ├── project_report.md
│   └── viva_question_bank.md
├── src/                  # Reusable Python modules
│   ├── __init__.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── Dockerfile            # Alternative deployment
├── requirements.txt      # Pinned dependencies
├── .gitignore
└── README.md
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

# 4. Generate synthetic data
python src/data_generation.py

# 5. Run notebooks in order (01 → 05)
jupyter notebook

# 6. Launch the Streamlit app locally
streamlit run app/app.py
```

## Techniques Used

| Category | Algorithms |
|----------|-----------|
| Classification | Decision Tree · Random Forest · Naïve Bayes · Logistic Regression · KNN · SVM · XGBoost |
| Clustering | K-Means (elbow + silhouette) |
| Association Rules | Apriori (mlxtend) |
| Oversampling | SMOTE (training set only) |
| Explainability | SHAP, feature importance |

## Results

> *Metrics are from SYNTHETIC data — not real clinical performance.*

The metrics comparison table (Accuracy, Precision, Recall, F1, ROC-AUC) for all
7 models is generated in Notebook 05 and displayed in the Streamlit app.

The model with the **highest recall on the HIGH-RISK class** is selected as the
deployment model, because a false negative (missing a high-risk pregnancy) is
far more dangerous than a false positive.

## Deployment

### Streamlit Community Cloud (Recommended)

**Step-by-step:**

1. Push this repository to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository, branch (`main` or `master`), and set the main file
   path to `app/app.py`.
5. Click **"Deploy!"** — Streamlit will install dependencies from
   `requirements.txt` and launch the app.
6. Your public URL will be:
   `https://<your-username>-maternal-infant-risk-app-app-<hash>.streamlit.app`

### Docker (Alternative)

```bash
# Build the image
docker build -t maternal-risk-app .

# Run the container
docker run -p 8501:8501 maternal-risk-app

# Open in browser: http://localhost:8501
```

## Live Demo

> 🔗 **[Streamlit Cloud URL]** — *(add after deployment)*

## Screenshots

> *(Insert screenshots of the Streamlit app after running it)*

## References

See [`reports/project_report.md`](reports/project_report.md) for the full
IEEE-format reference list.

## License

This project is submitted as academic coursework and is shared for educational
purposes only.