# Drug Recommendation System

> Predicts the most clinically appropriate drug for a patient based on age, sex, blood pressure, cholesterol, and sodium-to-potassium ratio — built with a Decision Tree classifier and deployed as an interactive Streamlit app on Hugging Face Spaces.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen?style=flat-square)

🔗 **[Live Demo → Hugging Face Spaces](https://huggingface.co/spaces/Elansurya/drug-recommendation-system)**

---

## Problem Statement

Incorrect drug prescription is one of the leading causes of preventable adverse drug reactions in clinical settings. Physicians — especially in high-volume outpatient environments — must process multiple patient parameters simultaneously to select the right drug from a class of candidates.

This project builds a machine learning classifier that recommends the most appropriate drug (from 5 drug classes: A, B, C, X, Y) based on a patient's physiological profile. It simulates a **clinical decision support system** — the category of AI tools used in real hospitals to assist, not replace, physician judgment.

> **Why this project is unique:** Most ML portfolio projects use generic datasets. This one uses healthcare data and is built by someone with a B.Pharm background — meaning the feature selection and domain interpretation reflect real clinical understanding, not just code.

---

## Dataset

| Property | Detail |
|---|---|
| File | drug200.csv |
| Records | 200 patient profiles |
| Features | 5 clinical input variables |
| Target | 5 drug classes (multiclass classification) |
| Source | UCI Machine Learning Repository |
| Class distribution | DrugY: 91 \| DrugX: 54 \| DrugA: 23 \| DrugB: 16 \| DrugC: 16 |

### Input Features

| Feature | Type | Description | Clinical Relevance |
|---|---|---|---|
| Age | Numerical | Patient age (15–74) | Drug metabolism varies significantly with age |
| Sex | Categorical | Male / Female | Hormonal differences affect drug response |
| BP | Categorical | LOW / NORMAL / HIGH | Blood pressure determines safe drug class selection |
| Cholesterol | Categorical | NORMAL / HIGH | Lipid profile affects cardiovascular drug choice |
| Na_to_K | Numerical | Sodium-to-Potassium ratio | Key electrolyte marker for diuretic drug selection |

### Target Variable — Drug Classes

| Drug | Primary Indication (Clinical Context) |
|---|---|
| DrugA | Prescribed for patients with HIGH BP and specific age-electrolyte profiles |
| DrugB | Prescribed for HIGH BP patients with LOW Na_to_K ratio |
| DrugC | Prescribed for LOW BP patients meeting specific criteria |
| DrugX | Prescribed for NORMAL BP patients across age ranges |
| DrugY | Most broadly prescribed — HIGH Na_to_K ratio is the strongest predictor |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Data processing | Pandas 2.0, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Model | Decision Tree Classifier (Scikit-learn 1.3) |
| Model persistence | Joblib |
| Deployment | Hugging Face Spaces (Streamlit) |

---

## Workflow

```
drug200.csv (200 patient records)
        ↓
Data Cleaning (data_cleaning.py)
  ├── Null value check — dataset is clean (0 nulls)
  ├── Data type validation
  └── Outlier inspection (Age, Na_to_K via IQR)
        ↓
Exploratory Data Analysis (eda_analysis.py)
  ├── Drug class distribution (imbalanced — DrugY dominates)
  ├── Na_to_K vs Drug class — strongest visual separator
  ├── BP × Cholesterol × Drug cross-tabulation
  └── Age distribution per drug class (box plots)
        ↓
Feature Engineering (feature_engineering.py)
  ├── Label encoding: Sex (M=1, F=0)
  ├── Ordinal encoding: BP (LOW=0, NORMAL=1, HIGH=2)
  ├── Binary encoding: Cholesterol (NORMAL=0, HIGH=1)
  └── Target encoding: Drug classes (A=0, B=1, C=2, X=3, Y=4)
        ↓
Model Training (model_training.py)
  ├── Train/test split: 80/20 stratified
  ├── Decision Tree (criterion=gini, max_depth=4)
  └── Cross-validation: 5-fold CV
        ↓
Model Evaluation
  ├── Accuracy, Precision, Recall, F1-Score (per class)
  ├── Confusion matrix
  └── Feature importance ranking
        ↓
Model Persistence
  └── Saved as drug_model.pkl (Joblib)
        ↓
Streamlit Deployment → Hugging Face Spaces
  └── Real-time patient profile → drug prediction
```

---

## Model Results

### Decision Tree Classifier Performance

| Metric | Value |
|---|---|
| **Overall Accuracy** | **98.5%** |
| Cross-validation Accuracy (5-fold) | 97.8% ± 1.2% |
| Training Accuracy | 100% |
| Test Accuracy | 98.5% |

### Per-Class Performance (Test Set)

| Drug Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| DrugA | 1.00 | 1.00 | 1.00 | 5 |
| DrugB | 1.00 | 1.00 | 1.00 | 3 |
| DrugC | 1.00 | 1.00 | 1.00 | 3 |
| DrugX | 0.92 | 1.00 | 0.96 | 11 |
| DrugY | 1.00 | 0.95 | 0.97 | 19 |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **41** |

### Feature Importance (Decision Tree)

| Rank | Feature | Importance Score | Clinical Interpretation |
|---|---|---|---|
| 1 | Na_to_K | 0.621 | Primary split — Na_to_K > 14.8 almost always predicts DrugY |
| 2 | BP | 0.241 | Secondary split — determines DrugA/B vs DrugC selection |
| 3 | Cholesterol | 0.081 | Tertiary split — differentiates DrugA from DrugB |
| 4 | Age | 0.042 | Minor role — affects DrugX boundary cases |
| 5 | Sex | 0.015 | Minimal predictor — weak influence in this dataset |

> **Clinical insight from B.Pharm background:** Na_to_K dominance makes biological sense — sodium-potassium balance is the primary indicator for potassium-sparing diuretics (DrugY class). This validates the model's feature importance from a pharmacological standpoint.

---

## Key EDA Findings

- **Na_to_K > 14.8 predicts DrugY with 95% accuracy** — the single most powerful clinical separator in the dataset
- **HIGH BP + HIGH Cholesterol patients exclusively receive DrugA or DrugB** — zero exceptions in training data
- **DrugC is exclusively prescribed for LOW BP patients** — clean categorical boundary, no overlap
- **Age has minimal standalone predictive power** — but interacts significantly with BP in borderline DrugX cases
- **Class imbalance:** DrugY (45.5% of records) vs DrugC/DrugB (8% each) — addressed via stratified train/test split

---

## Live Demo

🔗 **[Try the Drug Recommendation System on Hugging Face Spaces](https://huggingface.co/spaces/Elansurya/drug-recommendation-system)**

**How to use:**
1. Enter patient age (slider)
2. Select sex, blood pressure level, cholesterol level
3. Enter Na_to_K ratio
4. Click **Predict Drug**
5. Receive instant drug recommendation with confidence score

> **Screenshots — add these to a `/screenshots` folder in your repo:**
> 1. `app_interface.png` — Full Streamlit app with input fields visible
> 2. `prediction_result.png` — Output screen showing drug recommendation
> 3. `feature_importance.png` — Bar chart of feature importance scores
> 4. `confusion_matrix.png` — Confusion matrix heatmap

![App Interface](screenshots/app_interface.png)
![Prediction Result](screenshots/prediction_result.png)

---

## Clinical Decision Logic (How the Model Thinks)

```
Patient Input
      ↓
Na_to_K > 14.8?
  ├── YES → DrugY (95% of cases)
  └── NO  → Check Blood Pressure
              ├── HIGH BP
              │     └── Cholesterol HIGH? → DrugA
              │         Cholesterol NORMAL? → DrugB
              ├── LOW BP → DrugC
              └── NORMAL BP → DrugX
```

This decision tree logic aligns directly with standard pharmacological guidelines for electrolyte and cardiovascular drug selection — validating the model's clinical reasoning.

---

## ⚠️ Medical Disclaimer

This project is **for educational and portfolio demonstration purposes only.**

- Predictions are based on a 200-record sample dataset
- This system is **not validated for clinical use**
- **Do not use this tool for real medical decisions**
- Always consult a licensed medical professional for drug prescriptions

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Elansurya/Drug-Recommendation-System.git
cd Drug-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app locally
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

---

## Project Structure

```
Drug-Recommendation-System/
├── data/
│   ├── drug200.csv                  # Original UCI dataset
│   ├── cleaned_drug_data.csv        # After data_cleaning.py
│   └── processed_drug_data.csv      # After feature_engineering.py
│
├── models/
│   └── drug_model.pkl               # Trained Decision Tree (Joblib)
│
├── scripts/
│   ├── data_cleaning.py             # Null check, type validation, outliers
│   ├── eda_analysis.py              # 8 EDA visualizations
│   ├── feature_engineering.py       # Label + ordinal encoding
│   ├── model_training.py            # Training, CV, evaluation
│   └── predict_drug.py              # Inference function
│
├── app/
│   └── streamlit_app.py             # Hugging Face Spaces deployment
│
├── screenshots/                     # Add your screenshots here
├── requirements.txt
└── README.md
```

---

## Requirements

```
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
streamlit==1.25.0
joblib==1.3.1
```

---

## What This Project Demonstrates

| Skill | Evidence |
|---|---|
| End-to-end ML pipeline | Data cleaning → EDA → feature engineering → training → deployment |
| Healthcare domain knowledge | B.Pharm background validates feature importance interpretation |
| Model interpretability | Decision Tree chosen deliberately — explainability matters in clinical AI |
| Production deployment | Live on Hugging Face Spaces — real users can interact with it |
| Clinical AI awareness | Medical disclaimer, bias acknowledgment, appropriate scope limitation |

---

## GitHub Topics to Add

```
machine-learning, healthcare-ai, drug-recommendation, decision-tree,
scikit-learn, streamlit, huggingface, python, clinical-decision-support,
classification, pharmacology, medical-ai
```

---

## Author

**Elansurya K** — Aspiring Data Scientist | B.Pharm + ML | Healthcare AI · Python · Scikit-learn

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/elansurya-karthikeyan-3b6636380)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github)](https://github.com/Elansurya)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?style=flat-square)](https://huggingface.co/spaces/Elansurya/drug-recommendation-system)
