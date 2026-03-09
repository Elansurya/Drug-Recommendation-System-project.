# 💊 Drug Recommendation System

## 📌 Project Overview

The **Drug Recommendation System** is a Machine Learning project that predicts the most suitable drug for a patient based on medical attributes such as **Age, Sex, Blood Pressure, Cholesterol level, and Sodium-to-Potassium ratio (Na_to_K)**.

This project demonstrates a complete **Machine Learning pipeline**, including data preprocessing, exploratory data analysis, feature engineering, model training, and deployment using a **Streamlit web application**.

---

# 🚀 Features

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering using label encoding
* Decision Tree classification model
* Model evaluation
* Interactive **Streamlit web application**
* Predict drug recommendations in real time

---

# 📂 Project Structure

```
Drug Recommendation System
│
├── data
│   ├── drug200.csv
│   ├── cleaned_drug_data.csv
│   └── processed_drug_data.csv
│
├── models
│   └── drug_model.pkl
│
├── scripts
│   ├── data_cleaning.py
│   ├── eda_analysis.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── predict_drug.py
│
├── app
│   └── streamlit_app.py
│
└── README.md
```

---

# 📊 Dataset

The dataset used in this project is **drug200.csv**.

### Features

| Feature     | Description                        |
| ----------- | ---------------------------------- |
| Age         | Patient age                        |
| Sex         | Male or Female                     |
| BP          | Blood Pressure (LOW, NORMAL, HIGH) |
| Cholesterol | Cholesterol level (NORMAL, HIGH)   |
| Na_to_K     | Sodium to Potassium ratio          |

### Target Variable

| Drug Classes |
| ------------ |
| drugA        |
| drugB        |
| drugC        |
| drugX        |
| drugY        |

---

# ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

# 🧠 Machine Learning Model

The model used in this project:

**Decision Tree Classifier**

Why Decision Tree?

* Easy to interpret
* Works well for classification problems
* Handles categorical and numerical features

---

# 🔄 Machine Learning Workflow

1️⃣ Data Cleaning
2️⃣ Exploratory Data Analysis (EDA)
3️⃣ Feature Engineering
4️⃣ Model Training
5️⃣ Model Evaluation
6️⃣ Model Deployment using Streamlit

---

# 📈 Model Prediction Example

### Input

```
Age: 45
Sex: Female
BP: HIGH
Cholesterol: NORMAL
Na_to_K: 16
```

### Output

```
Recommended Drug: drugY
```

---

# 🖥️ Running the Streamlit Application

### Install dependencies

```
pip install streamlit pandas numpy scikit-learn joblib
```

### Run the application

```
streamlit run app.py
```

The application will open in your browser.

---

# 📷 Streamlit Interface

The Streamlit application allows users to:

* Enter patient details
* Click **Predict Drug**
* Get an instant drug recommendation

---

# ⚠️ Disclaimer

This project is **for educational purposes only**.

The drug predictions are based on a sample dataset and **should not be used for real medical decisions**.

---

# 👨‍💻 Author

ElanSurya

Machine Learning Enthusiast
