import sys
import warnings

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

print("=" * 70)
print("        DRUG RECOMMENDATION SYSTEM - DRUG PREDICTION")
print("=" * 70)
print("\n✅ Libraries imported successfully.\n")


# CONFIGURATION — Model Path & Encoding Maps

MODEL_PATH = r"C:\project\Drug Recommendation System\models\drug_model.pkl"

# These encoding maps MUST match exactly what LabelEncoder produced
# during feature_engineering.py (LabelEncoder sorts classes alphabetically)
ENCODING_MAP = {
    "Sex": {
        "Female" : 0,
        "Male"   : 1
    },
    "BP": {
        "High"   : 0,
        "Low"    : 1,
        "Normal" : 2
    },
    "Cholesterol": {
        "High"   : 0,
        "Normal" : 1
    }
}

# Feature order must exactly match the order used during model training
FEATURE_ORDER = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]


# STEP 2: Load the Trained Model

print("=" * 70)
print("STEP 1: LOADING TRAINED MODEL")
print("=" * 70)

try:
    model = joblib.load(MODEL_PATH)
    print(f"\n✅ Model loaded successfully from:\n   {MODEL_PATH}")
    print(f"\n   Model Type       : {type(model).__name__}")
    print(f"   Tree Depth       : {model.get_depth()}")
    print(f"   Number of Leaves : {model.get_n_leaves()}")
    print(f"   Drug Classes     : {list(model.classes_)}")
except FileNotFoundError:
    print(f"\n❌ ERROR: Model file not found at:\n   {MODEL_PATH}")
    print("   Please run model_training.py first to generate the model file.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error loading model: {e}")
    sys.exit(1)


# STEP 3: Define Sample Patient Input

print("\n" + "=" * 70)
print("STEP 2: DEFINING PATIENT INPUT")
print("=" * 70)

# -----------------------------------------------------------------------
# 🔧 MODIFY THESE VALUES to predict for a different patient
# -----------------------------------------------------------------------
patient_input = {
    "Age"         : 35,         # Integer (0–120)
    "Sex"         : "Female",   # "Female" or "Male"
    "BP"          : "High",     # "High", "Low", or "Normal"
    "Cholesterol" : "Normal",   # "High" or "Normal"
    "Na_to_K"     : 20.5        # Float (positive value)
}
# -----------------------------------------------------------------------

print(f"\n   Patient Details:")
print(f"   {'─' * 35}")
print(f"   {'Age':<15} : {patient_input['Age']}")
print(f"   {'Sex':<15} : {patient_input['Sex']}")
print(f"   {'Blood Pressure':<15} : {patient_input['BP']}")
print(f"   {'Cholesterol':<15} : {patient_input['Cholesterol']}")
print(f"   {'Na_to_K Ratio':<15} : {patient_input['Na_to_K']}")
print(f"   {'─' * 35}")


# STEP 4: Convert Patient Input to a Pandas DataFrame

print("\n" + "=" * 70)
print("STEP 3: CONVERTING INPUT TO DATAFRAME")
print("=" * 70)

# Wrap the dict in a list so pandas treats it as a single-row DataFrame
patient_df = pd.DataFrame([patient_input])

print(f"\n   ✅ Patient data converted to DataFrame.")
print(f"\n{patient_df.to_string(index=False)}")


# STEP 5: Encode Categorical Columns

print("\n" + "=" * 70)
print("STEP 4: ENCODING CATEGORICAL FEATURES")
print("=" * 70)

print(f"\n   Applying label encoding to: Sex, BP, Cholesterol")
print(f"   (Using same mapping as feature_engineering.py)\n")

for col, mapping in ENCODING_MAP.items():
    raw_value     = patient_df[col].iloc[0]

    # Validate the input value exists in the known encoding map
    if raw_value not in mapping:
        print(f"\n❌ ERROR: Invalid value '{raw_value}' for column '{col}'.")
        print(f"   Accepted values: {list(mapping.keys())}")
        sys.exit(1)

    encoded_value         = mapping[raw_value]
    patient_df[col]       = encoded_value

    print(f"   '{col}' : '{raw_value}' → {encoded_value}   (mapping: {mapping})")

print(f"\n   ✅ Encoding complete.")
print(f"\n   Encoded Patient DataFrame:")
print(f"\n{patient_df.to_string(index=False)}")

# STEP 6: Enforce Correct Feature Order

print("\n" + "=" * 70)
print("STEP 5: ENFORCING FEATURE ORDER")
print("=" * 70)

patient_df = patient_df[FEATURE_ORDER]

print(f"\n   ✅ Feature order aligned to training format.")
print(f"   Order: {FEATURE_ORDER}")
print(f"\n   Final Input Array:")
print(f"\n{patient_df.to_string(index=False)}")


# STEP 7: Predict the Recommended Drug

print("\n" + "=" * 70)
print("STEP 6: PREDICTING RECOMMENDED DRUG")
print("=" * 70)

# Predict the drug class
predicted_drug = model.predict(patient_df)[0]

# Get prediction probabilities for all drug classes
pred_probabilities = model.predict_proba(patient_df)[0]
prob_df = pd.DataFrame({
    "Drug"       : model.classes_,
    "Probability": (pred_probabilities * 100).round(2)
}).sort_values("Probability", ascending=False).reset_index(drop=True)

print(f"\n   ✅ Prediction complete.")
print(f"\n   Prediction Probabilities:")
print(f"   {'─' * 35}")
for _, row in prob_df.iterrows():
    bar    = "█" * int(row["Probability"] / 5)
    marker = " ◀ PREDICTED" if row["Drug"] == predicted_drug else ""
    print(f"   {row['Drug']:<12}: {row['Probability']:>6.2f}%  {bar}{marker}")
print(f"   {'─' * 35}")


# STEP 8 & 9: Print Patient Input and Predicted Drug

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(f"""
   ┌─────────────────────────────────────────────────┐
   │              PATIENT SUMMARY                    │
   ├─────────────────────────────────────────────────┤
   │  Age           : {str(patient_input['Age']):<30} │
   │  Sex           : {patient_input['Sex']:<30} │
   │  Blood Pressure: {patient_input['BP']:<30} │
   │  Cholesterol   : {patient_input['Cholesterol']:<30} │
   │  Na_to_K Ratio : {str(patient_input['Na_to_K']):<30} │
   ├─────────────────────────────────────────────────┤
   │  🩺 RECOMMENDED DRUG : {predicted_drug:<26} │
   └─────────────────────────────────────────────────┘
""")

print("=" * 70)
print("   ✅ PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 70 + "\n")