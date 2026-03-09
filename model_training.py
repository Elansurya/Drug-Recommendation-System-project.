import sys
import warnings

import pandas as pd
import joblib

from sklearn.model_selection   import train_test_split
from sklearn.tree              import DecisionTreeClassifier
from sklearn.metrics           import (accuracy_score,
                                       confusion_matrix,
                                       classification_report)

warnings.filterwarnings("ignore")

print("=" * 70)
print("       DRUG RECOMMENDATION SYSTEM - MODEL TRAINING PIPELINE")
print("=" * 70)
print("\n✅ All libraries imported successfully.\n")


# CONFIGURATION — File Paths & Hyperparameters

DATASET_PATH = r"C:\project\Drug Recommendation System\data\processed_drug_data.csv"
MODEL_DIR    = r"C:\project\Drug Recommendation System\models"
MODEL_PATH   = os.path.join(MODEL_DIR, "drug_model.pkl")

FEATURE_COLS = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
TARGET_COL   = "Drug"
TEST_SIZE    = 0.20       # 80% train / 20% test
RANDOM_STATE = 42         # Ensures reproducibility


# STEP 2: Load the Processed Dataset

print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"\n✅ Dataset loaded successfully from:\n   {DATASET_PATH}")
except FileNotFoundError:
    print(f"\n❌ ERROR: File not found at:\n   {DATASET_PATH}")
    print("   Please run feature_engineering.py first to generate the processed dataset.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error loading dataset: {e}")
    sys.exit(1)


# STEP 3: Display First Few Rows

print("\n" + "=" * 70)
print("STEP 2: DATASET PREVIEW")
print("=" * 70)

print(f"\n   Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns : {list(df.columns)}")
print(f"\n   First 5 rows:\n")
print(df.head().to_string(index=False))

print(f"\n   Data Types:")
for col, dtype in df.dtypes.items():
    print(f"      {col:<15}: {dtype}")

print(f"\n   Missing Values: {df.isnull().sum().sum()}")


# STEP 4: Separate Features (X) and Target Variable (y)

print("\n" + "=" * 70)
print("STEP 3: SEPARATING FEATURES AND TARGET VARIABLE")
print("=" * 70)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\n   ✅ Feature matrix  X  → Shape: {X.shape}")
print(f"      Features : {FEATURE_COLS}")

print(f"\n   ✅ Target vector   y  → Shape: {y.shape}")
print(f"      Target   : '{TARGET_COL}'")
print(f"      Classes  : {sorted(y.unique().tolist())}")
print(f"\n   Class distribution:")
for drug, count in y.value_counts().items():
    pct = (count / len(y)) * 100
    print(f"      {drug:<12}: {count:>3} samples  ({pct:.1f}%)")

# STEP 5 & 6: Train / Test Split (80/20, random_state=42)

print("\n" + "=" * 70)
print("STEP 4: SPLITTING DATASET INTO TRAIN AND TEST SETS")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y        # Preserves class distribution in both splits
)

print(f"\n   ✅ Split completed  (test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=True)")
print(f"\n   📐 Training set  : X_train → {X_train.shape}  |  y_train → {y_train.shape}")
print(f"   📐 Testing  set  : X_test  → {X_test.shape}   |  y_test  → {y_test.shape}")

print(f"\n   Training class distribution:")
for drug, count in y_train.value_counts().items():
    print(f"      {drug:<12}: {count:>3} samples")

print(f"\n   Testing class distribution:")
for drug, count in y_test.value_counts().items():
    print(f"      {drug:<12}: {count:>3} samples")

# STEP 7: Train the Decision Tree Classifier

print("\n" + "=" * 70)
print("STEP 5: TRAINING DECISION TREE CLASSIFIER")
print("=" * 70)

model = DecisionTreeClassifier(
    criterion    = "gini",      # Split quality metric
    max_depth    = None,        # Grow full tree (no depth limit)
    min_samples_split = 2,      # Min samples required to split a node
    min_samples_leaf  = 1,      # Min samples required at a leaf node
    random_state = RANDOM_STATE
)

print(f"\n   Model Configuration:")
print(f"      Algorithm    : Decision Tree Classifier")
print(f"      Criterion    : {model.criterion}")
print(f"      Max Depth    : {model.max_depth} (unrestricted)")
print(f"      Random State : {model.random_state}")

print(f"\n   🔄 Training model on {X_train.shape[0]} samples...")
model.fit(X_train, y_train)

print(f"   ✅ Model trained successfully.")
print(f"      Tree Depth      : {model.get_depth()}")
print(f"      Number of Leaves: {model.get_n_leaves()}")
print(f"      Number of Classes: {model.n_classes_}")

# Feature importance
print(f"\n   📊 Feature Importances:")
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importances_sorted = importances.sort_values(ascending=False)
for feat, score in importances_sorted.items():
    bar = "█" * int(score * 40)
    print(f"      {feat:<15}: {score:.4f}  {bar}")


# STEP 8: Make Predictions on the Test Set

print("\n" + "=" * 70)
print("STEP 6: MAKING PREDICTIONS ON TEST SET")
print("=" * 70)

y_pred = model.predict(X_test)

print(f"\n   ✅ Predictions generated for {len(y_pred)} test samples.")
print(f"\n   Sample Predictions (first 10):")
print(f"\n   {'Index':<8} {'Actual':<12} {'Predicted':<12} {'Match'}")
print(f"   {'-'*45}")
for i, (actual, predicted) in enumerate(zip(y_test.values[:10], y_pred[:10])):
    match = "✅" if actual == predicted else "❌"
    print(f"   {i:<8} {actual:<12} {predicted:<12} {match}")

# STEP 9 & 10: Evaluate the Model

print("\n" + "=" * 70)
print("STEP 7: MODEL EVALUATION")
print("=" * 70)

# --- Accuracy Score ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   ┌─────────────────────────────────────┐")
print(f"   │  Accuracy Score : {accuracy * 100:>6.2f}%             │")
print(f"   └─────────────────────────────────────┘")

# --- Confusion Matrix ---
print(f"\n   📊 Confusion Matrix:")
classes     = sorted(y.unique().tolist())
cm          = confusion_matrix(y_test, y_pred, labels=classes)
cm_df       = pd.DataFrame(cm, index=classes, columns=classes)

cm_df.index.name   = "Actual \\ Predicted"
print(f"\n{cm_df.to_string()}\n")

# Highlight correct predictions (diagonal)
print(f"   Diagonal values = correctly classified samples per class.")
for i, cls in enumerate(classes):
    correct = cm[i][i]
    total   = cm[i].sum()
    print(f"      {cls:<12}: {correct}/{total} correctly classified")

# --- Classification Report ---
print(f"\n   📋 Classification Report:")
print(f"\n{classification_report(y_test, y_pred, target_names=classes)}")


# STEP 11: Save the Trained Model

print("=" * 70)
print("STEP 8: SAVING TRAINED MODEL")
print("=" * 70)

try:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    model_size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"\n✅ Model saved successfully to:\n   {MODEL_PATH}")
    print(f"   File size : {model_size_kb:.2f} KB")
except PermissionError:
    print(f"\n❌ Permission denied while saving model to:\n   {MODEL_PATH}")
    print("   Please check folder permissions and try again.")
except Exception as e:
    print(f"\n❌ Error while saving model: {e}")

# FINAL SUMMARY

print("\n" + "=" * 70)
print("MODEL TRAINING SUMMARY")
print("=" * 70)

print(f"""
   📥 Input Dataset         : processed_drug_data.csv
   🤖 Model                 : Decision Tree Classifier
   🎯 Target Variable       : {TARGET_COL}

   📐 Dataset Split:
      Total Samples    : {len(df)}
      Training Samples : {X_train.shape[0]}  ({int((1 - TEST_SIZE) * 100)}%)
      Testing  Samples : {X_test.shape[0]}   ({int(TEST_SIZE * 100)}%)

   📊 Evaluation Results:
      Accuracy Score   : {accuracy * 100:.2f}%

   🏆 Most Important Feature : {importances_sorted.idxmax()}
      (Importance Score: {importances_sorted.max():.4f})

   💾 Model Saved To:
      {MODEL_PATH}
""")

print("=" * 70)
print("   ✅ MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70 + "\n")