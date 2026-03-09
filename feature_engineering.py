import sys
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

print("=" * 70)
print("    DRUG RECOMMENDATION SYSTEM - FEATURE ENGINEERING & ENCODING")
print("=" * 70)
print("\n✅ Libraries imported successfully.\n")


# CONFIGURATION — File Paths

INPUT_PATH  = r"C:\project\Drug Recommendation System\data\cleaned_drug_data.csv"
OUTPUT_PATH = r"C:\project\Drug Recommendation System\data\processed_drug_data.csv"

# STEP 2: Load the Cleaned Dataset

print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

try:
    df = pd.read_csv(INPUT_PATH)
    print(f"\n✅ Dataset loaded successfully from:\n   {INPUT_PATH}")
except FileNotFoundError:
    print(f"\n❌ ERROR: File not found at:\n   {INPUT_PATH}")
    print("   Please run data_cleaning.py first to generate the cleaned dataset.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error while loading dataset: {e}")
    sys.exit(1)

# STEP 3: Display First Few Rows

print("\n" + "=" * 70)
print("STEP 2: FIRST 5 ROWS OF THE DATASET")
print("=" * 70)
print(f"\n{df.head().to_string(index=False)}")
print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# STEP 4: Identify Categorical Columns

print("\n" + "=" * 70)
print("STEP 3: IDENTIFYING CATEGORICAL COLUMNS")
print("=" * 70)

categorical_cols = ["Sex", "BP", "Cholesterol"]
numerical_cols   = ["Age", "Na_to_K"]
target_col       = "Drug"

print(f"\n   🏷️  Categorical Columns (to encode) : {categorical_cols}")
print(f"   🔢  Numerical Columns (keep as-is)  : {numerical_cols}")
print(f"   🎯  Target Column                   : {target_col}")

print("\n   Unique values before encoding:")
for col in categorical_cols:
    print(f"      {col:<15}: {sorted(df[col].unique().tolist())}")


# STEP 5 & 6: Apply Label Encoding + Print Mappings

print("\n" + "=" * 70)
print("STEP 4: APPLYING LABEL ENCODING")
print("=" * 70)

# Dictionary to store each encoder for reference / inverse transform later
encoders = {}

# Work on a copy to preserve the original dataframe
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

    # Build a clean mapping table: original value → encoded integer
    mapping = {original: encoded
               for encoded, original in enumerate(le.classes_)}

    print(f"\n   ✅ '{col}' encoded successfully.")
    print(f"      Mapping  : {mapping}")
    print(f"      Classes  : {list(le.classes_)}")
    print(f"      Encoded  : {sorted(df_encoded[col].unique().tolist())}")

print("\n   📋 Dataset after encoding (first 5 rows):")
print(f"\n{df_encoded.head().to_string(index=False)}")

# STEP 7: Separate Features (X) and Target Variable (y)

print("\n" + "=" * 70)
print("STEP 5: SEPARATING FEATURES AND TARGET VARIABLE")
print("=" * 70)

# X — all columns except the target (Drug)
X = df_encoded.drop(columns=[target_col])

# y — target column (Drug) as string labels
y = df[target_col]

print(f"\n   ✅ Feature matrix  X  created.")
print(f"      Columns : {list(X.columns)}")
print(f"\n   ✅ Target vector   y  created.")
print(f"      Column  : '{target_col}'")
print(f"      Classes : {sorted(y.unique().tolist())}")

# STEP 8: Display Shape of X and y

print("\n" + "=" * 70)
print("STEP 6: SHAPE OF FEATURES AND TARGET")
print("=" * 70)

print(f"\n   📐 X shape : {X.shape}  → ({X.shape[0]} samples, {X.shape[1]} features)")
print(f"   📐 y shape : {y.shape}  → ({y.shape[0]} samples)")

print(f"\n   📋 Feature Matrix X — first 5 rows:")
print(f"\n{X.head().to_string(index=False)}")

print(f"\n   📋 Target Vector y — first 5 values:")
print(f"\n{y.head().to_string()}")


# STEP 9: Combine Encoded Features and Target Into Processed Dataset

print("\n" + "=" * 70)
print("STEP 7: COMBINING FEATURES AND TARGET INTO PROCESSED DATASET")
print("=" * 70)

# Concatenate X (encoded features) and y (original Drug labels) side by side
df_processed = pd.concat([X, y.reset_index(drop=True)], axis=1)

print(f"\n   ✅ Processed dataset created.")
print(f"      Shape   : {df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
print(f"      Columns : {list(df_processed.columns)}")

print(f"\n   📋 Processed Dataset — first 5 rows:")
print(f"\n{df_processed.head().to_string(index=False)}")

print(f"\n   📋 Data types in processed dataset:")
for col, dtype in df_processed.dtypes.items():
    print(f"      {col:<15}: {dtype}")


# STEP 10: Save Processed Dataset

print("\n" + "=" * 70)
print("STEP 8: SAVING PROCESSED DATASET")
print("=" * 70)

try:
    df_processed.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Processed dataset saved successfully to:\n   {OUTPUT_PATH}")
    print(f"   Saved shape: {df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
except PermissionError:
    print(f"\n❌ Permission denied while saving to:\n   {OUTPUT_PATH}")
    print("   Please check folder permissions and try again.")
except Exception as e:
    print(f"\n❌ Error while saving processed dataset: {e}")


# FINAL SUMMARY

print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print(f"""
   📥 Input Dataset         : cleaned_drug_data.csv
   📤 Output Dataset        : processed_drug_data.csv

   🏷️  Columns Encoded       : {categorical_cols}
   🔢  Columns Kept As-Is   : {numerical_cols}
   🎯  Target Variable      : {target_col} (kept as original string labels)

   Encoding Reference:
""")
for col, le in encoders.items():
    mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
    print(f"      {col:<15}: {mapping}")

print(f"""
   📐 Final Feature Matrix  : X → {X.shape}
   📐 Final Target Vector   : y → {y.shape}
   📐 Processed Dataset     : {df_processed.shape[0]} rows × {df_processed.shape[1]} columns
""")

print("=" * 70)
print("   ✅ FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70 + "\n")