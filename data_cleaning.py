
import pandas as pd
import numpy as np
import warnings
import os
import sys

# STEP 2: Suppress Warnings for Cleaner Output
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("=" * 70)
print("       DRUG RECOMMENDATION SYSTEM - DATA CLEANING PIPELINE")
print("=" * 70)
print("\n✅ Libraries imported successfully.")
print("✅ Warnings suppressed for cleaner output.\n")


# STEP 3 & 4: Load Dataset with Error Handling

print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

FILE_PATH = r"C:\project\Drug Recommendation System\data\drug200.csv"

try:
    df = pd.read_csv(FILE_PATH)
    print(f"\n✅ Dataset loaded successfully from:\n   {FILE_PATH}")
except FileNotFoundError:
    print(f"\n❌ ERROR: File not found at path:\n   {FILE_PATH}")
    print("   Please verify the file path and try again.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error while loading dataset: {e}")
    sys.exit(1)


# STEP 5: Display First 5 Rows

print("\n" + "=" * 70)
print("STEP 2: FIRST 5 ROWS OF THE DATASET")
print("=" * 70)
print(df.head())


# STEP 6: Display Dataset Shape

print("\n" + "=" * 70)
print("STEP 3: DATASET SHAPE")
print("=" * 70)
rows, cols = df.shape
print(f"\n📊 Total Rows    : {rows}")
print(f"📊 Total Columns : {cols}")

# STEP 7: Dataset Information (Data Types)
print("\n" + "=" * 70)
print("STEP 4: DATASET INFORMATION (DATA TYPES & NON-NULL COUNTS)")
print("=" * 70)
print()
df.info()

# STEP 8: Check for Missing Values

print("\n" + "=" * 70)
print("STEP 5: MISSING VALUES CHECK")
print("=" * 70)

missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_summary = pd.DataFrame({
    "Missing Count": missing_values,
    "Missing (%)": missing_percent.round(2)
})
print(f"\n{missing_summary}")

# STEP 9: Handle Missing Values

print("\n" + "=" * 70)
print("STEP 6: HANDLING MISSING VALUES")
print("=" * 70)

if missing_values.sum() == 0:
    print("\n✅ No missing values found. No imputation required.")
else:
    print(f"\n⚠️  Found {missing_values.sum()} missing value(s). Applying imputation strategy...")

    # Numerical columns → fill with median (robust to outliers)
    numerical_cols = ["Age", "Na_to_K"]
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   🔧 '{col}': Filled {df[col].isnull().sum()} missing value(s) with median ({median_val}).")

    # Categorical columns → fill with mode (most frequent value)
    categorical_cols = ["Sex", "BP", "Cholesterol", "Drug"]
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"   🔧 '{col}': Filled {df[col].isnull().sum()} missing value(s) with mode ('{mode_val}').")

    print("\n✅ Missing values handled successfully.")


# STEP 10: Check and Remove Duplicate Rows

print("\n" + "=" * 70)
print("STEP 7: DUPLICATE ROWS CHECK & REMOVAL")
print("=" * 70)

duplicate_count = df.duplicated().sum()
print(f"\n🔍 Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅ {duplicate_count} duplicate row(s) removed.")
    print(f"   Updated dataset shape: {df.shape}")
else:
    print("✅ No duplicate rows found. Dataset is clean.")

# STEP 11 & 12 & 13: Standardize Categorical Values
#   - Strip extra whitespace
#   - Convert to consistent Title Case format

print("\n" + "=" * 70)
print("STEP 8: STANDARDIZING CATEGORICAL VALUES")
print("=" * 70)

categorical_cols = ["Sex", "BP", "Cholesterol", "Drug"]

print("\n🔧 Before standardization — unique values per categorical column:")
for col in categorical_cols:
    print(f"   {col}: {df[col].unique().tolist()}")

for col in categorical_cols:
    # Remove leading/trailing whitespace
    df[col] = df[col].str.strip()
    # Convert to Title Case for consistency
    df[col] = df[col].str.title()

print("\n✅ After standardization — unique values per categorical column:")
for col in categorical_cols:
    print(f"   {col}: {df[col].unique().tolist()}")

print("\n✅ All categorical columns stripped and converted to Title Case.")

# STEP 14: Ensure Numerical Columns Have Correct Datatypes

print("\n" + "=" * 70)
print("STEP 9: VALIDATING NUMERICAL COLUMN DATA TYPES")
print("=" * 70)

numerical_cols = ["Age", "Na_to_K"]

for col in numerical_cols:
    original_dtype = df[col].dtype
    df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"\n   '{col}':")
    print(f"   - Original dtype : {original_dtype}")
    print(f"   - Current dtype  : {df[col].dtype}")

    # Handle any NaNs introduced by coercion
    coerced_nulls = df[col].isnull().sum()
    if coerced_nulls > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   - ⚠️  {coerced_nulls} value(s) coerced to NaN and filled with median ({median_val}).")
    else:
        print(f"   - ✅ No coercion errors. Data type is valid.")

# STEP 15 & 16: Detect and Handle Outliers Using IQR Method

print("\n" + "=" * 70)
print("STEP 10: OUTLIER DETECTION & HANDLING (IQR METHOD)")
print("=" * 70)

def cap_outliers_iqr(dataframe, column):
    """
    Detect outliers using the IQR method and cap them at the
    lower and upper whisker boundaries (Winsorization).

    Parameters:
        dataframe (pd.DataFrame): The dataset.
        column (str): The column to process.

    Returns:
        pd.DataFrame: Dataset with outliers capped.
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
    outlier_count = len(outliers)

    print(f"\n   📊 Column : '{column}'")
    print(f"      Q1              : {Q1}")
    print(f"      Q3              : {Q3}")
    print(f"      IQR             : {IQR:.4f}")
    print(f"      Lower Bound     : {lower_bound:.4f}")
    print(f"      Upper Bound     : {upper_bound:.4f}")
    print(f"      Outliers Found  : {outlier_count}")

    if outlier_count > 0:
        # Cap outliers at whisker boundaries (Winsorization)
        dataframe[column] = np.where(
            dataframe[column] < lower_bound, lower_bound,
            np.where(dataframe[column] > upper_bound, upper_bound, dataframe[column])
        )
        print(f"      ✅ {outlier_count} outlier(s) capped using Winsorization.")
    else:
        print(f"      ✅ No outliers detected.")

    return dataframe

for col in ["Age", "Na_to_K"]:
    df = cap_outliers_iqr(df, col)


# STEP 17: Validate Age Range (0–120)

print("\n" + "=" * 70)
print("STEP 11: VALIDATING AGE RANGE (0 – 120)")
print("=" * 70)

invalid_age = df[(df["Age"] < 0) | (df["Age"] > 120)]
print(f"\n🔍 Invalid Age records (outside 0–120): {len(invalid_age)}")

if len(invalid_age) > 0:
    print(f"\n   ⚠️  Removing {len(invalid_age)} record(s) with invalid Age values...")
    print(invalid_age[["Age"]].to_string())
    df = df[(df["Age"] >= 0) & (df["Age"] <= 120)]
    df.reset_index(drop=True, inplace=True)
    print(f"\n   ✅ Invalid Age records removed. Updated shape: {df.shape}")
else:
    print("   ✅ All Age values are within the valid range (0–120).")


# STEP 18: Validate Na_to_K Values

print("\n" + "=" * 70)
print("STEP 12: VALIDATING Na_to_K VALUES")
print("=" * 70)

# Na_to_K represents the ratio of Sodium to Potassium in the blood.
# Physiologically, values should be positive. Flag extreme negatives.
invalid_na_k = df[df["Na_to_K"] <= 0]
print(f"\n🔍 Invalid Na_to_K records (≤ 0): {len(invalid_na_k)}")

if len(invalid_na_k) > 0:
    print(f"\n   ⚠️  Removing {len(invalid_na_k)} record(s) with non-positive Na_to_K values...")
    df = df[df["Na_to_K"] > 0]
    df.reset_index(drop=True, inplace=True)
    print(f"   ✅ Invalid Na_to_K records removed. Updated shape: {df.shape}")
else:
    print("   ✅ All Na_to_K values are valid (> 0).")

# Additionally, flag statistically extreme values (beyond 3 standard deviations)
mean_nak = df["Na_to_K"].mean()
std_nak  = df["Na_to_K"].std()
extreme_nak = df[np.abs(df["Na_to_K"] - mean_nak) > 3 * std_nak]
print(f"\n🔍 Extreme Na_to_K outliers (> 3 std devs): {len(extreme_nak)}")

if len(extreme_nak) > 0:
    print(f"   Extreme values detected:")
    print(extreme_nak[["Na_to_K"]].to_string())
    print("   ℹ️  These values were already handled in the IQR capping step.")
else:
    print("   ✅ No extreme Na_to_K values beyond 3 standard deviations.")


# STEP 19: Summary Statistics for Numerical Columns

print("\n" + "=" * 70)
print("STEP 13: SUMMARY STATISTICS — NUMERICAL COLUMNS")
print("=" * 70)

summary_stats = df[["Age", "Na_to_K"]].describe().round(4)
print(f"\n{summary_stats}")

# STEP 20: Unique Values for Categorical Columns

print("\n" + "=" * 70)
print("STEP 14: UNIQUE VALUES — CATEGORICAL COLUMNS")
print("=" * 70)

categorical_cols = ["Sex", "BP", "Cholesterol", "Drug"]

for col in categorical_cols:
    unique_vals = df[col].unique().tolist()
    val_counts  = df[col].value_counts().to_dict()
    print(f"\n   📌 Column : '{col}'")
    print(f"      Unique Values  : {unique_vals}")
    print(f"      Value Counts   : {val_counts}")


# STEP 21: Final Dataset Verification
print("\n" + "=" * 70)
print("STEP 15: FINAL DATASET VERIFICATION")
print("=" * 70)

print(f"\n   📊 Final Shape             : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   🔍 Remaining Missing Values: {df.isnull().sum().sum()}")
print(f"   🔍 Remaining Duplicates    : {df.duplicated().sum()}")

print("\n   📋 Final Data Types:")
for col, dtype in df.dtypes.items():
    print(f"      - {col:<15}: {dtype}")

print("\n   🔎 Final Dataset Preview (first 5 rows):")
print(df.head().to_string(index=False))

all_checks_passed = (
    df.isnull().sum().sum() == 0 and
    df.duplicated().sum() == 0 and
    df["Age"].between(0, 120).all() and
    (df["Na_to_K"] > 0).all()
)

if all_checks_passed:
    print("\n   ✅ All verification checks PASSED. Dataset is clean and ready.")
else:
    print("\n   ⚠️  Some verification checks did NOT pass. Please review the steps above.")


# STEP 22: Save the Cleaned Dataset

print("\n" + "=" * 70)
print("STEP 16: SAVING CLEANED DATASET")
print("=" * 70)

OUTPUT_PATH = r"C:\project\Drug Recommendation System\data\cleaned_drug_data.csv"

try:
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Cleaned dataset saved successfully to:\n   {OUTPUT_PATH}")
    print(f"   Final saved shape: {df.shape[0]} rows × {df.shape[1]} columns")
except PermissionError:
    print(f"\n❌ Permission denied while saving to:\n   {OUTPUT_PATH}")
    print("   Please check folder permissions and try again.")
except Exception as e:
    print(f"\n❌ Error while saving cleaned dataset: {e}")

print("\n" + "=" * 70)
print("   ✅ DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70 + "\n")