
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")

# Apply seaborn styling globally for all plots
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

print("=" * 70)
print("      DRUG RECOMMENDATION SYSTEM - EXPLORATORY DATA ANALYSIS")
print("=" * 70)
print("\n✅ Libraries imported and seaborn styling applied.")
print("✅ Matplotlib Agg backend set — plots save directly to disk.\n")

# CONFIGURATION — File Paths

DATASET_PATH = r"C:\project\Drug Recommendation System\data\cleaned_drug_data.csv"
OUTPUT_DIR   = r"C:\project\Drug Recommendation System\outputs\eda_plots"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Plot output directory ready:\n   {OUTPUT_DIR}\n")

# HELPER FUNCTION: Save and Close Plot

def save_plot(filename: str) -> None:
    """Save the current matplotlib figure to the output directory and close it."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close("all")   # FIX: Always close all figures to free memory
    print(f"   💾 Plot saved → {filename}")

# STEP 2: Load the Cleaned Dataset

print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"\n✅ Dataset loaded successfully from:\n   {DATASET_PATH}")
except FileNotFoundError:
    print(f"\n❌ ERROR: File not found at:\n   {DATASET_PATH}")
    print("   Please verify the file path and run data_cleaning.py first.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error loading dataset: {e}")
    sys.exit(1)


# STEP 3: Display First Few Rows

print("\n" + "=" * 70)
print("STEP 2: FIRST 5 ROWS OF THE CLEANED DATASET")
print("=" * 70)
print(f"\n{df.head().to_string(index=False)}")

# STEP 4: Display Dataset Shape

print("\n" + "=" * 70)
print("STEP 3: DATASET SHAPE")
print("=" * 70)
print(f"\n📊 Rows    : {df.shape[0]}")
print(f"📊 Columns : {df.shape[1]}")
print(f"📊 Features: {list(df.columns)}")

# STEP 5: Summary Statistics

print("\n" + "=" * 70)
print("STEP 4: SUMMARY STATISTICS")
print("=" * 70)
print(f"\n{df.describe().round(4)}")

print("\n📋 Categorical Columns — Value Counts:")
for col in ["Sex", "BP", "Cholesterol", "Drug"]:
    print(f"\n   {col}:\n{df[col].value_counts().to_string()}")

# STEP 6: Correlation Matrix for Numerical Columns

print("\n" + "=" * 70)
print("STEP 5: CORRELATION MATRIX")
print("=" * 70)

num_cols    = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
corr_matrix = df[num_cols].corr()

print(f"\n{corr_matrix.round(4)}")
print("\n📊 Generating correlation heatmap...")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    linewidths=0.6,
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Correlation Matrix — Numerical Features", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
save_plot("01_correlation_matrix.png")


# STEP 7: Drug Distribution — Count Plot

print("\n" + "=" * 70)
print("STEP 6: DRUG DISTRIBUTION")
print("=" * 70)

drug_counts = df["Drug"].value_counts()
print(f"\n{drug_counts.to_string()}")
print(f"\nMost prescribed drug : {drug_counts.idxmax()} ({drug_counts.max()} patients)")
print(f"Least prescribed drug: {drug_counts.idxmin()} ({drug_counts.min()} patients)")
print("\n📊 Generating Drug distribution plot...")

fig, ax = plt.subplots(figsize=(9, 5))
sns.countplot(
    data=df,
    x="Drug",
    order=drug_counts.index,
    palette="Set2",
    edgecolor="black",
    linewidth=0.8,
    ax=ax
)
for container in ax.containers:
    ax.bar_label(container, fontsize=11, fontweight="bold", padding=3)
ax.set_title("Drug Distribution (Prescription Count per Drug)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Drug Type", fontsize=12)
ax.set_ylabel("Number of Patients", fontsize=12)
ax.set_ylim(0, drug_counts.max() * 1.15)
plt.tight_layout()
save_plot("02_drug_distribution.png")

# STEP 8: Age Distribution — Histogram with KDE

print("\n" + "=" * 70)
print("STEP 7: AGE DISTRIBUTION")
print("=" * 70)

print(f"\n   Min Age   : {df['Age'].min()}")
print(f"   Max Age   : {df['Age'].max()}")
print(f"   Mean Age  : {df['Age'].mean():.2f}")
print(f"   Median Age: {df['Age'].median()}")
print("\n📊 Generating Age distribution histogram with KDE...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["Age"], bins=20, kde=True, color="#2ecc71",
             edgecolor="black", linewidth=0.6, ax=axes[0])
axes[0].axvline(df["Age"].mean(),   color="red",    linestyle="--",
                linewidth=1.5, label=f"Mean: {df['Age'].mean():.1f}")
axes[0].axvline(df["Age"].median(), color="orange", linestyle="--",
                linewidth=1.5, label=f"Median: {df['Age'].median()}")
axes[0].set_title("Age Distribution (Histogram + KDE)", fontsize=13, fontweight="bold", pad=12)
axes[0].set_xlabel("Age", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].legend(fontsize=10)

sns.boxplot(y=df["Age"], color="#3498db", width=0.4,
            flierprops=dict(marker="o", color="red", markersize=6), ax=axes[1])
axes[1].set_title("Age — Box Plot (Spread & Outliers)", fontsize=13, fontweight="bold", pad=12)
axes[1].set_ylabel("Age", fontsize=12)

plt.suptitle("Age Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_plot("03_age_distribution.png")


# STEP 9: Blood Pressure (BP) vs Drug — Count Plot

print("\n" + "=" * 70)
print("STEP 8: BLOOD PRESSURE (BP) vs DRUG")
print("=" * 70)

bp_drug = df.groupby(["BP", "Drug"]).size().unstack(fill_value=0)
print(f"\n   BP vs Drug cross-tabulation:\n{bp_drug.to_string()}")
print("\n📊 Generating BP vs Drug countplot...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.countplot(data=df, x="BP", hue="Drug",
              palette="Set2", edgecolor="black", linewidth=0.7, ax=axes[0])
axes[0].set_title("Blood Pressure vs Drug Type", fontsize=13, fontweight="bold", pad=12)
axes[0].set_xlabel("Blood Pressure Level", fontsize=12)
axes[0].set_ylabel("Number of Patients", fontsize=12)
axes[0].legend(title="Drug", fontsize=9, title_fontsize=10)

bp_counts = df["BP"].value_counts()
sns.barplot(x=bp_counts.index, y=bp_counts.values,
            palette="Blues_d", edgecolor="black", linewidth=0.7, ax=axes[1])
for i, v in enumerate(bp_counts.values):
    axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=11, fontweight="bold")
axes[1].set_title("Blood Pressure Level Distribution", fontsize=13, fontweight="bold", pad=12)
axes[1].set_xlabel("Blood Pressure Level", fontsize=12)
axes[1].set_ylabel("Number of Patients", fontsize=12)

plt.suptitle("Blood Pressure Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_plot("04_bp_vs_drug.png")

# STEP 10: Cholesterol vs Drug — Count Plot

print("\n" + "=" * 70)
print("STEP 9: CHOLESTEROL vs DRUG")
print("=" * 70)

chol_drug = df.groupby(["Cholesterol", "Drug"]).size().unstack(fill_value=0)
print(f"\n   Cholesterol vs Drug cross-tabulation:\n{chol_drug.to_string()}")
print("\n📊 Generating Cholesterol vs Drug countplot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(data=df, x="Cholesterol", hue="Drug",
              palette="Set1", edgecolor="black", linewidth=0.7, ax=axes[0])
axes[0].set_title("Cholesterol Level vs Drug Type", fontsize=13, fontweight="bold", pad=12)
axes[0].set_xlabel("Cholesterol Level", fontsize=12)
axes[0].set_ylabel("Number of Patients", fontsize=12)
axes[0].legend(title="Drug", fontsize=9, title_fontsize=10)

sns.countplot(data=df, x="Cholesterol", hue="Sex",
              palette=["#e74c3c", "#3498db"], edgecolor="black", linewidth=0.7, ax=axes[1])
axes[1].set_title("Cholesterol Level by Sex", fontsize=13, fontweight="bold", pad=12)
axes[1].set_xlabel("Cholesterol Level", fontsize=12)
axes[1].set_ylabel("Number of Patients", fontsize=12)
axes[1].legend(title="Sex", fontsize=10, title_fontsize=10)

plt.suptitle("Cholesterol Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_plot("05_cholesterol_vs_drug.png")

# STEP 11: Na_to_K Distribution

print("\n" + "=" * 70)
print("STEP 10: Na_to_K DISTRIBUTION")
print("=" * 70)

print(f"\n   Min Na_to_K   : {df['Na_to_K'].min():.4f}")
print(f"   Max Na_to_K   : {df['Na_to_K'].max():.4f}")
print(f"   Mean Na_to_K  : {df['Na_to_K'].mean():.4f}")
print(f"   Median Na_to_K: {df['Na_to_K'].median():.4f}")
print("\n📊 Generating Na_to_K distribution plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df["Na_to_K"], bins=20, kde=True, color="#9b59b6",
             edgecolor="black", linewidth=0.6, ax=axes[0])
axes[0].axvline(df["Na_to_K"].mean(),   color="red",    linestyle="--",
                linewidth=1.5, label=f"Mean: {df['Na_to_K'].mean():.2f}")
axes[0].axvline(df["Na_to_K"].median(), color="orange", linestyle="--",
                linewidth=1.5, label=f"Median: {df['Na_to_K'].median():.2f}")
axes[0].set_title("Na_to_K Distribution (Histogram + KDE)", fontsize=12, fontweight="bold", pad=12)
axes[0].set_xlabel("Na_to_K Ratio", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)
axes[0].legend(fontsize=9)

sns.boxplot(y=df["Na_to_K"], color="#e67e22", width=0.4,
            flierprops=dict(marker="o", color="red", markersize=6), ax=axes[1])
axes[1].set_title("Na_to_K — Box Plot", fontsize=12, fontweight="bold", pad=12)
axes[1].set_ylabel("Na_to_K Ratio", fontsize=11)

drug_order_nak = df.groupby("Drug")["Na_to_K"].median().sort_values(ascending=False).index
sns.violinplot(data=df, x="Drug", y="Na_to_K", order=drug_order_nak,
               palette="Set2", inner="quartile", ax=axes[2])
axes[2].set_title("Na_to_K Distribution by Drug", fontsize=12, fontweight="bold", pad=12)
axes[2].set_xlabel("Drug Type", fontsize=11)
axes[2].set_ylabel("Na_to_K Ratio", fontsize=11)

plt.suptitle("Na_to_K (Sodium-to-Potassium Ratio) Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_plot("06_na_to_k_distribution.png")


# STEP 11B: Sex Distribution and Age vs Drug Analysis

print("\n" + "=" * 70)
print("STEP 11: SEX & AGE vs DRUG ANALYSIS")
print("=" * 70)
print("\n📊 Generating Sex and Age vs Drug analysis plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sex_counts = df["Sex"].value_counts()
axes[0].pie(
    sex_counts.values,
    labels=sex_counts.index,
    autopct="%1.1f%%",
    colors=["#e74c3c", "#3498db"],
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops={"fontsize": 12}
)
axes[0].set_title("Sex Distribution", fontsize=13, fontweight="bold", pad=12)

sns.countplot(data=df, x="Drug", hue="Sex",
              palette=["#e74c3c", "#3498db"],
              edgecolor="black", linewidth=0.7,
              order=df["Drug"].value_counts().index, ax=axes[1])
axes[1].set_title("Drug Prescription by Sex", fontsize=13, fontweight="bold", pad=12)
axes[1].set_xlabel("Drug Type", fontsize=12)
axes[1].set_ylabel("Number of Patients", fontsize=12)
axes[1].legend(title="Sex", fontsize=10, title_fontsize=10)

drug_order_age = df.groupby("Drug")["Age"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Drug", y="Age", order=drug_order_age,
            palette="Set2",
            flierprops=dict(marker="o", color="red", markersize=5), ax=axes[2])
axes[2].set_title("Age Distribution by Drug Type", fontsize=13, fontweight="bold", pad=12)
axes[2].set_xlabel("Drug Type", fontsize=12)
axes[2].set_ylabel("Age", fontsize=12)

plt.suptitle("Sex & Age vs Drug Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_plot("07_sex_age_vs_drug.png")


# STEP 12: Pairplot — Relationships Between All Variables
print("\n" + "=" * 70)
print("STEP 12: PAIRPLOT — FEATURE RELATIONSHIPS")
print("=" * 70)
print("\n📊 Generating pairplot (this may take a few seconds)...")

# Encode categorical columns numerically for pairplot only
df_pair = df.copy()
encode_map = {
    "Sex":         {"Male": 0, "Female": 1},
    "BP":          {"Low": 0, "Normal": 1, "High": 2},
    "Cholesterol": {"Normal": 0, "High": 1},
}
for col, mapping in encode_map.items():
    df_pair[col] = df_pair[col].map(mapping)

pair_cols = ["Age", "Na_to_K", "Sex", "BP", "Cholesterol"]

pair_grid = sns.pairplot(
    df_pair[pair_cols + ["Drug"]],
    hue="Drug",
    palette="Set2",
    diag_kind="kde",
    plot_kws={"alpha": 0.6, "edgecolor": "none", "s": 40},
    diag_kws={"fill": True}
)
pair_grid.figure.suptitle(
    "Pairplot — Feature Relationships Colored by Drug Type",
    fontsize=14, fontweight="bold", y=1.02
)
pair_grid.add_legend(title="Drug", bbox_to_anchor=(1.05, 0.5), loc="center left")
save_plot("08_pairplot.png")

# FINAL SUMMARY

print("\n" + "=" * 70)
print("EDA SUMMARY")
print("=" * 70)

print(f"""
   📊 Dataset Shape         : {df.shape[0]} rows × {df.shape[1]} columns
   🎯 Target Variable       : Drug ({df['Drug'].nunique()} unique classes)
   📌 Drug Classes          : {sorted(df['Drug'].unique().tolist())}

   🔢 Numerical Features    :
      - Age     → Range [{df['Age'].min()} – {df['Age'].max()}], Mean: {df['Age'].mean():.1f}
      - Na_to_K → Range [{df['Na_to_K'].min():.2f} – {df['Na_to_K'].max():.2f}], Mean: {df['Na_to_K'].mean():.2f}

   🏷️  Categorical Features  :
      - Sex         : {df['Sex'].unique().tolist()}
      - BP          : {df['BP'].unique().tolist()}
      - Cholesterol : {df['Cholesterol'].unique().tolist()}

   💡 Key Observations:
      - DrugY is the most frequently prescribed drug.
      - Na_to_K ratio is a strong predictor — DrugY patients have
        notably higher Na_to_K values than other drug groups.
      - High BP patients are predominantly prescribed DrugA or DrugB.
      - Low BP patients tend toward DrugC.
      - Cholesterol and Sex provide additional segmentation within classes.

   📁 All 8 plots saved to:
      {OUTPUT_DIR}
""")

print("=" * 70)
print("   ✅ EDA PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70 + "\n")