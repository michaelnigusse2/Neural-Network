"""
EDA Utilities.
Generates statistical summaries and plots for the dataset.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Append src to path so we can import config if run directly (though notebooks usually handle this)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import PRE_ACCIDENT_FEATURES, POST_ACCIDENT_FEATURES, TARGET_COL

OUTPUT_METRICS = "outputs/metrics"
OUTPUT_FIGURES = "outputs/figures"

def generate_eda_report(df: pd.DataFrame):
    """
    Generates all required EDA artifacts.
    """
    print("Generating EDA Report...")
    os.makedirs(OUTPUT_METRICS, exist_ok=True)
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    
    # 1. Dataset Shape & Types
    with open(os.path.join(OUTPUT_METRICS, "dataset_summary.txt"), "w") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Data Types:\n")
        f.write(df.dtypes.to_string())
        f.write("\n")
        
    # 2. Missing Value Summary
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing.to_csv(os.path.join(OUTPUT_METRICS, "missing_values.csv"))
    
    # 3. Target Distribution
    from src.preprocessing import normalize_target
    
    plt.figure(figsize=(10, 6))
    valid_targets = df[TARGET_COL].dropna()
    cleaned_targets = normalize_target(valid_targets)
    sns.countplot(y=cleaned_targets, order=cleaned_targets.value_counts().index)
    plt.title("Target Class Distribution (Cleaned)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, "target_distribution.png"))
    plt.close()
    
    # 4. Correlation Heatmap (Numeric)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title("Numeric Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES, "correlation_heatmap.png"))
        plt.close()
        
    # 5. Feature Distributions (Sample)
    # We'll stick to a few key numerical columns to avoid spamming 200 plots
    key_numeric = ["Driver age", "Vehicle Year", "Number of Casualties"] # Examples
    # Check which exist
    present_numeric = [c for c in key_numeric if c in df.columns]
    
    # Just plot top 5 numeric columns
    top_numeric = numeric_df.columns[:5]
    
    for col in top_numeric:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        safe_col = "".join(x for x in col if x.isalnum())
        plt.savefig(os.path.join(OUTPUT_FIGURES, f"dist_{safe_col}.png"))
        plt.close()
        
    print("EDA Generation Complete.")
