# %% [markdown]
# # 01. Exploratory Data Analysis (EDA)
# This notebook loads the raw data, analyzes its structure, and generates initial visualizations.

# %%
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from src.eda import generate_eda_report

# %%
# Load Data
RAW_DATA_PATH = "data/raw/raw_data.xlsx"
print(f"Loading {RAW_DATA_PATH}...")
df = pd.read_excel(RAW_DATA_PATH)

print(f"Loaded {len(df)} rows.")

# %%
# Generate Report
generate_eda_report(df)

# %%
print("EDA Complete. Check 'outputs/' directory.")
