"""
Preprocessing Utilities.
Handles data cleaning, transformation, and splitting.
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Append src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import PRE_ACCIDENT_FEATURES, POST_ACCIDENT_FEATURES, TARGET_COL, CLEAN_TARGET_COL, KEEP_LABELS

OUTPUT_METRICS = "outputs/metrics"
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "raw_data.xlsx")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "processed", "cleaned_data.csv")
RANDOM_STATE = 42

def normalize_target(value: pd.Series) -> pd.Series:
    """Normalize target labels to standard categories."""
    def _map(label: object) -> object:
        if pd.isna(label):
            return pd.NA
        text = str(label).strip()
        normalized = "".join(ch for ch in text.casefold() if ch.isalnum())
        if any(token in normalized for token in ("propertydamageonly", "pdo", "pod")):
            return "PDO"
        return KEEP_LABELS.get(text.casefold(), text)
    return value.map(_map)

def prepare_and_save(include_post_accident=False, ensure_both_outputs=True):
    """
    Loads raw data, processes it, saves cleaned CSV, and reports split stats.
    When called with the default arguments, this function also generates the
    complementary dataset (with post-accident features) so both cleaned CSVs
    stay in sync.
    """
    print("Preprocessing Data...")
    os.makedirs(OUTPUT_METRICS, exist_ok=True)
    
    # 1. Load
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Missing {RAW_DATA_PATH}")
    
    df = pd.read_excel(RAW_DATA_PATH)
    
    # 2. Target Cleaning
    target = df[TARGET_COL].astype("string").str.strip()
    df[CLEAN_TARGET_COL] = normalize_target(target)
    df = df.dropna(subset=[CLEAN_TARGET_COL])
    
    # 3. Feature Selection
    feature_cols = PRE_ACCIDENT_FEATURES.copy()
    if include_post_accident:
        feature_cols.extend(POST_ACCIDENT_FEATURES)
        
    # Validation
    valid_features = [c for c in feature_cols if c in df.columns]
    X = df[valid_features].copy()
    y = df[CLEAN_TARGET_COL].copy()
    
    # 4. Missing Values & Encoding
    # Impute
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
        
    for col in categorical_cols:
        mode_val = X[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
        X[col] = X[col].fillna(fill_val)
        
    # Encode
    dummy_X = pd.get_dummies(X[categorical_cols], drop_first=False)
    
    # Scale
    if numeric_cols:
        scaler = StandardScaler()
        scaled_numeric = pd.DataFrame(
            scaler.fit_transform(X[numeric_cols]),
            columns=numeric_cols,
            index=X.index
        )
        X_processed = pd.concat([scaled_numeric, dummy_X], axis=1)
    else:
        X_processed = dummy_X
        
    # Join Target for Saving
    cleaned_df = pd.concat([X_processed, y], axis=1)
    
    # Determine save path
    suffix = "cleaned_data_post.csv" if include_post_accident else "cleaned_data.csv"
    save_path = os.path.join(DATA_DIR, "processed", suffix)
    
    cleaned_df.to_csv(save_path, index=False)
    print(f"Saved cleaned data to {save_path}")
    
    # 5. Split & Summary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    summary_text = f"""
Split Summary (Random State={RANDOM_STATE})
-------------------------------------------
Total Samples: {len(X)}
Features:      {X_processed.shape[1]}

Training Set:   {len(X_train)} ({len(X_train)/len(X):.1%})
Validation Set: {len(X_val)} ({len(X_val)/len(X):.1%})
Test Set:       {len(X_test)} ({len(X_test)/len(X):.1%})

Class Distribution (Train):
{y_train.value_counts(normalize=True).to_string()}
    """
    
    summary_name = "split_summary_post.txt" if include_post_accident else "split_summary.txt"
    with open(os.path.join(OUTPUT_METRICS, summary_name), "w") as f:
        f.write(summary_text)
        
    print("Split summary saved.")
    
    if ensure_both_outputs and not include_post_accident:
        print("Preparing post-accident enriched dataset to keep artifacts complete...")
        prepare_and_save(include_post_accident=True, ensure_both_outputs=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    prepare_and_save()

def load_cleaned_data(filename="cleaned_data.csv"):
    """
    Loads the already processed dataset from CSV and returns the standard splits.
    Ensures '03_Model_Training' and '04_Evaluation' use the exact same data as '02_Data_Preparation'.
    """
    path = os.path.join(DATA_DIR, "processed", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned data not found at {path}. Run '02_Data_Preparation.py' first.")
    
    print(f"Loading processed data from {path}...")
    df = pd.read_csv(path)
    
    # Separate Features and Target
    X = df.drop(columns=[CLEAN_TARGET_COL])
    y = df[CLEAN_TARGET_COL]
    
    # Reproduce Split
    # Since we saved the encoded DataFrame, we don't need to re-encode.
    # Just split using the same random state.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
