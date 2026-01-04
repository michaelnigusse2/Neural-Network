# %% [markdown]
# # 03. Model Training
# This notebook builds the Unified Neural Network and trains it.

# %%
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from src.preprocessing import load_cleaned_data
from src.train import train_model

# %%
# 1. Get Data (Load processed data)
print("Loading Processed Data...")
X_train, X_val, X_test, y_train, y_val, y_test = load_cleaned_data()

# %%
# 2. Configure Training
config = {
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001
}

# %%
# 3. Train
print("Starting Training Process...")
model_wrapper, history = train_model(X_train, y_train, X_val, y_val, config, save_name="model_pre_accident.keras")

# %%
print("Training Complete. Model and curves saved.")
