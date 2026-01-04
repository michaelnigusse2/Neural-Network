# %% [markdown]
# # 02. Data Preparation
# This notebook cleans the data, handles missing values, encodes features, and generates the training splits.

# %%
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from src.preprocessing import prepare_and_save

# %%
# Run Preprocessing
# We strictly use Pre-Accident features for the main academic submission to avoid data leakage.
X_train, X_val, X_test, y_train, y_val, y_test = prepare_and_save(include_post_accident=False)

# %%
print("Data Preparation Complete.")
print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")
