# %% [markdown]
# # 04. Model Evaluation
# This notebook evaluates the trained model on the held-out test set.

# %%
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from src.preprocessing import load_cleaned_data
from src.evaluate import generate_results

# %%
# 1. Get Test Data
print("Loading Processed Data...")
_, _, X_test, _, _, y_test = load_cleaned_data()

# %%
# 2. Evaluate
generate_results(X_test, y_test, model_path="outputs/models/model_pre_accident.keras")

# %%
print("Evaluation Complete. Check 'outputs/' for results.")
