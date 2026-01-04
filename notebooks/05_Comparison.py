# %% [markdown]
# # 05. Model Comparison (Pre vs Post Accident)
# This notebook trains a second model using BOTH Pre-Accident and Post-Accident features (which contains outcome leakage).
# It then compares the performance of the "Deployable" (Pre-Only) model vs the "Upper Bound" (Pre+Post) model.

# %%
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from src.preprocessing import prepare_and_save, load_cleaned_data
from src.train import train_model
from src.evaluate import generate_results

OUTPUT_METRICS = "outputs/metrics"
OUTPUT_FIGURES = "outputs/figures"

# %% [markdown]
# ## 1. Train Model B (Pre + Post Accident)

# %%
# Process Data INCLUDING Post-Accident features
print("Preparing Data with Post-Accident Features...")
X_train_B, X_val_B, X_test_B, y_train_B, y_val_B, y_test_B = prepare_and_save(include_post_accident=True)

# %%
# Train Model B
print("Training Model B (Upper Bound)...")
config_B = {
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001
}
model_B, history_B = train_model(X_train_B, y_train_B, X_val_B, y_val_B, config_B, save_name="model_post_accident.keras", plot_name="training_curves_model_B.png")

# %%
# Evaluate Model B
print("Evaluating Model B...")
# Helper to get pure metrics without saving report to default location (we'll save comparison later)
# We can use generate_results but it overwrites standard report. That's fine, we will regenerate standard report for Model A later if needed,
# or we just rely on the comparison.
# Actually, let's manually predict to get metrics for comparison table.

def get_metrics(model, X, y):
    raw_preds = model.model.predict(X, verbose=0)
    sev_p = raw_preds["severity_head"].flatten()
    ns_p = raw_preds["non_severe_head"].flatten()
    s_p = raw_preds["severe_head"].flatten()
    
    final_preds = []
    for i in range(len(sev_p)):
        if sev_p[i] < 0.5:
            if ns_p[i] < 0.5: final_preds.append("Minor")
            else: final_preds.append("PDO")
        else:
            if s_p[i] < 0.5: final_preds.append("Serious")
            else: final_preds.append("Fatal")
            
    y_pred = pd.Series(final_preds)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    # Fatal Metrics
    is_fatal_true = (y == "Fatal").astype(int)
    is_fatal_pred = (y_pred == "Fatal").astype(int)
    fatal_rec = recall_score(is_fatal_true, is_fatal_pred, zero_division=0)
    fatal_prec = precision_score(is_fatal_true, is_fatal_pred, zero_division=0)
    
    return acc, f1, fatal_rec, fatal_prec

acc_B, f1_B, rec_B, prec_B = get_metrics(model_B, X_test_B, y_test_B)

# %% [markdown]
# ## 2. Load Model A (Pre-Accident Only)

# %%
# Load Data A
print("Loading Model A Data...")
X_train_A, X_val_A, X_test_A, y_train_A, y_val_A, y_test_A = load_cleaned_data("cleaned_data.csv")

# Load Model A (Assuming it was saved by 03_Model_Training.py as 'model_pre_accident.keras')
# We need to wrap it or just load raw.
from tensorflow.keras.models import load_model
from src.models import UnifiedMultitaskModel

# Quick hack: Instantiate wrapper to get helper methods, then replace internal model
model_A_wrapper = UnifiedMultitaskModel(input_dim=X_train_A.shape[1])
model_A_wrapper.model = load_model("outputs/models/model_pre_accident.keras")

acc_A, f1_A, rec_A, prec_A = get_metrics(model_A_wrapper, X_test_A, y_test_A)

# %% [markdown]
# ## 3. Comparison Results

# %%
comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "Macro F1", "Fatal Recall", "Fatal Precision"],
    "Model A (Pre-Only)": [acc_A, f1_A, rec_A, prec_A],
    "Model B (Pre+Post)": [acc_B, f1_B, rec_B, prec_B]
})

print("\nModel Comparison:")
print(comparison_df.round(4))

# Save
comparison_df.to_csv(os.path.join(OUTPUT_METRICS, "model_comparison_table.csv"), index=False)
print("Comparison table saved.")

# %%
# Comparison Plot
comparison_df_melted = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_df_melted, x="Metric", y="Score", hue="Model")
plt.title("Model Comparison: Deployable vs Upper Bound")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES, "comparison_chart.png"))
plt.close()
print("Comparison chart saved.")

# %%
# Confusion Matrix for Model B (Post-Accident)
from sklearn.metrics import confusion_matrix
raw_preds_B = model_B.model.predict(X_test_B, verbose=0)
sev_p = raw_preds_B["severity_head"].flatten()
ns_p = raw_preds_B["non_severe_head"].flatten()
s_p = raw_preds_B["severe_head"].flatten()

final_preds_B = []
for i in range(len(sev_p)):
    if sev_p[i] < 0.5:
        if ns_p[i] < 0.5: final_preds_B.append("Minor")
        else: final_preds_B.append("PDO")
    else:
        if s_p[i] < 0.5: final_preds_B.append("Serious")
        else: final_preds_B.append("Fatal")

labels = ["Minor", "PDO", "Serious", "Fatal"]
cm_B = confusion_matrix(y_test_B, final_preds_B, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_B, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Model B: Pre + Post)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES, "confusion_matrix_model_B.png"))
plt.close()
print("Model B Confusion Matrix saved.")
