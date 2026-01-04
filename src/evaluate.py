"""
Evaluation Utilities.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# Append src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import UnifiedMultitaskModel

OUTPUT_METRICS = "outputs/metrics"
OUTPUT_FIGURES = "outputs/figures"
OUTPUT_MODELS = "outputs/models"

def generate_results(X_test, y_test, model_path="outputs/models/final_model.keras"):
    """
    Evaluates the model and saves reports and plots.
    """
    print("Evaluating Model...")
    os.makedirs(OUTPUT_METRICS, exist_ok=True)
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    
    # Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model = load_model(model_path)
    
    # Predict
    # Need to reproduce prediction logic from UnifiedMultitaskModel wrapper
    raw_preds = model.predict(X_test, verbose=0)
    
    # Parse predictions (Shared logic with original wrapper)
    sev_p = raw_preds["severity_head"].flatten()
    ns_p = raw_preds["non_severe_head"].flatten()
    s_p = raw_preds["severe_head"].flatten()
    
    final_preds = []
    for i in range(len(sev_p)):
        if sev_p[i] < 0.5:
            if ns_p[i] < 0.5:
                final_preds.append("Minor")
            else:
                final_preds.append("PDO")
        else:
            if s_p[i] < 0.5:
                final_preds.append("Serious")
            else:
                final_preds.append("Fatal")
                
    y_pred = np.array(final_preds)
    
    # Metrics
    labels = ["Minor", "PDO", "Serious", "Fatal"]
    
    # 1. Classification Report (Overall)
    cr = classification_report(y_test, y_pred, target_names=labels, digits=4)
    
    # 2. Per-Head Metrics
    # Helper to calculate binary metrics
    def get_binary_metrics(y_true, y_prob, name):
        y_pred_bin = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred_bin)
        return f"\n--- {name} ---\nAccuracy: {acc:.4f}\n" + \
               classification_report(y_true, y_pred_bin, target_names=["Class 0", "Class 1"], digits=4)

    # Head 1: Severity (0=Non-Severe, 1=Severe)
    # Ground Truth
    y_test_lower = y_test.str.strip().str.casefold()
    is_severe_truth = y_test_lower.isin(["serious", "fatal"]).astype(int)
    head1_report = get_binary_metrics(is_severe_truth, sev_p, "Head 1: Severity (Non-Severe vs Severe)")
    
    # Head 2: Non-Severe Type (0=Minor, 1=PDO)
    # Evaluate ONLY on instances that are actually Non-Severe
    ns_mask = y_test_lower.isin(["minor", "pdo"])
    if ns_mask.sum() > 0:
        is_pdo_truth = (y_test_lower[ns_mask] == "pdo").astype(int)
        head2_report = get_binary_metrics(is_pdo_truth, ns_p[ns_mask], "Head 2: Non-Severe Type (Minor vs PDO)")
    else:
        head2_report = "\n--- Head 2 ---\nNo Non-Severe samples in test set.\n"
        
    # Head 3: Severe Type (0=Serious, 1=Fatal)
    # Evaluate ONLY on instances that are actually Severe
    s_mask = y_test_lower.isin(["serious", "fatal"])
    if s_mask.sum() > 0:
        is_fatal_truth = (y_test_lower[s_mask] == "fatal").astype(int)
        head3_report = get_binary_metrics(is_fatal_truth, s_p[s_mask], "Head 3: Severe Type (Serious vs Fatal)")
    else:
        head3_report = "\n--- Head 3 ---\nNo Severe samples in test set.\n"

    # Save Classification Reports
    with open(os.path.join(OUTPUT_METRICS, "classification_report.txt"), "w") as f:
        f.write("Test Set Classification Report (Final 4-Class)\n")
        f.write("============================================\n")
        f.write(cr)
        f.write("\n\n")
        f.write("Detailed Per-Head Performance\n")
        f.write("=============================\n")
        f.write(head1_report)
        f.write(head2_report)
        f.write(head3_report)
    print("Classification report saved.")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, "confusion_matrix.png"))
    plt.close()
    print("Confusion matrix saved.")
    
    # 3. Overall Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    return acc, cr
