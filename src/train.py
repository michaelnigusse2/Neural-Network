"""
Training Loop Utilities.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

# Append src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import UnifiedMultitaskModel

OUTPUT_METRICS = "outputs/metrics"
OUTPUT_FIGURES = "outputs/figures"

def encode_targets(y_series: pd.Series):
    """Encodes Multi-task targets and weights."""
    n = len(y_series)
    sev_target = np.zeros(n, dtype=np.float32)
    ns_target = np.zeros(n, dtype=np.float32)
    s_target = np.zeros(n, dtype=np.float32)
    
    sev_weight = np.ones(n, dtype=np.float32)
    ns_weight = np.zeros(n, dtype=np.float32)
    s_weight = np.zeros(n, dtype=np.float32)
    
    # Logic matching original implementation
    y_lower = y_series.str.strip().str.casefold()
    
    # Minor
    idx = y_lower == "minor"
    sev_target[idx] = 0; ns_target[idx] = 0; ns_weight[idx] = 1.0
    
    # PDO
    idx = y_lower == "pdo"
    sev_target[idx] = 0; ns_target[idx] = 1; ns_weight[idx] = 1.0
    
    # Serious
    idx = y_lower == "serious"
    sev_target[idx] = 1; s_target[idx] = 0; s_weight[idx] = 1.0
    
    # Fatal
    idx = y_lower == "fatal"
    sev_target[idx] = 1; s_target[idx] = 1; s_weight[idx] = 5.0 # Weighted Loss
    
    y_dict = {"severity_head": sev_target, "non_severe_head": ns_target, "severe_head": s_target}
    w_dict = {"severity_head": sev_weight, "non_severe_head": ns_weight, "severe_head": s_weight}
    return y_dict, w_dict

def train_model(X_train, y_train, X_val, y_val, config=None, save_name="final_model.keras", plot_name="training_curves.png"):
    """
    Trains the model and saves artifacts.
    """
    if config is None:
        config = {"epochs": 50, "batch_size": 32, "lr": 0.001}
        
    print("Initializing Model...")
    wrapper = UnifiedMultitaskModel(input_dim=X_train.shape[1], learning_rate=config["lr"])
    wrapper.save_summary()
    wrapper.save_architecture()
    
    # Save Config
    with open(os.path.join(OUTPUT_METRICS, "training_config.txt"), "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
            
    # Prepare Data
    train_targets, train_weights = encode_targets(y_train)
    val_targets, val_weights = encode_targets(y_val)
    
    # Train
    print("Starting Training...")
    history = wrapper.model.fit(
        X_train, train_targets,
        validation_data=(X_val, val_targets),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        sample_weight=train_weights,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )
    
    # Save Model
    wrapper.save_model(save_name)
    
    # Plot Curves
    print("Plotting Training Curves...")
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy (Severity Head as proxy for overall progress)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['severity_head_acc'], label='Train Sev Acc')
    plt.plot(history.history['val_severity_head_acc'], label='Val Sev Acc')
    plt.title('Severity Head Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, plot_name))
    plt.close()
    
    return wrapper, history
