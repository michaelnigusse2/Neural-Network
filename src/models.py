"""
Model Definition.
"""
import tensorflow as tf
import os
import sys
# Add Graphviz to PATH (common default install location)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision
from tensorflow.keras.utils import plot_model

# Append src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_METRICS = "outputs/metrics"
OUTPUT_FIGURES = "outputs/figures"
OUTPUT_MODELS = "outputs/models"

class UnifiedMultitaskModel:
    def __init__(self, input_dim: int, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        inputs = Input(shape=(self.input_dim,), name="input_features")
        
        x = Dense(128, activation="relu", name="shared_dense_1")(inputs)
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu", name="shared_dense_2")(x)
        x = Dropout(0.3)(x)
        
        severity_out = Dense(1, activation="sigmoid", name="severity_head")(x)
        non_severe_out = Dense(1, activation="sigmoid", name="non_severe_head")(x)
        severe_out = Dense(1, activation="sigmoid", name="severe_head")(x)
        
        model = Model(
            inputs=inputs, 
            outputs={
                "severity_head": severity_out,
                "non_severe_head": non_severe_out,
                "severe_head": severe_out
            },
            name="unified_accident_model"
        )
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                "severity_head": BinaryCrossentropy(),
                "non_severe_head": BinaryCrossentropy(),
                "severe_head": BinaryCrossentropy(),
            },
            metrics={
                "severity_head": [BinaryAccuracy(name="acc")],
                "non_severe_head": [BinaryAccuracy(name="acc")],
                "severe_head": [BinaryAccuracy(name="acc"), Recall(name="recall"), Precision(name="precision")],
            }
        )
        return model

    def save_summary(self):
        """Saves model summary to text file."""
        os.makedirs(OUTPUT_METRICS, exist_ok=True)
        path = os.path.join(OUTPUT_METRICS, "model_summary.txt")
        try:
            with open(path, "w", encoding='utf-8') as f:
                self.model.summary(print_fn=lambda x: f.write(x + "\n"), line_length=120)
            print(f"Model summary saved to {path}")
        except Exception as e:
            print(f"Warning: Could not save model summary: {e}")
            with open(path, "w") as f:
                f.write(f"Model Summary Failed: {e}")

    def save_architecture(self):
        """Saves model architecture diagram."""
        os.makedirs(OUTPUT_FIGURES, exist_ok=True)
        path_png = os.path.join(OUTPUT_FIGURES, "model_architecture.png")
        try:
            plot_model(self.model, to_file=path_png, show_shapes=True, show_layer_names=True)
            print(f"Model architecture saved to {path_png}")
        except Exception as e:
            print(f"Graphviz not found. Saving text description instead.")
            # Fallback: Save a detailed text description satisfying the architecture requirement
            path_txt = os.path.join(OUTPUT_FIGURES, "model_architecture_fallback.txt")
            with open(path_txt, "w", encoding='utf-8') as f:
                f.write("Model Architecture (Graphviz unavailable via plot_model)\n")
                f.write("======================================================\n\n")
                self.model.summary(print_fn=lambda x: f.write(x + "\n"), line_length=120, show_trainable=True)
            
            # Create a placeholder image saying "See text file" so the user knows
            # (Optional, but helps if they look for an image. Just text is safer for now)
            with open(path_png + ".error.txt", "w") as f:
                f.write(f"Graphviz not installed. See model_architecture_fallback.txt for details.")

    def save_model(self, filename="unified_model.keras"):
        os.makedirs(OUTPUT_MODELS, exist_ok=True)
        path = os.path.join(OUTPUT_MODELS, filename)
        self.model.save(path)
        print(f"Model saved to {path}")
