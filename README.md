# Traffic Accident Severity Classification

Multi-task neural network pipeline that predicts traffic accident severity across four discrete classes—**Minor**, **PDO**, **Serious**, **Fatal**—while handling class imbalance and strict feature leakage rules. The project targets an academic setting but follows production-grade hygiene: deterministic preprocessing, artifact tracking, and reproducible notebooks/scripts.

---

## Highlights
- **Multi-Task Architecture**: Shared backbone with three binary heads (`severity`, `non_severe`, `severe`) to decompose the four-class problem.
- **Leakage-Safe Feature Sets**: Pre-Accident predictors are kept separate from Post-Accident outcomes; both cleaned datasets are generated for auditing.
- **Weighted Lossing**: Fatal cases receive a 5× weight inside the severe head to prioritize recall on the rarest class.
- **Artifact-Rich Outputs**: Models, metrics, plots, and configs are versioned inside `outputs/` for downstream reporting.
- **Notebook + Script Parity**: Each stage can be run as a notebook (for demonstration) or invoked via the Python modules in `src/`.

---

## Repository Layout
```
.
├── data/
│   ├── raw/
│   │   └── raw_data.xlsx              # Original Excel dataset (place your copy here)
│   └── processed/
│       ├── cleaned_data.csv           # Pre-Accident cleaned dataset (auto-regenerated)
│       └── cleaned_data_post.csv      # Post-Accident enriched cleaned dataset
├── notebooks/
│   ├── 01_EDA.py                      # Exploratory analysis visuals + summaries
│   ├── 02_Data_Preparation.py         # Calls preprocessing.prepare_and_save
│   ├── 03_Model_Training.py           # Wraps src/train.py utilities
│   ├── 04_Evaluation.py               # Loads model & produces reports
│   └── 05_Comparison.py               # (Optional) scenario comparisons
├── outputs/
│   ├── figures/                       # Confusion matrices, training curves, EDA plots
│   ├── metrics/                       # Classification reports, split summaries, configs
│   └── models/                        # Saved Keras .keras files
└── src/
    ├── config.py                      # Feature lists, target column names
    ├── preprocessing.py               # Deterministic cleaning / splitting
    ├── models.py                      # UnifiedMultitaskModel definition
    ├── train.py                       # Training orchestration + plots
    ├── evaluate.py                    # Test-time metrics & plots
    └── audit_categorical_values.py    # Utility scripts
```

---

## Data Pipeline
1. **Raw ingestion** – place `raw_data.xlsx` under `data/raw/`.
2. **Normalization & Cleaning** – `src.preprocessing.prepare_and_save`:
   - Normalizes `Accident Type` strings (`Minor`, `PDO`, `Serious`, `Fatal`).
   - Selects configurable feature sets from `PRE_ACCIDENT_FEATURES` / `POST_ACCIDENT_FEATURES`.
   - Imputes numerical medians & categorical modes, one-hot encodes categoricals, scales numerics.
   - Writes two synchronized CSVs:
     - `cleaned_data.csv` (pre-accident only, default training input).
     - `cleaned_data_post.csv` (adds post-accident leakage features for analysis).
   - Produces split summaries (`split_summary.txt` + `split_summary_post.txt`) with class balance info.
3. **Deterministic splits** – stratified 70/15/15 (train/val/test) using `RANDOM_STATE = 42`.

> ⚠️ If the cleaned CSVs are deleted, rerun `python notebooks/02_Data_Preparation.py` (or call `prepare_and_save()` directly) to regenerate both files from scratch.

---

## Model Architecture & Training
- **Backbone**: Two dense layers (128 → 64 units, ReLU, Dropout 0.3).
- **Heads**:
  - `severity_head`: predicts severe vs non-severe.
  - `non_severe_head`: resolves Minor vs PDO conditional on being non-severe.
  - `severe_head`: resolves Serious vs Fatal conditional on being severe.
- **Optimization**: Adam (default LR 1e-3) with binary cross-entropy per head.
- **Sample Weights**: `src.train.encode_targets` injects per-head weights; Fatal class receives 5× penalty in `severe_head`.
- **Callbacks**: EarlyStopping on `val_loss` with patience 10 and best-weight restore.
- **Artifacts**:
  - `outputs/models/final_model.keras`
  - `outputs/figures/training_curves.png`
  - `outputs/metrics/training_config.txt`
  - `outputs/metrics/model_summary.txt`

Adjust hyperparameters by editing the `config` dict passed into `train.train_model` or the notebook harness (epochs, batch size, learning rate).

---

## Evaluation & Reporting
`src.evaluate.generate_results` loads a saved `.keras` file and:
- Reconstructs final four-class predictions from the three heads.
- Saves `classification_report.txt` with both the overall report and per-head binary diagnostics.
- Produces `confusion_matrix.png` plus any comparison charts housed under `outputs/figures/`.
- Prints aggregate accuracy for quick CLI inspection.

Additional comparative figures (e.g., pre- vs post-accident models) live in `outputs/figures/comparison_*`.

---

## End-to-End Execution
```bash
# (Optional) create environment
python -m venv .venv
.venv\Scripts\activate      # Windows

pip install -r requirements.txt  # If provided; otherwise install TensorFlow, pandas, numpy, matplotlib, seaborn, scikit-learn

# 1. Exploratory Data Analysis
python notebooks/01_EDA.py

# 2. Data Preparation (regenerates both cleaned CSVs)
python notebooks/02_Data_Preparation.py

# 3. Model Training
python notebooks/03_Model_Training.py

# 4. Evaluation
python notebooks/04_Evaluation.py
```

### Programmatic Usage
```python
from src import preprocessing, train, evaluate

X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.prepare_and_save()
wrapper, history = train.train_model(X_train, y_train, X_val, y_val)
acc, cr = evaluate.generate_results(X_test, y_test)
```

---

## Key Results
- Weighted multi-task setup improves Fatal recall while maintaining Minor/PDO precision.
- Final metrics (prec/rec/F1 per class and per head) are stored in `outputs/metrics/classification_report.txt`.
- Confusion matrices reveal most confusion occurs between neighboring severity tiers (e.g., Serious vs Fatal).

---

## Troubleshooting
| Issue | Fix |
| --- | --- |
| `raw_data.xlsx` missing | Ensure the raw Excel file exists under `data/raw/`. |
| `Cleaned data not found` | Run `02_Data_Preparation.py` to regenerate both CSVs. |
| `ModuleNotFoundError` for `src.*` | Execute scripts from repo root so Python can resolve the package path, or adjust `PYTHONPATH`. |
| TensorFlow GPU warnings | The project runs fine on CPU; ignore CUDA warnings if GPU is unavailable. |

---

## Dependencies
- Python 3.8+
- TensorFlow / Keras 2.x
- pandas, numpy, scikit-learn
- matplotlib, seaborn

Install via `pip install -r requirements.txt` if such a file is provided, or manually install the packages above.

---

## Acknowledgements
Data and problem framing stem from the accident-severity academic study (details redacted). The pipeline is organized to keep the submission transparent, reproducible, and audit-ready.
