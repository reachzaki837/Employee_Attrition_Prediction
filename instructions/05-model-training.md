# 05 — Model Training & Evaluation

> ⚡ Copilot: Train all 3 models, compare them, save the best one.
> ROC-AUC is the PRIMARY metric — not accuracy (due to class imbalance).

---

## Phase Task List

- [ ] Create `src/models/trainer.py` — train all 3 models
- [ ] Create `src/models/evaluator.py` — compute all metrics + charts
- [ ] Create `src/models/selector.py` — compare models, pick best
- [ ] Save best model to `models/best_model.pkl`
- [ ] Save comparison table to `reports/model_comparison.csv`
- [ ] Generate evaluation charts to `reports/evaluation/`

---

## Models to Train

### Model 1 — Logistic Regression (Baseline)
```python
from sklearn.linear_model import LogisticRegression

LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
```

### Model 2 — Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
```

### Model 3 — XGBoost (Expected Winner)
```python
import xgboost as xgb

xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,     # handles class imbalance
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
```

---

## Metrics to Compute for Each Model

```python
def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute full evaluation metrics.

    Returns dict with:
    - accuracy: float
    - precision: float
    - recall: float
    - f1: float
    - roc_auc: float        ← PRIMARY metric
    - avg_precision: float  ← Secondary (good for imbalanced)
    - confusion_matrix: np.ndarray
    """
```

---

## Required Evaluation Charts

### Chart 1 — ROC Curves (all 3 models on same plot)
```
Filename: reports/evaluation/01_roc_curves.png
Plot ROC curve for each model with AUC in legend
Add diagonal reference line
Title: "ROC Curves — Model Comparison"
```

### Chart 2 — Confusion Matrix (best model only)
```
Filename: reports/evaluation/02_confusion_matrix.png
Use seaborn heatmap, annot=True
Labels: ["Stayed", "Left"]
Title: "Confusion Matrix — [Best Model Name]"
```

### Chart 3 — Feature Importance (best model)
```
Filename: reports/evaluation/03_feature_importance.png
Top 15 features, horizontal bar chart
Sort descending by importance
Title: "Top 15 Feature Importances"
```

### Chart 4 — Precision-Recall Curve
```
Filename: reports/evaluation/04_precision_recall.png
Plot for best model
Mark the operating threshold point
Title: "Precision-Recall Curve"
```

---

## Model Comparison Table (Print + Save)

```
=== Model Comparison ===
Model                Accuracy  Precision  Recall   F1      ROC-AUC
Logistic Regression  0.XXX     0.XXX      0.XXX    0.XXX   0.XXX
Random Forest        0.XXX     0.XXX      0.XXX    0.XXX   0.XXX
XGBoost              0.XXX     0.XXX      0.XXX    0.XXX   0.XXX
========================
Best model: XGBoost (ROC-AUC: 0.XXX)
Saved to: models/best_model.pkl
```

---

## Model Saving

```python
import joblib

def save_model(model, scaler, feature_names: list[str]) -> None:
    """Save model, scaler, and feature names together."""
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": type(model).__name__,
    }, MODEL_PATH)
```

## Target Metric

ROC-AUC must be > 0.80 on test set.
If not, try: tuning XGBoost `scale_pos_weight`, adjusting threshold,
or adding more engineered features before escalating.
