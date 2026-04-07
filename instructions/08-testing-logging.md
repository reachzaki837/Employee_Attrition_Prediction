# 08 — Testing & Logging

---

## Phase Task List

- [ ] Create `tests/test_data.py` — test data loading and cleaning
- [ ] Create `tests/test_features.py` — test feature engineering
- [ ] Create `tests/test_model.py` — test model predictions
- [ ] Create `tests/test_api.py` — test FastAPI endpoints
- [ ] Create `src/utils/logger.py` — structured logger
- [ ] Run `pytest --cov=src --cov-report=term-missing`
- [ ] Achieve > 70% coverage

---

## Key Tests to Write

```python
# tests/test_data.py
def test_dataset_downloads_successfully():
def test_clean_data_has_no_nulls():
def test_target_column_is_binary():
def test_dropped_columns_not_present():

# tests/test_features.py
def test_income_per_year_exp_is_positive():
def test_satisfaction_score_in_range():
def test_smote_balances_classes():
def test_no_data_leakage_in_split():

# tests/test_model.py
def test_model_file_exists():
def test_model_predicts_probability():
def test_prediction_in_zero_one_range():
def test_roc_auc_above_threshold():   # must be > 0.80

# tests/test_api.py
def test_health_endpoint_returns_200():
def test_predict_returns_probability():
def test_predict_validates_input():    # bad input → 422
def test_risk_level_is_correct():
```

---

## Logger Setup

```python
# src/utils/logger.py
import logging
import json
from datetime import datetime
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes structured JSON to logs/app.log
    and human-readable output to console.
    """

# Usage in every module:
from src.utils.logger import get_logger
logger = get_logger(__name__)

logger.info("Dataset loaded", extra={"rows": 1470, "cols": 35})
logger.warning("Class imbalance detected", extra={"ratio": "16:84"})
logger.error("Model file not found", extra={"path": str(MODEL_PATH)})
```

---

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Run specific file
pytest tests/test_api.py -v
```
