# 13 — Copilot Guidance Protocol

---

## When to Ask for Guidance

Ask when:
- A requirement has two valid interpretations with different outcomes
- A data quality issue threatens the whole pipeline
- Model performance is far below the 0.80 ROC-AUC target
- A library version conflict blocks installation

Do NOT ask for guidance on:
- Which chart color to use (follow design tokens)
- Minor code style choices (follow PEP 8)
- How to write a for loop

---

## Guidance Request Format

Add this comment block in the relevant file:

```python
# 🤔 COPILOT NEEDS GUIDANCE
# File: src/models/trainer.py
# Phase: 4 — Model Training
#
# QUESTION:
# XGBoost ROC-AUC is only 0.74 on test set (target: 0.80).
# Option A: Tune hyperparameters (grid search — adds 30 min)
# Option B: Add more engineered features and retrain
# Option C: Lower the target threshold to 0.35 instead of 0.50
#
# CURRENT STATE:
# Logistic Regression: 0.71, Random Forest: 0.76, XGBoost: 0.74
#
# BLOCKER LEVEL: Medium — I can continue building the API while waiting
```

---

## Assumption Format

```python
# ⚠️ COPILOT ASSUMPTION
# Assumed: Use threshold=0.40 for binary classification (not default 0.50)
# Reason: Better recall for minority class (attrition=Yes)
# Revisit: If precision drops below 0.60, revert to 0.50
```
