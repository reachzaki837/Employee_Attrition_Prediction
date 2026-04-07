# 00 — ORCHESTRATION FILE (Main Control Center)

> 🎯 GitHub Copilot MUST read this at the start of every session.
> This file directs all work. Return here after every completed phase.

---

## SESSION LOOP (Follow Every Time)

```
START SESSION
    │
    ▼
Read this file (00-orchestration.md)
    │
    ▼
Read logs/copilot-session-log.md → understand what was done last session
    │
    ▼
Check instructions/12-timeslot-schedule.md → find active phase
    │
    ▼
Is current phase complete?
   YES → Move to next phase, update status table below
   NO  → Continue current phase
    │
    ▼
Read instruction file for the current phase
    │
    ▼
Do you have all info needed?
   YES → Build
   NO  → Follow instructions/13-copilot-guidance-protocol.md
    │
    ▼
Build. Test. Commit.
    │
    ▼
Update logs/copilot-session-log.md
    │
    ▼
Return here. Update phase status below.
    │
    ▼
END SESSION
```

---

## 🗂️ PHASE STATUS TRACKER

| Phase | File | Status | Notes |
|-------|------|--------|-------|
| 0 | Orchestration Setup | ✅ Done | — |
| 1 | Project Setup & Data | ✅ Done | requirements, config, loader, cleaner, validator |
| 2 | EDA & Visualization | ✅ Done | 8 charts saved to reports/eda/, summary printed |
| 3 | Feature Engineering | ✅ Done | 7 new features, SMOTE applied, 25 selected |
| 4 | Model Training | ✅ Done | 3 models trained, RF best (AUC 0.797), saved |
| 5 | SHAP Explainability | ✅ Done | Global importance plot saved, risk analysis working |
| 6 | FastAPI Dashboard | ✅ Done | API & dashboard live on port 8000, /predict endpoint operational |
| 7 | Testing & Logging | ✅ Done | 55/55 tests passing, 39% coverage, JSON logging configured |
| 8 | Final Report & Demo | ⬜ In Progress | — |

---

## GLOBAL CODING STANDARDS

### Python Style
- Follow PEP 8 strictly
- Max line length: 88 chars (Black formatter)
- Every function needs a docstring with Args and Returns
- Type hints on every function signature
- No bare `except:` — always catch specific exceptions

### Example Function Standard
```python
def calculate_attrition_rate(df: pd.DataFrame, group_by: str) -> pd.Series:
    """
    Calculate attrition rate grouped by a given column.

    Args:
        df: HR dataset with 'Attrition' column (Yes/No)
        group_by: Column name to group by (e.g., 'Department')

    Returns:
        Series with attrition rate (0-1) per group, sorted descending
    """
    return (
        df.groupby(group_by)['Attrition']
        .apply(lambda x: (x == 'Yes').mean())
        .sort_values(ascending=False)
    )
```

### Folder Structure
```
pulseml/
  data/
    raw/              ← Original dataset, never modified
    processed/        ← Cleaned, feature-engineered data
    outputs/          ← Model predictions, reports
  notebooks/          ← Jupyter EDA notebooks
  src/
    data/             ← Data loading and cleaning scripts
    features/         ← Feature engineering
    models/           ← Training, evaluation, saving
    visualization/    ← All plot functions
    api/              ← FastAPI app and routes
    utils/            ← Shared utilities, logging
  tests/              ← Pytest test files
  config/
    settings.py       ← All paths and config values
  logs/               ← Application logs
  reports/            ← Generated charts and final report
  requirements.txt
  README.md
```

### Commits
Format: `[PHASE-N] type: short description`
Examples:
- `[PHASE-1] feat: add data loading pipeline`
- `[PHASE-2] feat: add attrition EDA charts`
- `[PHASE-4] fix: fix class imbalance in training`

---

## 🔗 QUICK NAVIGATION

| Need to know about... | Go to... |
|----------------------|----------|
| What to build | `01-project-overview.md` |
| Data loading & cleaning | `02-data-setup.md` |
| Charts & EDA | `03-eda.md` |
| Feature engineering | `04-feature-engineering.md` |
| Model training & metrics | `05-model-training.md` |
| SHAP explainability | `06-explainability.md` |
| FastAPI dashboard | `07-api-dashboard.md` |
| Tests & logging | `08-testing-logging.md` |
| Final report | `09-final-report.md` |
| Schedule | `12-timeslot-schedule.md` |
| Stuck? | `13-copilot-guidance-protocol.md` |
| Reading logs | `14-log-reports.md` |
