# GitHub Copilot — Auto Instructions for PulseML

> ⚠️ This file is automatically read by GitHub Copilot on every session.
> You MUST follow all instructions here before writing any code.

## 🔴 FIRST ACTION — Always Do This

Before writing ANY code, read:
```
instructions/00-orchestration.md   ← START HERE ALWAYS
```

## Core Rules

1. Never skip the orchestration file — it is your source of truth
2. Always check the current phase in `instructions/12-timeslot-schedule.md`
3. Log every action in `logs/copilot-session-log.md`
4. Ask for guidance using `instructions/13-copilot-guidance-protocol.md`
5. Never hardcode file paths — use `config/settings.py` for all paths
6. Write docstrings for every function and class

## Project

You are building **PulseML** — an Employee Attrition Prediction system.
Full details: `instructions/01-project-overview.md`

## Technology Stack

- **Language**: Python 3.11+
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **ML**: Scikit-learn, XGBoost, SHAP
- **API**: FastAPI + Uvicorn + Jinja2
- **Testing**: Pytest + pytest-cov
- **Env management**: python-dotenv

## Build Order (Enforced)

```
Phase 1 → Project Setup & Data        (instructions/02-data-setup.md)
Phase 2 → EDA & Visualization         (instructions/03-eda.md)
Phase 3 → Feature Engineering         (instructions/04-feature-engineering.md)
Phase 4 → Model Training & Evaluation (instructions/05-model-training.md)
Phase 5 → SHAP Explainability         (instructions/06-explainability.md)
Phase 6 → FastAPI Dashboard           (instructions/07-api-dashboard.md)
Phase 7 → Testing & Logging           (instructions/08-testing-logging.md)
Phase 8 → Final Report & Demo         (instructions/09-final-report.md)
```

Do NOT jump phases.
