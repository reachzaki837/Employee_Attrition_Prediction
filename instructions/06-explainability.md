# 06 — SHAP Explainability

> ⚡ Copilot: SHAP shows WHY the model makes predictions.
> This is the most impressive part of the demo. Do it carefully.

---

## Phase Task List

- [ ] Create `src/models/explainer.py` with SHAP functions
- [ ] Generate global SHAP summary plot
- [ ] Generate per-employee waterfall chart function
- [ ] Save charts to `reports/shap/`

---

## SHAP Functions to Build

```python
import shap

def create_explainer(model, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """Create SHAP TreeExplainer for tree-based models (RF, XGBoost)."""
    return shap.TreeExplainer(model)

def get_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for a feature matrix."""
    return explainer.shap_values(X)

def plot_global_importance(shap_values, X: pd.DataFrame) -> None:
    """
    Global SHAP summary plot — shows which features matter most overall.
    Saved to: reports/shap/01_global_importance.png
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "01_global_importance.png", dpi=150)
    plt.close()

def plot_employee_explanation(explainer, employee_row: pd.DataFrame, employee_id: str) -> None:
    """
    Per-employee waterfall chart.
    Shows exactly why THIS employee is predicted at risk.
    Saved to: reports/shap/employee_{employee_id}.png
    """

def get_top_risk_factors(shap_values, X: pd.DataFrame, employee_idx: int, top_n: int = 5) -> list[dict]:
    """
    Return top N risk factors for one employee.

    Returns list of dicts:
    [{"feature": "OverTime", "value": 1, "impact": 0.34, "direction": "increases_risk"}, ...]
    """
```

---

## What SHAP Output Should Look Like

For the demo, when predicting an employee:

```
Employee #42 — Attrition Risk: 78% 🔴 HIGH RISK

Top reasons this employee may leave:
1. OverTime = Yes          → +34% risk
2. JobSatisfaction = 1     → +22% risk
3. YearsAtCompany = 1      → +15% risk
4. MonthlyIncome = $2,100  → +12% risk
5. DistanceFromHome = 29   → +8% risk

Recommendation: Address overtime and job satisfaction urgently.
```
