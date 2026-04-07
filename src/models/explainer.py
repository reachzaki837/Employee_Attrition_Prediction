from typing import Any, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from config import settings

STYLE = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor": "#1A1D27",
    "axes.edgecolor": "#2A2E42",
    "axes.labelcolor": "#94A3B8",
    "text.color": "#F1F5F9",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#2A2E42",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
}
plt.rcParams.update(STYLE)


def create_explainer(model: Any, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """
    Create SHAP TreeExplainer for tree-based models.

    Args:
        model: Fitted tree-based model (Random Forest, XGBoost).
        X_train: Training features (used as background for SHAP).

    Returns:
        TreeExplainer object.
    """
    return shap.TreeExplainer(model)


def get_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for a feature matrix.

    Args:
        explainer: SHAP explainer object.
        X: Feature matrix.

    Returns:
        SHAP values array.
    """
    return explainer.shap_values(X)


def plot_global_importance(
    shap_values: np.ndarray, X: pd.DataFrame, output_dir: str
) -> None:
    """
    Generate global SHAP summary plot (bar chart).

    Shows which features matter most overall for model predictions.

    Args:
        shap_values: SHAP values (array or list of arrays for multi-class).
        X: Feature matrix.
        output_dir: Directory to save chart.
    """
    # For binary classification, take the second class (risk class)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    plt.tight_layout()

    output_path = Path(output_dir) / "01_global_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Global SHAP importance saved to {output_path}")


def plot_employee_explanation(
    explainer: shap.TreeExplainer,
    employee_row: pd.DataFrame,
    feature_names: List[str],
    employee_id: str,
    output_dir: str,
) -> None:
    """
    Generate per-employee SHAP waterfall chart.

    Shows exactly why THIS employee is predicted at risk or not.

    Args:
        explainer: SHAP explainer object.
        employee_row: Single row DataFrame for one employee.
        feature_names: List of feature column names.
        employee_id: ID or index of employee.
        output_dir: Directory to save chart.
    """
    shap_values = explainer.shap_values(employee_row)

    # For binary classification, take class 1 (attrition)
    if isinstance(shap_values, list):
        shap_val = shap_values[1][0]
        base_val = explainer.expected_value[1]
    else:
        shap_val = shap_values[0]
        base_val = explainer.expected_value

    # Ensure base_val is scalar
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[0])
    else:
        base_val = float(base_val)

    fig = plt.figure(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        base_val,
        shap_val,
        employee_row.values[0],
        feature_names,
    )
    plt.tight_layout()

    output_path = Path(output_dir) / f"employee_{employee_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Employee {employee_id} waterfall saved to {output_path}")


def get_top_risk_factors(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    employee_idx: int,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get top risk factors for one employee.

    Args:
        explainer: SHAP explainer object.
        X: Full feature matrix.
        employee_idx: Row index of employee.
        top_n: Number of top factors to return.

    Returns:
        List of dicts with feature, value, impact, direction.
    """
    shap_vals = explainer.shap_values(X)

    # For binary, use class 1 - flatten if needed
    if isinstance(shap_vals, list):
        shap_val = shap_vals[1][employee_idx]
    else:
        shap_val = shap_vals[employee_idx]

    # Ensure 1D
    if hasattr(shap_val, 'shape') and len(shap_val.shape) > 1:
        shap_val = shap_val.flatten()

    # Sort by absolute impact
    impacts = np.abs(shap_val)
    top_indices = np.argsort(impacts)[-top_n:][::-1]

    risk_factors = []
    for idx in top_indices:
        try:
            impact_val = float(np.asarray(shap_val[idx]).flat[0])
            feature_val = float(X.iloc[employee_idx, idx])
        except (ValueError, TypeError):
            impact_val = float(shap_val[idx]) if hasattr(shap_val[idx], '__float__') else 0.0
            feature_val = float(X.iloc[employee_idx, idx])

        risk_factors.append(
            {
                "feature": X.columns[idx],
                "value": feature_val,
                "impact": impact_val,
                "direction": "increases_risk" if impact_val > 0 else "decreases_risk",
            }
        )

    return risk_factors

