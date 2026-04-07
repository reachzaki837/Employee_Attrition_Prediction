import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import joblib

from config import settings
from src.models.explainer import create_explainer, get_shap_values, plot_global_importance, plot_employee_explanation, get_top_risk_factors


def run_shap_explainability() -> None:
    """
    Execute the SHAP explainability pipeline.

    Steps:
    1. Load best model and test data
    2. Create SHAP explainer
    3. Generate global importance plot
    4. Print example risk factor analysis
    """

    # Load best model
    print("[1/3] Loading best model...")
    model_data = joblib.load(settings.MODEL_PATH)
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    model_name = model_data["model_name"]
    print(f"✅ Loaded {model_name}")

    # Load test data
    print("[2/3] Loading test data...")
    X_test = pd.read_csv(settings.PROCESSED_DATA_PATH.parent / "features_test.csv")
    y_test = X_test[settings.TARGET_COLUMN]
    X_test = X_test.drop(columns=[settings.TARGET_COLUMN])
    print(f"Test set shape: {X_test.shape}")

    # Create SHAP explainer
    print("[3/3] Creating SHAP explainer and generating importance...")
    explainer = create_explainer(model, X_test)
    shap_values = get_shap_values(explainer, X_test)
    
    shap_dir = settings.REPORTS_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    plot_global_importance(shap_values, X_test, str(shap_dir))

    # Print example explanation using top risk predictions
    print("\n" + "=" * 60)
    print("EXAMPLE EMPLOYEE RISK ANALYSIS (Top Risk Case)")
    print("=" * 60)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    top_idx = pd.Series(y_pred_proba).idxmax()
    risk_prob = y_pred_proba[top_idx]
    risk_factors = get_top_risk_factors(explainer, X_test, top_idx, top_n=5)

    print(f"\nEmployee #{top_idx} — Attrition Risk: {risk_prob*100:.0f}% 🔴 HIGH RISK\n")
    print("Top reasons this employee may leave:")
    for i, factor in enumerate(risk_factors, 1):
        impact_pct = abs(factor["impact"]) * 100
        direction = "→" if factor["direction"] == "increases_risk" else "←"
        print(f"{i}. {factor['feature']:20s} = {factor['value']:8.2f} {direction} {impact_pct:+.0f}% risk")

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY PIPELINE COMPLETE")
    print("=" * 60)
    print(f"✅ Global importance plot: {shap_dir}/01_global_importance.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_shap_explainability()
