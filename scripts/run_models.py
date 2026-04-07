import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import joblib

from config import settings
from src.models.trainer import train_all_models
from src.models.evaluator import (
    evaluate_model,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall,
)
from src.models.selector import compare_models, print_comparison_table, save_comparison_table


def run_model_training() -> None:
    """
    Execute the complete model training and evaluation pipeline.

    Steps:
    1. Load feature train/test data
    2. Train all 3 models
    3. Evaluate each on test set
    4. Compare and select best
    5. Generate comparison table and charts
    6. Save best model
    """

    # Load data
    print("[1/5] Loading feature data...")
    X_train = pd.read_csv(settings.PROCESSED_DATA_PATH.parent / "features_train.csv")
    X_test = pd.read_csv(settings.PROCESSED_DATA_PATH.parent / "features_test.csv")

    # Separate target
    y_train = X_train[settings.TARGET_COLUMN]
    y_test = X_test[settings.TARGET_COLUMN]
    X_train = X_train.drop(columns=[settings.TARGET_COLUMN])
    X_test = X_test.drop(columns=[settings.TARGET_COLUMN])

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    feature_names = X_train.columns.tolist()

    # Train models
    print("\n[2/5] Training models...")
    models = train_all_models(X_train, y_train)
    print(f"✅ Trained {len(models)} models")

    # Evaluate models
    print("\n[3/5] Evaluating models...")
    metrics = {}
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        metrics[name] = evaluate_model(model, X_test, y_test)

    # Compare and select best
    print("\n[4/5] Comparing models...")
    best_model_name, best_auc = compare_models(metrics)
    best_model = models[best_model_name]
    print(f"✅ Best model: {best_model_name} (ROC-AUC: {best_auc:.4f})")

    # Print comparison table
    df_comparison = print_comparison_table(metrics)

    # Generate charts
    print("\n[5/5] Generating evaluation charts...")
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.REPORTS_DIR / "evaluation").mkdir(parents=True, exist_ok=True)

    plot_roc_curves(models, X_test, y_test)
    print("  ✅ ROC curves saved")

    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
    print("  ✅ Confusion matrix saved")

    # Feature importance (for RF and XGB only)
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, feature_names, best_model_name)
        print("  ✅ Feature importance saved")

    plot_precision_recall(best_model, X_test, y_test, best_model_name)
    print("  ✅ Precision-recall curve saved")

    # Save comparison table
    comparison_path = settings.REPORTS_DIR / "model_comparison.csv"
    save_comparison_table(metrics, str(comparison_path))

    # Save best model
    settings.PROCESSED_DATA_PATH.parent.parent.mkdir(parents=True, exist_ok=True)
    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_names": feature_names,
            "model_name": best_model_name,
            "roc_auc": best_auc,
        },
        settings.MODEL_PATH,
    )
    print(f"\n✅ Best model saved to {settings.MODEL_PATH}")

    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best Model: {best_model_name}")
    print(f"ROC-AUC: {best_auc:.4f}")
    print(f"Test Set Accuracy: {metrics[best_model_name]['accuracy']:.4f}")
    print(f"Test Set Precision: {metrics[best_model_name]['precision']:.4f}")
    print(f"Test Set Recall: {metrics[best_model_name]['recall']:.4f}")
    print(f"Test Set F1: {metrics[best_model_name]['f1']:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_model_training()
