from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

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

PALETTE = ["#6366F1", "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"]


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Compute full evaluation metrics for a model.

    Args:
        model: Fitted sklearn or xgboost model.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc, avg_precision.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "avg_precision": average_precision_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def plot_roc_curves(
    models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """
    Plot ROC curves for all models on one chart.

    Args:
        models: Dict of model name -> fitted model.
        X_test: Test features.
        y_test: Test target.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, model) in enumerate(models.items()):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=PALETTE[idx])

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    settings.EDA_DIR.parent.mkdir(parents=True, exist_ok=True)
    eval_dir = settings.REPORTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_dir / "01_roc_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> None:
    """
    Plot confusion matrix for a model.

    Args:
        model: Fitted model.
        X_test: Test features.
        y_test: Test target.
        model_name: Name of model for title.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Left"],
        yticklabels=["Stayed", "Left"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    eval_dir = settings.REPORTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_dir / "02_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model: Any, feature_names: list, model_name: str
) -> None:
    """
    Plot top 15 feature importances.

    Args:
        model: Fitted model with feature_importances_ attribute.
        feature_names: List of feature column names.
        model_name: Name of model for title.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=True
    )
    top_15 = importances.tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_15.plot(kind="barh", ax=ax, color=PALETTE[0])
    ax.set_title(f"Top 15 Feature Importances — {model_name}")
    ax.set_xlabel("Importance")

    eval_dir = settings.REPORTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_dir / "03_feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> None:
    """
    Plot precision-recall curve for a model.

    Args:
        model: Fitted model.
        X_test: Test features.
        y_test: Test target.
        model_name: Name of model for title.
    """
    from sklearn.metrics import precision_recall_curve

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=model_name, color=PALETTE[0])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    eval_dir = settings.REPORTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_dir / "04_precision_recall.png", bbox_inches="tight")
    plt.close(fig)
