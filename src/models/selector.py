from typing import Any, Dict, Tuple

import pandas as pd


def compare_models(metrics: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
    """
    Compare models based on ROC-AUC (primary metric).

    Args:
        metrics: Dict of model_name -> metrics dict.

    Returns:
        Tuple of (best_model_name, best_roc_auc).
    """
    scores = {name: m["roc_auc"] for name, m in metrics.items()}
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    return best_model, best_score


def print_comparison_table(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted comparison table of all models.

    Args:
        metrics: Dict of model_name -> metrics dict.
    """
    data = []
    for name, m in metrics.items():
        data.append(
            {
                "Model": name,
                "Accuracy": f"{m['accuracy']:.3f}",
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1": f"{m['f1']:.3f}",
                "ROC-AUC": f"{m['roc_auc']:.3f}",
                "Avg Precision": f"{m['avg_precision']:.3f}",
            }
        )

    df = pd.DataFrame(data)
    print("\n=== Model Comparison ===")
    print(df.to_string(index=False))
    print("========================\n")

    return df


def save_comparison_table(
    metrics: Dict[str, Dict[str, float]], filepath: str
) -> None:
    """
    Save comparison table to CSV.

    Args:
        metrics: Dict of model_name -> metrics dict.
        filepath: Path to save CSV.
    """
    data = []
    for name, m in metrics.items():
        data.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "ROC-AUC": m["roc_auc"],
                "Avg Precision": m["avg_precision"],
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"✅ Comparison table saved to {filepath}")
