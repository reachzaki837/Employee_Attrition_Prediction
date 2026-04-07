from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import settings


def select_features(df: pd.DataFrame, target: str, top_n: int = 25) -> List[str]:
    """
    Select top N features using Random Forest feature importance.

    Args:
        df: Full feature matrix including target.
        target: Target column name.
        top_n: Number of top features to keep.

    Returns:
        List of selected feature column names (excluding target).
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Train a quick Random Forest to get feature importances
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=settings.RANDOM_STATE,
        n_jobs=-1,
        max_depth=10,
    )
    rf.fit(X, y)

    # Get importances and sort
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    # Select top N
    selected = importances.head(top_n).index.tolist()
    return selected
