from typing import Dict, Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from config import settings


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a Logistic Regression baseline model.

    Args:
        X_train: Training features.
        y_train: Training target (0/1).

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=settings.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a Random Forest model.

    Args:
        X_train: Training features.
        y_train: Training target (0/1).

    Returns:
        Fitted RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=settings.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train an XGBoost model.

    Args:
        X_train: Training features.
        y_train: Training target (0/1).

    Returns:
        Fitted XGBClassifier model.
    """
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5,
        eval_metric="auc",
        random_state=settings.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, Any]:
    """
    Train all 3 models and return as dict.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Dictionary with keys "Logistic Regression", "Random Forest", "XGBoost".
    """
    print("Training Logistic Regression...")
    lr = train_logistic_regression(X_train, y_train)

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    return {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb_model}
