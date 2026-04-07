from typing import Any

import pandas as pd

from config import settings


def validate_clean_data(df: pd.DataFrame) -> bool:
    """Validate cleaned dataframe according to project rules.

    Checks:
    - No null values remain
    - Target column is 0/1 integers
    - All numeric columns are float/int
    - Shape is (1470, expected_cols)

    Raises:
        ValueError: if any check fails, with details.

    Returns:
        True if all validations pass.
    """
    errors = []

    # nulls
    total_nulls = int(df.isnull().sum().sum())
    if total_nulls != 0:
        errors.append(f"Data contains {total_nulls} null values.")

    # target dtype and values
    if df[settings.TARGET_COLUMN].dtype not in (int, float):
        errors.append("Target column dtype is not numeric.")
    unique_vals = set(df[settings.TARGET_COLUMN].unique())
    if unique_vals != {0, 1} and unique_vals != {0} and unique_vals != {1}:
        errors.append(f"Target column unique values unexpected: {unique_vals}")
    if df[settings.TARGET_COLUMN].nunique() != 2:
        errors.append("Target column does not have two classes.")

    # numeric columns type
    for col in settings.NUMERIC_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col].dtype):
                errors.append(f"Column {col} is not numeric.")

    # shape
    expected_rows = 1470
    if df.shape[0] != expected_rows:
        errors.append(f"Row count expected {expected_rows}, got {df.shape[0]}")

    # categorical strings gone (we check by dtype object)
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if object_cols:
        errors.append(f"Remaining object dtype columns: {object_cols}")

    if errors:
        raise ValueError("; ".join(errors))
    return True
