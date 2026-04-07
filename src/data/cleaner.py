from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import settings


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns in COLUMNS_TO_DROP from settings."""
    return df.drop(columns=settings.COLUMNS_TO_DROP, errors="ignore")


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Attrition Yes/No to 1/0."""
    df = df.copy()
    df[settings.TARGET_COLUMN] = df[settings.TARGET_COLUMN].map(
        {settings.POSITIVE_CLASS: 1, settings.NEGATIVE_CLASS: 0}
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all CATEGORICAL_COLUMNS."""
    return pd.get_dummies(df, columns=settings.CATEGORICAL_COLUMNS, drop_first=True)


def scale_numerics(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standard scale all NUMERIC_COLUMNS. Return df + fitted scaler."""
    scaler = StandardScaler()
    df = df.copy()
    df[settings.NUMERIC_COLUMNS] = scaler.fit_transform(df[settings.NUMERIC_COLUMNS])
    return df, scaler


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order. Save to PROCESSED_DATA_PATH."""
    df = drop_useless_columns(df)
    df = encode_target(df)
    df = encode_categoricals(df)
    df, scaler = scale_numerics(df)

    # ensure processed directory exists and save
    settings.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    return df
