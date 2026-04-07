from pathlib import Path
from typing import Dict

import pandas as pd

from config import settings


def download_dataset() -> pd.DataFrame:
    """Download dataset using kagglehub, cache to RAW_DATA_PATH.

    Returns:
        DataFrame loaded from the downloaded CSV.
    """
    import kagglehub

    path = kagglehub.dataset_download(settings.KAGGLE_DATASET)
    # path contains the folder — find the CSV inside it
    csv_file = next(Path(path).glob("*.csv"))
    df = pd.read_csv(csv_file)

    # ensure raw directory exists and save a copy
    settings.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.RAW_DATA_PATH, index=False)
    return df


def load_raw_data() -> pd.DataFrame:
    """Load from cache if exists, else download.

    Returns:
        Raw dataset as pandas DataFrame.
    """
    if settings.RAW_DATA_PATH.exists():
        return pd.read_csv(settings.RAW_DATA_PATH)
    return download_dataset()


def get_data_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Return dict with shape, nulls, dtypes, target distribution.

    Args:
        df: Input dataframe.

    Returns:
        Dictionary containing metadata about the dataset.
    """
    summary = {
        "shape": df.shape,
        "nulls": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.apply(lambda dt: str(dt)).to_dict(),
        settings.TARGET_COLUMN: df[settings.TARGET_COLUMN].value_counts().to_dict(),
    }
    return summary
