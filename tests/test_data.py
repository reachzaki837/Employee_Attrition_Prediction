"""Tests for data loading and cleaning pipeline."""

import pytest
import pandas as pd
from pathlib import Path

from config.settings import (
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    COLUMNS_TO_DROP,
)
from src.data.loader import load_raw_data, get_data_summary
from src.data.cleaner import clean_pipeline
from src.data.validator import validate_clean_data


class TestDataLoading:
    """Tests for data loading."""

    def test_dataset_available_locally_or_downloadable(self):
        """Test that dataset can be loaded."""
        df = load_raw_data()
        assert df is not None
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_raw_dataset_has_expected_shape(self):
        """Test raw dataset has expected dimensions."""
        df = load_raw_data()
        assert df.shape[0] == 1470
        assert df.shape[1] == 35

    def test_raw_dataset_has_target_column(self):
        """Test that target column exists in raw data."""
        df = load_raw_data()
        assert TARGET_COLUMN in df.columns

    def test_get_data_summary_has_expected_keys(self):
        """Test that data summary contains expected fields."""
        df = load_raw_data()
        summary = get_data_summary(df)
        assert "shape" in summary or "rows" in summary


class TestDataCleaning:
    """Tests for data cleaning pipeline."""

    def test_clean_pipeline_creates_clean_csv(self):
        """Test that cleaning pipeline creates output file."""
        df = load_raw_data()
        clean_pipeline(df)
        assert PROCESSED_DATA_PATH.exists()

    def test_cleaned_data_has_expected_dimensions(self):
        """Test cleaned data has expected shape."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        assert df.shape[0] == 1470  # Same number of rows
        assert df.shape[1] == 45  # 35 - 4 dropped + 1-hot encoded


class TestDataValidation:
    """Tests for data validation."""

    def test_clean_data_has_no_nulls(self):
        """Test that cleaned data has no null values."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        assert df.isnull().sum().sum() == 0

    def test_target_column_is_binary(self):
        """Test that target column contains only 0 and 1."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        assert set(df[TARGET_COLUMN].unique()) == {0, 1}

    def test_dropped_columns_not_in_clean_data(self):
        """Test that dropped columns are not in clean data."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        for col in COLUMNS_TO_DROP:
            assert col not in df.columns

    def test_clean_data_no_object_dtypes(self):
        """Test that all columns are numeric (no object dtypes)."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        assert df.select_dtypes(include=['object']).shape[1] == 0

    def test_valid_clean_data_passes_validator(self):
        """Test that cleaned data passes validation."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        # Should not raise exception
        try:
            validate_clean_data(df)
        except ValueError as e:
            pytest.fail(f"Validation failed: {str(e)}")

    def test_target_column_is_numeric(self):
        """Test that target column is numeric."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        assert pd.api.types.is_numeric_dtype(df[TARGET_COLUMN])

    def test_target_column_class_distribution(self):
        """Test that target column has reasonable class distribution."""
        df = pd.read_csv(PROCESSED_DATA_PATH)
        ratio = df[TARGET_COLUMN].mean()  # Mean should be ~0.161 for attrition
        assert 0.1 < ratio < 0.2  # Allow some margin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
