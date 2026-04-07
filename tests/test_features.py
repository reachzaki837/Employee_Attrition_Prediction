"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import PROCESSED_DATA_PATH
from src.features.engineer import create_features
from src.data.loader import load_raw_data


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_engineered_features_are_created(self):
        """Test that feature engineering creates new columns."""
        df_raw = load_raw_data()
        df_engineered = create_features(df_raw)

        # Should have 7 new features
        new_cols = set(df_engineered.columns) - set(df_raw.columns)
        assert len(new_cols) == 7

    def test_income_per_year_exp_is_positive(self):
        """Test that IncomePerYearExp is always positive."""
        df_raw = load_raw_data()
        df_engineered = create_features(df_raw)

        assert "IncomePerYearExp" in df_engineered.columns
        assert (df_engineered["IncomePerYearExp"] > 0).all()

    def test_tenure_ratio_in_valid_range(self):
        """Test that TenureRatio is in [0, 1]."""
        df_raw = load_raw_data()
        df_engineered = create_features(df_raw)

        assert "TenureRatio" in df_engineered.columns
        assert (df_engineered["TenureRatio"] >= 0).all()
        assert (df_engineered["TenureRatio"] <= 1).all()

    def test_satisfaction_score_in_valid_range(self):
        """Test that SatisfactionScore is in [1, 4]."""
        df_raw = load_raw_data()
        df_engineered = create_features(df_raw)

        assert "SatisfactionScore" in df_engineered.columns
        assert (df_engineered["SatisfactionScore"] >= 1).all()
        assert (df_engineered["SatisfactionScore"] <= 4).all()

    def test_is_low_income_is_binary(self):
        """Test that IsLowIncome is binary (0 or 1)."""
        df_raw = load_raw_data()
        df_engineered = create_features(df_raw)

        assert "IsLowIncome" in df_engineered.columns
        assert set(df_engineered["IsLowIncome"].unique()) == {0, 1}

    def test_high_overtime_risk_is_binary(self):
        """Test that HighOvertimeRisk is binary (0 or 1)."""
        df_raw = load_raw_data()
        df_engineered = create_features(df_raw)

        assert "HighOvertimeRisk" in df_engineered.columns
        assert set(df_engineered["HighOvertimeRisk"].unique()).issubset({0, 1})

    def test_no_nulls_in_engineered_features(self):
        """Test that engineered features have no null values."""
        df_raw = load_raw_data()
        df_engineered = create_features(df_raw)

        # Check for nulls in new features
        new_cols = set(df_engineered.columns) - set(df_raw.columns)
        for col in new_cols:
            assert df_engineered[col].isnull().sum() == 0


class TestSMOTEBalancing:
    """Tests for class imbalance handling."""

    def test_smote_balances_classes(self):
        """Test that SMOTE creates balanced training set."""
        from src.features.selector import select_features
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE

        # Load cleaned data
        df = pd.read_csv(PROCESSED_DATA_PATH)
        X = df.drop("Attrition", axis=1)
        y = df["Attrition"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Check balance
        assert y_train_balanced.value_counts()[0] == y_train_balanced.value_counts()[1]
        assert len(X_train_balanced) > len(X_train)  # More samples due to oversampling


class TestDataLeakage:
    """Tests for data leakage prevention."""

    def test_no_data_leakage_in_split(self):
        """Test that train/test split is proper (no leakage indicators)."""
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(PROCESSED_DATA_PATH)
        X = df.drop("Attrition", axis=1)
        y = df["Attrition"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Check no overlap
        assert not set(X_train.index).intersection(set(X_test.index))

    def test_stratified_split_maintains_class_ratio(self):
        """Test that stratified split maintains class distribution."""
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(PROCESSED_DATA_PATH)
        X = df.drop("Attrition", axis=1)
        y = df["Attrition"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        overall_ratio = y.mean()

        # Ratios should be close (within 5%)
        assert abs(train_ratio - overall_ratio) < 0.05
        assert abs(test_ratio - overall_ratio) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
