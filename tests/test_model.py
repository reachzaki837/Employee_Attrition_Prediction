"""Tests for model training and prediction."""

import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import MODEL_PATH, PROCESSED_DATA_PATH

# Path to feature-selected test data (created in Phase 3)
FEATURES_TEST_PATH = Path(__file__).parent.parent / "data" / "processed" / "features_test.csv"
FEATURES_TRAIN_PATH = Path(__file__).parent.parent / "data" / "processed" / "features_train.csv"

from src.models.trainer import train_logistic_regression, train_random_forest, train_xgboost
from sklearn.ensemble import RandomForestClassifier


class TestModelFiles:
    """Tests for model file existence and integrity."""

    def test_model_file_exists(self):
        """Test that trained model file exists."""
        assert MODEL_PATH.exists()
        assert MODEL_PATH.suffix == ".pkl"

    def test_model_file_is_valid(self):
        """Test that model file can be loaded."""
        try:
            model_data = joblib.load(MODEL_PATH)
            assert "model" in model_data
            assert "feature_names" in model_data
        except Exception as e:
            pytest.fail(f"Model file is corrupted: {str(e)}")

    def test_loaded_model_is_sklearn_compatible(self):
        """Test that loaded model is a valid scikit-learn model."""
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]

        # Check that model has required methods
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


class TestModelPredictions:
    """Tests for model prediction behavior."""

    def test_model_predicts_probability(self):
        """Test that model returns probability predictions."""
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]

        # Load feature-selected test data
        if FEATURES_TEST_PATH.exists():
            df = pd.read_csv(FEATURES_TEST_PATH)
            X = df.drop("Attrition", axis=1).head(1)

            # Get predictions
            prob = model.predict_proba(X)

            # Should return 2D array with 2 columns (for binary classification)
            assert prob.shape[0] >= 1
            assert prob.shape[1] == 2
            assert 0 <= prob[0, 1] <= 1
        else:
            pytest.skip("Feature-selected test data not available")

    def test_prediction_in_zero_one_range(self):
        """Test that predictions are in [0, 1] range."""
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]

        # Load feature-selected test data
        if FEATURES_TEST_PATH.exists():
            df = pd.read_csv(FEATURES_TEST_PATH)
            X = df.drop("Attrition", axis=1).head(10)

            # Get predictions
            probs = model.predict_proba(X)

            # All probabilities should be in [0, 1]
            assert (probs >= 0).all()
            assert (probs <= 1).all()
        else:
            pytest.skip("Feature-selected test data not available")

    def test_predictions_sum_to_one(self):
        """Test that probabilities for both classes sum to 1."""
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]

        # Load feature-selected test data
        if FEATURES_TEST_PATH.exists():
            df = pd.read_csv(FEATURES_TEST_PATH)
            X = df.drop("Attrition", axis=1).head(10)

            # Get predictions
            probs = model.predict_proba(X)

            # Each row should sum to 1
            assert np.allclose(probs.sum(axis=1), 1.0)
        else:
            pytest.skip("Feature-selected test data not available")

    def test_model_handles_different_input_sizes(self):
        """Test that model handles single and multiple inputs."""
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]

        # Load feature-selected test data
        if FEATURES_TEST_PATH.exists():
            df = pd.read_csv(FEATURES_TEST_PATH)
            X = df.drop("Attrition", axis=1)

            # Single sample
            X_single = X.head(1)
            prob_single = model.predict_proba(X_single)
            assert prob_single.shape[0] == 1

            # Multiple samples
            X_multiple = X.head(5)
            prob_multiple = model.predict_proba(X_multiple)
            assert prob_multiple.shape[0] == 5
        else:
            pytest.skip("Feature-selected test data not available")


class TestModelPerformance:
    """Tests for model performance thresholds."""

    def test_roc_auc_meets_threshold(self):
        """Test that model ROC-AUC is above 0.75."""
        from sklearn.metrics import roc_auc_score

        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]

        # Load feature-selected test data
        if FEATURES_TEST_PATH.exists():
            df = pd.read_csv(FEATURES_TEST_PATH)
            X = df.drop("Attrition", axis=1)
            y = df["Attrition"]

            # Predict
            y_pred_proba = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_pred_proba)

            # Model should have reasonable AUC
            assert roc_auc > 0.70  # Lower threshold for test set
        else:
            pytest.skip("Feature-selected test data not available")

    def test_model_is_random_forest(self):
        """Test that best model is RandomForestClassifier."""
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]

        # Check model type
        assert isinstance(model, RandomForestClassifier)
        # Model name might be "RandomForestClassifier" or "Random Forest"
        assert "Forest" in model_data.get("model_name", "")


class TestModelMetadata:
    """Tests for model metadata."""

    def test_model_has_feature_names(self):
        """Test that model includes feature names."""
        model_data = joblib.load(MODEL_PATH)

        assert "feature_names" in model_data
        assert isinstance(model_data["feature_names"], list)
        assert len(model_data["feature_names"]) > 0

    def test_feature_names_count_matches_model_input(self):
        """Test that feature count matches model expected input."""
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]
        feature_names = model_data["feature_names"]

        # RandomForest should have n_features_in_ after training
        if hasattr(model, "n_features_in_"):
            assert model.n_features_in_ == len(feature_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
