"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_endpoint_returns_200(self):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_required_fields(self):
        """Test that health response has required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "version" in data

    def test_health_status_is_healthy(self):
        """Test that health status indicates healthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""

    def test_stats_endpoint_returns_200(self):
        """Test that stats endpoint returns 200 OK."""
        response = client.get("/api/stats")
        assert response.status_code == 200

    def test_stats_has_expected_fields(self):
        """Test that stats response has expected fields."""
        response = client.get("/api/stats")
        data = response.json()

        assert "dataset_size" in data
        assert "attrition_rate" in data
        assert "model_auc" in data

    def test_stats_values_are_reasonable(self):
        """Test that stats values are in reasonable ranges."""
        response = client.get("/api/stats")
        data = response.json()

        assert data["dataset_size"] == 1470
        assert 0 < data["attrition_rate"] < 1
        assert 0.75 < data["model_auc"] < 1


class TestReportImagesEndpoint:
    """Tests for /api/report-images endpoint."""

    def test_report_images_endpoint_returns_200(self):
        """Test that report images endpoint returns 200 OK."""
        response = client.get("/api/report-images")
        assert response.status_code == 200

    def test_report_images_has_expected_structure(self):
        """Test that response has correct structure."""
        response = client.get("/api/report-images")
        data = response.json()

        assert "total" in data
        assert "images" in data
        assert isinstance(data["images"], list)

    def test_report_images_count_is_eight(self):
        """Test that 8 EDA charts are available."""
        response = client.get("/api/report-images")
        data = response.json()
        assert data["total"] == 8


class TestPredictEndpoint:
    """Tests for /api/predict endpoint."""

    @pytest.fixture
    def valid_employee(self):
        """Valid employee request body with all required fields."""
        return {
            "Age": 35,
            "DailyRate": 1102,
            "Department": "Sales",
            "DistanceFromHome": 10,
            "EnvironmentSatisfaction": 2,
            "HourlyRate": 65,
            "JobInvolvement": 3,
            "JobRole": "Sales Executive",
            "JobSatisfaction": 2,
            "MaritalStatus": "Single",
            "MonthlyIncome": 5000,
            "MonthlyRate": 19479,
            "NumCompaniesWorked": 1,
            "OverTime": "Yes",
            "PercentSalaryHike": 15,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 2,
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 1,
            "TotalWorkingYears": 8,
            "BusinessTravel": "Travel_Frequently",
        }

    def test_predict_returns_200_with_valid_input(self, valid_employee):
        """Test that prediction endpoint returns 200 with valid input."""
        response = client.post("/api/predict", json=valid_employee)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, valid_employee):
        """Test that prediction response has required fields."""
        response = client.post("/api/predict", json=valid_employee)
        data = response.json()

        assert "attrition_probability" in data
        assert "risk_level" in data
        assert "risk_color" in data
        assert "top_risk_factors" in data
        assert "recommendation" in data

    def test_attrition_probability_in_range(self, valid_employee):
        """Test that attrition probability is in [0, 1]."""
        response = client.post("/api/predict", json=valid_employee)
        data = response.json()

        prob = data["attrition_probability"]
        assert 0 <= prob <= 1

    def test_risk_level_is_valid(self, valid_employee):
        """Test that risk level is one of LOW/MEDIUM/HIGH."""
        response = client.post("/api/predict", json=valid_employee)
        data = response.json()

        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_risk_color_is_valid_hex(self, valid_employee):
        """Test that risk color is valid hex code."""
        response = client.post("/api/predict", json=valid_employee)
        data = response.json()

        color = data["risk_color"]
        assert color.startswith("#")
        assert len(color) == 7
        try:
            int(color[1:], 16)
        except ValueError:
            pytest.fail(f"Invalid hex color: {color}")

    def test_top_risk_factors_is_list(self, valid_employee):
        """Test that top risk factors is a list."""
        response = client.post("/api/predict", json=valid_employee)
        data = response.json()

        assert isinstance(data["top_risk_factors"], list)
        assert len(data["top_risk_factors"]) >= 0

    def test_risk_factors_have_required_fields(self, valid_employee):
        """Test that each risk factor has required fields."""
        response = client.post("/api/predict", json=valid_employee)
        data = response.json()

        for factor in data["top_risk_factors"]:
            assert "feature" in factor
            assert "value" in factor
            assert "impact" in factor
            assert "direction" in factor

    def test_prediction_returns_422_on_invalid_input(self):
        """Test that invalid input returns 422 Unprocessable Entity."""
        invalid_input = {
            "Age": "not_a_number",
            "Department": "Sales",
            "JobRole": "Sales Executive",
        }
        response = client.post("/api/predict", json=invalid_input)
        assert response.status_code == 422

    def test_high_risk_recommendation_has_action_words(self, valid_employee):
        """Test that high-risk recommendation includes action items."""
        # Create high-risk employee
        high_risk_employee = valid_employee.copy()
        high_risk_employee["JobSatisfaction"] = 1
        high_risk_employee["WorkLifeBalance"] = 1
        high_risk_employee["MonthlyIncome"] = 2000
        high_risk_employee["YearsAtCompany"] = 1

        response = client.post("/api/predict", json=high_risk_employee)
        data = response.json()

        if data["risk_level"] == "HIGH":
            assert "HR intervention" in data["recommendation"] or "action" in data["recommendation"].lower()


class TestDashboardPages:
    """Tests for HTML dashboard and report pages."""

    def test_dashboard_returns_200(self):
        """Test that dashboard page returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_report_page_returns_200(self):
        """Test that report page returns 200."""
        response = client.get("/report")
        assert response.status_code == 200

    def test_auto_docs_available(self):
        """Test that auto-generated API docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
