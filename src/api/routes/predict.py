"""Prediction endpoint for employee attrition."""

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from config.settings import MODEL_PATH
from src.api.schemas import EmployeeInput, PredictionResponse, RiskFactor

router = APIRouter(prefix="/api", tags=["predictions"])

# Load model globally on startup
MODEL = None
FEATURE_NAMES = None
SCALER = None


def load_model():
    """Load the best trained model from disk."""
    global MODEL, FEATURE_NAMES
    
    model_file = MODEL_PATH
    if not model_file.exists():
        print(f"⚠️ Model not found at {model_file}")
        return False
    
    try:
        model_data = joblib.load(model_file)
        
        # Handle both dict format and direct model format
        if isinstance(model_data, dict):
            MODEL = model_data.get("model")
            FEATURE_NAMES = model_data.get("feature_names", [])
        else:
            MODEL = model_data
            FEATURE_NAMES = list(MODEL.feature_names_in_) if hasattr(MODEL, "feature_names_in_") else []
        
        if MODEL is None:
            print("⚠️ Could not load model from disk")
            return False
        
        print(f"✅ Model loaded: {MODEL.__class__.__name__} with {len(FEATURE_NAMES)} features")
        return True
    except Exception as e:
        print(f"⚠️ Error loading model: {str(e)}")
        return False


def get_risk_level(probability: float) -> tuple[str, str]:
    """
    Determine risk level and color based on attrition probability.
    
    Args:
        probability: Attrition probability (0-1)
    
    Returns:
        Tuple of (risk_level, risk_color_hex)
    """
    if probability >= 0.65:
        return "HIGH", "#EF4444"
    elif probability >= 0.35:
        return "MEDIUM", "#F59E0B"
    else:
        return "LOW", "#10B981"


def prepare_employee_features(employee: EmployeeInput) -> pd.DataFrame:
    """
    Convert EmployeeInput to feature matrix matching model training.
    Applies feature engineering and encoding transformations.
    
    Args:
        employee: Validated employee input
    
    Returns:
        DataFrame with single row, engineered features in correct order
    """
    # Create raw data from input - include all fields needed for feature engineering
    raw_data = {
        "Age": employee.Age,
        "DailyRate": employee.DailyRate,
        "DistanceFromHome": employee.DistanceFromHome,
        "EnvironmentSatisfaction": employee.EnvironmentSatisfaction,
        "HourlyRate": employee.HourlyRate,
        "JobInvolvement": employee.JobInvolvement,
        "JobSatisfaction": employee.JobSatisfaction,
        "MonthlyIncome": employee.MonthlyIncome,
        "MonthlyRate": employee.MonthlyRate,
        "NumCompaniesWorked": employee.NumCompaniesWorked,
        "OverTime": employee.OverTime,
        "PercentSalaryHike": employee.PercentSalaryHike,
        "RelationshipSatisfaction": employee.RelationshipSatisfaction,
        "StockOptionLevel": employee.StockOptionLevel,
        "TrainingTimesLastYear": employee.TrainingTimesLastYear,
        "WorkLifeBalance": employee.WorkLifeBalance,
        "YearsAtCompany": employee.YearsAtCompany,
        "YearsInCurrentRole": employee.YearsInCurrentRole,
        "YearsSinceLastPromotion": employee.YearsSinceLastPromotion,
        "YearsWithCurrManager": employee.YearsWithCurrManager,
        "TotalWorkingYears": employee.TotalWorkingYears,
    }
    
    df = pd.DataFrame([raw_data])
    
    # Apply feature engineering (matching src/features/engineer.py)
    # 1. Income per year of experience
    df["IncomePerYearExp"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    
    # 2. Tenure ratio (years at company vs total career)
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    
    # 3. Satisfaction score (average of satisfaction metrics)
    satisfaction_cols = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    df["SatisfactionScore"] = df[satisfaction_cols].mean(axis=1)
    
    # 4. Promotion lag (years since last promotion relative to tenure)
    df["PromotionLag"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    
    # 5. Manager stability (years with current manager vs tenure)
    df["ManagerStability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    
    # 6. Encode OverTime as binary (Yes=1, No=0)
    df["OverTime_Yes"] = (df["OverTime"] == "Yes").astype(int)
    
    # 7. High overtime risk (OverTime=Yes AND JobSatisfaction <= 2)
    df["HighOvertimeRisk"] = (
        (df["OverTime_Yes"] == 1) & (df["JobSatisfaction"] <= 2)
    ).astype(int)
    
    # Expected features in exact order (matching model training)
    EXPECTED_FEATURES = [
        "MonthlyIncome", "Age", "TotalWorkingYears", "IncomePerYearExp",
        "DailyRate", "MonthlyRate", "DistanceFromHome", "SatisfactionScore",
        "HourlyRate", "OverTime_Yes", "NumCompaniesWorked", "YearsWithCurrManager",
        "YearsAtCompany", "ManagerStability", "StockOptionLevel", "TenureRatio",
        "WorkLifeBalance", "PercentSalaryHike", "JobInvolvement",
        "EnvironmentSatisfaction", "PromotionLag", "TrainingTimesLastYear",
        "YearsInCurrentRole", "RelationshipSatisfaction", "HighOvertimeRisk",
    ]
    
    # Return features in expected order
    return df[EXPECTED_FEATURES]


def extract_risk_factors(employee: EmployeeInput, probability: float) -> list:
    """
    Extract top risk factors for an employee (simplified version).
    
    Args:
        employee: Employee input features
        probability: Model's attrition probability
    
    Returns:
        List of RiskFactor dicts
    """
    risk_factors = []
    
    # Simplified heuristic-based risk factors
    if employee.JobSatisfaction <= 2:
        risk_factors.append(
            {
                "feature": "JobSatisfaction",
                "value": employee.JobSatisfaction,
                "impact": 0.15,
                "direction": "increases_risk",
            }
        )
    
    if employee.WorkLifeBalance <= 2:
        risk_factors.append(
            {
                "feature": "WorkLifeBalance",
                "value": employee.WorkLifeBalance,
                "impact": 0.12,
                "direction": "increases_risk",
            }
        )
    
    if employee.OverTime == "Yes":
        risk_factors.append(
            {
                "feature": "OverTime",
                "value": "Yes",
                "impact": 0.10,
                "direction": "increases_risk",
            }
        )
    
    if employee.YearsAtCompany < 2:
        risk_factors.append(
            {
                "feature": "YearsAtCompany",
                "value": employee.YearsAtCompany,
                "impact": 0.08,
                "direction": "increases_risk",
            }
        )
    
    if employee.MonthlyIncome < 3000:
        risk_factors.append(
            {
                "feature": "MonthlyIncome",
                "value": employee.MonthlyIncome,
                "impact": 0.06,
                "direction": "increases_risk",
            }
        )
    
    # Sort by impact descending, take top 5
    risk_factors = sorted(risk_factors, key=lambda x: x["impact"], reverse=True)[:5]
    
    return [RiskFactor(**rf) for rf in risk_factors]


@router.post("/predict", response_model=PredictionResponse)
async def predict_attrition(employee: EmployeeInput) -> PredictionResponse:
    """
    Predict attrition risk for a single employee.
    
    Args:
        employee: Employee features from request body
    
    Returns:
        PredictionResponse with probability, risk level, and top factors
    """
    try:
        # Prepare features
        X = prepare_employee_features(employee)
        
        # Get prediction probability
        if MODEL is not None:
            prob = MODEL.predict_proba(X)[0][1]  # Class 1 = "Attrition: Yes"
        else:
            # Fallback: heuristic-based prediction
            prob = calculate_risk_heuristic(employee)
        
        # Determine risk level
        risk_level, risk_color = get_risk_level(prob)
        
        # Extract risk factors
        risk_factors = extract_risk_factors(employee, prob)
        
        # Generate recommendation
        if risk_level == "HIGH":
            recommendation = (
                "🔴 IMMEDIATE ACTION REQUIRED: Schedule HR intervention, "
                "review compensation, and discuss career development."
            )
        elif risk_level == "MEDIUM":
            recommendation = (
                "🟡 MONITOR: Regular check-ins with manager, "
                "consider mentorship or skill development programs."
            )
        else:
            recommendation = (
                "🟢 STABLE: Continue standard engagement practices. "
                "Monitor for changes in satisfaction scores."
            )
        
        return PredictionResponse(
            attrition_probability=round(prob, 3),
            risk_level=risk_level,
            risk_color=risk_color,
            top_risk_factors=risk_factors,
            recommendation=recommendation,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Prediction error: {str(e)}"
        )


def calculate_risk_heuristic(employee: EmployeeInput) -> float:
    """
    Calculate attrition risk using simple heuristic rules (when model unavailable).
    
    Args:
        employee: Employee features
    
    Returns:
        Risk probability (0-1)
    """
    risk = 0.16  # Base rate
    
    if employee.JobSatisfaction <= 2:
        risk += 0.20
    if employee.WorkLifeBalance <= 2:
        risk += 0.15
    if employee.OverTime == "Yes":
        risk += 0.15
    if employee.YearsAtCompany < 2:
        risk += 0.18
    if employee.MonthlyIncome < 3000:
        risk += 0.12
    if employee.JobRole == "Sales Rep":
        risk += 0.10
    if employee.DistanceFromHome > 20:
        risk += 0.08
    
    return min(risk, 0.95)  # Cap at 95%
