"""Pydantic schemas for FastAPI requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Dict


class EmployeeInput(BaseModel):
    """Input schema for employee attrition prediction."""

    Age: int = Field(..., ge=18, le=70, description="Age in years")
    DailyRate: float = Field(..., gt=0, description="Daily rate")
    Department: str = Field(..., description="Department name")
    DistanceFromHome: int = Field(..., ge=0, description="Distance from home (km)")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, description="Environment satisfaction (1-4)")
    HourlyRate: float = Field(..., gt=0, description="Hourly rate")
    JobInvolvement: int = Field(..., ge=1, le=4, description="Job involvement (1-4)")
    JobRole: str = Field(..., description="Job role")
    JobSatisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction (1-4)")
    MaritalStatus: str = Field(..., description="Marital status")
    MonthlyIncome: float = Field(..., gt=0, description="Monthly income in dollars")
    MonthlyRate: float = Field(..., gt=0, description="Monthly rate")
    NumCompaniesWorked: int = Field(..., ge=0, description="Number of companies worked for")
    OverTime: str = Field(..., pattern="^(Yes|No)$", description="Overtime: Yes or No")
    PercentSalaryHike: int = Field(..., ge=0, le=25, description="Percent salary hike")
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, description="Relationship satisfaction (1-4)")
    StockOptionLevel: int = Field(..., ge=0, le=3, description="Stock option level (0-3)")
    TrainingTimesLastYear: int = Field(..., ge=0, description="Training times last year")
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="Work-life balance (1-4)")
    YearsAtCompany: int = Field(..., ge=0, description="Years at company")
    YearsInCurrentRole: int = Field(..., ge=0, description="Years in current role")
    YearsSinceLastPromotion: int = Field(..., ge=0, description="Years since last promotion")
    YearsWithCurrManager: int = Field(..., ge=0, description="Years with current manager")
    TotalWorkingYears: int = Field(..., ge=0, description="Total years of work experience")
    BusinessTravel: str = Field(..., description="Business travel frequency")


class RiskFactor(BaseModel):
    """A single risk factor contributing to attrition probability."""

    feature: str
    value: str | int | float
    impact: float
    direction: str  # "increases_risk" or "decreases_risk"


class PredictionResponse(BaseModel):
    """Response from /predict endpoint."""

    attrition_probability: float = Field(..., ge=0, le=1)
    risk_level: str  # LOW / MEDIUM / HIGH
    risk_color: str  # hex color
    top_risk_factors: List[RiskFactor]
    recommendation: str


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str
    model_loaded: bool
    model_name: str
    version: str
