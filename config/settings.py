from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "hr_attrition.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clean.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
SCALER_PATH = ROOT_DIR / "models" / "scaler.pkl"
ENCODER_PATH = ROOT_DIR / "models" / "encoder.pkl"

# Reports
REPORTS_DIR = ROOT_DIR / "reports"
EDA_DIR = REPORTS_DIR / "eda"
SHAP_DIR = REPORTS_DIR / "shap"

# Dataset
KAGGLE_DATASET = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
TARGET_COLUMN = "Attrition"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"

# Columns to drop (constants, no variance, leakage)
COLUMNS_TO_DROP = [
    "EmployeeCount",   # always 1
    "EmployeeNumber",  # just an ID
    "Over18",          # always Y
    "StandardHours",   # always 80
]

# Categorical columns
CATEGORICAL_COLUMNS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]

# Numeric columns to scale
NUMERIC_COLUMNS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]
