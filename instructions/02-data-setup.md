# 02 — Data Setup & Cleaning

> ⚡ Copilot: This is Phase 1. Complete ALL tasks here before EDA.
> The rest of the pipeline depends on clean, consistent data output.

---

## Phase Task List

- [ ] Create `requirements.txt` with all dependencies
- [ ] Create `config/settings.py` with all paths and constants
- [ ] Create `src/data/loader.py` — downloads and caches dataset
- [ ] Create `src/data/cleaner.py` — cleans raw data
- [ ] Create `src/data/validator.py` — validates cleaned data shape
- [ ] Run the pipeline and confirm output saved to `data/processed/clean.csv`

---

## requirements.txt

```
kagglehub==0.3.3
pandas==2.1.0
numpy==1.24.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
scikit-learn==1.3.0
xgboost==1.7.6
shap==0.42.1
imbalanced-learn==0.11.0
fastapi==0.103.0
uvicorn==0.23.2
jinja2==3.1.2
python-dotenv==1.0.0
requests==2.31.0
joblib==1.3.2
pytest==7.4.0
pytest-cov==4.1.0
httpx==0.24.1
```

---

## config/settings.py

```python
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
```

---

## src/data/loader.py — Key Functions to Build

```python
def download_dataset() -> pd.DataFrame:
    """Download dataset using kagglehub, cache to RAW_DATA_PATH."""
    import kagglehub
    path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
    # path contains the folder — find the CSV inside it
    csv_file = next(Path(path).glob("*.csv"))
    return pd.read_csv(csv_file)

def load_raw_data() -> pd.DataFrame:
    """Load from cache if exists, else download."""

def get_data_summary(df: pd.DataFrame) -> dict:
    """Return dict with shape, nulls, dtypes, target distribution."""
```

---

## src/data/cleaner.py — Key Functions to Build

```python
def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns in COLUMNS_TO_DROP from settings."""

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Attrition Yes/No to 1/0."""

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all CATEGORICAL_COLUMNS."""

def scale_numerics(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standard scale all NUMERIC_COLUMNS. Return df + fitted scaler."""

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order. Save to PROCESSED_DATA_PATH."""
```

---

## src/data/validator.py — Key Functions to Build

```python
def validate_clean_data(df: pd.DataFrame) -> bool:
    """
    Check:
    - No null values remain
    - Target column is 0/1 integers
    - All numeric columns are float/int
    - Shape is (1470, expected_cols)
    Raise ValueError with details if any check fails.
    """
```

---

## Validation Checks (Must All Pass)

```
✓ df.isnull().sum().sum() == 0
✓ df[TARGET_COLUMN].dtype == int
✓ df[TARGET_COLUMN].nunique() == 2
✓ df.shape[0] == 1470
✓ No original categorical string columns remain
```