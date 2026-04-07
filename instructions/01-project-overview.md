# 01 — Project Overview: PulseML

## What Is PulseML?

PulseML is an **Employee Attrition Prediction System** for HR teams.
It ingests HR data, explores it visually, trains an ML model to predict
which employees are at risk of leaving, explains WHY using SHAP values,
and serves predictions through a live FastAPI web dashboard.

---

## Dataset

**IBM HR Analytics Employee Attrition Dataset**
- Source: Kaggle (via `kagglehub`)
- Download using:
  ```python
  import kagglehub
  # Download latest version
  path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
  print("Path to dataset files:", path)
  ```
- 1,470 rows × 35 columns
- Target column: `Attrition` (Yes / No)
- Class imbalance: ~16% Yes, ~84% No

### Key Columns
| Column | Type | Description |
|--------|------|-------------|
| Attrition | Target | Yes / No |
| Age | Numeric | Employee age |
| Department | Categorical | Sales, R&D, HR |
| JobRole | Categorical | 9 job roles |
| MonthlyIncome | Numeric | Salary |
| OverTime | Binary | Yes / No |
| YearsAtCompany | Numeric | Tenure |
| JobSatisfaction | Ordinal | 1–4 scale |
| WorkLifeBalance | Ordinal | 1–4 scale |
| DistanceFromHome | Numeric | Miles |

---

## What the System Does

### 1. Data Pipeline
- Downloads and caches the dataset
- Cleans nulls, fixes types, drops useless columns
- Encodes categoricals, scales numerics

### 2. EDA (Exploratory Data Analysis)
- Attrition rate by Department, JobRole, Age group
- Correlation heatmap
- Distribution plots for key features
- All charts saved to `reports/eda/`

### 3. Feature Engineering
- Creates new features: `IncomePerYear`, `TenureRatio`, `SatisfactionScore`
- Handles class imbalance with SMOTE
- Outputs processed dataset to `data/processed/`

### 4. Model Training
- Trains 3 models: Logistic Regression, Random Forest, XGBoost
- Evaluates with: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Selects best model, saves to `models/best_model.pkl`

### 5. SHAP Explainability
- Global feature importance plot
- Per-prediction waterfall chart (why THIS employee is at risk)
- Saves charts to `reports/shap/`

### 6. FastAPI Dashboard
- `GET /` → HTML dashboard with overview metrics
- `POST /predict` → Takes employee JSON, returns attrition probability
- `GET /report` → Shows EDA charts in browser
- `GET /health` → Health check

### 7. Final Report
- Auto-generated PDF/HTML report
- Contains: EDA findings, model metrics, top risk factors, recommendations

---

## Success Criteria

After 2 days, the project must:
- [ ] Load and clean the dataset automatically
- [ ] Generate at least 6 EDA charts
- [ ] Train and compare 3 models
- [ ] Achieve ROC-AUC > 0.80 on test set
- [ ] Show SHAP explanations for predictions
- [ ] Serve live predictions via FastAPI
- [ ] Have test coverage > 70%
- [ ] Generate a final summary report