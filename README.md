# PulseML — Employee Attrition Prediction System

A machine learning system that predicts employee attrition risk using historical HR data. Combines Random Forest modeling with SHAP explainability to provide actionable insights for HR teams.

## 🎯 Quick Start

### Prerequisites
- Python 3.11+
- pip (Python package manager)
- ~2GB disk space for dataset + models

### Installation

```bash
# Clone the repository
cd path/to/PulseML

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/run_pipeline.py    # Load & clean data
python scripts/run_eda.py         # Generate visualizations
python scripts/run_features.py    # Engineer features
python scripts/run_models.py      # Train models
python scripts/run_shap.py        # Generate explanations

# Start the dashboard
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser to **http://localhost:8000**

## 📊 Key Features

### 1. **Data Pipeline**
- Automatic dataset download from Kaggle (1,470 employees)
- Automated data cleaning, encoding, and scaling
- Feature engineering with domain-specific features
- SMOTE balancing for class imbalance handling

### 2. **ML Models**
- **Random Forest** (Best) — ROC-AUC: 79.7%, Accuracy: 84%
- Logistic Regression (Baseline) — ROC-AUC: 79.6%
- XGBoost (Alternative) — ROC-AUC: 76.4%

### 3. **Interactive Dashboard**
- Predict attrition risk for any employee
- Real-time risk assessment (Low/Medium/High)
- Top 5 risk factors with impact percentages
- Personalized HR recommendations

### 4. **Explainability**
- SHAP global feature importance
- Per-employee risk factor breakdown
- Top 10 attrition drivers identified

### 5. **Comprehensive Testing**
- 55 unit & integration tests
- 39% code coverage
- Automated test suite with pytest

## 📁 Project Structure

```
PulseML/
├── config/
│   └── settings.py              # Central configuration hub
├── data/
│   ├── raw/                     # Raw dataset from Kaggle
│   └── processed/               # Cleaned & engineered data
├── src/
│   ├── api/                     # FastAPI endpoints
│   ├── data/                    # Data pipeline (load, clean, validate)
│   ├── features/                # Feature engineering & selection
│   ├── models/                  # Model training & evaluation
│   ├── reports/                 # Report generation
│   ├── utils/                   # Logging and utilities
│   └── visualization/           # EDA charts
├── scripts/                     # Runnable pipeline scripts
├── tests/                       # Test suite (55 tests)
├── templates/                   # HTML dashboard templates
├── reports/                     # Generated reports & charts
├── logs/                        # Session logs & app logs
├── instructions/                # Phase-by-phase guidance
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── .github/copilot-instructions.md  # AI automation rules
```

## 🚀 API Endpoints

### /health (GET)
Health check endpoint. Returns model status.

```bash
curl http://localhost:8000/health
```

### /api/stats (GET)
Dataset and model statistics.

```bash
curl http://localhost:8000/api/stats
```

### /api/predict (POST)
Predict attrition risk for an employee.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Department": "Sales",
    "JobRole": "Sales Executive",
    "MonthlyIncome": 5000,
    "OverTime": "Yes",
    "YearsAtCompany": 2,
    "JobSatisfaction": 2,
    "WorkLifeBalance": 2,
    "DistanceFromHome": 10,
    "TotalWorkingYears": 8,
    "MaritalStatus": "Single",
    "BusinessTravel": "Travel_Frequently"
  }'
```

**Response:**
```json
{
  "attrition_probability": 0.78,
  "risk_level": "HIGH",
  "risk_color": "#EF4444",
  "top_risk_factors": [
    {
      "feature": "JobSatisfaction",
      "value": 2,
      "impact": 0.15,
      "direction": "increases_risk"
    }
  ],
  "recommendation": "IMMEDIATE ACTION REQUIRED..."
}
```

### /api/report-images (GET)
List all EDA chart images.

```bash
curl http://localhost:8000/api/report-images
```

### / (GET)
Interactive dashboard with prediction form.

```bash
# Open in browser
http://localhost:8000
```

### /report (GET)
EDA report with all 8 charts.

```bash
# Open in browser
http://localhost:8000/report
```

### /docs (GET)
Auto-generated API documentation (Swagger).

```bash
# Open in browser
http://localhost:8000/docs
```

## 📈 Key Findings

### Highest Risk Profiles
1. **Sales Reps** — 39.8% attrition rate (highest department/role combination)
2. **Overtime Workers** — 30.5% attrition vs 10.4% non-overtime (3x risk)
3. **Low Job Satisfaction** — Employees with satisfaction ≤2 leave at much higher rates

### Protective Factors
- **Tenure** — >5 years tenure reduces attrition risk by 8%
- **Job Level** — Higher positions (Manager+) more stable
- **IT Department** — 13.9% attrition (most stable)

### Compensation Insight
- Bottom 25% earn $4,787/month vs $6,833 for stayers
- Income-to-experience ratio shows role-specific pay equity issues

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

**Results:**
- ✅ **55 / 55 tests passing**
- 📊 **39% code coverage** (core modules: 80-100%)

## 📝 Logging

Application logs are written to:
- **Console**: Human-readable format
- **File**: `logs/app.log` (JSON format for structured analysis)

Example log entry:
```json
{
  "timestamp": "2024-09-20T15:30:45.123456",
  "level": "INFO",
  "logger": "src.api.routes.predict",
  "message": "Prediction generated",
  "attrition_probability": 0.78,
  "risk_level": "HIGH"
}
```

## 🔧 Configuration

Edit `config/settings.py` to customize:
- Kaggle dataset path
- Model hyperparameters
- Feature selection criteria
- Data paths

## 📊 Reports

All reports are saved in `reports/`:

- **eda/**: 8 EDA visualization charts
  - Attrition distribution
  - By department/job role
  - Income vs attrition
  - Correlation heatmap
  - Overtime impact
  - Satisfaction patterns

- **evaluation/**: Model evaluation charts
  - ROC curves (all 3 models)
  - Confusion matrix (best model)
  - Feature importance
  - Precision-recall curve

- **shap/**: SHAP explainability
  - Global feature importance
  - Per-employee risk factors

- **model_comparison.csv**: Metrics for all 3 models

- **final_report.html**: Complete project report

## 🎓 Architecture & Phases

The project follows 8 distinct phases (see `instructions/`):

1. **Project Setup** — Configuration, requirements, file structure
2. **EDA** — 8 exploratory data analysis charts
3. **Feature Engineering** — 7 new features, SMOTE balancing, top-25 selection
4. **Model Training** — 3 models, Random Forest selected
5. **SHAP Explainability** — Feature importance & per-employee explanations
6. **FastAPI Dashboard** — Interactive web interface
7. **Testing & Logging** — Comprehensive test suite, structured logging
8. **Final Report** — Project summary and recommendations

Each phase has detailed instructions in `instructions/[phase#].md`

## 🤖 AI-Driven Development

This project demonstrates AI-assisted development with GitHub Copilot:
- Orchestration system in `.github/copilot-instructions.md`
- Phase-based guidance in `instructions/00-orchestration.md`
- Session logging in `logs/copilot-session-log.md`

All code follows PEP 8 with:
- Full docstrings on every function
- Type hints on all signatures
- Comprehensive error handling
- Structured logging throughout

## 📚 Dependencies

Core packages:
- `pandas` — Data manipulation
- `scikit-learn` — ML models & pipeline
- `xgboost` — Gradient boosting
- `shap` — Model explainability
- `fastapi` — Web API framework
- `uvicorn` — ASGI server
- `pytest` — Testing framework

See `requirements.txt` for full list.

## 🚨 Demo Script (What to Show)

1. **Show orchestration** — `instructions/00-orchestration.md` details all phases
2. **Show session log** — `logs/copilot-session-log.md` proves systematic execution
3. **Run data pipeline** — `python scripts/run_pipeline.py` downloads & cleans data
4. **Display EDA charts** — Open `reports/eda/` to see patterns
5. **Show model comparison** — `reports/model_comparison.csv` with 3 models
6. **Start API** — `uvicorn src.api.main:app --reload`
7. **Open dashboard** — http://localhost:8000 (live prediction form)
8. **Predict employee** — Fill form → get risk + recommendations
9. **View explanations** — Show SHAP importance
10. **API docs** — http://localhost:8000/docs (auto-generated)

## ❓ FAQ

**Q: How long does the full pipeline take?**
A: ~2-3 minutes (depends on network for Kaggle download)

**Q: Can I use my own data?**
A: Yes, modify `config/settings.py` to point to your CSV file

**Q: What if the model doesn't load?**
A: API has a heuristic fallback for predictions. Check `logs/app.log` for details

**Q: How accurate are predictions?**
A: ROC-AUC 0.797 on test set. True positive rate ~47%, false positive rate manageable

**Q: How do I interpret SHAP values?**
A: Positive values increase risk, negative decrease. Magnitude shows impact strength

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 👤 Authors

Built with GitHub Copilot (Claude Haiku 3.5) — September 2024

## 📞 Support

- Documentation: `instructions/` folder
- Examples: `scripts/` folder
- Q&A: See FAQ above  
- Logs: `logs/` folder for debugging

---

**PulseML v1.0** — Making employee attrition predictable and preventable.
