# 04 — Feature Engineering

> ⚡ Copilot: Create new features BEFORE training. Better features
> beat better models. Output must be saved to data/processed/features.csv

---

## Phase Task List

- [ ] Create `src/features/engineer.py` with all feature functions
- [ ] Create `src/features/selector.py` for feature selection
- [ ] Handle class imbalance with SMOTE
- [ ] Save final feature matrix to `data/processed/features.csv`
- [ ] Print feature importance preview to console

---

## New Features to Create

```python
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all engineered features. Returns df with new columns."""

    # 1. Income per year of experience
    df["IncomePerYearExp"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

    # 2. Tenure ratio (years at company vs total career)
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

    # 3. Composite satisfaction score (average of all satisfaction metrics)
    satisfaction_cols = [
        "EnvironmentSatisfaction", "JobSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance"
    ]
    df["SatisfactionScore"] = df[satisfaction_cols].mean(axis=1)

    # 4. Promotion lag (years since last promotion relative to tenure)
    df["PromotionLag"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

    # 5. Manager stability (years with current manager vs tenure)
    df["ManagerStability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)

    # 6. Is low income flag (below 25th percentile)
    df["IsLowIncome"] = (df["MonthlyIncome"] < df["MonthlyIncome"].quantile(0.25)).astype(int)

    # 7. Is high overtime risk (OverTime=Yes AND JobSatisfaction <= 2)
    # Note: OverTime already encoded as 0/1 by this phase
    df["HighOvertimeRisk"] = ((df["OverTime"] == 1) & (df["JobSatisfaction"] <= 2)).astype(int)

    return df
```

---

## Feature Selection

```python
def select_features(df: pd.DataFrame, target: str, top_n: int = 25) -> list[str]:
    """
    Select top N features using Random Forest feature importance.

    Args:
        df: Full feature matrix including target
        target: Target column name
        top_n: Number of top features to keep

    Returns:
        List of selected feature column names (excluding target)
    """
```

---

## Class Imbalance — SMOTE

The dataset has ~16% attrition (Yes). Without handling this, the model
will be biased toward predicting "No" for everyone.

```python
from imblearn.over_sampling import SMOTE

def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Apply SMOTE to training data only (NEVER to test data).

    Returns:
        X_resampled, y_resampled with balanced classes
    """
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    return smote.fit_resample(X_train, y_train)
```

⚠️ SMOTE is applied AFTER train/test split, on training data ONLY.
Applying it before splitting causes data leakage.

---

## Train/Test Split

```python
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame) -> tuple:
    """
    Split into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
```

Use `stratify=y` to preserve the class ratio in both splits.
