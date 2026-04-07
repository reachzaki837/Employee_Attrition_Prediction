import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features.

    Args:
        df: Input dataframe with raw and cleaned columns.

    Returns:
        DataFrame with added engineered feature columns.
    """
    df = df.copy()

    # 1. Income per year of experience
    df["IncomePerYearExp"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

    # 2. Tenure ratio (years at company vs total career)
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

    # 3. Composite satisfaction score (average of all satisfaction metrics)
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

    # 6. Is low income flag (below 25th percentile)
    df["IsLowIncome"] = (
        df["MonthlyIncome"] < df["MonthlyIncome"].quantile(0.25)
    ).astype(int)

    # 7. Is high overtime risk (OverTime=Yes AND JobSatisfaction <= 2)
    # OverTime is encoded as Yes/No in raw data, needs mapping
    overtime_flag = (df["OverTime"] == "Yes").astype(int)
    df["HighOvertimeRisk"] = ((overtime_flag == 1) & (df["JobSatisfaction"] <= 2)).astype(
        int
    )

    return df
