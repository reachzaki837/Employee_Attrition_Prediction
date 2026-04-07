import sys
from pathlib import Path

# ensure imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from config import settings
from src.data.loader import load_raw_data
from src.features.selector import select_features


def run_feature_pipeline() -> None:
    """
    Execute the complete feature engineering pipeline.

    Steps:
    1. Load cleaned data
    2. Engineer new features using raw numeric context
    3. Split into train/test
    4. Select top N features
    5. Apply SMOTE to training data
    6. Save feature matrices with target
    """

    # Load cleaned data (scaled, encoded)
    print("[1/6] Loading cleaned data...")
    df = pd.read_csv(settings.PROCESSED_DATA_PATH)
    print(f"Cleaned data shape: {df.shape}")

    # Load raw for engineering context
    df_raw = load_raw_data()

    print("[2/6] Creating engineered features...")

    # Extract raw numeric columns for feature engineering
    df_eng = df_raw[
        [
            "MonthlyIncome",
            "TotalWorkingYears",
            "YearsAtCompany",
            "EnvironmentSatisfaction",
            "JobSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance",
            "YearsSinceLastPromotion",
            "YearsWithCurrManager",
            "OverTime",
        ]
    ].copy()

    # Create 7 engineered features
    df_eng["IncomePerYearExp"] = df_eng["MonthlyIncome"] / (df_eng["TotalWorkingYears"] + 1)
    df_eng["TenureRatio"] = df_eng["YearsAtCompany"] / (df_eng["TotalWorkingYears"] + 1)

    satisfaction_cols = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    df_eng["SatisfactionScore"] = df_eng[satisfaction_cols].mean(axis=1)

    df_eng["PromotionLag"] = df_eng["YearsSinceLastPromotion"] / (df_eng["YearsAtCompany"] + 1)
    df_eng["ManagerStability"] = df_eng["YearsWithCurrManager"] / (df_eng["YearsAtCompany"] + 1)

    df_eng["IsLowIncome"] = (
        df_eng["MonthlyIncome"] < df_eng["MonthlyIncome"].quantile(0.25)
    ).astype(int)

    overtime_flag = (df_eng["OverTime"] == "Yes").astype(int)
    df_eng["HighOvertimeRisk"] = (
        (overtime_flag == 1) & (df_eng["JobSatisfaction"] <= 2)
    ).astype(int)

    # Add engineered features to cleaned data
    new_features = [
        "IncomePerYearExp",
        "TenureRatio",
        "SatisfactionScore",
        "PromotionLag",
        "ManagerStability",
        "IsLowIncome",
        "HighOvertimeRisk",
    ]
    for feat in new_features:
        df[feat] = df_eng[feat]

    print(f"After feature engineering: {df.shape}")

    # Target is already 0/1 in cleaned data
    print("[3/6] Preparing target...")
    print(f"Target distribution: {df[settings.TARGET_COLUMN].value_counts().to_dict()}")

    # Split into train/test (stratified)
    print("[4/6] Splitting into train/test...")
    X = df.drop(columns=[settings.TARGET_COLUMN])
    y = df[settings.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Train target distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # Select top features using Random Forest on train set
    print("[5/6] Selecting top features...")
    df_with_target = X_train.copy()
    df_with_target[settings.TARGET_COLUMN] = y_train
    selected_features = select_features(df_with_target, settings.TARGET_COLUMN, top_n=25)
    print(f"Selected {len(selected_features)} features")

    # Apply SMOTE to training data only
    print("[6/6] Applying SMOTE to training data...")
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    smote = SMOTE(random_state=settings.RANDOM_STATE, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

    print(
        f"After SMOTE: Train set {X_train_balanced.shape}, "
        f"balanced classes: {pd.Series(y_train_balanced).value_counts().to_dict()}"
    )

    # Combine back with target
    df_train_final = pd.DataFrame(X_train_balanced, columns=selected_features)
    df_train_final[settings.TARGET_COLUMN] = y_train_balanced

    df_test_final = X_test_selected.copy()
    df_test_final[settings.TARGET_COLUMN] = y_test

    # Save feature matrices
    settings.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_path = settings.PROCESSED_DATA_PATH.parent / "features_train.csv"
    test_path = settings.PROCESSED_DATA_PATH.parent / "features_test.csv"

    df_train_final.to_csv(train_path, index=False)
    df_test_final.to_csv(test_path, index=False)

    print(f"\n✅ Feature engineering pipeline complete!")
    print(f"Train features saved: {train_path}")
    print(f"Test features saved: {test_path}")
    print(f"Final train shape: {df_train_final.shape}")
    print(f"Final test shape: {df_test_final.shape}")
    print(f"\nFeature summary:")
    print(f"- Original features: 35 (raw)")
    print(f"- Cleaned/encoded features: {X.shape[1]}")
    print(f"- Engineered features: 7")
    print(f"- Selected for modeling: {len(selected_features)}")


if __name__ == "__main__":
    run_feature_pipeline()
