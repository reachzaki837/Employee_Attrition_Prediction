from pathlib import Path
import pandas as pd

# ensure imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization import eda_plots as plots
from config import settings


def summary_stats(df: pd.DataFrame) -> None:
    # map target to numeric flag for calculations
    df_flag = df.copy()
    df_flag["AttritionFlag"] = df_flag[settings.TARGET_COLUMN].map({settings.POSITIVE_CLASS: 1, settings.NEGATIVE_CLASS: 0})
    overall = df_flag["AttritionFlag"].mean() * 100
    dept = df_flag.groupby('Department')["AttritionFlag"].mean().idxmax()
    dept_rate = df_flag.groupby('Department')["AttritionFlag"].mean().max() * 100
    role = df_flag.groupby('JobRole')["AttritionFlag"].mean().idxmax()
    role_rate = df_flag.groupby('JobRole')["AttritionFlag"].mean().max() * 100
    ot_yes = df_flag[df_flag['OverTime'] == 'Yes']["AttritionFlag"].mean() * 100
    ot_no = df_flag[df_flag['OverTime'] == 'No']["AttritionFlag"].mean() * 100
    avg_age_left = df_flag[df_flag[settings.TARGET_COLUMN] == settings.POSITIVE_CLASS]['Age'].mean()
    avg_age_stay = df_flag[df_flag[settings.TARGET_COLUMN] == settings.NEGATIVE_CLASS]['Age'].mean()
    avg_inc_left = df_flag[df_flag[settings.TARGET_COLUMN] == settings.POSITIVE_CLASS]['MonthlyIncome'].mean()
    avg_inc_stay = df_flag[df_flag[settings.TARGET_COLUMN] == settings.NEGATIVE_CLASS]['MonthlyIncome'].mean()
    num = df_flag.select_dtypes(include="number")
    corr = num.corr()["AttritionFlag"].drop("AttritionFlag").abs()
    most_corr = corr.idxmax()
    most_corr_val = corr.max()

    print("=== PulseML EDA Summary ===")
    print(f"Overall attrition rate: {overall:.1f}%")
    print(f"Highest risk department: {dept} ({dept_rate:.1f}%)")
    print(f"Highest risk job role: {role} ({role_rate:.1f}%)")
    print(f"Overtime workers attrition rate: {ot_yes:.1f}%")
    print(f"Non-overtime attrition rate: {ot_no:.1f}%")
    print(f"Avg age (left): {avg_age_left:.1f} | Avg age (stayed): {avg_age_stay:.1f}")
    print(f"Avg income (left): ${avg_inc_left:.0f} | Avg income (stayed): ${avg_inc_stay:.0f}")
    print(f"Most correlated feature with attrition: {most_corr} (r={most_corr_val:.2f})")
    print("===========================")


def main():
    # load raw dataset for categorical columns
    from src.data.loader import load_raw_data
    df = load_raw_data()
    plots.plot_attrition_rate(df)
    plots.plot_attrition_by_department(df)
    plots.plot_attrition_by_jobrole(df)
    plots.plot_age_distribution(df)
    plots.plot_income_vs_attrition(df)
    plots.plot_correlation_heatmap(df)
    plots.plot_overtime_impact(df)
    plots.plot_satisfaction_heatmap(df)
    summary_stats(df)


if __name__ == '__main__':
    main()
