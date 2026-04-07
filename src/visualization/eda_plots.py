import matplotlib.pyplot as plt
import seaborn as sns

from config import settings

STYLE = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor": "#1A1D27",
    "axes.edgecolor": "#2A2E42",
    "axes.labelcolor": "#94A3B8",
    "text.color": "#F1F5F9",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#2A2E42",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
}
plt.rcParams.update(STYLE)

PALETTE = ["#6366F1", "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"]


def _prepare(df):
    """Return copy with numeric attrition flag column for computations."""
    df = df.copy()
    df["AttritionFlag"] = df[settings.TARGET_COLUMN].map(
        {settings.POSITIVE_CLASS: 1, settings.NEGATIVE_CLASS: 0}
    )
    return df


def save_fig(fig, filename: str) -> None:
    """Utility to save a figure under reports/eda."""
    path = settings.EDA_DIR / filename
    settings.EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def plot_attrition_rate(df):
    """Chart 1: Overall attrition rate donut."""
    counts = df[settings.TARGET_COLUMN].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%.1f%%", colors=PALETTE)
    centre = plt.Circle((0, 0), 0.70, fc='black')
    fig.gca().add_artist(centre)
    ax.set_title("Overall Attrition Rate")
    save_fig(fig, "01_attrition_rate.png")


def plot_attrition_by_department(df):
    """Chart 2: Bar of attrition rate by department."""
    d = _prepare(df)
    rates = (
        d.groupby('Department')['AttritionFlag']
        .mean()
        .sort_values(ascending=False) * 100
    )
    fig, ax = plt.subplots()
    bars = ax.bar(rates.index, rates.values, color=PALETTE[0])
    ax.set_title("Attrition Rate by Department")
    ax.set_ylabel("Rate (%)")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}%", ha='center', va='bottom')
    save_fig(fig, "02_attrition_by_department.png")


def plot_attrition_by_jobrole(df):
    """Chart 3: Horizontal bar of attrition rate by job role."""
    d = _prepare(df)
    rates = (
        d.groupby('JobRole')['AttritionFlag']
        .mean() * 100
    ).sort_values()
    fig, ax = plt.subplots()
    rates.plot.barh(ax=ax, color=PALETTE)
    ax.set_title("Attrition Rate by Job Role")
    ax.set_xlabel("Rate (%)")
    save_fig(fig, "03_attrition_by_jobrole.png")


def plot_age_distribution(df):
    """Chart 4: Age KDE overlay."""
    fig, ax = plt.subplots()
    sns.kdeplot(df[df[settings.TARGET_COLUMN] == settings.POSITIVE_CLASS]['Age'], label="Left Company", ax=ax)
    sns.kdeplot(df[df[settings.TARGET_COLUMN] == settings.NEGATIVE_CLASS]['Age'], label="Stayed", ax=ax)
    ax.set_title("Age Distribution by Attrition")
    ax.set_xlabel("Age")
    ax.legend()
    save_fig(fig, "04_age_distribution.png")


def plot_income_vs_attrition(df):
    """Chart 5: Monthly income boxplot."""
    fig, ax = plt.subplots()
    sns.boxplot(x=settings.TARGET_COLUMN, y='MonthlyIncome', data=df,
                palette=PALETTE, ax=ax)
    ax.set_title("Monthly Income: Stayed vs Left")
    save_fig(fig, "05_income_vs_attrition.png")


def plot_correlation_heatmap(df):
    """Chart 6: Correlation of top 15 numeric features with target."""
    d = _prepare(df)
    # only numeric columns for correlation
    num = d.select_dtypes(include="number")
    corr = num.corr()["AttritionFlag"].abs().sort_values(ascending=False)
    top = corr.iloc[1:16].index
    mat = num[top.tolist() + ["AttritionFlag"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation with Attrition")
    save_fig(fig, "06_correlation_heatmap.png")


def plot_overtime_impact(df):
    """Chart 7: Overtime vs attrition by department grouped bar."""
    d = _prepare(df)
    temp = (
        d.groupby(['OverTime', 'Department'])['AttritionFlag']
        .mean().unstack()
    ) * 100
    fig, ax = plt.subplots()
    temp.T.plot(kind='bar', ax=ax)
    ax.set_title("Overtime vs Attrition by Department")
    ax.set_ylabel("Rate (%)")
    save_fig(fig, "07_overtime_impact.png")


def plot_satisfaction_heatmap(df):
    """Chart 8: Attrition rate heatmap by satisfaction scores."""
    d = _prepare(df)
    pivot = (
        d.groupby(['JobSatisfaction', 'WorkLifeBalance'])['AttritionFlag']
        .mean().unstack()
    ) * 100
    fig, ax = plt.subplots()
    sns.heatmap(pivot, annot=True, cmap="viridis", ax=ax)
    ax.set_title("Attrition Rate by Satisfaction Scores")
    save_fig(fig, "08_satisfaction_heatmap.png")
