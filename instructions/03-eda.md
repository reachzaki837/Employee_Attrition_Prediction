# 03 — EDA & Visualization

> ⚡ Copilot: Run EDA on the CLEAN dataset from data/processed/clean.csv.
> Every chart must be saved to reports/eda/ as a PNG. Never just plt.show().

---

## Phase Task List

- [ ] Create `src/visualization/eda_plots.py` with all plot functions
- [ ] Create `notebooks/01_eda.ipynb` that calls all plot functions
- [ ] Generate and save all 8 charts below
- [ ] Print a text summary of key findings to console
- [ ] Confirm all 8 PNGs exist in `reports/eda/`

---

## Chart Style — Apply to Every Plot

```python
# At top of eda_plots.py
import matplotlib.pyplot as plt
import seaborn as sns

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
```

---

## 8 Required Charts

### Chart 1 — Overall Attrition Rate (Pie/Donut)
```
Filename: reports/eda/01_attrition_rate.png
Shows: % of employees who left vs stayed
Use: plt.pie with a center circle for donut style
Title: "Overall Attrition Rate"
```

### Chart 2 — Attrition by Department (Bar)
```
Filename: reports/eda/02_attrition_by_department.png
X: Department, Y: Attrition Rate (%)
Color: Use PALETTE[0]
Add value labels on each bar
Title: "Attrition Rate by Department"
```

### Chart 3 — Attrition by Job Role (Horizontal Bar)
```
Filename: reports/eda/03_attrition_by_jobrole.png
Y: JobRole (sorted by rate), X: Attrition Rate (%)
Title: "Attrition Rate by Job Role"
```

### Chart 4 — Age Distribution by Attrition (KDE/Histogram)
```
Filename: reports/eda/04_age_distribution.png
Overlay two KDE curves: Attrition=Yes vs Attrition=No
X: Age, Y: Density
Legend: "Left Company" vs "Stayed"
Title: "Age Distribution by Attrition"
```

### Chart 5 — Monthly Income vs Attrition (Boxplot)
```
Filename: reports/eda/05_income_vs_attrition.png
X: Attrition (0/1), Y: MonthlyIncome
Use seaborn boxplot with PALETTE
Title: "Monthly Income: Stayed vs Left"
```

### Chart 6 — Correlation Heatmap (Numeric Features)
```
Filename: reports/eda/06_correlation_heatmap.png
Only top 15 most correlated features with target
Use seaborn heatmap, annot=True, cmap="coolwarm"
Title: "Feature Correlation with Attrition"
```

### Chart 7 — Overtime Impact (Grouped Bar)
```
Filename: reports/eda/07_overtime_impact.png
X: OverTime (Yes/No), Y: Attrition Rate (%)
Split further by Department (grouped bars)
Title: "Overtime vs Attrition by Department"
```

### Chart 8 — Satisfaction Scores Heatmap
```
Filename: reports/eda/08_satisfaction_heatmap.png
X: JobSatisfaction (1-4), Y: WorkLifeBalance (1-4)
Color: Attrition rate in each cell
Use seaborn heatmap
Title: "Attrition Rate by Satisfaction Scores"
```

---

## Key Findings to Print (Console Output)

After generating charts, print:

```
=== PulseML EDA Summary ===
Overall attrition rate: X%
Highest risk department: X (X%)
Highest risk job role: X (X%)
Overtime workers attrition rate: X%
Non-overtime attrition rate: X%
Avg age (left): X | Avg age (stayed): X
Avg income (left): $X | Avg income (stayed): $X
Most correlated feature with attrition: X (r=X)
===========================
```
