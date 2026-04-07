# 09 — Final Report & Demo

---

## Phase Task List

- [ ] Create `src/reports/generator.py` — auto-generates HTML report
- [ ] Generate `reports/final_report.html` with all findings
- [ ] Update `README.md` with setup instructions and screenshots
- [ ] Do a full end-to-end run from raw data to prediction

---

## Final Report Must Include

1. **Project Summary** — what PulseML does, dataset used
2. **EDA Findings** — all 8 charts + 3 bullet point insights each
3. **Model Comparison Table** — all 3 models with all metrics
4. **Best Model Summary** — why it was chosen
5. **Top 10 Attrition Risk Factors** (global SHAP)
6. **Sample Predictions** — 3 employees: Low/Medium/High risk
7. **Recommendations** — 3 actionable HR recommendations
8. **How to Run** — step by step instructions

---

## Demo Script (What to Show)

```
1. Show the instructions/ folder → explain the orchestration system
2. Show logs/copilot-session-log.md → prove Copilot followed the plan
3. Run: python src/data/loader.py → data downloads automatically
4. Open: reports/eda/ → show the 8 charts
5. Show: reports/model_comparison.csv → compare all 3 models
6. Run: uvicorn src.api.main:app → start the dashboard
7. Open: http://localhost:8000 → show live dashboard
8. POST /predict → predict a high-risk employee live
9. Show SHAP explanation → "here's WHY this person is at risk"
10. Open: http://localhost:8000/docs → show auto-generated API docs
```
