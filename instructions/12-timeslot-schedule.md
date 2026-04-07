# 12 — Timeslot Schedule (2-Day Plan)

> ⚡ Copilot: Check this at the START of every session.
> Find the current day and time block. Work ONLY on that phase.

---

## DAY 1 — Data + Models (7 hours)

| Time | Phase | Instruction File | Goal |
|------|-------|-----------------|------|
| 9:00–9:30 | Setup | `00-orchestration.md` | Read all files, confirm plan |
| 9:30–11:00 | Phase 1 | `02-data-setup.md` | requirements.txt, loader, cleaner, validator |
| 11:00–12:00 | Phase 2 | `03-eda.md` | All 8 EDA charts saved to reports/eda/ |
| 12:00–1:00 | 🍽️ Lunch | — | — |
| 1:00–2:00 | Phase 3 | `04-feature-engineering.md` | 7 new features + SMOTE |
| 2:00–4:00 | Phase 4 | `05-model-training.md` | Train 3 models, compare, save best |
| 4:00–5:00 | Phase 5 | `06-explainability.md` | SHAP global + per-employee charts |
| 5:00–5:30 | Log | `logs/copilot-session-log.md` | End of day log + status update |

---

## DAY 2 — API + Polish + Demo (7 hours)

| Time | Phase | Instruction File | Goal |
|------|-------|-----------------|------|
| 9:00–9:15 | Resume | `logs/copilot-session-log.md` | Read yesterday's log, continue |
| 9:15–11:30 | Phase 6 | `07-api-dashboard.md` | FastAPI + HTML dashboard + /predict |
| 11:30–12:00 | Phase 7 | `08-testing-logging.md` | Core tests + logger setup |
| 12:00–1:00 | 🍽️ Lunch | — | — |
| 1:00–2:00 | Phase 7 | `08-testing-logging.md` | Run pytest, hit 70% coverage |
| 2:00–3:00 | Phase 8 | `09-final-report.md` | Generate HTML report |
| 3:00–4:30 | End-to-end | All | Full run: raw data → dashboard → prediction |
| 4:30–5:30 | Demo prep | `09-final-report.md` | Polish, README, demo script |

---

## ⏰ If Running Behind Schedule

**Behind by 30 min:** Skip Jupyter notebook, work only in .py files
**Behind by 1 hour:** Skip Chart 7 and Chart 8 from EDA
**Behind by 2 hours:** Skip SHAP per-employee charts (keep global only)
**Behind by 3 hours:** Skip HTML report, just demo the API live

The non-negotiables (must have for demo):
1. ✅ Clean data pipeline running
2. ✅ At least 1 trained model saved
3. ✅ FastAPI /predict endpoint working
4. ✅ Session log showing Copilot followed the orchestration system
