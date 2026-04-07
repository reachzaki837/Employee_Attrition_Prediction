# 14 — Reading Log Reports

---

## Where Are the Logs?

```
logs/
  copilot-session-log.md   ← Copilot's session notes
  app.log                  ← Structured JSON application logs
```

---

## Reading copilot-session-log.md

Look for:
- ✅ Completed — what was built
- ⚠️ Blockers — what was stuck and why
- 📌 Decisions — autonomous choices Copilot made
- 🔜 Next Session — where to resume

---

## Reading app.log (JSON logs)

```bash
# Show all errors
cat logs/app.log | python3 -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin if json.loads(l).get('level')=='ERROR']"

# Show last 20 lines pretty
tail -20 logs/app.log | python3 -m json.tool

# Count log events by level
cat logs/app.log | python3 -c "
import sys, json
from collections import Counter
levels = Counter(json.loads(l).get('level') for l in sys.stdin)
print(levels)
"
```

---

## After Each Copilot Session — Checklist

```
□ Read copilot-session-log.md — did it complete the scheduled tasks?
□ Check for any COPILOT NEEDS GUIDANCE comments in .py files
□ Check for any COPILOT ASSUMPTION comments — validate them
□ Run: pytest tests/ -v
□ Check model ROC-AUC is still > 0.80
□ Verify all expected chart PNGs exist in reports/
□ Update phase status in 00-orchestration.md
```
