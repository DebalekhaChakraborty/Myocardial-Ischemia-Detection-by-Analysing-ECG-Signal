# Contributing to CardioSentinel

Keep changes small, reviewable, and tied to a documented research question.
Read `AGENTS.md`, `docs/RESEARCH_SCOPE.md`, and
`docs/EXPERIMENT_CONTRACT.md` before changing research logic.

Do not commit raw ECG, patient-derived data, credentials, checkpoints, or
experiment outputs. Add or update tests for behavior changes, then run:

```bash
python -m ruff check .
python -m pytest -q
```

Document assumptions, data provenance, and unresolved decisions in the relevant
configuration or documentation change. Do not modify `legacy/college-v1/`
without an explicit archival task.

