# CardioSentinel Agent Instructions

This repository is research software, not a medical device. Before making ML,
data, evaluation, or clinical-language changes, read `docs/RESEARCH_SCOPE.md`
and `docs/EXPERIMENT_CONTRACT.md`.

1. Never fabricate measurements, labels, experiment results, or clinical claims.
2. Never place a patient or subject in more than one data partition.
3. Never tune models, thresholds, calibration, routing, or personalization on
   the fixed test partition.
4. Never commit raw physiological data, patient-derived outputs, credentials,
   checkpoints, or untracked manual result edits.
5. Keep claims bounded by recorded evidence. Raw softmax scores are not
   calibrated confidence.
6. Use validated configuration, record assumptions and unresolved decisions, and
   preserve run provenance including commit, configuration, seed, and data
   provenance.
7. Add tests for new logic and run linting and tests before reporting completion.
8. Prefer small, reviewable changes over broad rewrites. Preserve
   `legacy/college-v1/` unchanged unless an explicit archival task says otherwise.
9. Do not add an LLM, dashboard, API, cloud service, or deployment infrastructure
   unless a later task explicitly requests it.

