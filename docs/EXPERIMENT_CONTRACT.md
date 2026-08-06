# Experiment Contract

Every experiment must be reproducible, traceable, and bounded by the evidence.

## Prohibitions

- Do not leak patients or subjects across train, validation, and test partitions.
- Do not tune architectures, thresholds, calibration, routing, or personalization
  using the fixed test set.
- Do not manually edit outputs, invent numbers, or report untracked results.
- Do not combine window-level and episode-level conclusions.
- Do not report accuracy alone; include class-aware metrics and prevalence-aware
  interpretation.
- Do not treat raw softmax values as calibrated confidence.
- Do not infer clinical utility or effectiveness from a hardware demonstration.

## Required record

Each run must emit machine-readable outputs containing the resolved
configuration, Git commit and dirty state, random seed, runtime environment,
dataset and annotation provenance, split-manifest digest, command, and timing.
The test partition is fixed before model selection and never changes during a
study.

Use subject-wise partitions. Use deterministic execution where technically
possible and explicitly record exceptions. Report calibration metrics before
confidence-driven routing. Report window and episode metrics separately,
including false alarms per hour and event-onset delay when applicable.

Every novelty claim requires error analysis and ablation evidence. Preserve
predictions and artefacts outside Git according to dataset access conditions.

