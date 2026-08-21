# Implementation Plan

Future changes must be independently reviewable and must follow the experiment
contract. See `docs/CURRENT_STATE.md` for the current experiment ladder,
selected models, and open work; this file tracks which plan items have been
addressed, not day-to-day status.

1. **Dataset ingestion and annotation validation:** implementation and
   annotation-semantic validation complete. Versioned EDB and LTSTDB contracts,
   strict WFDB metadata inspection, annotation preservation, manifest
   generation, leakage validation, remote header/annotation validation, and
   synthetic tests are available.
2. **Signal-processing pipeline:** causal implementation and bounded waveform
   integration validation complete. Includes physical-unit loading, raw and
   optional stateful SOS profiles, causal windows, descriptive quality metrics,
   response audits, and provenance.
3. **Reproducible baselines:** Phase 3A benchmark protocol, frozen subject split,
   leakage-safe window targets, sampling policy, and metrics protocol complete.
   Phase 3B-1 implements frozen waveform-only signal and R-aligned morphology
   schemas, resumable external feature materialization, B0--B3 global classical
   baselines, validation-only threshold locks, and sealed-test reporting. Full
   results are complete: each frozen B0--B3 baseline received one sealed-test
   evaluation (`PHASE3B1_CLASSICAL_BASELINE_RESULTS.md`). The B4 neural
   baseline that follows it is also complete on validation: compact CNN,
   CNN-Transformer, and CNN-SSM candidates were trained and compared, and
   B4-B (CNN-Transformer) is the selected official model
   (`B4_GLOBAL_ENCODER_SELECTION_V1.md`).
4. **Patient-adaptive memory:** contamination-safe short- and long-term baseline
   mechanisms. Complete: short-memory, dual-memory, and long-memory variants
   (M1S/M1D/M1L) were implemented and evaluated; M1L is the selected variant
   (`M1_MEMORY_RETENTION_DECISION_V1.md`), after two earlier attempts failed
   and were documented rather than silently retried.
5. **Physiology-guided model:** justified fusion of ECG representations and ST-T
   morphology. Complete: a plain neural head (P1A) and a physiology-fusion
   model (P1B) were compared, and P1-B is selected
   (`P1_PHYSIOLOGY_RETENTION_DECISION_V1.md`).
6. **Uncertainty calibration:** held-out calibration, reliability metrics, and
   abstention/routing controls. Complete: Platt calibration with selective
   routing is selected and frozen (`U1_CALIBRATION_ROUTING_RETENTION_DECISION_V1.md`).
7. **Temporal episode reasoning:** event construction, onset-delay, and
   false-alarm-per-hour evaluation. Partial: the longitudinal half (causal S4D
   vs. a GRU baseline) is complete, trained, and one-shot outer-validated
   (`T2_LONGITUDINAL_TEMPORAL_RETENTION_DECISION_V1.md`). The episodic/alerting
   state-machine half (internally "T1") has a complete canonical execution
   harness merged to master, with authorization to run it pending human
   decision and zero attempts executed as of this update.
8. **Edge/cloud routing:** confidence-aware policy evaluated without clinical
   claims. Complete, as part of item 6's selective-routing protocol.
9. **Edge benchmarking:** reproducible latency, energy, and hardware-in-the-loop
   measurements. Partial: B4 latency and parameter-count benchmarking is
   complete on a fixed benchmark host (`B4_RESOURCE_BENCHMARK_V1.md`);
   no on-device or edge-hardware measurement exists yet.
10. **Final ablation and external validation:** pre-specified comparisons,
    confounder analysis, and bounded reporting. Not started.
