# Implementation Plan

Future changes must be independently reviewable and must follow the experiment
contract.

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
   Non-neural and simple learned baselines remain pending Phase 3B.
4. **Patient-adaptive memory:** contamination-safe short- and long-term baseline
   mechanisms.
5. **Physiology-guided model:** justified fusion of ECG representations and ST-T
   morphology.
6. **Uncertainty calibration:** held-out calibration, reliability metrics, and
   abstention/routing controls.
7. **Temporal episode reasoning:** event construction, onset-delay, and
   false-alarm-per-hour evaluation.
8. **Edge/cloud routing:** confidence-aware policy evaluated without clinical
   claims.
9. **Edge benchmarking:** reproducible latency, energy, and hardware-in-the-loop
   measurements.
10. **Final ablation and external validation:** pre-specified comparisons,
    confounder analysis, and bounded reporting.
