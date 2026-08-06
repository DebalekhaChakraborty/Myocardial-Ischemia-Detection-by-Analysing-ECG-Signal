# Implementation Plan

Future changes must be independently reviewable and must follow the experiment
contract.

1. **Dataset ingestion and annotation validation:** acquisition contracts,
   record metadata, subject manifests, and annotation checks.
2. **Signal-processing pipeline:** deterministic loading, quality checks, and
   documented preprocessing.
3. **Reproducible baselines:** non-neural and simple learned baselines with
   subject-wise evaluation.
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

