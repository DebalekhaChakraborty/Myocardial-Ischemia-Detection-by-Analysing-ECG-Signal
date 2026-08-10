# P1 Physiology Retention Decision V1

## 1. Nature of this document

This is a **human governance decision**, not a new scientific experiment. It
records a bounded Pareto judgement over the already-frozen canonical P1 Stage-1
evidence. No metric here was recomputed; every value is read from immutable
artifacts.

The canonical Stage-1 suite deliberately did **not** make this decision and must
continue to record `physiology_retained: null` and
`retention_decision_performed: false`. That file is not modified by this
document.

## 2. Evidence sources (immutable)

| Artifact | SHA-256 |
|---|---|
| P1 protocol | `f48ffc66e52649d74a8286182d5e7220f78abdd6c12a7ebfe04f116b853337f1` |
| P1 Stage-1 suite | `cc354ef64415d9c0dafcffdc0fdfa2446cd81a7d0c30fa9c58b0095cbc0be772` |
| P1-A lock (`P1A_neural_head_v1`) | `969af95dd21cd946e736600b890b722c08cb8d075574a492c0190320d2a876c9` |
| P1-B lock (`P1B_phys_fusion_v1`) | `796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0` |
| Physiology transform | `cc6bd3a353f0ac6cad342114ed96e135cbf3c61e2946f847d5b95358b6bd51a9` |
| TRAIN embedding cache | `0a5f021b89597d245a2afdc51fe1a65ba5cd6a090beba429f38bbccff8c372dd` |
| VALIDATION embedding cache | `c533db3acfdfa1057c2ac9d8e77d011d3ac5f87fc7a872399227f94f526db0c3` |
| B4-B checkpoint | `b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9` |

Re-verified read-only with `validate_p1_stage1_results`: **PASS**,
`test_accessed: false`, `sealed_test_state: unopened`.

## 3. Canonical P1 evidence

P1-A is the **matched neural-only control** for P1-B. Both consume the same
frozen 128-d B4-B embedding under one common training contract; only P1-B adds
the 18-d transformed physiology vector.

### Pooled validation (473,897 windows, 12 subjects)

| Metric | P1-A | P1-B |
|---|---|---|
| AUPRC | 0.3372201051523283 | 0.3752480844977594 |
| AUROC | 0.8865373476753099 | 0.9045063238671038 |
| F1 | 0.3752448726662136 | 0.3944669687574529 |
| Sensitivity | 0.43397447752912893 | 0.45884963935638984 |
| Specificity | 0.9579630706504315 | 0.9585114168780084 |
| PPV | 0.3305162335375731 | 0.3459286112660346 |
| MCC | 0.3446324902394315 | 0.365246791848724 |

### Subject-macro AUPRC (9 contributing subjects)

| | |
|---|---|
| P1-A | 0.3940333427586504 |
| P1-B | 0.4095404575163482 |

### Matched deltas (B − A), derived descriptive

| Quantity | Δ |
|---|---|
| Pooled AUPRC | **+0.03802797934543112** |
| Subject-macro AUPRC | **+0.015507114757697837** |
| Rate-related challenge FPR | **+0.006032575909913518** (worse) |
| Axis-shift challenge FPR | −0.0030000000000000027 (better) |
| Conduction challenge FPR | +0.006097560975609755 (**descriptive only**) |

Deployment increment: **+1,152 parameters**, **+4,608 FP32 payload bytes**.

Scores are **uncalibrated sigmoid model scores**, not calibrated probabilities.

## 4. Decision

**RETAIN the complete frozen 18-dimension `morphology_v1` physiology vector for
downstream patient-memory research.**

The M1 base representation is therefore:

```
z_t = [ frozen B4-B pooled embedding (128) ; frozen transformed morphology_v1 (18) ]
dim(z_t) = 146
```

## 5. Reasoning

P1-B produced a meaningful matched development improvement in the primary metric
(pooled AUPRC, +0.0380) and in subject-macro AUPRC (+0.0155), at a very small
model-size cost (+1,152 parameters). Axis-shift challenge FPR improved slightly.

**Retained caveat — rate-related challenge FPR degraded by +0.00603.** This is
quantitative secondary evidence and is carried forward explicitly rather than
netted away. If a later phase shows rate-related false alarms are operationally
costly, this decision should be revisited.

Conduction-change evidence is **exploratory/descriptive only** — one validation
subject — and carried **no weight** in this decision.

## 6. Limits of this claim

This decision explicitly does **not** assert:

- that P1-B is superior to the historical end-to-end B4-B detector (P1-A, not
  B4-B, is the matched control; the training procedures differ);
- statistical significance — no hypothesis test, confidence interval or paired
  subject-level analysis was performed, and none is implied by the arithmetic;
- any test-set or generalisation improvement — the sealed test remains unopened;
- any clinical benefit.

The morphology features remain **R-aligned algorithmic waveform proxies**, not
validated J-point measurements, ST-segment delineation, clinically validated
morphology, or diagnostic measurements.

## 7. Scope and deferrals

- The canonical P1 Stage-1 result, both arm locks, both embedding caches, the
  physiology transform and all B4 artifacts are **unmodified** by this document.
- **Feature-group ablations are authorized by the P1 protocol but DEFERRED.**
  They do not block M1 and remain available as a later, separately authorised
  step.
- This document authorises no sealed-test access.
