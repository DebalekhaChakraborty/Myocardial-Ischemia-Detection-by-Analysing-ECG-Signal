# M2 Update-Policy Retention Decision V1

## 0. Nature of this document

**THIS IS A HUMAN GOVERNANCE DECISION, NOT A NEW SCIENTIFIC EXPERIMENT.**

It records a bounded Pareto judgement over the frozen M2-v1 development
evidence produced by the completed canonical suite
`m2-v1-development-two-arm-recovery2`. No metric here was recomputed; every
value is read from immutable promoted artifacts.

The canonical suite deliberately did not make this decision and continues to
record `automatic_arm_preference_applied: false`,
`memory_selection_performed: false`, `memory_selected: null`,
`human_review_required: true`, `test_accessed: false` and
`sealed_test_state: "unopened"`. **This document does not modify it.**

This document is not a rerun, not threshold tuning, not retraining, not a
weighted-score selection, not a statistical-significance claim, and not a TEST
decision.

## 1. Bound identities

| Artifact | SHA-256 |
|---|---|
| Execution / master tree | `cdc33797c3b7eb8a2c337c64a7f22a92f05d83a5` |
| M2-v1 protocol | `a8ba6fad038ed0ec01156b6959239f489426d55db8ad73a0c704fd527e7db91c` |
| M2 TRAIN gate derivation receipt | `5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24` |
| Recovery2 decision | `93e53d3c8281d922823d48b73712a2a1ede1c5b0f5bc9f41694af563e1a2fca4` |
| Canonical suite result | `8a6b0a1c64da72fc0f4573c742bef491b01b4eb8179f0759da3c537a01939a02` |
| **`M2-G` arm result (RETAINED)** | **`a061d4d8c5211381c18baa228436bb9abc78b2f87f71fe4cab6ca71b2d15cf75`** |
| **`M2-G` experiment lock (RETAINED)** | **`5ac07d9f1ea3859e046c84fb91f22cee1bb20ef4857837b1c82fcd944dbf0fe8`** |
| `M2-0` arm result (control/ablation) | `37a6e9d4c01b823e407addbb897d14f6f54835a347a6557a745890585395644c` |
| `M2-0` experiment lock (control/ablation) | `8f7109494efc243613046dd57bcdece80491cf6897c867e224235eb8480c1461` |
| Retained M1 arm lock (`M1L_long_memory_v2`) | `a2636855e14bdd54ff3b0a17f238579d097366bb64761e723003b6d6a13c75a5` |

Canonical suite id: `m2-v1-development-two-arm-recovery2`.
Partition accessed: `validation`. Sealed TEST: **unopened**.

Execution history: the original attempt `m2-v1-development-two-arm` and
`m2-v1-development-two-arm-recovery1` were each consumed by a documented
pre-scoring failure that produced **no scientific evidence whatsoever** — no arm
result, no lock, no suite result, no scored row. Both remain preserved and
immutable. Recovery2 produced this evidence in a single invocation.

## 2. Decision

**RETAIN `M2-G`** as the contamination-safe update policy carried into the
downstream uncertainty / longitudinal / operational phases.

| Arm | Retained |
|---|---|
| `M2-0` (naive) | **false** — immutable frozen control / ablation evidence |
| **`M2-G` (gated)** | **true** |

| Property | Value |
|---|---|
| Selection basis | **DEVELOPMENT evidence only** |
| `test_accessed` | **false** |
| `weighted_score_used` | **false** |
| `statistical_significance_claim` | **false** |
| M2 rerun permitted | **no** |

`M2-0` is **not** deleted and **not** rerun. It remains the frozen control that
makes the gated arm's effect measurable.

## 3. Evidence — PRIMARY population

Frozen PRIMARY population: 473,897 rows (21,628 ischemic positive, 452,269
background negative, 12 subjects). Threshold `0.7554003000259399`, inherited
frozen from the retained M1L arm and **not** selected here.

| Metric | M2-0 | M2-G | Δ (M2-G − M2-0) |
|---|---|---|---|
| AUPRC | 0.3847955698 | 0.3845274603 | **−0.0002681095** |
| AUROC | 0.9075699068 | 0.9084480510 | **+0.0008781442** |
| Sensitivity | 0.4535324579 | 0.4683280932 | **+0.0147956353** |
| Specificity | 0.9606053035 | 0.9575120117 | **−0.0030932918** |
| PPV | 0.3550640701 | 0.3451695348 | **−0.0098945353** |
| Balanced accuracy | 0.7070688807 | 0.7129200524 | **+0.0058511717** |
| MCC | 0.3688867751 | 0.3687438704 | **−0.0001429047** |

## 4. False-alarm trade-off — recorded as a real cost

| Measure | M2-0 | M2-G | Δ (M2-G − M2-0) |
|---|---|---|---|
| Background FPR | 0.0393946965 | 0.0424879883 | **+0.0030932918** |
| Rate-related challenge FPR | 0.3939272069 | 0.4080032174 | **+0.0140760105** |
| Axis-shift challenge FPR | 0.0713333333 | 0.0786666667 | **+0.0073333334** |
| Conduction change | 5 / 164 | 5 / 164 | descriptive only |

Subject-level false-positive distribution, upper tail (also less favourable
under M2-G):

| Statistic | M2-0 | M2-G |
|---|---|---|
| median | 0.0070991941 | 0.0071137299 |
| q75 | 0.0954972058 | 0.1014977761 |
| IQR | 0.0939194039 | 0.0997219465 |
| p90 | 0.1408588365 | 0.1588952248 |
| max | 0.1918313570 | 0.2030632411 |

**M2-G DID NOT improve false-alarm behaviour.** Its retention must not be
justified by such a claim, and no part of this decision rests on one.
Conduction-change evidence is **exploratory / descriptive only** (a single
validation subject, 164 windows, identical 5/164 in both arms) and carried no
weight.

## 5. Contamination / drift safety evidence

Frozen drift definition, unchanged and not renormalised:

```
sqrt(mean((mu_long(t) - mu_ref) ** 2))
```

Stress intervals are **source-defined only**: 203 eligible intervals
(ischemic 163, heart-rate-related 36, unreadable-quality 4), selection digest
`d66b1f521c3399358a0465c3deb5a6b73a77e6b980c9b13c612a8096fddfbf63`. Zero
intervals were excluded at the drift stage in either arm.

### Maximum observed peak drift

| Family | M2-0 | M2-G | Approximate reduction |
|---|---|---|---|
| Ischemic | 1.3088318203 | 0.0023193737 | **99.82 %** |
| Heart-rate related | 1.0076068363 | 0.0398963001 | **96.04 %** |
| Unreadable quality | 1.1560887735 | 0.4041660010 | **65.0 %** |

### Persisted mean / end / residual evidence

Read from the immutable arm results, not recomputed:

| Family | Arm | n | peak max / mean | mean-drift max / mean | end max / mean |
|---|---|---|---|---|---|
| Ischemic | M2-0 | 163 | 1.3088318203 / 0.0968941937 | 0.8248143317 / 0.0569076564 | 1.2942110520 / 0.0956872129 |
| Ischemic | **M2-G** | 163 | 0.0023193737 / 0.0000359580 | 0.0022782494 / 0.0000341525 | 0.0023193737 / 0.0000359356 |
| HR-related | M2-0 | 36 | 1.0076068363 / 0.0843176546 | 0.5946078060 / 0.0465112692 | 1.0076068363 / 0.0841806342 |
| HR-related | **M2-G** | 36 | 0.0398963001 / 0.0017936583 | 0.0261383329 / 0.0011325902 | 0.0398963001 / 0.0017936583 |
| Unreadable | M2-0 | 4 | 1.1560887735 / 0.4340901213 | 0.9132203060 / 0.3144228512 | 1.1448810704 / 0.4307363863 |
| Unreadable | **M2-G** | 4 | 0.4041660010 / 0.1010415002 | 0.1931298534 / 0.0482824634 | 0.4041660010 / 0.1010415002 |

Residual drift, maximum over evaluated intervals:

| Family | ≥300 s (n) | M2-0 | M2-G | ≥1800 s (n) | M2-0 | M2-G |
|---|---|---|---|---|---|---|
| Ischemic | 163 | 1.2552774515 | 0.0302824411 | 158 | 1.3998993375 | 0.1162195251 |
| HR-related | 36 | 0.9678683579 | 0.0403344916 | 35 | 0.7603842779 | 0.0585508213 |
| Unreadable | 4 | 1.0869388456 | 0.3948374230 | 4 | 0.8442170087 | 0.3489008809 |

The ≥1800 s counts are lower than the interval counts because source follow-up
does not extend that far for every interval. No follow-up was fabricated
(`follow_up_fabricated: false`), and one right-censored ischemic episode was
excluded at source-eligibility stage with no fabricated boundary.

This is a **material reduction in stress-associated prototype drift**, a
contamination proxy. It is **not** proven clinical safety and **not** proven
physiological contamination elimination.

## 6. Non-trivial adaptation

Over the frozen full-replay denominator of 492,904 timeline rows:

| | M2-0 | M2-G |
|---|---|---|
| Admitted updates | 492,898 | **107,671** |
| Admission fraction | 0.9999878272 | **0.2184421307** |
| Freeze fraction | 0.0000121728 | 0.7815578693 |

**`M2-G` is NOT a trivial never-update policy.** It admits 107,671 updates,
21.84 % of all timeline rows, and continues to adapt patient prototypes
throughout the replay.

Gate refusal accounting for `M2-G` (overlapping conditions; a row failing
several applicable conditions is counted in each, so these do not sum to the
freeze fraction):

| Condition | Failed | Evaluated | Fraction |
|---|---|---|---|
| G3 SQI | 30,322 | 492,898 | 0.0615177988 |
| G4 normal evidence | 264,137 | 492,898 | 0.5358857208 |
| G5 refractory | 369,263 | 492,898 | 0.7491671705 |
| G6 morphology | 2 | 492,898 | 0.0000040576 |

Six rows were non-applicable in both arms — the source-null
`UNAVAILABLE_EXACT_FLAT` rows, where `G1 = false` and G2–G6 are not applicable.
This is the frozen source-null semantics behaving exactly as specified, not a
G3 refusal.

`classification_threshold_used_for_admission: false` in both arms: the
classification threshold is not the memory-admission threshold.

## 7. Bounded-Pareto rationale

`M2-G` provides a very large improvement in the **prespecified M2 objective** —
contamination-safe continual patient adaptation, as measured by
stress-associated prototype drift — while preserving primary discrimination and
increasing sensitivity.

- The AUPRC difference is extremely small: **−0.0002681095**.
- MCC is effectively unchanged in magnitude: **−0.0001429047**.
- AUROC and balanced accuracy are marginally higher.
- Sensitivity increases by **+0.0147956353** absolute.

The safety gain is accepted **despite a real false-positive trade-off**:
background FPR, rate-related FPR, axis FPR, PPV and the subject-FPR upper tail
are all less favourable under `M2-G` (§4).

The human decision prioritises the prespecified M2 objective while requiring
that useful detection remains intact, that there is no unacceptable sensitivity
loss, and that adaptation continues non-trivially. Those requirements are met:
sensitivity did not fall, discrimination is preserved to four decimal places on
AUPRC and MCC, and 107,671 updates were admitted.

### Explicit non-claims

- **No strict Pareto dominance is claimed.** `M2-G` is worse on several
  measured axes and this is recorded, not minimised.
- **No universal superiority is claimed.**
- **No improvement in false-alarm performance is claimed** — the opposite is
  true and recorded in §4.
- **No statistical significance is claimed.** No hypothesis test, confidence
  interval or paired subject-level analysis was performed, and none is implied
  by the arithmetic. The primary deltas are small.
- No test-set or generalisation claim; the sealed TEST remains unopened.
- No clinical benefit is claimed. Scores are uncalibrated model scores, not
  calibrated probabilities, confidences, uncertainties or conformal scores.

## 8. Known cold-start limitation — carried forward, not reinterpreted

**0–5 minute sensitivity remains 0 for both arms** at the frozen operating
threshold:

| Stratum | n | M2-0 sens | M2-G sens | M2-0 spec | M2-G spec |
|---|---|---|---|---|---|
| 0–5 min | 1,798 | **0.000000** | **0.000000** | 0.958820 | 0.958264 |
| 5–60 min | 19,637 | 0.311214 | 0.318416 | 0.965975 | 0.964619 |
| >60 min | 452,462 | 0.467612 | 0.483158 | 0.960393 | 0.957218 |

**M2 does not solve M1 cold start.** The limitation is inherited from M1 and
gating can only make early adaptation more conservative. No cold-start
threshold was tuned, the limitation is not reinterpreted, and the retention was
**not** altered because of it. **No new cold-start threshold is authorised** by
this document. It is carried forward as a later operational-design limitation.

## 9. Non-estimable stress families

| Family | Source markers observed | Status |
|---|---|---|
| Axis shift | 223 | `not_estimable_from_source_defined_LTSTDB_intervals` |
| Conduction change | 12 | `not_estimable_from_source_defined_LTSTDB_intervals` |
| Point noise | 1 | `not_estimable_from_source_defined_LTSTDB_intervals` |

These families produced no source-defined interval, so no drift value exists
for them in either arm: `drift_value_produced: false`,
`zero_drift_asserted: false`, `stress_end_fabricated: false`,
`marker_vicinity_reused_as_stress_duration: false`.

**Their absence from the drift analysis does NOT mean zero drift.** It means
the source does not define an interval over which drift could be estimated. No
duration was manufactured for a marker-only event.

## 10. Defensible wording

The defensible claim is approximately:

> A contamination-safe admission policy substantially reduced stress-associated
> patient-prototype drift while preserving development discrimination, at the
> cost of a modest increase in false-positive rates.

This must **not** be upgraded to clinical safety, statistically significant
superiority, better false-alarm performance, elimination of contamination, or
test generalisation.

## 11. Scope

`M2-0` remains **immutable frozen control / ablation evidence**. It is not
deleted, not rerun, and not reopened as a candidate downstream policy.

No M2 artifact is modified by this decision. **No M2 rerun is permitted.** The
sealed TEST partition remains unopened, and completion of DEVELOPMENT does not
authorise TEST.
