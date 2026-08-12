# M1 Memory-Architecture Retention Decision V1

## 0. Nature of this document

A **human governance decision**, not a new scientific experiment. It records a
bounded Pareto judgement over the frozen M1-v2 Stage-1 development evidence.
No metric here was recomputed; every value is read from immutable artifacts.

The canonical suite deliberately did not make this decision and continues to
record `memory_selection_performed: false`, `memory_selected: null`,
`weighted_score_used: false`. **This document does not modify it.**

## 1. Bound identities

| Artifact | SHA-256 |
|---|---|
| Merged scientific tree | `8260b718ab235873bd8067ca3fbf14f158c71dcd` |
| M1-v2 protocol | `31a81358870cd23c2258cf4f307ab8c4dc7bf245bc4bf18a4d1f48fe2aada39c` |
| M1-v2 Stage-1 suite | `be36f0743dad649756626a981c3dd05ec6f54dc9c01150e70bb3caeb407bac0e` |
| **`M1L_long_memory_v2` lock (RETAINED)** | **`a2636855e14bdd54ff3b0a17f238579d097366bb64761e723003b6d6a13c75a5`** |
| `M1L_long_memory_v2` checkpoint | `a26b6a18db8c005a051054417156068174a166062a5498f32fd48e473ad58510` |
| `M1S_short_memory_v2` lock (ablation) | `e9fd43f7920686c8f14cdf3da7ca2e2a5e6553289c638263e9c57e54be593a65` |
| `M1S_short_memory_v2` checkpoint | `3caa391cbade99fab4988274f22fe3000854c273fcb620d92457987b03f94bbd` |
| `M1D_dual_memory_v2` lock (ablation) | `2d08ffbbbb3fcd962f3abec99d7b2f97823b6ccaafb85fa681dc05363af1a3c1` |
| `M1D_dual_memory_v2` checkpoint | `5c677bcfeb56a888e335525322e63b612dcb2b41e4a3c27f0bf4e8b7c61201e4` |
| P1-B global control lock | `796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0` |
| Physical-observation decision | `ba9be6de0da7037e0d99b7c619aabbb09c44f84a32c04e2241a61d8277ed5ce7` |

Execution history: Authorizations 1 and 2 were consumed by documented pre-claim
failures producing zero arm claims; Authorization 3 produced this evidence in a
single invocation.

## 2. Decision

**RETAIN `M1L_long_memory_v2`** as the memory architecture carried into the
next contamination-safe adaptation phase (M2).

| Arm | Retained |
|---|---|
| `M1S_short_memory_v2` | **false** — frozen ablation evidence |
| **`M1L_long_memory_v2`** | **true** |
| `M1D_dual_memory_v2` | **false** — frozen ablation evidence |

| Property | Value |
|---|---|
| Selection basis | **DEVELOPMENT evidence only** |
| `test_accessed` | **false** |
| `weighted_score_used` | **false** |
| `statistical_significance_claim` | **false** |
| M1 rerun permitted | **no** |

## 3. Evidence

Frozen P1-B global control versus the retained M1L:

| Metric | P1-B control | M1L | Δ (M1L − P1-B) |
|---|---|---|---|
| pooled AUPRC | 0.375248 | **0.384796** | **+0.009548** |
| subject-macro AUPRC | 0.409540 | **0.415833** | **+0.006293** |
| sensitivity | 0.458850 | 0.453532 | **−0.005318** |
| specificity | 0.958511 | 0.960605 | +0.002094 |
| PPV | 0.345929 | 0.355064 | +0.009135 |
| MCC | 0.365247 | 0.368887 | +0.003640 |
| primary-background FPR | — | 0.039395 | **−0.002094** abs |
| rate-related challenge FPR | 0.398753 | 0.393927 | **−0.004826** abs |
| axis-shift challenge FPR | 0.075333 | 0.071333 | **−0.004000** abs |

Conduction-change evidence is **exploratory/descriptive only** (one validation
subject) and carried no weight.

## 4. Bounded-Pareto rationale

M1L provides the strongest alignment with the prespecified M1 objective:
**improved primary discrimination together with more favourable false-alarm
behaviour**, including subject-level false-positive stability, at the cost of a
small sensitivity reduction.

M1L's subject-level false-positive distribution is the tightest of the three
arms (median 0.007099, q75 0.095497, IQR 0.093919, p90 0.140859, max 0.191831),
which matters for a patient-adaptive baseline whose purpose is fewer false
alarms per patient rather than a better pooled average alone.

**M1S is not retained** because pooled AUPRC (0.365077) fell *below* the global
control and its principal false-alarm measures were the worst of the three
(background FPR 0.046247, rate 0.434547, axis 0.092000).

**M1D remains Pareto-relevant and is explicitly not dominated.** It achieved
higher sensitivity (0.477390 vs 0.453532), AUROC (0.912372 vs 0.907570), MCC
(0.371579 vs 0.368887) and balanced accuracy (0.716903 vs 0.707069). It was not
retained because its pooled AUPRC (0.381417) was below M1L's and its
false-alarm behaviour was worse on every measured axis (background 0.043585,
rate 0.427307, axis 0.083000).

The decision prioritises the prespecified M1 purpose — patient-specific
baseline modelling that reduces false alarms without an unacceptable
sensitivity penalty — not a universal claim of superiority.

### Explicit non-claims

- **No statistical significance is claimed.** No hypothesis test, confidence
  interval or paired subject-level analysis was performed, and none is implied
  by the arithmetic. The deltas are small.
- **M1L is not claimed to dominate M1D.**
- No test-set or generalisation claim; the sealed test remains unopened.
- No clinical benefit is claimed. Scores are uncalibrated model scores.

## 5. Known cold-start limitation — carried forward, not reinterpreted

**All three memory arms had very weak 0–5 minute cold-start evidence, including
zero sensitivity at their frozen operating thresholds** on the reported
development subset:

| Arm | 0–5 min (n=1,798) | 5–60 min (n=19,637) | >60 min (n=452,462) |
|---|---|---|---|
| M1S | AUPRC 0.001361, **sens 0.000000** | AUPRC 0.459282 | AUPRC 0.365917 |
| **M1L** | AUPRC 0.001362, **sens 0.000000** | AUPRC 0.464888 | AUPRC 0.385815 |
| M1D | AUPRC 0.001318, **sens 0.000000** | AUPRC 0.464612 | AUPRC 0.382755 |

This is expected in direction — a patient prototype has no patient-specific
history in the first minutes of a stream — but the magnitude is a **real
limitation of M1-v1/v2 as an operating system component**, and it is recorded
here rather than explained away.

It is a **later operational-design consideration**. The M1 selection was **not**
altered because of it, no post-hoc cold-start tuning was performed, and **no new
cold-start threshold is authorised** by this document.

## 6. Scope

`M1S_short_memory_v2` and `M1D_dual_memory_v2` remain **immutable frozen
ablation evidence**. They are not deleted, not rerun, and not re-opened as
candidate architectures in M2 — M2 isolates *update policy*, not memory family.

No M1 artifact is modified by this decision. No M1 rerun is permitted.
