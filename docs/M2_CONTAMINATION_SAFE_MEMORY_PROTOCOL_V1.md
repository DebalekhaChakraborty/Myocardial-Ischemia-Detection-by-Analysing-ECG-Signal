# M2 Contamination-Safe Continual Adaptation Protocol V1

> **STATUS: FROZEN SCIENTIFIC PROTOCOL — IMPLEMENTATION NOT YET AUTHORIZED.**
> Every scientific choice is closed. No M2 scientific execution has occurred and
> no M2 result exists. Implementation requires separate human authorization.

## 0. Purpose and scope

M1 provided development evidence that the retained long-timescale patient-memory
arm improved the prespecified operating trade-off, and established that M1-v2 is
**explicitly not contamination-safe**: any available finite observation updates
the prototypes, so a developing ischemic event, a severe artifact or a
confounded window can be learned as the patient's new normal.

**M2 exists to prevent that.** It isolates the **UPDATE POLICY** and nothing
else.

M2 **does not** reopen memory-family selection. `M1L_long_memory_v2` is frozen
and retained by
`docs/M1_MEMORY_RETENTION_DECISION_V1.md`
(`a3685fc0f8ff1fa0dce2bf9954bb28a925787070c021f3e80ca5716a4fa5f0ed`).
M1S and M1D are frozen ablation evidence and are **not** re-compared as
candidate architectures.

### Bound upstream identities

| Artifact | SHA-256 |
|---|---|
| Scientific tree | `8260b718ab235873bd8067ca3fbf14f158c71dcd` |
| M1-v2 protocol | `31a81358870cd23c2258cf4f307ab8c4dc7bf245bc4bf18a4d1f48fe2aada39c` |
| M1-v2 Stage-1 suite | `be36f0743dad649756626a981c3dd05ec6f54dc9c01150e70bb3caeb407bac0e` |
| Retained M1L lock | `a2636855e14bdd54ff3b0a17f238579d097366bb64761e723003b6d6a13c75a5` |
| P1-B global control lock | `796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0` |
| `SIGNAL_V1` schema | `25d05f8716340e0fcc9950590025e7c58dccbfe8d0e0475ccd36bd629d4c57d4` |
| `MORPHOLOGY_V1` schema | `13f60be400b5b957c1eb592bbafd8206d4d2855c1aa657a058671fb8d7cab434` |
| `COMBINED_V1` schema | `6b1517cb6ffd5d113a385bb252a90630f75beec4da7345185e74dda0eff98a34` |
| **TRAIN-only gate derivation receipt** | **`5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24`** |

`docs/M2_GATE_DERIVATION_RECEIPT_V1.json` binds every constant below to the
TRAIN population it was derived from. **No validation or test data was accessed
in any derivation.**

**Provenance note (canonical-runtime reproduction).** The receipt above
supersedes an earlier receipt
(`3befd05dc7e9c51ddfed99078d3020375fd610b328d19e64fc7ee3cc745f398e`) generated
while the shared scientific interpreter transiently carried five unrelated
distributions from a concurrent, unrelated application-side session (see
`docs/RUNTIME_INTEGRITY_SENTINEL_V1.md`). Every TRAIN-only constant in this
document was independently recomputed under the canonical, isolated `tactics`
runtime (`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`)
using the exact historical arithmetic paths of the original derivation and
reproduced bit-exactly. **No scientific choice below changed.**

## 1. Causal signal inventory (measured from this repository)

Everything below was verified by inspection at tree `8260b718…71dcd`.

### A. Physical availability — **AVAILABLE NOW**

`observation_state` ∈ {`UNINITIALIZED`(0), `AVAILABLE`(1),
`UNAVAILABLE_EXACT_FLAT`(2)}, persisted per row in the schema-3 store.
Frozen by M1-v2 §5, decided from waveform samples before encoder inference via
`np.ptp(values) <= np.finfo(np.float64).eps`. Fully causal and deployment-observable.

### B. Waveform-only quality — **AVAILABLE NOW**

`SIGNAL_V1`: 22 frozen columns, `COMBINED_V1` indices **0–21**, computed from
the window's own samples only. Directly relevant candidates:

| Index | Column | Physical meaning |
|---|---|---|
| 0 | `finite_sample_fraction` | fraction of finite samples |
| 1 | `flatline_fraction` | fraction in flat runs |
| 2 | `repeated_value_fraction` | fraction of repeated ADC values (saturation/clipping) |
| 5 | `derivative_outlier_fraction` | motion/step artifact proxy |
| 7 | `high_frequency_power_ratio` | EMG/noise proxy |
| 8, 9 | `powerline_ratio_50hz`, `powerline_ratio_60hz` | mains interference |
| 3, 4 | `robust_amplitude_range_mv`, `robust_derivative_scale_mv_per_s` | robust scale |

**Frozen status:** the schema is frozen and the values are in the frozen feature
corpus. **They have never been used as a gate in any protocol** — B4, P1 and M1
all use them only as model input or provenance. Using them to gate memory is
new, so any threshold requires the prospective derivation in §4.

### C. Retained-M1L causal quantities at time *t* — **AVAILABLE NOW**

`z_t` (146-d), `d_long(t)`, the M1L raw logit/score, the frozen validation
threshold `0.7554003000259399`, `past_observed_count`, `past_update_count`,
`recording_age_seconds`, `prototype_disagreement`, and the physical
availability state. All are past-only by construction (score-before-update).

### D. Morphology computability — **AVAILABLE NOW, but distinct**

`morphology_valid` (`COMBINED_V1` index 24) is a **feature-computability flag**,
not a physical-availability flag and not a signal-quality score. A window can be
physically available and perfectly readable yet have `morphology_valid = 0`
because fewer than two usable beats were detected.

**Scientific judgement:** using it in an update gate is *defensible* — it is
waveform-derived, causal and deployment-observable, and a window whose
morphology cannot be computed contributes an 18-d physiology block made entirely
of train-median imputations, which is a weak basis for moving a patient
prototype. It is **not** a quality score and must never be described as one.
**Resolved: included as G6** (§4.4). The trade-off is real — gating on it could refuse to learn during exactly the rhythms a patient-specific baseline should represent — and the human decision accepted that cost so a prototype is never moved by an imputation-dominated representation.

### E. Deployment-observable confounder warnings — **DO NOT EXIST**

`rate_related_confounder`, `axis_shift_confounder` and
`conduction_change_confounder` exist **only as annotation-derived target
families** in `evaluation/models.py`. There is **no** rate warning, axis warning
or conduction warning computed causally from the signal. **A challenge
annotation label is not a deployment-observable warning and must never be used
as one.**

### F. Calibrated uncertainty — **DOES NOT EXIST**

`src/cardiosentinel/uncertainty/` is a **2-line placeholder package**
(`"""Future calibrated uncertainty estimation and confidence controls."""`).
No temperature scaling, isotonic regression or conformal machinery exists. Every
frozen lock records scores as *"uncalibrated model score, not a calibrated
probability"*.

**Therefore M2-v1 must not use the phrase "low uncertainty gate."**

### G. Temporal state machine — **DOES NOT EXIST**

`src/cardiosentinel/episodes/` is a **2-line placeholder package**
(`"""Future temporal reasoning for ST-event and episode construction."""`).
There is no NORMAL / WATCH / EVENT / RECOVERY machine. M2 must not invent one.

## 2. Phase-separation table

| Eventual gate signal | Status in M2-v1 | Why |
|---|---|---|
| Physical availability | **AVAILABLE NOW** | frozen in M1-v2 schema 3 |
| Waveform-only SQI (`SIGNAL_V1`) | **AVAILABLE NOW** | frozen schema, causal, in corpus |
| Normal-evidence margin (M1L score) | **AVAILABLE NOW** | causal, past-only |
| Memory-update refractory | **AVAILABLE NOW** (mechanism only) | derivable from causal state alone |
| `morphology_valid` | **NOW**, included as G6 | computability, not quality |
| **Calibrated uncertainty admission** | **DEFERRED → U1/U2** | no calibration exists |
| **WATCH/EVENT episode state** | **DEFERRED → T1** | no state machine exists |
| **Learned longitudinal confounder state** | **DEFERRED → T1/T2 or MT1** | no causal confounder warning exists |
| **Cloud/delayed contradictory signal** | **DEFERRED** | no such channel exists (§6) |

Later phases may add freeze conditions **without rewriting M2 history**.

## 3. Label firewall

Target and challenge annotations **MAY** be used to: evaluate contamination,
construct controlled contamination stress experiments, measure event
detectability, measure false alarms, measure prototype drift, and stratify
reporting.

They **MUST NOT** be used to: decide whether an online memory update is
admitted; initialize prototypes; select a patient-specific state during
deployment simulation; or trigger rollback in the operational policy.

Any experiment that violates this must be explicitly labelled an **ORACLE
MECHANISM STRESS TEST** and may never be presented as deployable behaviour.

Patient identity is never a predictive or gating feature. No test label, test
waveform, test prediction or test metric is accessed.

## 4. M2-G core admission gate (FROZEN)

A memory update at time *t* is admitted **only if all six conditions hold**.
Every condition is causal, past-only and label-free at runtime.

| # | Condition | Source |
|---|---|---|
| G1 | `observation_state == AVAILABLE` | schema-3 state (inherited from M1-v2) |
| G2 | fused `z_t` finite | representation (inherited) |
| G3 | waveform SQI admissible | `SIGNAL_V1` — §4.2 |
| G4 | `score_t <= NORMAL_EVIDENCE_THRESHOLD` | frozen M1L score — §4.3 |
| G5 | not currently in memory-update refractory | §5 |
| G6 | `morphology_valid == 1` | §4.4 |

### 4.2 G3 — waveform SQI (FROZEN)

Hard precondition: **`finite_sample_fraction == 1.0`**.

Then each of six `SIGNAL_V1` columns must be **at or below** its frozen TRAIN
upper bound, derived as `numpy.quantile(values, 0.99, method="linear")` over the
**full TRAIN timeline, physically AVAILABLE rows only**, with **no** target
family, ischemic/background, challenge or quality-label filtering, and **no**
validation or test rows.

| Column | Frozen Q99 upper bound |
|---|---|
| `flatline_fraction` | `0.4853941576630652` |
| `repeated_value_fraction` | `0.4853941576630652` |
| `derivative_outlier_fraction` | `0.12404961984793918` |
| `high_frequency_power_ratio` | `0.026922298961394597` |
| `powerline_ratio_50hz` | `0.0017282393761769012` |
| `powerline_ratio_60hz` | `0.0012844103306429878` |

`robust_amplitude_range_mv` and `robust_derivative_scale_mv_per_s` are
**deliberately excluded**: they vary legitimately with patient physiology, and
G3 screens artifact/noise rather than selecting a physiological phenotype.

**Recorded observation.** `flatline_fraction` and `repeated_value_fraction` are
**bitwise identical** in the frozen corpus, so the six declared columns impose
**five independent constraints**. Both are retained as declared; this is noted
so the gate is never described as six independent checks.

Combined G3 TRAIN rejection fraction: **0.038969**.

### 4.3 G4 — deterministic normal-evidence margin (FROZEN)

**The M1L classification threshold `0.7554003000259399` is NOT the
memory-admission threshold.** Choosing an operating point for *classification*
and deciding what is safe to *learn as normal* are different problems, and
admitting normal memory is deliberately the stricter of the two.

```
NORMAL_EVIDENCE_THRESHOLD = numpy.quantile(
    M1L_score_on_PRIMARY_TRAIN_background_negative, 0.50, method="linear")
```

**`NORMAL_EVIDENCE_THRESHOLD = 0.0002997174742631614`**

Derived over the 280,839 frozen PRIMARY TRAIN background-negative rows using the
frozen retained M1L model and the frozen M1-v2 naive TRAIN representation and
memory features. No retraining, no new memory replay, no validation scores, no
test scores. It is strictly below the classification threshold (margin
0.755101).

Runtime condition: `score_t <= NORMAL_EVIDENCE_THRESHOLD`.

This is a **DETERMINISTIC NORMAL-EVIDENCE MARGIN**. It is **not** a calibrated
probability, confidence, uncertainty or conformal score, and the underlying
score is never called a probability.

### 4.3.1 Label-derivation boundary

Runtime memory admission uses **no labels whatsoever** — the gate observes only
the frozen model score.

The G4 constant is derived **once, offline, prospectively** from frozen PRIMARY
TRAIN background-negative membership. That supervised TRAIN membership is
permitted **solely** for this global fixed development-time threshold
derivation. Validation and test labels never derive or alter it. Labels are
never operational gate inputs.

### 4.4 G6 — morphology computability admission (FROZEN)

**Included:** `morphology_valid == 1`.

Named **MORPHOLOGY COMPUTABILITY ADMISSION** — it is not SQI, not physical
availability, not normality and not uncertainty. When `morphology_valid == 0`
the 18-d physiology block rests substantially on the frozen TRAIN-median
imputation policy rather than a valid window-specific morphology estimate. Such
a representation may still be **scored** by the frozen classifier, but it must
not **move the patient prototype**.

Runtime: `morphology_valid == 0` → score still produced → memory update refused
→ **no refractory is triggered by this alone**. No morphology threshold other
than the frozen binary flag exists.

## 5. G5 — memory-update safety refractory (FROZEN)

**`REFRACTORY_DURATION_SECONDS = 60.0`**, measured in **real elapsed physical
time**, not in AVAILABLE-row counts and not in admitted updates. Window
availability time is `(start_sample + 2500) / 250.0` seconds.

When an AVAILABLE finite row is scored and `score_t > NORMAL_EVIDENCE_THRESHOLD`:

- the current row is **not** updated;
- the refractory is set or **re-armed**:
  `refractory_until = max(refractory_until, available_time_t + 60.0)`.

Any later row whose availability time is before `refractory_until` fails G5.
Scores are still computed for AVAILABLE finite rows during refractory, so
**sustained suspicious evidence keeps re-arming the freeze** without creating
episode semantics.

A row failing **only** SQI, morphology or physical availability does **not** by
itself start a refractory. If G4 also fails, G4 re-arms it.

> **THIS IS A MEMORY-UPDATE SAFETY REFRACTORY. IT IS NOT NORMAL/WATCH/EVENT/
> RECOVERY, NOT EPISODE REASONING AND NOT CLINICAL PERSISTENCE LOGIC.**

Rationale: 60 s is short relative to the retained 720-update (~60 minute)
long-memory half-life, while preventing immediate window-by-window re-entry
after suspicious evidence. **No validation tuning was used.**

## 5.1 Exact causal update order (FROZEN)

For every timeline row:

**A.** row becomes available → **B.** determine physical observation state.
If unavailable: no representation, no score, no update; real time still
advances, the refractory clock advances naturally, and **no new refractory is
triggered**.

If available: **C.** construct the frozen M1 representation `z_t` → **D.**
compute `d_long(t)` against the current prototype **before** any update →
**E.** compute the frozen M1L score from `[z_t ; d_long(t)]` → **F.**
independently evaluate G3, G4, G5 (prior state) and G6 → **G.** update the
prototype **only if G1–G6 all hold** → **H.** *after* deciding the current
row, if `score_t > NORMAL_EVIDENCE_THRESHOLD`, re-arm the refractory for
**future** rows.

No row can affect its own `d_long` or its own score. No future information is
used.

## 6. Rollback — excluded from the claim-bearing core (FROZEN)

**There is no M2-GR claim-bearing arm in M2-v1**, because no real
deployment-observable delayed contradictory or cloud confirmation signal exists
in this system.

Rollback is **DEFERRED FOR OPERATIONAL INTEGRATION**. An optional future
`M2-RB-ORACLE` may test snapshot/restore mechanics as **MECHANISM EVIDENCE
ONLY**: it does not participate in M2 retention, cannot support a deployable
rollback claim, and must not delay M2-G. It is **not** implemented here.

## 7. Prospective experiment design (FROZEN)

Two arms, both on the **frozen retained M1L**. The encoder, physiology
transform, representation, alphas and **head weights** are unchanged and **no
classifier is retrained** — M2 isolates UPDATE POLICY.

| Arm | Policy |
|---|---|
| **M2-0** | frozen M1L naive always-update control (reproduces M1-v2 behaviour) |
| **M2-G** | identical system with the frozen G1–G6 gate |

### 7.1 Primary contamination evidence — natural longitudinal stress

Arbitrary synthetic corruption severity is **not** part of core gate selection.
Primary evaluation uses **real frozen DEVELOPMENT longitudinal stress
intervals already present in the corpus**. Gate execution stays entirely
**label-blind**; annotations are used only *after* replay to define evaluation
strata:

1. annotated ischemic ST-event intervals;
2. rate-related challenge intervals;
3. axis-shift challenge intervals;
4. quality/noise intervals where the existing annotation source permits a
   reproducible evaluation definition;
5. conduction-change — **exploratory/descriptive only** (one-subject support).

These annotation states are **never** used to admit or refuse an update.

### 7.2 Prototype contamination metric (FROZEN)

For each eligible annotated stress interval and each policy, `mu_ref` is the
long-memory prototype immediately **before** the first stress window, using only
past causal history. At later time *t*, in the same standardized 146-d memory
space:

```
prototype_drift(t) = sqrt(mean((mu_long(t) - mu_ref) ** 2))
```

Report: peak drift during stress; mean drift during stress; drift at stress end;
residual drift at the first eligible point ≥ 5 minutes after stress end; and
≥ 30 minutes after stress end, where sufficient causal follow-up exists.

If no valid pre-stress prototype or follow-up exists, that interval is **excluded
from that specific statistic with a recorded reason**. Follow-up is never
fabricated. No tuned recovery threshold is defined.

### 7.3 Core evidence for both arms

Pooled and subject-macro validation AUPRC · AUROC · sensitivity · specificity ·
PPV · MCC · background FPR · subject-level FPR distribution · rate challenge FPR
· axis challenge FPR · conduction descriptive FPR · cold-start evidence ·
update-admission fraction · freeze fraction · SQI-refusal fraction ·
normal-evidence-refusal fraction · morphology-refusal fraction ·
refractory-refusal fraction · memory update count · time since last admitted
update · prototype drift evidence per §7.2.

## 8. Exit rule (prospective, FROZEN)

> **M2-G may be retained only if human bounded-Pareto review finds development
> evidence that it is materially safer than M2-0** — through reduced prototype
> contamination/drift and/or improved false-alarm behaviour — **while preserving
> event detectability without unacceptable sensitivity loss.**
>
> **The gate must additionally not collapse adaptation into a trivial
> never-update policy.** Admission and update coverage must be reported
> explicitly.

No weighted score. No automatic preference for M2-G. No significance claim
unless later statistics actually support one. **No test access.**

### 8.1 TRAIN-only sanity evidence (descriptive; no rule was altered by it)

| Quantity | TRAIN |
|---|---|
| physically AVAILABLE | 1.000000 |
| G3 SQI pass | 0.961031 |
| G4 normal-evidence pass (where a score exists) | 0.462278 |
| G6 `morphology_valid` | 0.999976 |
| combined pre-refractory admission | 0.439617 |
| **final M2-G update fraction after causal refractory replay** | **0.201222** |
| per-stream update fraction (132 streams) | min 0.000000, q10 0.001282, median 0.126352, q90 0.509003, max 0.819401 |
| cold-start update fraction 0–5 / 5–60 / >60 min | 0.100631 / 0.122234 / 0.204855 |

The gate does **not** collapse adaptation: about one in five timeline rows still
admits an update. One stream reaches a 0.000000 update fraction, which must be
reported in M2 results rather than smoothed over.

## 9. Known inherited limitation

M1's cold-start weakness (zero sensitivity in the 0–5 minute bin at the frozen
thresholds) is inherited by every M2 arm and is **not** addressed by this
protocol. Gating can only make early adaptation more conservative, so M2 should
be expected to leave cold-start behaviour unchanged or slightly worse, and must
report it rather than obscure it. No cold-start threshold change is authorized.

## 10. Implementation readiness

**Every scientific choice is closed.** The SQI columns and bounds, the
normal-evidence margin, `morphology_valid` inclusion, the refractory duration
and semantics, and the exclusion of rollback from the claim-bearing core are all
frozen above and bound by the derivation receipt.

**No unresolved scientific choice remains for implementing M2-G.**
Implementation is nonetheless **not yet authorized** and requires a separate
human decision.
