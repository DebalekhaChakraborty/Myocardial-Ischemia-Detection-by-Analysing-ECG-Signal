# M2 Contamination-Safe Continual Adaptation Protocol V1

> **STATUS: PROPOSED — HUMAN REVIEW REQUIRED.**
> No M2 scientific execution has occurred. No M2 result exists. Sections marked
> **OPEN** contain unresolved scientific choices and must be closed by human
> decision before any M2 implementation or run is authorized.

## 0. Purpose and scope

M1 established that patient-specific memory helps, and that M1-v2 is
**explicitly not contamination-safe**: any available finite observation updates
the prototypes, so a developing ischemic event, a severe artifact or a
confounded window can be learned as the patient's new normal.

**M2 exists to prevent that.** It isolates the **UPDATE POLICY** and nothing
else.

M2 **does not** reopen memory-family selection. `M1L_long_memory_v2` is frozen
and retained by
`docs/M1_MEMORY_RETENTION_DECISION_V1.md`
(`45b29cd83ecfc60b43639be5569075a9cf561650f58a9812ade3051467f11b51`).
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
Whether to include it is **OPEN** (§4.4).

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
| `morphology_valid` | **AVAILABLE NOW**, inclusion **OPEN** | computability, not quality |
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

## 4. Proposed M2 core admission gate

A memory update at time *t* is admitted **only if all** of the following hold.
Every condition is causal, past-only and label-free.

| # | Condition | Source | Status |
|---|---|---|---|
| G1 | `observation_state == AVAILABLE` | schema-3 state | **FROZEN** (inherited from M1-v2) |
| G2 | fused `z_t` finite | representation | **FROZEN** (inherited) |
| G3 | waveform SQI admissible | `SIGNAL_V1` | **OPEN** — §4.2 |
| G4 | normal-evidence margin satisfied | M1L score | **OPEN** — §4.3 |
| G5 | not in memory-update refractory | causal state | **OPEN** — §5 |
| G6 | `morphology_valid == 1` | `MORPHOLOGY_V1` | **OPEN** — §4.4, may be excluded |

G1 and G2 are already frozen M1-v2 behaviour. **G3–G6 are the new content and
none may be finalized without the derivations below.**

### 4.2 SQI gate derivation rule (OPEN)

No arbitrary "good SQI" constant is permitted. Any threshold must state its
source feature, physical meaning, and a **prospective TRAIN-only** derivation.

Proposed form — to be frozen before any M2 result exists:

> For a chosen subset of `SIGNAL_V1` columns, compute the distribution over the
> **frozen primary TRAIN population only** and admit a window when each selected
> column lies within a fixed, prospectively declared quantile bound of that
> TRAIN distribution.

**Unresolved:** which columns; one-sided or two-sided; the exact quantile. These
must be chosen on physical reasoning and TRAIN distribution shape, **never** by
sweeping M2 validation outcomes. Quality *annotations* remain evaluation
evidence only and are not gate inputs.

### 4.3 Normal-evidence gate derivation rule (OPEN)

**The M1 validation-selected F1 threshold must NOT be reused as the
memory-admission threshold.** They are conceptually different objects: one
chooses an operating point for *classification*, the other decides what is safe
to *learn as normal*. Admitting normal memory should be strictly more
conservative than declaring a window negative.

Proposed form:

> Derive a conservative normal-admission threshold from the **frozen TRAIN**
> score distribution of the retained M1L model — for example a low quantile of
> the TRAIN background-negative score distribution, or a fixed margin below the
> classification threshold — declared prospectively.

**Unresolved:** the exact quantile or margin. **No threshold sweep using M2
validation outcomes is permitted.**

Because no calibrated uncertainty exists, this is a **deterministic
normal-evidence margin**, not an uncertainty gate, and must be named that way.
The score is never called a probability.

### 4.4 `morphology_valid` (OPEN)

Include as G6 or not. Argument for: a window whose 18-d physiology is entirely
train-median imputation is a weak basis for moving a patient prototype.
Argument against: it is a computability flag, its failure correlates with
genuinely abnormal rhythm, and gating on it could systematically refuse to learn
during exactly the rhythms a patient-specific baseline should represent. This is
a real scientific trade-off and is left to human decision.

## 5. Memory-update refractory / suspicion freeze (OPEN)

Purpose: prevent immediate re-entry into memory updating right after a
suspicious observation, so a developing event cannot be absorbed window by
window as its score drifts back under the margin.

Explicitly named a **memory-update refractory/freeze state**. It is **not**
WATCH/EVENT episode reasoning, carries no clinical semantics, and does not
anticipate T1.

Proposed form: after a window fails G4, refuse updates for a fixed number of
subsequent windows in the same `(record_id, channel_index)` stream, prospectively
declared and **never tuned on validation**.

**Unresolved:** the duration. Rationale for any choice must be argued from the
M1 memory time constants — the short half-life is 60 updates (~5 min) and the
long half-life 720 updates (~60 min) at the 5 s stride — not from M2 outcomes.

## 6. Rollback inventory and semantics (OPEN)

**Finding: no real delayed contradictory or cloud signal exists in this system.**
There is no channel that could operationally trigger rollback in deployment.

Therefore M2-v1 **must not claim deployable automatic rollback.** What may
legitimately be built and evaluated:

- **snapshot mechanics** — periodic immutable prototype snapshots with a
  deterministic restore API;
- **a deterministic rollback API** operating on those snapshots;
- **a synthetic/oracle contamination-recovery stress test** measuring whether
  rollback *can* restore a contaminated prototype.

Oracle-triggered rollback is **MECHANISM EVIDENCE ONLY** and must be labelled as
such wherever reported. Per the exit rule, rollback is retained only if it adds
recovery value beyond gating alone.

## 7. Prospective experiment design

Three arms, all built on the **frozen retained M1L** — the encoder, physiology
transform, representation, alphas and head are unchanged. Only the update policy
differs.

| Arm | Policy |
|---|---|
| **M2-0** | frozen M1L naive always-update control (reproduces M1-v2 behaviour) |
| **M2-G** | gated adaptation (G1–G5, plus G6 if adopted) |
| **M2-GR** | gated + rollback — **conditional** on §6 resolving to legitimate semantics |

### Evidence to report for every arm

Prototype displacement/drift · recovery behaviour · pooled and subject-macro
AUPRC · sensitivity preservation · background FPR · subject-wise FPR
distribution · rate FPR · axis FPR · conduction descriptive evidence ·
**update-admission fraction** · **freeze fraction** · **memory update count** ·
**time since last admitted update** · cold-start behaviour by the frozen bins.

### Controlled contamination injections

Artifact/noise corruption · rate-related stress · axis-shift stress · event-like
sustained abnormal segments.

**Injection rules must be defined prospectively.** Contamination severity must
**not** be optimized against validation outcomes. Labels may identify evaluation
or stress regions but **never** gate updates.

## 8. Exit rule (prospective)

> **M2-G is retained only if it is materially safer than M1L naive
> always-update**, demonstrated by reduced prototype contamination/drift and/or
> an improved false-alarm versus event-detectability trade-off, **without
> unacceptable sensitivity loss**.
>
> **M2-GR is retained only if rollback provides additional recovery value beyond
> M2-G.**

No weighted score. No automatic preference for the more complex policy. Bounded
Pareto judgement by a human. **No test partition is involved at any point.**

## 9. Known inherited limitation

M1's cold-start weakness (zero sensitivity in the 0–5 minute bin at the frozen
thresholds) is inherited by every M2 arm and is **not** addressed by this
protocol. Gating can only make early adaptation more conservative, so M2 should
be expected to leave cold-start behaviour unchanged or slightly worse, and must
report it rather than obscure it. No cold-start threshold change is authorized.

## 10. What must be closed before implementation

1. §4.2 SQI columns, bound direction and quantile.
2. §4.3 normal-evidence margin derivation.
3. §4.4 whether `morphology_valid` enters the gate.
4. §5 refractory duration and its rationale.
5. §6 whether rollback proceeds as mechanism-only evidence.

**No M2 implementation code should be written until these are frozen**, because
each is a scientific choice that would otherwise be made implicitly by whoever
writes the code first.
