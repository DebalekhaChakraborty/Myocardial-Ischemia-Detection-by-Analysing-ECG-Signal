# T2 longitudinal temporal retention decision, V1

This is a **human decision record**, not a protocol and not a computation. It
retains one of the two frozen T2 candidates on the evidence the one-shot outer
VALIDATION already produced, and it freezes what that retention does and does
not authorize. No score, metric, threshold or selection is recomputed anywhere
in the change set that carries this file.

## RETAINED T2 ARM

**`causal_s4d_longitudinal_v1`**

### Role

Causal **longitudinal temporal evidence for the CURRENT ECG window**.

The retained object is the **continuous** score:

```
uncalibrated_temporal_model_score = sigmoid(current_window_t2_logit)
```

### What the retained arm is NOT

- not the primary ECG detector;
- not a calibrated probability;
- not a confidence;
- not an uncertainty;
- not NORMAL / WATCH / EVENT / RECOVERY;
- not an episode alarm;
- not a routing policy.

## Comparator

**`causal_gru_longitudinal_v1`** remains **immutable comparator / ablation
evidence**. Its checkpoint, threshold, row evidence and metrics are preserved
unchanged. No later phase may silently switch T2 from S4D to GRU.

## The frozen selection basis

The canonical selector decided this prospectively, under the rule frozen before
any outer number existed.

| | |
|---|---|
| GRU pooled outer PRIMARY AUPRC | `0.29486969381230116` |
| S4D pooled outer PRIMARY AUPRC | `0.388084635785268` |
| Absolute difference | `0.09321494197296681` |
| Frozen stage-1 boundary | `0.002` |
| Selection terminated at | **STAGE 1 — pooled PRIMARY outer-VALIDATION AUPRC** |
| `selection_basis` | `pooled_primary_validation_auprc` |
| Canonical selected arm | `causal_s4d_longitudinal_v1` |

Explicitly, and as recorded in the immutable outer artifacts:

- **TRAIN / internal-dev evidence was NOT a selection input.**
- **Challenge evidence was NOT a selection input.**
- **Latency was NOT a selection input.**
- **No weighted composite was used.**

The human decision agrees with the canonical selector. It does not substitute
for it, and it introduces no second selection procedure.

## Supporting evidence (descriptive only)

These did not select the arm and do not replace the frozen basis above.

| | GRU | S4D |
|---|---|---|
| Subject-macro AUPRC | `0.4097370037090087` | `0.4281524927446359` |
| Pooled AUROC | `0.9180555592455625` | `0.929154904230353` |

S4D at its frozen threshold: F1 `0.36312751443064545`, sensitivity
`0.2879600517847235`, specificity `0.9857474202299958`, PPV
`0.4913997159539214`, balanced accuracy `0.6368537360073596`, MCC
`0.35406330160806493`.

## Retention rationale

1. S4D was selected **prospectively** under the frozen stage-1 rule.
2. Its pooled outer AUPRC exceeds GRU by approximately `0.0932`, far beyond the
   `0.002` boundary.
3. Directional support is also present in subject-macro AUPRC, pooled AUROC,
   F1, sensitivity, specificity, PPV, balanced accuracy and MCC.
4. T2's scientific role is longitudinal temporal **evidence**, not system state.
5. Therefore the **continuous** S4D score is retained for downstream T1
   development, rather than its binary threshold being converted directly into
   system state.

## Temporal fragmentation

Recorded because it matters for how T1 must consume this evidence.

| | GRU | S4D |
|---|---|---|
| Positive prediction runs | `1081` | `1787` |
| Median positive-run duration | `25.0 s` | `10.0 s` |
| Isolated single-window positive fraction | `0.15911` | `0.49636` |
| Transitions | `2161` | `3571` |
| Transitions per hour | `3.15664` | `5.21627` |

**At their frozen thresholds, S4D predictions are temporally more fragmented
than GRU under these descriptive measures.**

That is the whole permitted claim. No post-hoc smoothing was applied, no
threshold was changed, and no run was repaired. These descriptors are not
episode statements: formal episode reasoning belongs to T1. This evidence is
precisely why T2 evidence and T1 deterministic state semantics remain separate —
a fragmented binary trace at a reporting threshold is not an argument against
the continuous score, and it is not something to fix by moving the threshold.

## Cold start

S4D sensitivity by recording age:

| stratum | sensitivity |
|---|---|
| 0–5 min | `0.0` |
| 5–60 min | `0.18673` |
| >60 min | `0.29797` |

No warmup repair, no new threshold and no alternative initial state was applied
or is authorized. **Future T1 must address cold-start semantics
prospectively**, before it consumes this evidence near the start of a stream.

## Challenge trade-offs

| subset | GRU | S4D |
|---|---|---|
| RATE | `944 / 4973`, FPR `0.18983` | `1022 / 4973`, FPR `0.20551` |
| AXIS | `87 / 3000`, FPR `0.029` | `55 / 3000`, FPR `0.018333` |
| CONDUCTION | `0` FP over `164` rows | `0` FP over `164` rows |

**S4D is worse than GRU on RATE false-positive rate and better on AXIS
false-positive rate.** Conduction support is sparse and its evidence is
descriptive / exploratory only.

No weighted challenge score is constructed, and challenge evidence does not
override the retention. It is recorded so the trade-off is visible rather than
averaged away.

## No significance claim

```
T2_RETENTION_STATISTICAL_SIGNIFICANCE_CLAIM = False
```

No prospective paired superiority procedure was frozen before the outer run, so
no significance claim is made and none is computed after the fact. The subject
bootstrap remains what it was defined to be: **between-subject variation
conditional on the fitted temporal model** — not uncertainty from independent
windows, and not a hypothesis test.

## Development evidence only

The outer VALIDATION remains **DEVELOPMENT evidence**. It is **not** unseen
generalization, **not** external validation and **not** independent clinical
validation. The sealed B4 TEST partition remains **unopened** and is untouched
by this decision.

## Row evidence retained for T1

The already-promoted per-row evidence store is bound by this decision. Its score
semantics are `uncalibrated_temporal_model_score`, and it records
`supports_t1_without_rerunning_outer_validation: true`.

**Future T1 development must consume the existing persisted S4D score stream.
It must NOT rerun T2 over VALIDATION.** Both arms' per-row streams are
preserved, so the comparator remains inspectable without re-execution.

## Threshold governance

```
T2_RETAINED_THRESHOLD_IS_T1_POLICY        = False
T2_RETAINED_THRESHOLD_MAY_SELECT_T1_STATE = False
```

The frozen S4D threshold `0.8972153067588806` remains immutable T2
experiment/reporting evidence **only**. It is valid for:

- T2 experiment metrics;
- challenge reporting;
- temporal descriptive evidence;
- future frozen ablation / reference reporting.

It does **not** automatically choose NORMAL, WATCH, EVENT or RECOVERY. Any T1
state rule is a separate, separately-authorized decision, and the fragmentation
evidence above is a direct warning against adopting this threshold as a state
rule unexamined.

## No rerun, no extended training

```
T2_RERUN_PERMITTED              = False
T2_EXTENDED_TRAINING_PERMITTED  = False
```

S4D's best internal-dev epoch was epoch 10, which was also the prospective
maximum. That the budget bound the best epoch is **not** authorization to extend
it: doing so after seeing the outer result would convert a prospective protocol
into a retrospectively tuned one. No epoch 11+, no new patience, no new seed, no
new learning rate, no new state size and no new checkpoint.

## T1 is not started

This decision **closes T2**. It defines no NORMAL / WATCH / EVENT / RECOVERY
threshold, no duration, no hysteresis, no onset or recovery confirmation and no
routing. T1 begins only after this decision is reviewed and merged.
