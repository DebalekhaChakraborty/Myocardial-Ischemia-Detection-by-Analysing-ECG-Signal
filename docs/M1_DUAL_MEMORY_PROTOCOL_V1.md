# M1 Dual-Timescale Causal Patient Memory Protocol V1

## 0. Status

This protocol is frozen **before any real M1 evidence exists**. Every execution
semantic below was fixed prospectively. No M1 metric, cache, arm claim or run
directory existed when this document was written.

### Revision record

| Digest | State |
|---|---|
| `52eedc628d906ac02619264fc26cd4629e56f05d6c1916448d62a2844c9815f4` | **SUPERSEDED BEFORE USE — NO M1 SCIENTIFIC EVIDENCE GENERATED UNDER THIS SHA** |

The superseded draft stated the recording-age formula as
`window_start_samples / sampling_rate`, which is absolute rather than
stream-relative and contradicted both the surrounding sentence and the
implementation. It also left the standard-deviation convention unstated and did
not define the subject-wise false-positive summary. All three are corrected
below. The correction is **prospective**: no M1 cache, arm claim, metric or run
directory existed under either digest.

> **THIS M1 UPDATE RULE IS INTENTIONALLY NOT CONTAMINATION-SAFE.**
>
> **AN ABNORMAL OR CONFOUNDED WINDOW MAY ENTER MEMORY.**
>
> **M2 IS REQUIRED BEFORE ANY SAFE-ADAPTATION OR DEPLOYMENT-SAFE
> PERSONALIZATION CLAIM.**

## 1. Scientific question

> Does strictly causal, past-only, patient-specific memory improve the
> **development** operating trade-off relative to the frozen global P1-B
> representation?

M1 studies the **memory mechanism**. M1 does **not** establish contamination
safety, uncertainty calibration, conformal validity, episode reasoning,
longitudinal SSM performance, cloud routing, or clinical diagnosis.

## 2. Bound upstream identities

M1 consumes these frozen artifacts and never refits or re-derives them.

| Artifact | SHA-256 |
|---|---|
| B4 protocol | `f6f5e9ed728c86a9b2bd75b2327b9199f0e097b91387525a192c212e6771b28b` |
| B4 split | `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7` |
| Feature corpus | `f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5` |
| B4-B checkpoint | `b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9` |
| B4-B experiment lock | `58e44a09ce3ebffecfcd49d957acfa368fc03b534fdcd990aedb9b6b0e9bda7b` |
| Frozen training selection | `318da148da5d638af44e73c06c00cc4df2815017d4ce8bb1a1b864e53eda8009` |
| P1 protocol | `f48ffc66e52649d74a8286182d5e7220f78abdd6c12a7ebfe04f116b853337f1` |
| P1 Stage-1 suite | `cc354ef64415d9c0dafcffdc0fdfa2446cd81a7d0c30fa9c58b0095cbc0be772` |
| P1-B lock (frozen global control) | `796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0` |
| P1 physiology transform | `cc6bd3a353f0ac6cad342114ed96e135cbf3c61e2946f847d5b95358b6bd51a9` |
| P1 physiology retention decision | `7b403709fa0fb12eef65423d830c121fc3ada904266a1b47931d438f5e797d68` |
| Challenge selection | `49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a` |
| Dependency digest | `b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a` |

## 3. Base representation (frozen)

The P1 physiology-retention decision retained the complete 18-dimension
`morphology_v1` vector. The M1 base representation is therefore:

```
z_t = [ frozen B4-B pooled embedding (128) ; frozen transformed morphology_v1 (18) ]
dim(z_t) = 146
```

The embedding tap is `B4BTransformerCNN.encode:pooled_post_final_norm`. The
encoder runs under `eval()`, `requires_grad_(False)` and `torch.no_grad()`, and
is never fine-tuned. The physiology transform is loaded from the frozen P1
artifact and is **never refitted** by M1. No second morphology implementation
exists.

## 4. Chronology contract (established by read-only audit)

| Property | Value |
|---|---|
| Causal order field | `window_start_samples` |
| Stream key | **`(record_id, channel_index)`** |
| Channels per record | 2 (`{0, 1}`) |
| Train population | 56 subjects / 60 records |
| Validation population | 12 subjects / 13 records |
| Cross-record acquisition chronology | **NOT AVAILABLE** |

Rows sharing a start sample on different channels are **simultaneous**, not
sequential. Both channels of one record must therefore **never** be merged into
a single sequential history; each `(record_id, channel_index)` pair is an
independent causal state unit.

`metadata_json.elapsed_seconds` is **feature-generation wall-clock** and MUST
NOT be used as acquisition chronology.

Because no cross-record acquisition chronology exists, **memory resets at every
`(record_id, channel_index)` boundary.** Subject ID is retained for provenance,
subject-wise metrics and namespace audit only, and is **never** a classifier
feature.

**Scientific wording.** M1-v1 is patient-adaptive **within a continuous
recording/lead stream**. M1 does not carry learned state across separate
recordings from the same subject.

## 5. Update policy: finite-observation always-update

The chronology audit established that no reviewed, deployment-observable LTSTDB
SQI admission rule is suitable for M1:

- EDB `quality_states` is a **different dataset contract**;
- LTSTDB `quality_excluded` is **annotation-derived** and therefore cannot gate
  memory without label leakage;
- `morphology_valid` is observable but is a **morphology-computability flag**,
  not a validated general-purpose signal-quality admission rule.

M1-v1 therefore uses **FINITE-OBSERVATION ALWAYS-UPDATE**. For every causal
stream observation whose fused representation `x_t` is finite:

1. compute memory deviations from state built only from windows `< t`;
2. expose those deviation features for the current window;
3. **after** feature construction, update both prototypes with `x_t`.

The update MUST NOT be gated on `morphology_valid`, the ischemic label, the
background label, the rate label, the axis label, the conduction label, the
model score, uncertainty, a threshold, WATCH/EVENT state, or any future
information.

If a fused representation is non-finite despite the frozen P1 transformation,
M1 **refuses** with an integrity error. It is never silently skipped.

## 6. Full development stream

Memory trajectories are generated over the **full causal development stream**,
not only the primary ischemic/background rows. For each
`(record_id, channel_index)`, rows are sorted ascending by
`window_start_samples`, and start samples are required to be strictly
increasing.

All causally representable development windows participate in the memory
trajectory **independent of their benchmark target family**.

Evaluation labels determine only supervised training membership, primary
validation membership, and challenge reporting strata. **Labels MUST NOT
determine whether a window exists in memory history.** A rate/axis/conduction
challenge row therefore affects later memory exactly as it would in a real
unlabelled stream.

This is intentional in M1. M2 later addresses unsafe contamination.

## 7. Memory distance standardizer

A **TRAIN-ONLY** global distance-space transform is frozen from the canonical
frozen primary TRAIN fused P1-B representation: **374,452 rows × 146
dimensions**.

It is **not** fitted on all full-stream rows. Rationale: this is a model
statistic and must stay tied to the prospectively defined supervised TRAIN
development population rather than to validation or challenge composition.

Per dimension `j`: `mean_j` and `std_j` are the TRAIN mean and standard
deviation; a zero-variance dimension takes `scale = 1`.

**Standard-deviation convention: NumPy population standard deviation,
`numpy.std(..., axis=0, ddof=0)`.** This is stated explicitly so the frozen
distance space cannot drift with a sample-variance reading.

```
x_t = (z_t - mean) / scale
```

Persisted as `M1_DISTANCE_STANDARDIZER.json` with 146 means, 146 scales, the
zero-variance dimension list, the fitted population, input identities and the
artifact SHA. No validation statistic and no patient-specific normalization
enters it.

Classifier heads consume the **original** `z_t` plus memory distances. Only the
memory distance uses standardized `x_t`.

## 8. Cold start

At each `(record_id, channel_index)` boundary both prototypes are initialized
to the **global TRAIN standardized prior** persisted in the standardizer
artifact. The exact persisted prior vector is used; hard-coded zeros are not
assumed even where numerically near zero.

The first window of a stream is compared against the global prior **before any
update**, and is updated afterwards. There is no future bootstrap interval and
no label-informed initialization.

`past_observed_count` and `past_update_count` are persisted for every row.
Under finite-observation always-update these ordinarily agree.

## 9. Dual-timescale prototypes

Frozen stride: **5 seconds**.

| Timescale | Half-life | Updates `H` | `alpha = 1 - 2^(-1/H)` |
|---|---|---|---|
| SHORT | 5 minutes | 60 | **`0.01148597964710385`** |
| LONG | 60 minutes | 720 | **`0.0009622411662165709`** |

`alpha_short > alpha_long > 0` is required and asserted. There is no sweep and
no tuning.

For an admitted finite `x_t`:

```
mu_short <- (1 - alpha_short) * mu_short + alpha_short * x_t
mu_long  <- (1 - alpha_long)  * mu_long  + alpha_long  * x_t
```

## 10. Score-before-update

For the current window `t`, `state_t` is the state after windows **strictly
earlier** than `t`. M1 computes

```
d_short(t) = sqrt(mean((x_t - mu_short)^2))
d_long(t)  = sqrt(mean((x_t - mu_long)^2))
```

records the M1 features, and **only afterwards** updates `mu_short` and
`mu_long` using `x_t`.

A window may never influence the prototype used to compute its own distance,
and no future row may affect any feature at `t`.

## 11. Memory features

M1-v1 uses exactly `d_short` and `d_long`.

No covariance, no Mahalanobis, no cosine-distance sweep, no prototype-angle
feature, no learned patient embedding, no patient-ID embedding, and no
short/long prototype disagreement as an **input** feature. Short/long
disagreement may be reported descriptively but is not a classifier feature in
M1-v1.

## 12. Comparators

Frozen global control: **`P1B_phys_fusion_v1`**, lock
`796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0`. No other
"global" model is retrained.

| Arm | Classifier input | Dim | Trainable parameters |
|---|---|---|---|
| `M1S_short_memory_v1` | `[z_t ; d_short]` | 147 | 9,537 |
| `M1L_long_memory_v1` | `[z_t ; d_long]` | 147 | 9,537 |
| `M1D_dual_memory_v1` | `[z_t ; d_short ; d_long]` | 148 | 9,601 |

Head family (identical across arms):
`Linear(d, 64) -> SiLU -> Dropout(0.10) -> Linear(64, 1)`. Parameter counts are
derived by code and asserted against this table. There is no architecture
sweep.

## 13. Training contract

Reused identically from P1 across M1S/M1L/M1D:

- seed **2026**; `BCEWithLogitsLoss(reduction=mean)`
- `AdamW`, lr `1e-3`, weight decay `1e-4`, betas `(0.9, 0.999)`, eps `1e-8`,
  `amsgrad=False`, `foreach=False`, `fused=False`
- batch size **256**, `drop_last=False`, max epochs **30**
- no scheduler, no augmentation, no class weighting, AMP off
- checkpoint: strict maximum **full primary validation AUPRC**, earliest exact
  tie
- early stopping: patience **4**, reference improvement `> 1e-6`
- checkpoint maximum and patience reference remain **separate** quantities
- threshold: selected checkpoint, full primary validation, maximum F1, highest
  observed score wins an exact tie

No calibration is performed. Scores are **uncalibrated model scores**, not
calibrated probabilities.

## 14. Primary / challenge membership

After full-stream causal memory features are materialized:

- M1 head training uses **only** frozen PRIMARY TRAIN membership;
- primary validation metrics and the threshold use **only** frozen PRIMARY
  VALIDATION membership;
- challenge rows are scored at their **actual causal stream positions**, using
  memory state produced only from prior full-stream observations.

Frozen challenge selection `49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a`:

| Family | Windows / subjects | Evidence weight |
|---|---|---|
| rate_related | 4,973 / 4 | quantitative_secondary |
| axis_shift | 3,000 / 8 | quantitative_secondary |
| conduction_change | 164 / 1 | **exploratory_descriptive only** |

## 15. Cold-start bins (frozen now)

Recording age is **stream-relative**: it is measured from the first available
window of the same `(record_id, channel_index)` stream, not from the absolute
start of the record.

```
recording_age_seconds =
    (window_start_samples - first_stream_start_sample) / sampling_rate
```

with `sampling_rate = 250.0 Hz`. The first window of every stream therefore has
`recording_age_seconds == 0.0` by construction. Bins:

- `0-5 minutes`
- `5-60 minutes`
- `>60 minutes`

These bins are **not** redefined after M1 metrics exist. They are supporting
evidence only.

## 16. Evidence and exit rule

Primary metric: **pooled validation AUPRC**. Supporting: AUROC, F1,
sensitivity, specificity, PPV, NPV, MCC, balanced accuracy. Subject-macro
handling follows existing repository convention.

Personalization evidence: primary-background FPR; subject-wise false-positive
distribution; sensitivity preservation; cold-start bins; rate FPR; axis FPR;
conduction descriptive FPR; finite representation rate; update counts;
descriptive short/long prototype disagreement.

### 16.1 Subject-wise false-positive evidence (frozen definition)

Computed on **PRIMARY VALIDATION only**, at the arm's **already selected**
threshold. It is **supporting evidence only**: it never selects a threshold,
never enters a weighted score, and is never used for tuning.

A false positive is a `background_negative` window scored at or above the
selected threshold. Only subjects with at least one background-negative window
("negative support") contribute.

```
pooled_background_fpr = FP(all background_negative rows) / N(background_negative rows)
subject_fpr[s]        = FP(background_negative rows of subject s) / N(background_negative rows of subject s)
```

Reported fields, over the contributing subjects' `subject_fpr` values sorted
ascending:

| Field | Definition |
|---|---|
| `pooled_background_negative_fpr` | as above |
| `background_negative_count` | pooled denominator |
| `contributing_subject_count` | subjects with negative support |
| `subject_fpr_median` | `numpy.median` |
| `subject_fpr_q25`, `subject_fpr_q75` | `numpy.quantile`, **linear** interpolation |
| `subject_fpr_iqr` | `q75 - q25` |
| `subject_fpr_p90` | `numpy.quantile(0.90)`, linear interpolation |
| `subject_fpr_max` | maximum |
| `subject_false_positive_rates` | exact per-subject values, keyed by subject ID |

The interpolation method is named because quantile conventions differ between
libraries and the frozen definition must be reproducible. Subject IDs appear
here as **reporting keys only**; they are never model inputs.

Comparison is across the frozen P1-B global control, M1S, M1L and M1D, by
**bounded Pareto judgement only**. There is **no weighted score**. Dual memory
is **not** automatically selected. No test partition is involved.

## 17. Stream cache integrity

The canonical M1 stream cache binds at minimum: the M1 protocol SHA; the P1
Stage-1 suite SHA; the P1-B lock; the B4-B checkpoint; the P1 physiology
transform SHA; the P1 train/validation embedding-cache SHAs; feature, source
and split identities; the distance-standardizer SHA; partition; full stream row
count; stream count; record IDs; channel indices; the ordered stable-ID digest;
the ordered `(record_id, channel_index, window_start_samples)` digest; the
representation-content digest; the `d_short` and `d_long` content digests; the
history-count digest; the update policy; `alpha_short`; `alpha_long`; the Git
SHA and clean state; the dependency digest; and `test_accessed: false`.

**Crash semantics.** A partial existing cache means **STOP / human review**. It
is never overwritten and never automatically repaired.

## 18. Run claim semantics

M1 follows the P1/B4 one-shot conventions. A canonical M1 run directory is an
**irreversible claim**: no force, no retry, no alternate seed, no selective
rerun. If execution fails after a claim, the arm records
`FAILED_OR_INTERRUPTED` with `human_review_required: true` and
`repeat_attempt_permitted: false`, and the claim is never released.

## 19. Execution route

`cardiosentinel m1 preflight` is read-only: it validates this protocol, the P1
suite / P1-B lock / physiology transform, chronology, source and feature
identities, inspects stream-cache state, M1 arm claims and test artifacts, and
enforces the scientific runtime and clean Git state. It creates **zero** models
and **zero** artifacts. An initial healthy status of
`stream_cache_materialization_required` is analogous to P1's initial cache
state and is **not** a failure.

`cardiosentinel m1 run-stage1` is the **only** canonical M1 Stage-1 route. It
validates all gates; constructs or reuses the immutable full-stream
representation and memory cache; validates it; then claims, runs and freezes
M1S, M1L and M1D in that order; then writes the combined M1 Stage-1 result.
There is **no single-arm public route**.

## 20. Test firewall

No test partition is accepted anywhere in M1. No sealed-test evaluator is
imported or invoked. `TEST_ATTEMPT.json` is never created. Every M1 artifact
records `test_accessed: false`.

## 21. M1 / M2 boundary

M1-v1 contains **no** rollback, **no** uncertainty admission, **no**
event-state admission, **no** conformal admission and **no** label-based update
gate. Those belong to M2 and are out of scope here.
