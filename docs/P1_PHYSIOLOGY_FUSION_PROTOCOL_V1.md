# P1 Physiology-Fusion Protocol V1

## 0. Revision note

A first draft (SHA-256
`f4ad68bac1474e0de920ac7e0b67dd3be108324452ef135889c6d1496260b20e`) froze the P1
scientific design correctly but described execution semantics the implementation
did not yet provide, and mis-stated the claim mechanism as an `O_EXCL` file
claim. It was **superseded before use** and produced **no scientific evidence**.
A second revision (`66e91c6cda73ac66c7dfddb2cf25a601af383ed8a84ba9f24dfed82519d8f256`)
described the execution path but preceded six execution-integrity corrections
found in re-review: the official `run-stage1` route was a stub, preflight could
report ready without caches, cache materialization accepted an arbitrary encoder
while stamping the frozen checkpoint SHA, partial caches were not blocked,
physiology was passed as bare arrays with no stable-ID binding, the challenge set
was accepted unverified, and checkpoint selection was conflated with the
early-stopping delta. It too was **superseded before use** and produced **no
scientific evidence**. This revision states the corrected path.

## 1. Scientific question

Does adding already-frozen explicit physiology/morphology proxy information to
the frozen selected B4-B neural representation improve the **development**
operating profile of short-window ischemia detection?

P1-v1 does **not** invent a clinical delineation algorithm and does not change
the neural backbone. It isolates the incremental value of explicit physiology.

## 2. Selected encoder (frozen, not retrained)

| | |
|---|---|
| Official model | `B4-B` |
| Experiment ID | `B4B_cnn_transformer_v1` |
| Architecture | `B4BTransformerCNN` |
| Experiment-lock SHA | `58e44a09ce3ebffecfcd49d957acfa368fc03b534fdcd990aedb9b6b0e9bda7b` |
| Checkpoint SHA | `b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9` |
| Locked validation threshold (B4-B's own) | `0.8329097628593445` |

### 2.1 Embedding tap

`B4BTransformerCNN.encode()` returns the pooled **128-d** representation: after
the final `LayerNorm` and `AdaptiveAvgPool1d(1)` over the 79 tokens, and
**before** the classifier MLP's first dropout site. `forward()` is defined in
terms of `encode()`, so they cannot drift; `forward()` is bitwise unchanged by
that extraction.

This tap is chosen over the head's internal 64-d activation because the latter
sits inside the ischemia-specific MLP after a dropout site and would bake that
head's task compression into every downstream reuse.

### 2.2 Frozen-encoder rule

B4-B is a **fixed representation extractor** for P1-v1: `eval()`,
`requires_grad_(False)`, `torch.no_grad()`. It is **not fine-tuned**. Its state
digest must be identical before and after any extraction. No B4-B weight may be
written.

## 3. Physiology source

`morphology_v1`, schema SHA
**`13f60be400b5b957c1eb592bbafd8206d4d2855c1aa657a058671fb8d7cab434`**,
18 features, in exactly this repository-defined order (columns 22–39 of
`COMBINED_V1`):

| # | Feature | Unit |
|---|---|---|
| 0 | `detected_r_peak_count` | count |
| 1 | `usable_beat_count` | count |
| 2 | `morphology_valid` | binary |
| 3 | `rr_mean_ms` | ms |
| 4 | `rr_median_ms` | ms |
| 5 | `rr_std_ms` | ms |
| 6 | `rr_cv` | ratio |
| 7 | `estimated_hr_bpm` | beats/min |
| 8 | `pre_r_baseline_median_mv` | mV |
| 9 | `qrs_proxy_peak_to_peak_mv` | mV |
| 10 | `post_r_80ms_delta_mv` | mV |
| 11 | `post_r_120ms_delta_mv` | mV |
| 12 | `post_r_160ms_delta_mv` | mV |
| 13 | `post_r_200ms_delta_mv` | mV |
| 14 | `post_r_80_160_slope_mv_per_s` | mV/s |
| 15 | `post_r_80_200_area_mv_s` | mV·s |
| 16 | `beat_template_correlation_median` | correlation |
| 17 | `beat_template_variability` | mV |

**These are R-aligned algorithmic waveform proxies.** They are **not** validated
J-point measurements, not validated ST-segment delineation, not clinically
validated morphology, and not diagnostic measurements. No feature is renamed,
recomputed or redefined by this protocol.

### 3.1 Frozen feature groups

Group names are **engineering/scientific proxy groupings, not validated clinical
physiological measurements**.

| Group | Features |
|---|---|
| `detection_quality` | 0, 1, 2 |
| `rr_rhythm` | 3, 4, 5, 6, 7 |
| `qrs_amplitude` | 8, 9 |
| `st_t_proxy` | 10, 11, 12, 13, 14, 15 |
| `beat_template` | 16, 17 |

Complete (18 of 18) and non-overlapping. Frozen **before** any P1 metric exists.

## 4. Missingness and validity — frozen rule

The production extractor initialises all morphology descriptors as `NaN` and
returns most of them `NaN` when fewer than two usable beats exist;
`beat_template_correlation_median` can additionally be `NaN` in other
circumstances. **`morphology_valid` alone therefore does not resolve
missingness, and imputation is required.**

Train-only audit over the frozen B4 training selection (374,452 windows):

| Observation | Count |
|---|---:|
| `morphology_valid == 0` (15 continuous features `NaN`) | 4 |
| `morphology_valid == 1` with an isolated `NaN` (`beat_template_correlation_median`) | 1 |
| Rows with any non-finite morphology value | 5 (0.0013%) |
| Features with zero finite support | 0 |
| Features with zero variance over finite train values | 0 |

**Frozen rule:**

1. `morphology_valid` is **retained as an explicit reliability feature**; it is
   never imputed and never dropped.
2. Every non-finite value in any other feature is replaced by that feature's
   **median over finite TRAIN values only**.
3. Statistics are estimated on the frozen B4 train selection **only**. Validation
   and challenge rows only *apply* them.
4. **No row is dropped** for missing physiology.
5. Zero finite train support for a feature is a **refusal**, not a silent fill.
6. Zero train variance yields a frozen constant-zero output column after
   centering (the feature carries no information; it is retained for width and
   order stability rather than dropped).
7. Imputed counts are recorded per feature and in total.

**No additional per-feature missingness indicators are introduced.** With 5
affected rows in 374,452 (0.0013%), such indicators would be near-constant
columns carrying essentially no information, and `morphology_valid` already
covers 4 of the 5 cases explicitly. This choice is frozen here, prospectively,
before any P1 metric exists.

## 5. Normalization — train only

Per-feature standardisation `(x − mean) / std`, with mean and std computed on
the frozen train selection **after** imputation. Zero std yields a constant-zero
column (§4.6). Validation and challenge rows only apply the frozen transform.

No validation-derived statistics. No challenge-derived statistics. No
subject-specific normalization — patient-specific normalization belongs to
M1/M2 if it is ever justified.

The fitted transform is a persisted artifact binding: the 18-feature order, the
schema SHA, the training population identity, imputation statistics,
normalization statistics, the zero-variance and missingness policies, and its
own digest.

**Transformed physiology dimension: 18.**

## 6. Models

Both consume the **frozen** 128-d B4-B embedding. B4-B is never fine-tuned.

| | Input | Dimension |
|---|---|---|
| **P1-A** (neural-only control) | embedding | 128 |
| **P1-B** (neural + physiology) | `[embedding ; transformed physiology]` | 146 |

### 6.1 Matched head

Identical family and capacity logic for both; only the input width differs:

```
Linear(d, 64) -> SiLU -> Dropout(0.10) -> Linear(64, 1)
```

Parameter counts: **P1-A 8,321** · **P1-B 9,473**.

No attention, no SSM, no mixture of experts, no architecture sweep, no
independent tuning of the two arms.

P1-A is the **matched control** for P1-B. The historical end-to-end B4-B metrics
are useful context but are **not** the matched control, because the training
procedure differs.

## 7. Common training contract

One procedure, applied identically to P1-A and P1-B. Every deterministic detail
is frozen here before any P1 metric exists:

| | |
|---|---|
| seed | 2026 |
| loss | `BCEWithLogitsLoss(reduction=mean)` |
| optimizer | AdamW, lr 1e-3, weight decay 1e-4, betas (0.9, 0.999), eps 1e-8 |
| batch size | 256, `drop_last=False` |
| max epochs | 30 |
| scheduler / augmentation / class weighting / AMP | none |
| checkpoint selection | maximum full primary-validation AUPRC; earliest epoch wins an exact tie |
| early stopping | 4 completed epochs without improvement > 1e-6 |

Additional frozen determinism:

| | |
|---|---|
| `amsgrad` / `foreach` / `fused` | false / false / false |
| Head initialization | standard PyTorch `nn.Linear` initialization under `torch.manual_seed(2026)` applied **immediately before** head construction, so nothing may consume RNG in between |
| Train shuffle | `torch.randperm` under a generator seeded `2026 + epoch`; deterministic and epoch-dependent |
| Inference batching | 256, `torch.no_grad()`, `eval()` |

Max epochs is 30 rather than B4's 15: this is a small head over precomputed
embeddings, so an epoch is far cheaper and the frozen early-stopping rule, not
the ceiling, is expected to terminate training. No hyperparameter sweep, no
alternate seed after observing P1, no validation-driven tweaking, no arm-specific
tuning.

### 7.1 Numerical integrity

Training aborts without repair on a non-finite mean training loss, a non-finite
prediction, a non-finite head parameter or a non-finite validation metric. There
is no restart, retry, learning-rate reduction or alternate seed.

## 8. Populations

| Partition | Windows | Positive | Negative | Subjects |
|---|---:|---:|---:|---:|
| Train (selection SHA `318da148…a009`) | 374,452 | 93,613 | 280,839 | 56 |
| Primary validation | 473,897 | 21,628 | 452,269 | 12 |

Binary primary ischemia/background only. Label semantics unchanged. Rate, axis
and conduction labels are **never** training supervision.

## 9. Threshold

Selected on the **full primary validation** partition from the selected
checkpoint: maximum F1 over exact observed scores, highest threshold winning an
exact tie. Each arm selects its own. Scores are **uncalibrated sigmoid model
scores**, not calibrated probabilities.

## 10. Metrics

**Primary:** pooled validation AUPRC.

**Supporting:** subject-macro AUPRC, AUROC, F1, sensitivity, specificity, PPV,
MCC.

**Challenge robustness**, over the frozen validation challenge population
(selection SHA `49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a`):

| Family | Windows | Subjects | Status |
|---|---:|---:|---|
| `rate_related_confounder` | 4,973 | 4 | quantitative_secondary |
| `axis_shift_confounder` | 3,000 | 8 | quantitative_secondary |
| `conduction_change_confounder` | 164 | 1 | **exploratory_descriptive only** |

Challenge physiology is read from the **frozen** `morphology_v1` values by stable
ID. No second morphology implementation is computed. Conduction is never
bootstrapped and never headline.

Also recorded: head parameter count, serialized head size, and any measurable
inference overhead attributable to fusion.

## 11. Retention rule — frozen before results

P1 physiology fusion **may be retained** if it produces credible incremental
value in at least one claim-relevant dimension — pooled AUPRC, subject-macro
behaviour, rate-related FPR, axis-shift FPR, or later interpretability/evidence
value — **without unacceptable degradation in the others**.

It must **not** be retained merely because explicit features "look
interpretable", and must **not** be retained for a tiny numerical gain that
carries a material robustness or deployment penalty.

The decision is a bounded Pareto/evidence judgement. **No weighted scalar score
may be defined.** No test evidence may participate.

## 12. Feature-group ablation authorization

**Stage P1-1 only:** neural-only (P1-A) versus all frozen morphology_v1
physiology (P1-B).

Predeclared feature-group ablations are authorized **only if** stage P1-1 shows a
retained-value signal. This prevents multiple-comparison fishing. The groups are
nevertheless frozen now (§3.1), before any result exists.

## 13. Partitions and test prohibition

Permitted: `train`, `validation`. **`test` is prohibited on every P1 path** —
embedding cache creation, transform fitting, transform application, head
training, validation metrics, challenge evaluation and selection. No sealed-test
import, no `TEST_ATTEMPT`, no test cache, metadata or waveform.

Historical B0–B3 test evidence is closed and may not inform P1 design.

## 14. Execution path, provenance and one-shot semantics

Experiment IDs `P1A_neural_head_v1` and `P1B_phys_fusion_v1`, run root
`cardiosentinel-runs/phase4-p1-physiology-v1`.

### 14.1 Embedding cache

Train and validation embeddings are materialized once with the frozen B4-B
checkpoint and are the only input to head training; the encoder never executes
inside training. Each cache binds, and re-verifies on load:

- an **order-sensitive** stable-ID digest (a sorted digest alone would not detect
  a row-order change, which would silently misalign labels and physiology);
- an embedding **content** digest over shape, dtype and contiguous bytes;
- a label content digest;
- the persisted artifact's file SHA-256;
- the exact population (train 374,452 / 93,613 / 280,839 / 56; validation
  473,897 / 21,628 / 452,269 / 12);
- the B4-B experiment-lock and checkpoint SHAs, embedding tap, dimension 128;
- split, feature-corpus and training-selection identities;
- Git SHA with `git_dirty: false`, runtime environment and dependency digest;
- this protocol's SHA.

An existing canonical cache is refused, never overwritten. Any altered row, row
order, embedding value, artifact or bound identity fails validation.

### 14.2 Claim semantics

The canonical experiment **directory** is the claim: it is created with
`mkdir(exist_ok=False)` and its parent is fsynced. That atomic directory
creation is the irreversible claim — this is not an `O_EXCL` file claim. An
existing directory **in any state** consumes and blocks the attempt. It is never
deleted, reset or renamed, and there is no force, overwrite, retry,
rerun-candidate or selective-arm path.

On any post-claim exception a `FAILED_OR_INTERRUPTED` receipt is written with
`error_type`, `error`, `traceback`, `human_review_required: true` and
`repeat_attempt_permitted: false`. **Human review is required.**

### 14.3 Gates before any claim

The exact scientific runtime (Python 3.12.6, PyTorch 2.13.0+cpu, NumPy 2.3.2,
scikit-learn 1.9.0, SciPy 1.18.0, WFDB 4.3.1, CPU, AMP off, dependency digest
`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`) and a clean
Git checkout are verified **before** any attempt is claimed, reusing the already
reviewed environment gate. Nothing is installed or repaired automatically.

### 14.4 Artifacts and lock

Each completed arm writes `RUN_STATUS.json`, `RUN_MANIFEST.json`,
`EPOCH_HISTORY.json`, `PHYSIOLOGY_TRANSFORM.json` (explicit null for P1-A),
`VALIDATION_METRICS.json`, `VALIDATION_THRESHOLD.json`,
`VALIDATION_PREDICTIONS.npz`, `CHALLENGE_METRICS.json`, `model_selected.pt`,
`training_checkpoint.pt` and `EXPERIMENT_LOCK.json`.

The lock binds the experiment ID, head architecture/input dimension/parameter
count/payload, the B4-B lock and checkpoint SHAs with `encoder_fine_tuned:
false`, both embedding-cache digests and ordered-ID digests, the physiology
transform digest for P1-B, this protocol SHA, Git SHA and clean state, the
environment dependency digest, split/corpus/training-selection identities,
populations, selected epoch, selected validation AUPRC, locked threshold, epoch
history digest, artifact hashes, challenge evidence identity, and `test: null`.
A lock validator re-derives the canonical digest and re-checks the bound
checkpoint, so no self-consistent altered artifact validates silently.

### 14.4a Corrected execution integrity

- **Encoder provenance.** Cache materialization loads the canonical B4-B model
  through the reviewed locked-model loader, verifying the experiment lock, the
  checkpoint SHA-256, `test: null` and the constructed architecture before use.
  An arbitrary preconstructed encoder can never be stamped as official.
- **Crash safety.** Any pre-existing canonical cache directory — complete or
  partial, manifest or not — blocks materialization and requires human review or
  explicit read-only validation. The array is written to a temporary file and
  atomically published; `p1_embeddings.npz` is never silently overwritten.
- **Physiology binding.** Physiology is carried as a bundle holding the ordered
  stable IDs, content digest, schema SHA and transform digest, joined from the
  frozen corpus **by stable ID in the cache's exact row order**. A row
  permutation or a partition mismatch is refused.
- **Challenge identity.** The challenge population is **rebuilt** through the
  reviewed B4 validation-challenge index and must reproduce rate 4,973/4, axis
  3,000/8, conduction 164/1, total 8,137 and selection digest
  `49899d1b…e72a`. It is a **mandatory** keyword argument of the Stage-1 suite:
  there is no `challenge=None` path.
- **Cache load.** Validation requires the exact frozen population, the training
  selection SHA for train, split/corpus/protocol identities, the frozen
  dependency digest, encoder identity, and the ordered stable-ID, embedding,
  label **and subject-ID** content digests. Subject IDs are bound because
  subject-macro metrics depend on them.
- **Artifact locking.** The lock binds a file SHA-256 for every claim-bearing
  artifact — epoch history, physiology transform, validation metrics and
  threshold, validation predictions, challenge metrics, both checkpoints and the
  run manifest — and the validator re-checks each one.

### 14.4b Checkpoint selection versus patience

These are **separate**, following the reviewed `CheckpointTracker`:

- a checkpoint is saved whenever `validation_auprc > best_auprc` (strict
  numerical maximum, no delta);
- patience resets **only** when
  `validation_auprc > early_stopping_reference + 1e-6`.

An improvement of, say, 5e-7 therefore becomes the selected maximum checkpoint
**without** resetting patience. The earliest epoch wins an exact tie.

### 14.5 Stage-1 suite

One official controller runs **both** arms in the frozen order P1-A then P1-B and
writes a combined `P1_STAGE1_RESULTS.json`. There is deliberately **no official
route that runs a single arm** and calls Stage P1-1 complete, and no selective
retry. The suite records `physiology_retained: null` and
`retention_decision_performed: false`: **it never retains or rejects
physiology** — that is the later human decision of §11.

### 14.6 Entry points

`cardiosentinel p1 preflight` is read-only: it validates the protocol, the B4-B
identities, the runtime and Git state, the expected populations and schema, and
reports whether the canonical attempts and caches exist. It constructs no
scientific model and creates no artifact.

`cardiosentinel p1 run-stage1` is the only official Stage P1-1 route.

## 15. Next gate

After this protocol and its implementation are reviewed and merged, the next
step is the **one canonical P1-A vs P1-B train/validation development run**.
Physiology retention, any feature-group ablation, M1 patient memory and sealed
test access are all separate, separately authorised decisions.
