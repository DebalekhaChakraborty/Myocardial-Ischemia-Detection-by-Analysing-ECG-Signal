# B4 Validation Challenge Evidence Protocol V1

## 0. Revision note

An earlier draft of this procedure specified a prediction-only design. That was
**scientifically invalid**: the locked prediction artifacts contain primary rows
only (§3). The error was found in review **before any real challenge evidence
was generated**, so the procedure was corrected prospectively. The superseded
draft (SHA-256 `5478c46fe5013d1d893f1f134d35af68ec3007105c542a2115718c0431858692`)
produced no scientific evidence and must not be cited as a frozen protocol.

A second revision (`4ab7e2e6adcf1e4d4e88a4ca5114515abc917aef6c1927ee52aaf79b16a81be1`)
corrected the design but still carried stale prediction-only language in §5, §11
and §13. It too was **superseded before use** and produced no scientific
evidence. Every superseded SHA in this history was replaced before any real
challenge evidence existed.

## 1. Purpose

This document freezes the **execution procedure** for producing the missing
validation-partition challenge evidence for the three Phase 3B-2 global encoder
candidates (B4-A, B4-B, B4-C) before the architecture-selection gate.

This protocol is an execution procedure only. It does **not** define, redefine or
extend any metric. The scientific meaning of every quantity reported here is
already fixed by:

- `docs/BENCHMARK_PROTOCOL_V1.md`
- `docs/METRICS_PROTOCOL.md`
- `docs/B4_PROTOCOL_V1.md`
- `docs/B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.md`

Those documents outrank this one. Where this document appears to say anything
different about a metric's meaning, they win and this procedure is wrong.

## 2. Scope: development only

The evidence produced under this protocol is **development validation evidence**.

It is not test evidence, it is not a headline metric, and it does not constitute
an architecture-selection decision.

The B4 sealed test is out of scope in every respect. See §10.

## 3. Why this is not a prediction-only procedure

The canonical B4 `validation_predictions.npz` artifacts contain **primary
validation rows only**. `build_validation_index` calls `load_b4_references`
with its `primary_only=True` default, which drops every family outside
`PRIMARY_FAMILIES`, and both runners persist predictions from
`prepared.indexes["validation"]`. The three locked artifacts therefore contain
exactly `ischemic_positive` (21,628) and `background_negative` (452,269) and
**no confounder rows at all**.

Negative challenge evidence consequently **cannot** be derived from those files.
It requires scoring the frozen challenge rows with each already-locked model.
Positive-context descriptives are the exception: those ischemic-positive rows,
with their context flags, are already present in the locked primary predictions
and are read from there.

## 4. Inputs

Exactly three locked candidate runs, in the frozen order below.

| Order | Official model | Experiment | Architecture |
|---|---|---|---|
| 1 | `B4-A` | `B4_raw_compact_cnn_v1` | `B4CompactCNN` |
| 2 | `B4-B` | `B4B_cnn_transformer_v1` | `B4BTransformerCNN` |
| 3 | `B4-C` | `B4C_cnn_ssm_v1` | `B4CSSMCNN` |

For each candidate the evaluator consumes:

1. `EXPERIMENT_LOCK.json` — the immutable development lock;
2. `model_selected.pt` — the locked inference model, **for inference only**;
3. `validation_predictions.npz` — the locked primary predictions, used **only**
   for positive-context descriptives;
4. the frozen validation metadata and the validated waveform source, for the
   challenge rows.

### 4.1 Frozen validation challenge population

Rebuilt from validation metadata via
`load_b4_references(feature_root, "validation", primary_only=False)`, keeping
only the three confounder families. `quality_excluded`, `boundary_ambiguous` and
`source_censored_or_unknown` are **excluded**.

| Family | Windows | Subjects |
|---|---:|---:|
| `rate_related_confounder` | 4,973 | 4 |
| `axis_shift_confounder` | 3,000 | 8 |
| `conduction_change_confounder` | 164 | 1 |
| **Total** | **8,137** | 9 distinct |

Window counts are Benchmark V1; subject counts are the denominators already
recorded in the frozen B0-B3 `challenge_metrics_validation.json` evidence.

Canonical sorted stable-ID selection digest:
`49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a`

Conduction is supported by a **single subject**, which is the standing reason it
is exploratory-only (§8.3).

### 4.2 Waveform contract

Challenge windows are read through the existing validated physical path
(`B4WaveformDataset.read_waveform`): single channel, 10 s, 250 Hz, 2,500
samples, physical mV, float32 model input, raw identity, no filtering, no
normalisation, no handcrafted features. `B4WindowReference.binary_label` is
**not** used for challenge rows — it is defined for primary families only and
that invariant is preserved.

### 4.3 Locked model inference

`model_selected.pt` only. No optimizer, no training checkpoint. The model is
`eval()` with `requires_grad_(False)` and every forward runs under
`torch.no_grad()`. The complete model state is digested before and after
inference and the two digests must match, so inference cannot silently mutate a
locked model. No backward pass, augmentation, calibration fitting or threshold
search exists on this path.

## 5. Partition and its two subsets

Everything here lives inside the frozen **development validation** split. Two
disjoint subsets of that split are used for different purposes.

**Primary validation subset** — `ischemic_positive` + `background_negative`,
473,897 windows over 12 subjects, recorded in each lock as `validation_rows`.
This is the subset that produced checkpoint selection and threshold selection,
and it is the source of the positive-context descriptives (§10).

**Validation challenge subset** — the prospectively frozen
`rate_related_confounder`, `axis_shift_confounder` and
`conduction_change_confounder` rows (§4.1). These sit in the same validation
split but outside the primary task subset, and are used only for secondary
architecture-selection evidence.

Both subsets are validation. Neither is test. The `train` partition is not
evaluated and the `test` partition is prohibited (§6).

## 6. Test prohibition

The evaluator must not, on any code path:

- import or call the sealed-test module;
- evaluate, resolve or name the `test` partition;
- read a test prediction, label, cache, waveform or metadata file;
- create, read or reference `TEST_ATTEMPT.json`;
- produce any test metric.

A partition value of `test` entering any public execution path is a hard error.
The challenge index is built from validation metadata only.

## 7. Artifact integrity

Before any quantity is computed, the evaluator must verify:

1. the candidate lock re-derives its own recorded `experiment_lock_sha256`;
2. `test` in the lock is `null`;
3. the lock's `status` is `locked_for_one_shot_test`;
4. the lock's `experiment_id` and architecture match the requested official model;
5. `validation_predictions.npz` hashes exactly to the lock's
   `validation_predictions_sha256`;
6. the prediction arrays are mutually length-aligned and non-empty;
7. `stable_id` values are unique;
8. every `score` and `label` is finite; labels are exactly `{0, 1}`;
9. the prediction row count, positive count, negative count **and distinct
   subject count** equal the lock's `validation_rows`;
10. the primary population equals the frozen canonical identity — 473,897
    windows, 21,628 positive, 452,269 negative, 12 subjects — so a
    self-consistent but altered lock cannot redefine it.

Any failure is refused. Nothing is repaired, coerced, filled or dropped.

### 7.1 Historical binding (B4-A)

B4-A predates `environment_dependency_digest`, `candidate_architecture` and
`architecture_protocol_sha256`. The evaluator validates the **strongest
historically available** binding for B4-A — `split_sha256`,
`feature_corpus_sha256`, `training_selection_sha256`,
`development_feature_integrity_sha256` and `validation_predictions_sha256` — and
records which fields were unavailable.

B4-B and B4-C are **not** weakened to match B4-A. Fields that exist for them are
required for them.

## 8. Threshold source

Each candidate is evaluated at its own **already locked** validation threshold,
read from `validation_threshold` in that candidate's `EXPERIMENT_LOCK.json`.

The evaluator must not:

- select a threshold;
- optimise a threshold;
- retune a threshold per challenge stratum;
- sweep, search or tie-break a threshold;
- share one candidate's threshold with another.

`threshold_source` is recorded as `locked_experiment_lock.validation_threshold`.

## 9. Challenge definitions

The evaluator reuses the already-frozen production implementation
`cardiosentinel.baseline.metrics.challenge_metrics` and the frozen policy table
`cardiosentinel.evaluation.protocol.CHALLENGE_EVIDENCE_POLICIES`. No second,
numerically different definition is introduced by this protocol.

For each challenge, the principal quantity is the false-positive fraction among
the validation windows whose `target_family` equals that challenge's confounder
family. Those families are disjoint from `ischemic_positive`, so the denominator
is by construction a **non-ischemic / negative-context** population.

| Challenge | Target family | Evidence status | Bootstrap |
|---|---|---|---|
| Rate-related | `rate_related_confounder` | `quantitative_secondary` | permitted |
| Axis-shift | `axis_shift_confounder` | `quantitative_secondary` | permitted |
| Conduction-change | `conduction_change_confounder` | `exploratory_descriptive` | **never** |

### 9.1 Rate-related interpretation

Quantitative secondary evidence. May inform the selection gate alongside the
other required dimensions. It is not a headline metric and is never a
tie-breaker on its own.

### 9.2 Axis-shift interpretation

Quantitative secondary evidence, on the same terms as §9.1.

### 9.3 Conduction-change interpretation

**Exploratory / descriptive only.** Subject support for this stratum is sparse.

The conduction figure must be reported with its supporting window and subject
counts and must be labelled `exploratory_descriptive` wherever it appears. It
must not be bootstrapped, must not carry a confidence interval, and must not be
used as quantitative selection evidence.

## 10. Positive-context handling

Positive ischemic windows that also carry a confounder context remain **positive**
and remain in the primary task. They are never removed, reweighted or converted
into a negative for any challenge denominator.

Where the frozen machinery already provides positive-context descriptives
(`cardiosentinel.baseline.metrics.positive_context_analysis` over
`POSITIVE_CONTEXT_FLAGS`), those are preserved and reported separately at
evidence level `descriptive_error_analysis`. They are descriptive error analysis,
not challenge FPR.

## 11. Prohibited operations

The evaluator must not:

- train, optimise or call `backward()`;
- load `training_checkpoint.pt` or any optimizer state;
- modify, rewrite or re-derive any locked candidate artifact;
- regenerate or rewrite validation predictions;
- use or modify the primary B4 waveform cache for challenge evidence;
- read any **test** waveform, cache or metadata;
- read challenge waveforms through any path other than the already validated
  raw physical-mV source path (§4.2);
- access the sealed test (§5);
- select an architecture, rank candidates, or emit a winner;
- combine challenge strata into a single scalar score.

## 12. Required supporting counts

Every reported challenge stratum must carry, without exception:

- `challenge_window_count` — the denominator;
- `false_positive_count` — the numerator;
- `false_positive_fraction` — `null` when the denominator is zero, never `0.0`
  and never imputed;
- `contributing_subject_count` — distinct subjects supporting that stratum.

A fraction is never reported without its denominator. An empty stratum is
reported as empty; it is never silently omitted and never fabricated.

## 13. Artifact schema

The official suite writes one directory:

```
<run_root>/B4_architecture_validation_challenge_v1/
    VALIDATION_CHALLENGE_ATTEMPT.json
    VALIDATION_CHALLENGE_RESULTS.json
```

Per candidate the results record: `official_model`, `experiment_id`,
`architecture`, `experiment_lock_sha256`, `checkpoint_sha256`,
`locked_inference_model`, `locked_validation_threshold`, `threshold_source`,
`threshold_selected_by_evaluator`, `partition`, `dataset`, `dataset_version`,
`split_sha256`, `feature_corpus_sha256`, `training_selection_sha256`,
`development_feature_integrity_sha256`, `development_source_integrity_sha256`,
`provenance_binding`, `challenge_selection_sha256`, `challenge_population`,
`challenge_window_total`, `challenges` (per family: `target_family`,
`evidence_status`, `is_headline_metric`, `challenge_window_count`,
`false_positive_count`, `false_positive_fraction`, `supporting_subject_count`,
`bootstrap_permitted`, `frozen_metric`), `challenge_evidence_source`,
`inference_receipt`, `positive_context` (with `evidence_source`,
`validation_prediction_sha256` and `primary_window_count`),
`metric_definitions`, `validation_challenge_protocol_sha256`,
`b4_protocol_sha256` and a canonical `challenge_result_sha256`.

The suite binds: this protocol's SHA-256, the B4 protocol SHA, the
architecture-selection protocol SHA, `metric_implementation_sha256` (the actual
metric source, not merely function names), the frozen challenge policy table,
`challenge_selection_sha256`, `challenge_population`, the frozen candidate
order, `git_sha` and `git_dirty`, the verified `runtime_environment` and its
`dependency_digest`, all three experiment-lock SHAs, all three checkpoint SHAs,
all three locked thresholds, all three candidate evidence digests, and a
combined `validation_challenge_suite_sha256`.

Honest access flags are mandatory:

| Flag | Required value |
|---|---|
| `dataset_accessed` | `true` |
| `waveform_accessed` | `true` |
| `model_inference_performed` | `true` |
| `training_performed` | `false` |
| `test_accessed` | `false` |
| `threshold_search_performed` | `false` |
| `architecture_selection_performed` | `false` |

The suite must never claim to be prediction-only.

## 13.1 Runtime environment gate

Because challenge evidence performs real inference, the **current runtime** must
equal the frozen scientific environment — Python 3.12.6, PyTorch 2.13.0+cpu,
NumPy 2.3.2, scikit-learn 1.9.0, SciPy 1.18.0, WFDB 4.3.1, dependency digest
`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`, AMP off.

The already-reviewed `require_exact_scientific_environment` gate is reused; no
second comparison is defined. **The gate runs before the one-shot attempt is
claimed**, so a wrong environment can never consume the official attempt.

This runtime environment is distinct from the historical environment recorded in
a candidate lock. Both are retained; the runtime one is verified.

## 14. One-suite execution semantics

The official suite requires exactly B4-A, B4-B and B4-C, measured in that frozen
order. No omission, no fourth model, no candidate-only official mode.

The official attempt is claimed atomically via `O_EXCL` creation of
`VALIDATION_CHALLENGE_ATTEMPT.json`, fsynced on write. Execution is also refused
if `VALIDATION_CHALLENGE_RESULTS.json` already exists, even should the attempt
receipt have disappeared. The run root must be a non-versioned path, and the
checkout must be clean.

If any exception occurs after the claim, the attempt is rewritten in place to
`FAILED_OR_INTERRUPTED` recording `error_type`, `error`, `traceback`,
`human_review_required: true`, `repeat_attempt_permitted: false` and
`selective_candidate_retry_permitted: false`. The claim is **never** unlinked.

An existing attempt in **any** state — started, complete, partial or corrupt —
blocks automatic re-execution. There is no `--force`, `--overwrite`,
`--retry-one`, `--rerun-candidate`, `--best-of`, reset or delete path, and none
may be added.

If a real official run fails after the claim, the attempt is consumed and human
review is required. Recovery semantics are deliberately not defined here.

## 15. Selection is out of scope

This evaluator produces evidence. It does not select.

It contains no ranking function, no weighted score, no Pareto rule and no
statement that any candidate wins or loses. Architecture selection is a separate,
separately authorised step that considers this evidence together with validation
AUPRC, subject-macro behaviour, model size, latency/RAM feasibility and training
stability, per the architecture-selection protocol.

No interpretation in this document may be revised after observing B4 candidate
challenge results.
