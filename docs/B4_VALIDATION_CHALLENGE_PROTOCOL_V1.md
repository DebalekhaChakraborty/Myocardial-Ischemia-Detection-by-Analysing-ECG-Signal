# B4 Validation Challenge Evidence Protocol V1

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

## 3. Inputs

Exactly three locked candidate runs, in the frozen order below.

| Order | Official model | Experiment | Architecture |
|---|---|---|---|
| 1 | `B4-A` | `B4_raw_compact_cnn_v1` | `B4CompactCNN` |
| 2 | `B4-B` | `B4B_cnn_transformer_v1` | `B4BTransformerCNN` |
| 3 | `B4-C` | `B4C_cnn_ssm_v1` | `B4CSSMCNN` |

For each candidate the evaluator consumes only:

1. `EXPERIMENT_LOCK.json` — the immutable development lock;
2. `validation_predictions.npz` — the immutable locked validation predictions.

No other artifact is read for measurement purposes.

## 4. Partition

The evaluation partition is exactly the frozen **primary validation** partition
recorded in each candidate lock (`validation_rows`).

The `train` partition is not evaluated. The `test` partition is prohibited.

## 5. Test prohibition

The evaluator must not, on any code path:

- import or call the sealed-test module;
- evaluate, resolve or name the `test` partition;
- read a test prediction, label, cache, waveform or metadata file;
- create, read or reference `TEST_ATTEMPT.json`;
- produce any test metric.

A partition value of `test` entering any public execution path is a hard error.

## 6. Prediction-artifact integrity

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
9. the prediction row count and subject count equal the lock's `validation_rows`.

Any failure is refused. Nothing is repaired, coerced, filled or dropped.

### 6.1 Historical binding (B4-A)

B4-A predates `environment_dependency_digest`, `candidate_architecture` and
`architecture_protocol_sha256`. The evaluator validates the **strongest
historically available** binding for B4-A — `split_sha256`,
`feature_corpus_sha256`, `training_selection_sha256`,
`development_feature_integrity_sha256` and `validation_predictions_sha256` — and
records which fields were unavailable.

B4-B and B4-C are **not** weakened to match B4-A. Fields that exist for them are
required for them.

## 7. Threshold source

Each candidate is evaluated at its own **already locked** validation threshold,
read from `validation_threshold` in that candidate's `EXPERIMENT_LOCK.json`.

The evaluator must not:

- select a threshold;
- optimise a threshold;
- retune a threshold per challenge stratum;
- sweep, search or tie-break a threshold;
- share one candidate's threshold with another.

`threshold_source` is recorded as `locked_experiment_lock.validation_threshold`.

## 8. Challenge definitions

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

### 8.1 Rate-related interpretation

Quantitative secondary evidence. May inform the selection gate alongside the
other required dimensions. It is not a headline metric and is never a
tie-breaker on its own.

### 8.2 Axis-shift interpretation

Quantitative secondary evidence, on the same terms as §8.1.

### 8.3 Conduction-change interpretation

**Exploratory / descriptive only.** Subject support for this stratum is sparse.

The conduction figure must be reported with its supporting window and subject
counts and must be labelled `exploratory_descriptive` wherever it appears. It
must not be bootstrapped, must not carry a confidence interval, and must not be
used as quantitative selection evidence.

## 9. Positive-context handling

Positive ischemic windows that also carry a confounder context remain **positive**
and remain in the primary task. They are never removed, reweighted or converted
into a negative for any challenge denominator.

Where the frozen machinery already provides positive-context descriptives
(`cardiosentinel.baseline.metrics.positive_context_analysis` over
`POSITIVE_CONTEXT_FLAGS`), those are preserved and reported separately at
evidence level `descriptive_error_analysis`. They are descriptive error analysis,
not challenge FPR.

## 10. Prohibited operations

The evaluator must not:

- construct a neural model or call `forward()`;
- load `model_selected.pt` or `training_checkpoint.pt` for inference;
- train, optimise or call `backward()`;
- regenerate or rewrite validation predictions;
- read a waveform cache, WFDB source or LTSTDB signal file;
- access the sealed test (§5);
- select an architecture, rank candidates, or emit a winner;
- combine challenge strata into a single scalar score.

## 11. Required supporting counts

Every reported challenge stratum must carry, without exception:

- `challenge_window_count` — the denominator;
- `false_positive_count` — the numerator;
- `false_positive_fraction` — `null` when the denominator is zero, never `0.0`
  and never imputed;
- `contributing_subject_count` — distinct subjects supporting that stratum.

A fraction is never reported without its denominator. An empty stratum is
reported as empty; it is never silently omitted and never fabricated.

## 12. Artifact schema

The official suite writes one directory:

```
<run_root>/B4_architecture_validation_challenge_v1/
    VALIDATION_CHALLENGE_ATTEMPT.json
    VALIDATION_CHALLENGE_RESULTS.json
```

Per candidate, the results record at minimum: `experiment_id`, `architecture`,
`official_model`, `experiment_lock_sha256`, `validation_prediction_sha256`,
`locked_validation_threshold`, `threshold_source`, `split_sha256`,
`feature_corpus_sha256`, `validation_window_count`, `validation_subject_count`,
the three challenge strata of §11 with their evidence statuses, the
positive-context descriptives of §9, `dataset_accessed`, `test_accessed`,
`model_inference_performed`, and a canonical `challenge_result_sha256`.

The suite binds: this protocol's SHA-256, the B4 protocol SHA, the
architecture-selection protocol SHA, the benchmark and metrics definitions relied
upon, the frozen candidate order, all three experiment-lock SHAs, all three
validation-prediction hashes, all three locked thresholds, all three candidate
evidence digests, the metric definition and status labels, `dataset_accessed`,
`test_accessed`, and a combined `validation_challenge_suite_sha256`.

Both `dataset_accessed` and `test_accessed` must be `false`.

## 13. One-suite execution semantics

The official suite requires exactly B4-A, B4-B and B4-C, measured in that frozen
order. No omission, no fourth model, no candidate-only official mode.

The official attempt is claimed atomically via `O_EXCL` creation of
`VALIDATION_CHALLENGE_ATTEMPT.json`. The directory and attempt file together are
the claim.

An existing attempt in **any** state — started, complete, partial or corrupt —
blocks automatic re-execution. There is no `--force`, `--overwrite`,
`--retry-one`, `--rerun-candidate`, `--best-of`, reset or delete path, and none
may be added.

If a real official run fails after the claim, the attempt is consumed and human
review is required. Recovery semantics are deliberately not defined here.

## 14. Selection is out of scope

This evaluator produces evidence. It does not select.

It contains no ranking function, no weighted score, no Pareto rule and no
statement that any candidate wins or loses. Architecture selection is a separate,
separately authorised step that considers this evidence together with validation
AUPRC, subject-macro behaviour, model size, latency/RAM feasibility and training
stability, per the architecture-selection protocol.

No interpretation in this document may be revised after observing B4 candidate
challenge results.
