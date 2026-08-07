# Baseline Protocol V1

This protocol freezes the CardioSentinel Phase 3B-1 global classical baselines
before model performance is obtained. It is research software protocol, not a
diagnostic, clinical-effectiveness, or medical-device claim. Results cannot be
used to rewrite this V1 protocol or the frozen benchmark.

## Benchmark identity

- Dataset: Long-Term ST Database v1.0.0 (`ltstdb`).
- Primary annotation: `.stb`, 100 uV / 30 seconds.
- Window and stride: completed causal 10-second windows every 5 seconds.
- Split SHA-256:
  `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7`.
- Primary metric: average precision (AUPRC).
- Threshold: maximum validation F1 with highest-threshold tie breaking.
- Test bootstrap: 1000 subject replicates with seed `2026`.
- Signal path: physical mV through the `raw` identity profile. Filtering is off.

Any other split hash fails validation. Expert ST measurements, `.stf`,
GRST/LRST, corrected beats, expert beat locations, event deviations, event
endpoints, labels, identity metadata, and future waveform samples are prohibited
predictive inputs.

## Frozen feature schemas

`signal_v1` contains the ten Phase-2 waveform-only quality descriptors and 12
pre-specified physical statistics: median, mean, standard deviation, RMS,
5th/25th/75th/95th percentiles, interquartile and peak-to-peak ranges, and
median/95th-percentile absolute physical first derivatives. Derivative units
are mV/s.

`morphology_v1` uses WFDB XQRS within each completed window only. A beat is
usable only when every sample from R-200 ms through R+400 ms is in that window.
At least two usable beats are required for a valid morphology row. The
waveform-derived baseline proxy is the median from R-200 to R-80 ms. Post-QRS
proxies at R+80/120/160/200 ms, an 80-to-160 ms slope, 80-to-200 ms area, a QRS
peak-to-peak proxy, rhythm summaries, template correlation, and template
variability are reported. These are R-aligned waveform morphology proxies, not
validated J-point or clinical ST-deviation measurements. Detection failures
retain the row, counts, a zero validity field, and missing proxy values.

`combined_v1` is the ordered concatenation of `signal_v1` and `morphology_v1`.
Every schema includes ordered names, units, descriptions, and a canonical
SHA-256 digest. Record, subject, channel, lead, stable ID, partition, target,
and context remain metadata and never enter the numeric model matrix.

## Frozen baseline ladder

### B0_constant_prior

Fit the positive prior on every eligible, unsampled primary training window and
emit that constant for all evaluation rows. No validation or test value enters
the prior.

### B1_signal_logreg

Input `signal_v1`. Fit a median `SimpleImputer(keep_empty_features=True)`, then
`StandardScaler`, then L2 `LogisticRegression` with `solver="lbfgs"`, `C=1.0`,
`max_iter=1000`, `class_weight=None`, and seed `2026`.

### B2_morphology_logreg

Input `combined_v1`. Use the identical frozen imputation, scaling, and logistic
regression configuration as B1.

### B3_morphology_hgb

Input `combined_v1`. Fit a median
`SimpleImputer(keep_empty_features=True)`, then
`HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200,
max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False,
random_state=2026)`. There is no grid search.

All learned imputer and scaler state is fitted on selected training rows only.
No personalization, neural model, compact raw-waveform model, calibration, or
foundation model belongs to Phase 3B-1.

## Sampling and held-out use

B0 estimates its prior from the unsampled eligible training primary benchmark.
B1/B2/B3 retain every eligible ischemic-positive training window and use the
frozen deterministic subject-aware sampler for at most three background
windows per positive. Rate, axis, conduction, boundary, unreadable, and
source-censored windows do not train these models.

Training selection uses two metadata-only passes over per-record caches. The
first derives the exact frozen group quotas; the second retains the same
seeded stable-ID choices as the canonical `WindowTarget` reference sampler.
Only selected B1/B2/B3 rows are then loaded into one numeric training matrix.
B0 reads unsampled primary label counts and calculates its scalar prior without
loading or copying the training feature matrix.

Validation and test primary metrics use all eligible ischemic-positive and
background-negative rows without sampling. Fit reads train and validation only.
Each model selects its threshold from validation predictions, serializes the
model and learned transforms, hashes them, and writes `experiment_lock.json`.
Test evaluation requires a valid lock, matching clean Git revision, frozen split
and schema, and artifact hashes. It refuses a second evaluation for the lock.

## Metrics and challenge interpretation

Report pooled-window and subject-macro AUPRC, AUROC, F1, sensitivity,
specificity, PPV, NPV, balanced accuracy, and MCC. A mathematically undefined
subject metric remains undefined and includes contributing and
non-contributing subject counts. Subject-level AUPRC and AUROC require both
classes; all-positive and all-negative subjects do not establish discrimination.
The same rule applies to subject-bootstrap replicates. Test confidence intervals
resample subjects, not windows, and retain undefined or degenerate replicate
counts.

At the validation-frozen threshold, rate-related and axis-shift false-positive
fractions are `quantitative_secondary` with subject/window/FP denominators and
subject-bootstrap uncertainty when supported. Conduction change is
`exploratory_descriptive`, reports only subjects, windows, and `FP / N`, is not
bootstrapped, and is never a selection criterion or combined robustness score.
Positive windows are also described in no-axis/conduction, axis, conduction,
and point-noise context strata with subject and window denominators.

## Materialization and artifacts

Waveforms and derived rows live outside Git. Materialization reads deterministic
bounded chunks, applies raw identity processing, emits existing causal windows,
and writes atomic compressed per-record caches. Record-level processes own
disjoint records, limit nested numerical thread pools, and return completed
metadata to the parent, which alone updates the manifest. A record resumes only
when its source, split, schema, geometry, annotation definition, processing
profile, and actual cache-file SHA-256 match. `--force` is required to replace
stale output. Worker count does not alter scientific output.

Once all 86 record caches are complete, `feature_corpus_sha256` canonically
binds dataset/version, split and combined schema hashes, processing profile,
window/stride, annotation definition, and every sorted record's identity,
source hash, row and target counts, and exact cache-file hash. Runtime,
timestamps, and resume bookkeeping are excluded. Fit records this corpus hash
in the experiment lock; sealed-test evaluation re-hashes every cache and rejects
any changed corpus before loading test rows.

Runs remain external and contain machine-readable configuration, environment,
feature and training manifests, model and transform artifacts, validation and
test predictions, the experiment lock, metrics, challenge/context reports, and
`RESULTS_SUMMARY.json`. Predictions are derived physiological research data.
No waveform, cache, prediction, checkpoint, or run artifact is committed.

The pinned acquisition is explicit and resumable through `wget --continue`. It
downloads only `SHA256SUMS.txt`, `RECORDS`, and the frozen benchmark records'
`.hea`, `.dat`, and `.stb` files. Before full materialization,
`verify-source` requires the local manifest to match PhysioNet's pinned V1
manifest hash, verifies all required source bytes, validates the `RECORDS` set,
and writes an external `source_verification.json` receipt.

```bash
python -m cardiosentinel baseline acquire \
  --destination /external/data/ltstdb/1.0.0
python -m cardiosentinel baseline acquire \
  --destination /external/data/ltstdb/1.0.0 --execute

python -m cardiosentinel baseline verify-source \
  --source /external/data/ltstdb/1.0.0
```

The first command reports the source, destination, exact pinned file count, and
available disk without downloading. Full acquisition occurs only with
`--execute`; import, tests, and help never initiate it.

## Execution stages

```bash
python -m cardiosentinel baseline preflight \
  --source /external/data/ltstdb/1.0.0 \
  --feature-root /external/features/ltstdb-baseline-v1 \
  --workers 2

python -m cardiosentinel baseline materialize \
  --source /external/data/ltstdb/1.0.0 \
  --feature-root /external/features/ltstdb-baseline-v1 \
  --workers 2

python -m cardiosentinel baseline smoke-remote \
  --output-root /external/runs/phase-3b-smoke

python -m cardiosentinel baseline fit \
  --feature-root /external/features/ltstdb-baseline-v1 \
  --run-root /external/runs --experiment-id b1-signal-v1 \
  --baseline B1_signal_logreg

python -m cardiosentinel baseline evaluate-test \
  --feature-root /external/features/ltstdb-baseline-v1 \
  --run-dir /external/runs/b1-signal-v1
```

Materialization defaults to one worker. Preflight reports host CPU and disk,
source/cache readiness, frozen counts, conservative matrix-memory estimates,
and an engineering-only projected runtime. It does not read waveform samples,
load sealed-test features, or train models. It blocks full execution unless all
86 source records are present and the matching official checksum receipt exists.

EDB performance is outside this phase. B4 belongs to Phase 3B-2 and is not
implemented here.
