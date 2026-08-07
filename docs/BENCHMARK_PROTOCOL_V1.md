# Benchmark Protocol V1

This document freezes the CardioSentinel Phase-3 causal-window benchmark before
any model result exists. It is research software protocol, not a diagnostic or
medical-device standard. A material later change requires
`BENCHMARK_PROTOCOL_V2.md`; V1 must not be silently rewritten after results are
known.

## Primary benchmark

- Dataset: Long-Term ST Database v1.0.0 (`ltstdb`).
- Annotation definition: `.stb`, `Vmin = 100 uV`, `Tmin = 30 seconds`.
- `.sta` and `.stc` remain separate future sensitivity analyses and are never
  combined with `.stb`.
- Split unit: subject. All records from one subject remain together.
- Split seed: `2026`.
- Frozen split: `protocols/splits/ltstdb_v1.json`.
- Split hash:
  `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7`.
- Source metadata hash:
  `2ccd1908f3fb1887aef25273bf8039e48e478c751693394da4ad72d155970913`.
- Generator code hash:
  `0211c84004ff5dd97d8e6e99418d5c779d65d18fe4f05bc807415211bb2db52d`.
- Test partition: sealed before model training or performance inspection.

The deterministic assignment uses subject-level source metadata only: record
count, total duration, `.stb` ischemic episode count and duration, rate-related
episode count and duration, channel count, axis-shift marker count/presence, and
conduction-change marker count/presence. Every feature enters the same normalized
objective without a special multiplier. The algorithm uses exact 56/12/12
subject capacities, deterministic greedy assignment, improving pairwise swaps,
and SHA-256 seed tie-breaking. Waveform embeddings, learned features,
predictions, errors, and performance are prohibited inputs.

## Pre-model review correction

The initial split was generated before any model result. A final reviewer audit
then found that its conduction challenge windows were distributed 4,290/3,603/
164 across train/validation/test. Before training or inspection of predictions,
source-level axis and conduction marker burden was therefore added to the
deterministic subject-stratification objective. Marker presence is balanced as
well as count because challenge analysis requires contributing subjects, not
only aggregate marker volume. No waveform feature, model prediction, error, or
performance metric was available or used. The corrected split in this document
is the final V1 split and becomes immutable when Phase 3A is merged.

Only four subjects contain conduction-change markers. The corrected split puts
two in train, one in validation, and one in test, but the two train subjects
contain 879 of 895 source markers and the largest subject alone contains 450
(50.28%). By comparison, 54 subjects contain axis markers and the largest has
232 of 1,493 (15.54%). Consequently, conduction challenge windows remain
concentrated at 7,883/164/10. This residual is a source-cohort limitation;
subjects were not moved manually and no feature received a tuned weight.

The final metadata-only enumeration is:

| Partition | Subjects | Ischemic episodes (s) | Positive | Background | Prevalence | Rate | Axis | Conduction | Unreadable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 56 | 791 (475,102) | 93,613 | 2,049,986 | 4.3671% | 24,512 | 13,630 | 7,883 | 13,099 |
| Validation | 12 | 163 (109,590) | 21,628 | 452,269 | 4.5639% | 4,973 | 3,000 | 164 | 10,112 |
| Test | 12 | 164 (106,242) | 20,899 | 432,905 | 4.6053% | 5,068 | 3,058 | 10 | 350 |

Rate/axis/conduction challenge-window subject contributors are respectively
18/38/2 in train, 4/8/1 in validation, and 4/8/1 in test.

## Causal window definition

The primary window length is 10 seconds and stride is 5 seconds. Samples are
derived from each WFDB header's sampling frequency; 2500/1250 samples at 250 Hz
are consequences, not hardcoded assumptions. Windows are indexed from the
record start and become available only at their exclusive end sample.

The choice was made before model results. A minimum 30-second `.stb` episode can
contain multiple 10-second observations, while a 5-second stride retains useful
temporal resolution and remains practical for later edge studies. Alternative
geometry may be investigated on training/validation only and cannot replace V1
without a new protocol version.

Physical window geometry is generated only from record identity, waveform
extent, sampling frequency, length, stride, channel, and stream position.
Annotations are assigned only after indices exist. Event onset, peak, endpoint,
axis shift, or annotation proximity never creates, centers, shifts, or changes
the stride of a physical window.

## Target assignment

Targets retain dataset, record, subject, channel/lead, half-open sample bounds,
semantic family/subtype, eligibility flags, annotation definition, overlapping
event and marker identities, source-supported context flags, quality state, and
exclusion reason. They are not immediately reduced to a binary value.

The canonical states are:

- `ischemic_positive`: the complete 10-second window is fully contained in a
  complete `.stb` ischemic episode on the same lead.
- `background_negative`: zero overlap with ischemic or rate-related episodes,
  unreadable intervals, source-censored regions, and configured marker
  challenges. This means background, not healthy or normal patient.
- `boundary_ambiguous`: partial overlap with an ischemic episode boundary.
- `rate_related_confounder`: overlap with a heart-rate-related ST episode,
  retaining whether the window is fully contained or boundary-overlapping.
- `axis_shift_confounder`: overlap with the pre-specified axis-marker vicinity.
- `conduction_change_confounder`: overlap with the pre-specified conduction-
  marker vicinity.
- `quality_excluded`: overlap with an expert unreadable interval.
- `source_censored_or_unknown`: overlap with a region whose source annotation
  lacks an onset or endpoint at a record boundary.

Expert unreadable state has exclusion precedence. Complete ischemic containment
then precedes ischemic-boundary handling, rate-related episodes, duration-
bounded EDB apparent ST changes, and LTSTDB marker challenges. No endpoint is
invented for source-censored annotations.

Marker metadata is collected independently before applying this precedence. An
ischemic-positive window in an axis/conduction vicinity remains positive and
retains `axis_shift_context` or `conduction_change_context`. A source point-noise
marker inside a window adds `point_noise_context`; it does not invent a noise
duration, exclusion, or disease class.

Noisy-but-readable reference intervals remain eligible and retain their quality
state. Phase-2 waveform-only SQIs are independent measurements and are never
replaced by expert quality labels.

## Confounder subsets

An axis or conduction marker receives a pre-specified plus/minus 30-second
challenge vicinity. This is an engineering challenge interval around a change
marker and is not asserted to represent the clinical duration of the
phenomenon. Point noise markers are not automatically expanded or excluded.

Rate-related, axis, and conduction targets remain separate challenge subsets.
They are not ischemic positives or clean background negatives. Their future
training participation remains a Phase-3B decision; V1 marks them ineligible
for training by default. Ischemic positives carrying marker context never enter
challenge false-positive-rate denominators.

## Training sampling

No training occurs in Phase 3A. The frozen V1 sampler interface retains all
eligible ischemic-positive training windows and draws at most three eligible
background negatives per positive. Sampling is seed-controlled, without
replacement, and round-robin distributed across subject/record groups. Subjects
with no positive window may contribute at most 30 background windows, within
the global 3:1 budget.

Validation and test composition are not training-sampled. Primary test
evaluation retains every eligible positive and every eligible background
window. Exploratory subsampling can never replace the headline test benchmark.

## Metrics and thresholds

The primary metric is AUPRC. Secondary metrics, subject-macro aggregation,
subject bootstrap confidence intervals, challenge false-positive rates, and the
validation-only threshold rule are frozen in `docs/METRICS_PROTOCOL.md`.
Thresholds, model choices, calibration, and personalization cannot use test
predictions. Raw sigmoid or softmax output is not calibrated confidence.

Phase 3 is window-level only. Episode sensitivity, false alarms per hour,
onset delay, and event-duration accuracy are reserved for the Phase-7 temporal
protocol.

## Test access and versioning

The test subject list and canonical hash must not change because of poor model
performance, discovered prevalence, hyperparameter behavior, calibration, or
presentation preferences. The test partition is evaluated only after model,
threshold, and all training/validation decisions are frozen for an experiment.

Changing target semantics, primary annotation definition, window geometry,
split, marker vicinity, headline metric, bootstrap unit, threshold rule, or
cross-dataset exclusion requires a V2 protocol with a documented reason. A Git
diff to V1 is not an acceptable silent protocol change.

## Leakage firewall

Expert ST deviations, `.stf`, GRST/LRST, corrected beats, episode endpoints,
future samples, and target labels are ground truth or evaluation metadata only.
They cannot enter signal preprocessing, window generation, feature extraction,
models, personalization state, or threshold selection. Future morphology must
be derived causally from waveform data.

The evaluation package may import annotations. Predictive-input packages may
not import `STEvent`, `WindowTarget`, expert deviations, or evaluation targets;
architecture tests enforce this dependency direction.

## Cross-dataset policy

EDB is a compatible secondary benchmark using its reference ST episodes,
duration-bounded apparent axis-shift episodes, quality intervals, documented
subject groups, and the same 10-second/5-second geometry. EDB and LTSTDB subject
identifiers remain namespaced and are never merged.

The explicit `full` EDB secondary cohort has 90 records and is marked
`contains_known_source_overlap = true`. The `overlap_clean` secondary cohort
excludes 15 verified or same-EDB-subject records and has 75 records. A model
trained on LTSTDB must use the overlap-clean cohort for recommended EDB
evaluation. Neither cohort is described as fully independent external
validation. The correspondences, rescaling limitation, and conservative policy
are recorded in `docs/CROSS_DATASET_PROVENANCE.md`.

## Provenance and artefacts

Aggregate summaries record CardioSentinel version, Git SHA and dirty state,
Python version, dataset/version, annotation definition, protocol and label
policy versions, split hash, geometry, seed, and command. Full deterministic
window indices are generated lazily. Million-row manifests, raw physiological
signals, and patient-derived outputs remain outside Git.

The manifest keeps three distinct hashes. `split_sha256` covers only canonical
subject/record assignment. `source_metadata_sha256` covers canonical record and
subject-burden metadata. `generator_code_sha256` covers normalized LF contents
of `evaluation/models.py` and `evaluation/splits.py`, in that path order, plus a
canonical object containing the exact assignment constants and marker subtype
names. Frozen result hashes are excluded from generator inputs to avoid
self-reference. `generation_git_sha` and `generation_git_dirty` independently
record the checkout state used for generation; no timestamp enters assignment
identity.
