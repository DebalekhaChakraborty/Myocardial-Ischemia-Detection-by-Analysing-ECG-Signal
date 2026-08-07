# Phase 3B-1 Classical Baseline Results

## Scope

Phase 3B-1 evaluates frozen classical global baselines for research on transient
ischemic ST episode detection and monitoring decision support. These results are
from public-dataset validation only. CardioSentinel is research software, not a
diagnostic system or medical device, and these results do not establish clinical
effectiveness.

This document is a compact, version-controlled evidence summary. The complete
machine-readable run artifacts, predictions, models, and physiological data
remain external to Git.

## Frozen experimental contract

- Dataset: Long-Term ST Database (LTSTDB) v1.0.0.
- Partitioning: frozen subject-disjoint 56/12/12 train/validation/test split.
- Split SHA-256:
  `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7`.
- Feature corpus SHA-256:
  `f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5`.
- Feature materialization provenance: clean commit
  `4b20a284aac91155bbeefc134fd5fb448028fa8a`. This identifies the code that
  created the unchanged feature corpus.
- Experiment implementation provenance: clean commit
  `4f57ba38d4df593abd9fdd77d5544931b8255534`. This identifies the later
  implementation used to fit, lock, and evaluate B0-B3 without rematerializing
  the corpus.
- Geometry: completed causal 10-second windows with a 5-second stride.
- Signal processing: `raw` identity profile; filtering disabled.
- Operating point: maximum validation F1 over exact observed validation scores,
  with the highest threshold winning a tie.
- Test access: each frozen experiment received one sealed-test evaluation after
  its experiment lock was written. No test result was used for tuning.

The four validation-selected thresholds were `0.04367094778454366` (B0),
`0.7627586011738117` (B1), `0.8579205687821696` (B2), and
`0.8651536386570748` (B3). All locks record the implementation commit above,
`git_dirty=false`, and the same feature-corpus and split hashes.

## Baselines

- **B0 population prior:** constant positive prior fitted on every eligible,
  unsampled primary training window.
- **B1 signal logistic regression:** frozen `signal_v1` descriptors with median
  imputation, standardization, and L2 logistic regression.
- **B2 morphology logistic regression:** frozen `combined_v1` signal and
  morphology descriptors with the same logistic-regression pipeline.
- **B3 morphology histogram gradient boosting:** frozen `combined_v1` inputs
  with the pre-specified histogram gradient boosting model.

`morphology_v1` contains R-aligned waveform proxies. It is not validated
clinical J-point, ST-segment, or T-wave delineation.

## Validation and sealed-test results

All values below were verified against the frozen run summaries. Confidence
intervals are 95% subject-bootstrap intervals with 1,000 replicates and seed
`2026`.

### Ranking metrics

| Baseline | Val AUPRC | Val AUROC | Val F1 | Test AUPRC | Test AUPRC 95% CI | Test AUROC | Subject-macro AUPRC | Subject-macro AUROC |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| B0 | 0.04563860923365204 | 0.5 | 0.08729327480954543 | 0.046052921525592545 | [0.009012349072748565, 0.08780316993202182] | 0.5 | 0.04256120183565847 | 0.5 |
| B1 | 0.421196476883858 | 0.6993048106645885 | 0.4610234600082316 | 0.11729892544635499 | [0.019708935756577554, 0.3357848740857507] | 0.789999793855206 | 0.3342473148024008 | 0.8567497487584544 |
| B2 | 0.47710708202188506 | 0.7977486716738781 | 0.4948146572355475 | 0.16401169749689845 | [0.025301739228663522, 0.503757285914506] | 0.8226971154643274 | 0.405035165175352 | 0.8245275890258019 |
| B3 | 0.6800929449133636 | 0.939145920765906 | 0.650408083948698 | 0.16829012858659798 | [0.029100552763739743, 0.5434208898484981] | 0.8359558626626343 | 0.4364103740418013 | 0.9059971257915307 |

B3 test AUROC had a subject-bootstrap 95% CI of
`[0.6798726686158643, 0.9163211989259034]`.

### Validation-selected operating point

| Baseline | Test F1 | Sensitivity | Specificity | PPV | MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.08805084442272326 | 1.0 | 0.0 | 0.046052921525592545 | undefined |
| B1 | 0.13549084123164562 | 0.13201588592755634 | 0.9605733359513057 | 0.13915367932617137 | 0.09494183651736642 |
| B2 | 0.1739865757623804 | 0.15751949854060002 | 0.9684665226781858 | 0.19429853036652306 | 0.1392882946993911 |
| B3 | 0.17816839149201727 | 0.1639312885784009 | 0.9673531144246428 | 0.19511361694857338 | 0.1426780177408285 |

### Challenge-window false-positive fractions

| Baseline | Rate-related FPR | Axis-shift FPR |
| --- | ---: | ---: |
| B0 | 1.0 | 1.0 |
| B1 | 0.14917127071823205 | 0.35807717462393723 |
| B2 | 0.07241515390686662 | 0.25768476128188356 |
| B3 | 0.10556432517758485 | 0.23250490516677566 |

The conduction-change test stratum contains only 10 windows from one subject.
Its results are exploratory and descriptive only (B0: 10/10 false positives;
B1: 3/10; B2: 4/10; B3: 0/10). It was not bootstrapped and does not establish
robustness.

## Subject-level descriptive findings

The frozen B3 test predictions reproduced the following descriptive results for
the eight test subjects containing both positive and negative primary windows:

| Subject | Positive prevalence | AUPRC | AUROC |
| --- | ---: | ---: | ---: |
| s3074 | 0.1480 | 0.6382 | 0.9196 |
| s2032 | 0.0018 | 0.0897 | 0.9653 |
| s2055 | 0.0596 | 0.7033 | 0.9178 |
| s2015 | 0.0424 | 0.3670 | 0.7069 |
| s2060 | 0.0022 | 0.0937 | 0.9904 |
| s2035 | 0.0317 | 0.7996 | 0.9875 |
| s2029 | 0.0427 | 0.5263 | 0.8900 |
| s2051 | 0.0120 | 0.2735 | 0.8706 |

Subjects `s2065`, `s2009`, `s2024`, and `s2022` have no ischemic-positive
primary test windows, so subject-level AUPRC and AUROC are undefined. AUPRC is
prevalence-sensitive, particularly for subjects with very low event prevalence.
The largest test subject, `s3074`, performs reasonably well; pooled performance
cannot be attributed to one large poorly performing subject.

## Scientific interpretation

Classical global baselines demonstrated meaningful within-subject
discriminative capability in several subjects but substantial between-subject
heterogeneity. Morphology-informed models improved sealed-test pooled
discrimination over generic signal statistics, while a single global model and
validation-selected operating point remained insufficient for robust
cross-subject performance.

The higher subject-macro than pooled discrimination is consistent with
between-subject score-distribution heterogeneity and motivates prospective
evaluation of patient-adaptive methods. It does not prove calibration failure;
no probability-calibration experiment was performed.

AUPRC is threshold-free. Its validation-to-test degradation therefore reflects
ranking/generalization behavior and cannot be explained merely by the frozen
validation-selected threshold. The low test sensitivity and high specificity at
that threshold separately describe operating-point transfer. Neither observation
supports causal or clinical-effectiveness claims.

## Limitations

- The test partition contains only 12 subjects; four have no positive primary
  windows.
- Subject-bootstrap confidence intervals are wide, and prevalence varies
  substantially across subjects.
- The conduction challenge is too sparse for inference: 10 windows from one
  subject.
- Morphology features are R-aligned waveform proxies, not clinical ST
  delineation.
- Overlapping windows are not independent observations; the effective
  inferential unit is the subject.
- No external overlap-clean European ST-T Database result is available yet.
- Probability calibration has not been evaluated.
- No episode-level temporal model has been evaluated.
- No patient personalization has been evaluated.
- No hardware validation has been performed.

## Phase 3B-1 closure

B0-B3 are frozen historical benchmark evidence. No further test-guided tuning,
threshold changes, feature changes, model changes, or retrospective protocol
changes are permitted. Phase 3B-2/B4 is a separate prospective experiment; its
architecture, training, and evaluation contract must be frozen before training
or sealed-test access.
