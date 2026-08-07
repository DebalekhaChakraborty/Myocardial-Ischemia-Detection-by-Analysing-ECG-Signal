# Metrics Protocol

This protocol is frozen before Phase-3 model development. It covers causal
window classification only and does not authorize episode-level claims.

## Primary metric

The primary discrimination metric is area under the precision-recall curve
(AUPRC), implemented as average precision using the step-wise precision-recall
integral. Positive ischemic windows are expected to be much less common than
background windows, so prevalence and class counts must accompany AUPRC.

## Secondary metrics

Report AUROC, F1, sensitivity/recall, specificity, PPV/precision, NPV, balanced
accuracy, and Matthews correlation coefficient. Accuracy alone is insufficient.
PPV, NPV, and AUPRC must be interpreted with observed benchmark prevalence.

For each metric report:

1. pooled-window performance over the complete eligible partition; and
2. subject-macro performance so long recordings do not dominate conclusions.

A subject-level metric requiring both classes is undefined for a subject lacking
one class. Do not replace it with zero; report the number of contributing and
non-contributing subjects for each macro metric.

## Binary threshold

Choose a binary threshold using validation predictions only. Evaluate every
unique validation score as a candidate, maximize validation-set F1, and break
ties by selecting the highest threshold. Freeze that threshold before test
evaluation. Test predictions may never select or revise it.

Raw sigmoid, softmax, or other model scores are not calibrated confidence.
Calibration metrics and methods remain Phase 6.

## Confidence intervals

Use 1000 subject-level bootstrap replicates with seed `2026`. Resample subjects
with replacement, preserving all eligible windows for each sampled subject and
its multiplicity. Use the 2.5th and 97.5th percentiles for a 95% interval and
report degenerate or undefined replicates. Independent window-level bootstrap
is prohibited because overlapping windows from one subject are correlated.

## Challenge subsets

At the frozen validation-selected threshold, separately report:

- rate-related FPR as a `quantitative_secondary` challenge metric;
- axis-shift FPR in the plus/minus 30-second marker vicinity as a
  `quantitative_secondary` challenge metric; and
- conduction-change `FP = x / N` as an `exploratory_descriptive` stress test.

Do not merge these challenge results into clean-background specificity without
also reporting the components. They are not the primary positive target.
Only non-ischemic targets whose primary family is the corresponding confounder
enter a challenge false-positive-rate denominator. An `ischemic_positive` with
axis or conduction context remains a positive and is excluded from that
denominator.

Every challenge report must include its evidence level, contributing-subject
count, challenge-window count, false-positive count, and denominator. A fraction
may be shown only as `FP / N`, never without those counts. Conduction-change
evidence has one held-out subject in V1, so it is not bootstrapped, does not
receive an inferential confidence interval, is not a headline or model-selection
criterion, and is never included in an overall confounder-robustness score.

Descriptive/error-analysis strata report ischemic positives with no axis or
conduction marker context, with axis context, with conduction context, and with
point-noise context. These strata may overlap and are not new disease classes,
headline metrics, threshold-selection inputs, or formal FPR challenges. Every
context report includes contributing-subject and window counts; it must not show
an aggregate percentage without those denominators.

## Test composition

Headline validation and test metrics use every eligible ischemic-positive and
background-negative window. The test set is not artificially balanced or
negative-sampled. Training sampling has no effect on evaluation prevalence.

## Metric hierarchy

- Primary metric: AUPRC.
- Secondary model metrics: AUROC, F1, sensitivity, specificity, PPV, NPV,
  balanced accuracy, and MCC.
- Quantitative challenge metrics: rate-related FPR and axis-shift FPR.
- Exploratory challenge: conduction-change descriptive `FP / N`.
- Descriptive positive-context strata: axis, conduction, and point noise.

## Deferred metrics

Episode sensitivity, false alarms per hour, event-onset delay, and duration
accuracy require temporal evidence accumulation and are reserved for Phase 7.
No Phase-3A output is an episode-performance or clinical-effectiveness result.
