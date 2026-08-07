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

- false-positive rate on rate-related ST windows;
- false-positive rate in the plus/minus 30-second axis-marker vicinity; and
- false-positive rate in the plus/minus 30-second conduction-marker vicinity.

Do not merge these challenge results into clean-background specificity without
also reporting the components. They are not the primary positive target.

## Test composition

Headline validation and test metrics use every eligible ischemic-positive and
background-negative window. The test set is not artificially balanced or
negative-sampled. Training sampling has no effect on evaluation prevalence.

## Deferred metrics

Episode sensitivity, false alarms per hour, event-onset delay, and duration
accuracy require temporal evidence accumulation and are reserved for Phase 7.
No Phase-3A output is an episode-performance or clinical-effectiveness result.
