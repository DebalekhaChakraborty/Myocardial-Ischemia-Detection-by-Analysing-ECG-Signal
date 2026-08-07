"""Frozen pooled, subject-macro, bootstrap, challenge, and context metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.evaluation.metrics import subject_bootstrap_plan
from cardiosentinel.evaluation.protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    challenge_evidence_policy,
)

METRIC_NAMES = (
    "auprc",
    "auroc",
    "f1",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "balanced_accuracy",
    "mcc",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def binary_metrics(
    labels: Sequence[int] | NDArray[np.int64],
    scores: Sequence[float] | NDArray[np.float64],
    threshold: float,
) -> dict[str, float | int | None]:
    """Compute the frozen binary metrics while retaining undefined values."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.size == 0 or labels_array.shape != scores_array.shape:
        raise ValueError("Metric labels and scores must be non-empty and aligned.")
    if not np.isin(labels_array, (0, 1)).all() or not np.isfinite(scores_array).all():
        raise ValueError("Metrics require binary labels and finite scores.")
    predictions = scores_array >= threshold
    positive = labels_array == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    tn = int(np.sum(~predictions & negative))
    fp = int(np.sum(predictions & negative))
    fn = int(np.sum(~predictions & positive))
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    ppv = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    balanced = (
        None
        if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2.0
    )
    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (
        None
        if mcc_denominator == 0
        else (tp * tn - fp * fn) / float(np.sqrt(mcc_denominator))
    )
    class_count = np.unique(labels_array).size
    return {
        "window_count": int(labels_array.size),
        "positive_count": int(np.sum(positive)),
        "negative_count": int(np.sum(negative)),
        "positive_prevalence": float(np.mean(positive)),
        "auprc": (
            None
            if not positive.any()
            else float(average_precision_score(labels_array, scores_array))
        ),
        "auroc": (
            None
            if class_count < 2
            else float(roc_auc_score(labels_array, scores_array))
        ),
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def subject_macro_metrics(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    subjects: NDArray[np.str_],
    threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    """Average each metric over only subjects for which it is defined."""
    unique_subjects = tuple(sorted(set(subjects.tolist())))
    per_subject = [
        binary_metrics(
            labels[subjects == subject], scores[subjects == subject], threshold
        )
        for subject in unique_subjects
    ]
    report: dict[str, dict[str, float | int | None]] = {}
    for metric in METRIC_NAMES:
        values = [item[metric] for item in per_subject if item[metric] is not None]
        report[metric] = {
            "value": None if not values else float(np.mean(values)),
            "contributing_subject_count": len(values),
            "non_contributing_subject_count": len(unique_subjects) - len(values),
        }
    return report


def subject_bootstrap_confidence_intervals(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    subjects: NDArray[np.str_],
    threshold: float,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float | int | None]]:
    """Resample subjects with multiplicity and retain degenerate replicates."""
    index_by_subject = {
        subject: np.flatnonzero(subjects == subject)
        for subject in sorted(set(subjects.tolist()))
    }
    collected: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    undefined = {metric: 0 for metric in METRIC_NAMES}
    for sampled_subjects in subject_bootstrap_plan(
        subjects.tolist(), replicates=replicates, seed=seed
    ):
        indices = np.concatenate(
            [index_by_subject[subject] for subject in sampled_subjects]
        )
        metrics = binary_metrics(labels[indices], scores[indices], threshold)
        for metric in METRIC_NAMES:
            value = metrics[metric]
            if value is None:
                undefined[metric] += 1
            else:
                collected[metric].append(float(value))
    return {
        metric: {
            "lower_95": (None if not values else float(np.percentile(values, 2.5))),
            "upper_95": (None if not values else float(np.percentile(values, 97.5))),
            "successful_replicates": len(values),
            "undefined_replicates": undefined[metric],
            "requested_replicates": replicates,
            "seed": seed,
        }
        for metric, values in collected.items()
    }


def challenge_metrics(
    target_families: NDArray[np.str_],
    scores: NDArray[np.float64],
    subjects: NDArray[np.str_],
    threshold: float,
) -> dict[str, dict[str, object]]:
    """Report typed challenge false positives without combining subsets."""
    report: dict[str, dict[str, object]] = {}
    for challenge in ("rate_related", "axis_shift", "conduction_change"):
        policy = challenge_evidence_policy(challenge)
        selected = target_families == policy.target_family
        count = int(np.sum(selected))
        false_positives = int(np.sum(scores[selected] >= threshold))
        subject_count = len(set(subjects[selected].tolist()))
        report[challenge] = {
            "evidence_level": policy.evidence_level,
            "contributing_subject_count": subject_count,
            "challenge_window_count": count,
            "false_positive_count": false_positives,
            "false_positive_fraction": (
                None if count == 0 else false_positives / count
            ),
            "bootstrap_permitted": (
                policy.supports_inferential_bootstrap and subject_count >= 2
            ),
        }
    return report


def challenge_bootstrap_confidence_intervals(
    target_families: NDArray[np.str_],
    scores: NDArray[np.float64],
    subjects: NDArray[np.str_],
    threshold: float,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float | int | None]]:
    """Bootstrap quantitative challenge FPRs by subject, never conduction."""
    output: dict[str, dict[str, float | int | None]] = {}
    for challenge in ("rate_related", "axis_shift"):
        policy = challenge_evidence_policy(challenge)
        selected = target_families == policy.target_family
        selected_subjects = subjects[selected]
        unique_subjects = tuple(sorted(set(selected_subjects.tolist())))
        if len(unique_subjects) < 2:
            output[challenge] = {
                "lower_95": None,
                "upper_95": None,
                "successful_replicates": 0,
                "undefined_replicates": replicates,
                "requested_replicates": replicates,
                "seed": seed,
            }
            continue
        selected_scores = scores[selected]
        index_by_subject = {
            subject: np.flatnonzero(selected_subjects == subject)
            for subject in unique_subjects
        }
        fractions: list[float] = []
        undefined = 0
        for sampled in subject_bootstrap_plan(
            unique_subjects, replicates=replicates, seed=seed
        ):
            indices = np.concatenate([index_by_subject[subject] for subject in sampled])
            if indices.size == 0:
                undefined += 1
            else:
                fractions.append(float(np.mean(selected_scores[indices] >= threshold)))
        output[challenge] = {
            "lower_95": (
                None if not fractions else float(np.percentile(fractions, 2.5))
            ),
            "upper_95": (
                None if not fractions else float(np.percentile(fractions, 97.5))
            ),
            "successful_replicates": len(fractions),
            "undefined_replicates": undefined,
            "requested_replicates": replicates,
            "seed": seed,
        }
    return output


def positive_context_analysis(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    subjects: NDArray[np.str_],
    context_flags: NDArray[np.str_],
    threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    """Report descriptive positive strata with mandatory denominators."""
    positive = labels == 1
    has_axis = np.char.find(context_flags, "axis_shift_context") >= 0
    has_conduction = np.char.find(context_flags, "conduction_change_context") >= 0
    has_noise = np.char.find(context_flags, "point_noise_context") >= 0
    strata = {
        "no_axis_or_conduction_context": positive & ~has_axis & ~has_conduction,
        "axis_shift_context": positive & has_axis,
        "conduction_change_context": positive & has_conduction,
        "point_noise_context": positive & has_noise,
    }
    return {
        name: {
            "contributing_subject_count": len(set(subjects[selected].tolist())),
            "window_count": int(np.sum(selected)),
            "true_positive_count": int(np.sum(scores[selected] >= threshold)),
            "sensitivity": (
                None
                if not selected.any()
                else float(np.mean(scores[selected] >= threshold))
            ),
        }
        for name, selected in strata.items()
    }
