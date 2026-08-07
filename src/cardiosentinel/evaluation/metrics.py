"""Pre-model metric protocol helpers with subject-level resampling."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Literal

from cardiosentinel.evaluation.models import WindowTarget
from cardiosentinel.evaluation.protocol import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED

ChallengeFamily = Literal[
    "rate_related_confounder",
    "axis_shift_confounder",
    "conduction_change_confounder",
]


def subject_bootstrap_plan(
    subject_ids: Iterable[str],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[str, ...], ...]:
    """Return subject resamples; windows are never independent bootstrap units."""
    subjects = tuple(sorted(set(subject_ids)))
    if not subjects or replicates <= 0:
        raise ValueError("Bootstrap requires subjects and positive replicate count.")
    generator = random.Random(seed)
    return tuple(
        tuple(generator.choice(subjects) for _ in subjects)
        for _ in range(replicates)
    )


def select_validation_f1_threshold(
    labels: Sequence[int], scores: Sequence[float], *, partition: str
) -> float:
    """Maximize validation F1; ties select the highest, more specific threshold."""
    if partition != "validation":
        raise ValueError("Threshold selection may use validation predictions only.")
    if len(labels) != len(scores) or not labels:
        raise ValueError("Threshold labels and scores must be non-empty and aligned.")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("Threshold labels must be binary.")
    if not any(labels):
        raise ValueError("Validation labels must contain at least one positive.")
    candidates = sorted(set(float(score) for score in scores), reverse=True)

    def f1(threshold: float) -> float:
        predicted = tuple(score >= threshold for score in scores)
        pairs = tuple(zip(predicted, labels, strict=True))
        true_positive = sum(
            prediction and label == 1 for prediction, label in pairs
        )
        false_positive = sum(
            prediction and label == 0 for prediction, label in pairs
        )
        false_negative = sum(
            not prediction and label == 1 for prediction, label in pairs
        )
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if denominator == 0 else 2 * true_positive / denominator

    return max(candidates, key=lambda threshold: (f1(threshold), threshold))


def challenge_false_positive_rate(
    targets: Sequence[WindowTarget],
    scores: Sequence[float],
    threshold: float,
    challenge_family: ChallengeFamily,
) -> float | None:
    """Compute FPR over one explicitly non-ischemic challenge family only."""
    if len(targets) != len(scores):
        raise ValueError("Challenge targets and scores must be aligned.")
    selected_scores = tuple(
        float(score)
        for target, score in zip(targets, scores, strict=True)
        if target.target_family == challenge_family
        and target.eligible_for_confounder_evaluation
    )
    if not selected_scores:
        return None
    return sum(score >= threshold for score in selected_scores) / len(selected_scores)


def ischemic_positive_context_strata(
    targets: Iterable[WindowTarget],
) -> dict[str, tuple[WindowTarget, ...]]:
    """Return overlapping descriptive strata without creating disease classes."""
    positives = tuple(
        target for target in targets if target.target_family == "ischemic_positive"
    )
    return {
        "no_axis_or_conduction_context": tuple(
            target
            for target in positives
            if "axis_shift_context" not in target.context_flags
            and "conduction_change_context" not in target.context_flags
        ),
        "axis_shift_context": tuple(
            target
            for target in positives
            if "axis_shift_context" in target.context_flags
        ),
        "conduction_change_context": tuple(
            target
            for target in positives
            if "conduction_change_context" in target.context_flags
        ),
        "point_noise_context": tuple(
            target
            for target in positives
            if "point_noise_context" in target.context_flags
        ),
    }
