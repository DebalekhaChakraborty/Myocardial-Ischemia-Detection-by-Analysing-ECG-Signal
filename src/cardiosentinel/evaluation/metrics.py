"""Pre-model metric protocol helpers with subject-level resampling."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from cardiosentinel.evaluation.protocol import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED


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
