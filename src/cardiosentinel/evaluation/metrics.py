"""Pre-model metric protocol helpers with subject-level resampling."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

from cardiosentinel.evaluation.models import (
    ChallengeName,
    ChallengeReport,
    WindowTarget,
)
from cardiosentinel.evaluation.protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    challenge_evidence_policy,
)


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
    """Maximize validation F1 with an exact O(N log N) cumulative sweep."""
    if partition != "validation":
        raise ValueError("Threshold selection may use validation predictions only.")
    if len(labels) != len(scores) or not labels:
        raise ValueError("Threshold labels and scores must be non-empty and aligned.")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("Threshold labels must be binary.")
    if not any(labels):
        raise ValueError("Validation labels must contain at least one positive.")
    candidates = sorted(set(float(score) for score in scores), reverse=True)
    positive_counts = dict.fromkeys(candidates, 0)
    negative_counts = dict.fromkeys(candidates, 0)
    for label, score in zip(labels, scores, strict=True):
        score_value = float(score)
        if math.isnan(score_value):
            continue
        if label == 1:
            positive_counts[score_value] += 1
        else:
            negative_counts[score_value] += 1

    total_positive = sum(labels)
    true_positive = 0
    false_positive = 0
    best_f1 = -1.0
    best_threshold = candidates[0]
    for threshold in candidates:
        true_positive += positive_counts[threshold]
        false_positive += negative_counts[threshold]
        false_negative = total_positive - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def challenge_false_positive_rate(
    targets: Sequence[WindowTarget],
    scores: Sequence[float],
    threshold: float,
    challenge: ChallengeName,
) -> float | None:
    """Return the count-based false-positive fraction for one challenge."""
    return challenge_report(
        targets, scores, threshold, challenge
    ).false_positive_fraction


def challenge_report(
    targets: Sequence[WindowTarget],
    scores: Sequence[float],
    threshold: float,
    challenge: ChallengeName,
) -> ChallengeReport:
    """Attach V1 evidence policy and denominators to a challenge result."""
    if len(targets) != len(scores):
        raise ValueError("Challenge targets and scores must be aligned.")
    policy = challenge_evidence_policy(challenge)
    selected = tuple(
        (target, float(score))
        for target, score in zip(targets, scores, strict=True)
        if target.target_family == policy.target_family
        and target.eligible_for_confounder_evaluation
    )
    window_count = len(selected)
    false_positive_count = sum(score >= threshold for _, score in selected)
    subject_count = len({target.subject_id for target, _ in selected})
    return ChallengeReport(
        challenge=challenge,
        evidence_level=policy.evidence_level,
        contributing_subject_count=subject_count,
        challenge_window_count=window_count,
        false_positive_count=false_positive_count,
        false_positive_fraction=(
            None if window_count == 0 else false_positive_count / window_count
        ),
        is_headline_metric=policy.is_headline_metric,
        inference_available=(
            policy.supports_inferential_bootstrap and subject_count >= 2
        ),
    )


def challenge_subject_bootstrap_plan(
    targets: Iterable[WindowTarget],
    challenge: ChallengeName,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[str, ...], ...]:
    """Return a plan only for sufficiently supported quantitative challenges."""
    policy = challenge_evidence_policy(challenge)
    if not policy.supports_inferential_bootstrap:
        raise ValueError(
            f"{challenge} is exploratory descriptive and cannot be bootstrapped."
        )
    subjects = tuple(
        target.subject_id
        for target in targets
        if target.target_family == policy.target_family
        and target.eligible_for_confounder_evaluation
    )
    if len(set(subjects)) < 2:
        raise ValueError(
            "Challenge bootstrap inference requires at least two contributing subjects."
        )
    return subject_bootstrap_plan(subjects, replicates=replicates, seed=seed)


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
