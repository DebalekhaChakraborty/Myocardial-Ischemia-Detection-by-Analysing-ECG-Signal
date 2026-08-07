"""SYNTHETIC training-sampling and metrics-protocol tests."""

import math
import random

import pytest

from cardiosentinel.evaluation.metrics import (
    challenge_false_positive_rate,
    challenge_report,
    challenge_subject_bootstrap_plan,
    ischemic_positive_context_strata,
    select_validation_f1_threshold,
    subject_bootstrap_plan,
)
from cardiosentinel.evaluation.models import WindowTarget
from cardiosentinel.evaluation.protocol import challenge_evidence_policy
from cardiosentinel.evaluation.sampling import (
    primary_evaluation_targets,
    sample_training_targets,
)


def reference_validation_f1_threshold(
    labels: list[int], scores: list[float], *, partition: str
) -> float:
    """Test-only copy of the original exhaustive threshold implementation."""
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


def target(
    family: str,
    index: int,
    subject: str,
    record: str,
) -> WindowTarget:
    primary = family in {"ischemic_positive", "background_negative"}
    return WindowTarget(
        dataset="ltstdb",
        record_id=record,
        subject_id=subject,
        channel_index=0,
        lead_name="I",
        window_start_sample=index * 10,
        window_end_sample=index * 10 + 10,
        target_family=family,
        target_subtype="SYNTHETIC",
        eligible_for_training=primary,
        eligible_for_primary_evaluation=primary,
        eligible_for_confounder_evaluation=family.endswith("_confounder"),
        annotation_definition="ltstdb.stb",
        overlapping_event_ids=(),
        overlapping_marker_ids=(),
        context_flags=(),
        quality_state=None,
        exclusion_reason=None,
    )


def test_training_sampler_is_deterministic_and_at_most_three_to_one() -> None:
    targets = [
        target("ischemic_positive", index, "subject-positive", "record-positive")
        for index in range(2)
    ]
    targets.extend(
        target(
            "background_negative",
            index + 10,
            f"subject-{index % 4}",
            f"record-{index % 4}",
        )
        for index in range(20)
    )
    first = sample_training_targets(targets, partition="train", seed=2026)
    second = sample_training_targets(reversed(targets), partition="train", seed=2026)
    assert first == second
    positives = [item for item in first if item.target_family == "ischemic_positive"]
    negatives = [item for item in first if item.target_family == "background_negative"]
    assert len(positives) == 2
    assert len(negatives) == 6
    assert len({item.subject_id for item in negatives}) > 1


def test_test_evaluation_is_full_and_cannot_use_training_sampler() -> None:
    targets = tuple(
        target("background_negative", index, "subject", "record")
        for index in range(20)
    )
    assert primary_evaluation_targets(targets, partition="test") == targets
    with pytest.raises(ValueError, match="training"):
        sample_training_targets(targets, partition="test")


def test_bootstrap_resamples_subjects_not_windows() -> None:
    plan = subject_bootstrap_plan(["a", "a", "b", "b", "c"], replicates=25, seed=7)
    assert len(plan) == 25
    assert all(len(replicate) == 3 for replicate in plan)
    assert all(set(replicate) <= {"a", "b", "c"} for replicate in plan)
    assert any(len(set(replicate)) < 3 for replicate in plan)


def test_threshold_selection_is_validation_only_with_deterministic_ties() -> None:
    threshold = select_validation_f1_threshold(
        [0, 1, 1, 0], [0.1, 0.7, 0.8, 0.2], partition="validation"
    )
    assert threshold == 0.7
    with pytest.raises(ValueError, match="validation"):
        select_validation_f1_threshold(
            [0, 1], [0.1, 0.9], partition="test"
        )


@pytest.mark.parametrize("seed", range(20))
def test_optimized_threshold_matches_exhaustive_reference_on_random_scores(
    seed: int,
) -> None:
    generator = random.Random(seed)
    labels = [generator.randrange(2) for _ in range(97)]
    labels[0] = 1
    scores = [round(generator.random(), 3) for _ in labels]

    assert select_validation_f1_threshold(
        labels, scores, partition="validation"
    ) == reference_validation_f1_threshold(labels, scores, partition="validation")


@pytest.mark.parametrize(
    ("labels", "scores"),
    (
        ([1, 0, 1, 0, 1], [0.8, 0.8, 0.5, 0.5, 0.2]),
        ([0, 1, 0, 1], [0.4, 0.4, 0.4, 0.4]),
        ([1, 0, 0, 0, 0, 0, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
        ([1, 1, 0, 0], [0.9, 0.8, 0.8, 0.8]),
        ([1, 0, 1, 0], [0.1, 0.4, 0.7, 0.9]),
        ([0, 1, 0, 1], [0.9, 0.7, 0.4, 0.1]),
    ),
)
def test_optimized_threshold_matches_exhaustive_reference_on_edge_cases(
    labels: list[int], scores: list[float]
) -> None:
    assert select_validation_f1_threshold(
        labels, scores, partition="validation"
    ) == reference_validation_f1_threshold(labels, scores, partition="validation")


def test_threshold_ties_select_the_highest_observed_score() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.8, 0.8]

    assert select_validation_f1_threshold(labels, scores, partition="validation") == 0.9


def test_threshold_selection_retains_nan_score_semantics() -> None:
    labels = [1, 0, 1]
    scores = [float("nan"), 0.8, 0.7]

    assert select_validation_f1_threshold(
        labels, scores, partition="validation"
    ) == reference_validation_f1_threshold(labels, scores, partition="validation")

    all_nan = [float("nan")]
    assert math.isnan(
        select_validation_f1_threshold([1], all_nan, partition="validation")
    )


@pytest.mark.parametrize(
    ("labels", "scores", "match"),
    (
        ([], [], "non-empty and aligned"),
        ([1], [], "non-empty and aligned"),
        ([2], [0.5], "binary"),
        ([0, 0], [0.2, 0.8], "at least one positive"),
    ),
)
def test_threshold_selection_retains_input_validation_contract(
    labels: list[int], scores: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        select_validation_f1_threshold(labels, scores, partition="validation")


def test_threshold_selection_handles_many_unique_scores_without_quadratic_sweep(
) -> None:
    count = 50_000
    labels = [1 if index % 97 == 0 else 0 for index in range(count)]
    scores = [index / count for index in range(count)]

    threshold = select_validation_f1_threshold(labels, scores, partition="validation")

    assert threshold in scores


def test_mixed_positives_do_not_enter_challenge_fpr_denominators() -> None:
    mixed_positive = target("ischemic_positive", 0, "subject-a", "record-a")
    mixed_positive = WindowTarget(
        **{
            **mixed_positive.__dict__,
            "context_flags": ("axis_shift_context", "conduction_change_context"),
        }
    )
    axis_challenge = target("axis_shift_confounder", 1, "subject-b", "record-b")
    targets = (mixed_positive, axis_challenge)
    assert challenge_false_positive_rate(
        targets, (0.99, 0.10), 0.5, "axis_shift"
    ) == 0.0
    assert challenge_false_positive_rate(
        targets, (0.99, 0.10), 0.5, "conduction_change"
    ) is None


def test_positive_context_strata_are_descriptive_and_overlapping() -> None:
    plain = target("ischemic_positive", 0, "subject-a", "record-a")
    mixed = WindowTarget(
        **{
            **target("ischemic_positive", 1, "subject-b", "record-b").__dict__,
            "context_flags": (
                "axis_shift_context",
                "conduction_change_context",
                "point_noise_context",
            ),
        }
    )
    strata = ischemic_positive_context_strata((plain, mixed))
    assert strata["no_axis_or_conduction_context"] == (plain,)
    assert strata["axis_shift_context"] == (mixed,)
    assert strata["conduction_change_context"] == (mixed,)
    assert strata["point_noise_context"] == (mixed,)


def test_challenge_evidence_policy_bounds_conduction_claims() -> None:
    rate = challenge_evidence_policy("rate_related")
    axis = challenge_evidence_policy("axis_shift")
    conduction = challenge_evidence_policy("conduction_change")
    assert rate.evidence_level == "quantitative_secondary"
    assert axis.evidence_level == "quantitative_secondary"
    assert conduction.evidence_level == "exploratory_descriptive"
    assert all(
        policy.is_headline_metric is False for policy in (rate, axis, conduction)
    )
    assert conduction.supports_inferential_bootstrap is False


def test_conduction_report_keeps_fp_numerator_and_denominator() -> None:
    targets = (
        target("conduction_change_confounder", 0, "subject-a", "record-a"),
        target("conduction_change_confounder", 1, "subject-a", "record-a"),
    )
    report = challenge_report(targets, (0.9, 0.1), 0.5, "conduction_change")
    assert report.contributing_subject_count == 1
    assert report.challenge_window_count == 2
    assert report.false_positive_count == 1
    assert report.false_positive_fraction == 0.5
    assert report.evidence_level == "exploratory_descriptive"
    assert report.is_headline_metric is False
    assert report.inference_available is False
    with pytest.raises(ValueError, match="exploratory descriptive"):
        challenge_subject_bootstrap_plan(targets, "conduction_change")


def test_quantitative_challenge_bootstrap_requires_two_subjects() -> None:
    one_subject = (
        target("axis_shift_confounder", 0, "subject-a", "record-a"),
    )
    with pytest.raises(ValueError, match="at least two"):
        challenge_subject_bootstrap_plan(one_subject, "axis_shift")

    two_subjects = one_subject + (
        target("axis_shift_confounder", 1, "subject-b", "record-b"),
    )
    plan = challenge_subject_bootstrap_plan(
        two_subjects, "axis_shift", replicates=5, seed=7
    )
    assert len(plan) == 5
    assert all(len(replicate) == 2 for replicate in plan)
