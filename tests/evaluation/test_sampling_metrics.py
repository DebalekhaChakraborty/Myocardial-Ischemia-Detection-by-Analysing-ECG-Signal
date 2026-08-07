"""SYNTHETIC training-sampling and metrics-protocol tests."""

import pytest

from cardiosentinel.evaluation.metrics import (
    select_validation_f1_threshold,
    subject_bootstrap_plan,
)
from cardiosentinel.evaluation.models import WindowTarget
from cardiosentinel.evaluation.sampling import (
    primary_evaluation_targets,
    sample_training_targets,
)


def target(
    family: str,
    index: int,
    subject: str,
    record: str,
) -> WindowTarget:
    primary = family in {"ischemic_positive", "background_negative"}
    return WindowTarget(
        "ltstdb",
        record,
        subject,
        0,
        "I",
        index * 10,
        index * 10 + 10,
        family,
        "SYNTHETIC",
        primary,
        primary,
        False,
        "ltstdb.stb",
        (),
        None,
        None,
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
