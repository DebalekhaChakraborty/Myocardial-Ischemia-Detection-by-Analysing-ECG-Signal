"""SYNTHETIC equivalence tests for metadata-only training sampling."""

from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.baseline.cache import (
    FeatureTable,
    write_feature_table_atomic,
    write_json_atomic,
)
from cardiosentinel.baseline.sampling import (
    build_training_selection_plan,
    sample_training_indices,
)
from cardiosentinel.evaluation.models import WindowTarget
from cardiosentinel.evaluation.sampling import sample_training_targets
from cardiosentinel.features.schema import COMBINED_V1


def target(family: str, index: int, subject: str, record: str) -> WindowTarget:
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
        eligible_for_training=True,
        eligible_for_primary_evaluation=True,
        eligible_for_confounder_evaluation=False,
        annotation_definition="ltstdb.stb",
        overlapping_event_ids=(),
        overlapping_marker_ids=(),
        context_flags=(),
        quality_state=None,
        exclusion_reason=None,
    )


def assert_fast_matches_reference(targets: list[WindowTarget]) -> None:
    stable_ids = np.asarray([item.stable_id for item in targets])
    indices = sample_training_indices(
        stable_ids,
        np.asarray([item.subject_id for item in targets]),
        np.asarray([item.record_id for item in targets]),
        np.asarray([item.target_family for item in targets]),
    )
    fast_ids = tuple(stable_ids[indices].tolist())
    reference_ids = tuple(
        item.stable_id
        for item in sample_training_targets(targets, partition="train", seed=2026)
    )
    assert fast_ids == reference_ids
    repeated = sample_training_indices(
        stable_ids,
        np.asarray([item.subject_id for item in targets]),
        np.asarray([item.record_id for item in targets]),
        np.asarray([item.target_family for item in targets]),
    )
    assert np.array_equal(indices, repeated)


def test_fast_sampler_multiple_subjects_records_and_small_budget() -> None:
    targets = [
        target("ischemic_positive", index, "positive", "p1") for index in range(3)
    ]
    targets += [
        target("background_negative", 100 + index, "positive", f"p{index % 2 + 1}")
        for index in range(30)
    ]
    targets += [
        target("background_negative", 200 + index, "zero", f"z{index % 3 + 1}")
        for index in range(30)
    ]
    assert_fast_matches_reference(targets)


def test_fast_sampler_budget_larger_than_available_negatives() -> None:
    targets = [
        target("ischemic_positive", index, "positive", "p1") for index in range(5)
    ]
    targets += [
        target("background_negative", 100 + index, "positive", "p1")
        for index in range(4)
    ]
    assert_fast_matches_reference(targets)


def test_fast_sampler_preserves_zero_positive_subject_cap() -> None:
    targets = [
        target("ischemic_positive", index, "positive", "p1") for index in range(20)
    ]
    targets += [
        target("background_negative", 100 + index, "zero", f"z{index % 2 + 1}")
        for index in range(100)
    ]
    targets += [
        target("background_negative", 300 + index, "positive", "p1")
        for index in range(100)
    ]
    assert_fast_matches_reference(targets)


def test_metadata_plan_matches_canonical_sampler_exactly(tmp_path: Path) -> None:
    targets = [
        target("ischemic_positive", index, "positive", "p1") for index in range(7)
    ]
    targets += [
        target("background_negative", 100 + index, "positive", f"p{index % 2 + 1}")
        for index in range(50)
    ]
    targets += [
        target("background_negative", 300 + index, "zero", f"z{index % 3 + 1}")
        for index in range(80)
    ]
    root = tmp_path / "features"
    entries = []
    for record_id in sorted({item.record_id for item in targets}):
        record_targets = [item for item in targets if item.record_id == record_id]
        count = len(record_targets)
        path = root / "train" / f"{record_id}.npz"
        table = FeatureTable(
            features=np.zeros((count, len(COMBINED_V1.names))),
            stable_ids=np.asarray([item.stable_id for item in record_targets]),
            record_ids=np.asarray([item.record_id for item in record_targets]),
            subject_ids=np.asarray([item.subject_id for item in record_targets]),
            channel_indices=np.zeros(count, dtype=np.int64),
            lead_names=np.asarray(["I"] * count),
            window_start_samples=np.arange(count, dtype=np.int64),
            window_end_samples=np.arange(count, dtype=np.int64) + 1,
            partitions=np.asarray(["train"] * count),
            target_families=np.asarray([item.target_family for item in record_targets]),
            context_flags=np.asarray([""] * count),
        )
        write_feature_table_atomic(path, table, {"SYNTHETIC": True})
        entries.append(
            {
                "record_id": record_id,
                "partition": "train",
                "status": "complete",
                "cache_path": f"train/{record_id}.npz",
            }
        )
    write_json_atomic(root / "manifest.json", {"records": entries})
    plan = build_training_selection_plan(root, sampled=True)
    expected = frozenset(
        item.stable_id
        for item in sample_training_targets(targets, partition="train", seed=2026)
    )
    assert plan.selected_stable_ids == expected


def test_b0_metadata_plan_never_loads_numeric_feature_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "features"
    cache_path = root / "train" / "record.npz"
    table = FeatureTable(
        features=np.full((4, len(COMBINED_V1.names)), 99.0),
        stable_ids=np.asarray(
            [f"ltstdb:r:0:{index}:{index + 1}" for index in range(4)]
        ),
        record_ids=np.asarray(["r"] * 4),
        subject_ids=np.asarray(["s"] * 4),
        channel_indices=np.zeros(4, dtype=np.int64),
        lead_names=np.asarray(["I"] * 4),
        window_start_samples=np.arange(4, dtype=np.int64),
        window_end_samples=np.arange(4, dtype=np.int64) + 1,
        partitions=np.asarray(["train"] * 4),
        target_families=np.asarray(
            ["ischemic_positive", "background_negative", "background_negative", "other"]
        ),
        context_flags=np.asarray([""] * 4),
    )
    write_feature_table_atomic(cache_path, table, {"SYNTHETIC": True})
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        '{"records":[{"record_id":"r","partition":"train",'
        '"status":"complete","cache_path":"train/record.npz"}]}',
        encoding="utf-8",
    )

    def fail_if_full_table_is_loaded(*args, **kwargs):
        raise AssertionError("B0 metadata planning must not load numeric features")

    monkeypatch.setattr(
        "cardiosentinel.baseline.sampling.read_feature_table",
        fail_if_full_table_is_loaded,
    )
    plan = build_training_selection_plan(root, sampled=False)
    assert plan.unsampled_positive_count == 1
    assert plan.unsampled_negative_count == 2
    assert plan.selected_stable_ids == frozenset()
