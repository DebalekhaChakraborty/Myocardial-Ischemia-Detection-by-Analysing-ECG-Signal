from pathlib import Path
from types import SimpleNamespace

import pytest

from cardiosentinel.neural import metadata
from cardiosentinel.neural.metadata import (
    B4WindowReference,
    build_training_index,
    canonical_selection_sha256,
    load_b4_references,
)


def make_reference(index: int, family: str, subject: str) -> B4WindowReference:
    start = index * 2500
    return B4WindowReference(
        stable_id=f"ltstdb:r1:0:{start}:{start + 2500}",
        record_id="r1",
        subject_id=subject,
        channel_index=0,
        start_sample=start,
        end_sample=start + 2500,
        partition="train",
        target_family=family,
        context_flags=(),
    )


def test_selection_digest_is_order_independent_and_rejects_duplicates() -> None:
    assert canonical_selection_sha256(["b", "a"]) == canonical_selection_sha256(
        ["a", "b"]
    )
    with pytest.raises(ValueError, match="duplicate"):
        canonical_selection_sha256(["a", "a"])


def test_test_partition_rejected_before_any_path_resolution(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("artifact path was resolved")

    monkeypatch.setattr(metadata, "require_nonversioned_path", forbidden)
    monkeypatch.setattr(metadata, "read_json", forbidden)
    with pytest.raises(ValueError, match="train and validation only"):
        load_b4_references(Path("/sealed/test"), "test")


def test_exact_training_selection_uses_existing_sampler(monkeypatch, tmp_path) -> None:
    references = (
        make_reference(0, "ischemic_positive", "s1"),
        make_reference(1, "background_negative", "s1"),
        make_reference(2, "background_negative", "s2"),
    )
    selected_ids = frozenset(item.stable_id for item in references)
    plan = SimpleNamespace(
        selected_positive_count=1,
        selected_negative_count=2,
        selected_count=3,
        selected_subject_count=2,
        selected_stable_ids=selected_ids,
    )
    monkeypatch.setitem(
        metadata.EXPECTED_COUNTS,
        "train",
        {"positive": 1, "negative": 2, "total": 3, "subjects": 2},
    )
    monkeypatch.setattr(metadata, "require_nonversioned_path", lambda path, _: path)
    monkeypatch.setattr(metadata, "build_training_selection_plan", lambda *a, **k: plan)
    monkeypatch.setattr(metadata, "load_b4_references", lambda *a, **k: references)

    index = build_training_index(tmp_path)

    assert index.references == references
    assert index.selection_sha256 == canonical_selection_sha256(selected_ids)
    assert not hasattr(index.references[0], "features")
