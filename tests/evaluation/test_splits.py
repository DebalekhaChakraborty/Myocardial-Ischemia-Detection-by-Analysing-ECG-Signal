"""SYNTHETIC split-integrity and frozen-registry tests."""

from pathlib import Path

import pytest

from cardiosentinel.data.models import DatasetRecord, ParsedAnnotations
from cardiosentinel.evaluation.models import SubjectBurden
from cardiosentinel.evaluation.protocol import LTSTDB_V1_SPLIT_SHA256
from cardiosentinel.evaluation.provenance import (
    conservative_edb_overlap_exclusions,
    verified_overlap_registry,
)
from cardiosentinel.evaluation.splits import (
    PARTITIONS,
    assign_subjects,
    build_subject_burdens,
    generate_split_manifest,
    load_split_manifest,
    validate_split_manifest,
)


def burden(index: int, records: int = 1) -> SubjectBurden:
    return SubjectBurden(
        subject_id=f"ltstdb:s{index:04d}",
        records=tuple(f"s{index:04d}{item}" for item in range(1, records + 1)),
        recording_seconds=86_400.0 * records,
        ischemic_episode_count=index % 5 + 1,
        ischemic_episode_seconds=float((index % 5 + 1) * 60),
        rate_related_episode_count=index % 3,
        rate_related_episode_seconds=float((index % 3) * 45),
        signal_channel_count=2 * records,
    )


def test_subject_assignment_is_deterministic_disjoint_and_capacity_exact() -> None:
    burdens = tuple(burden(index, 2 if index < 6 else 1) for index in range(80))
    first = assign_subjects(burdens, 2026)
    second = assign_subjects(reversed(burdens), 2026)
    assert first == second
    assert [len(first[partition]) for partition in PARTITIONS] == [56, 12, 12]
    assigned = [subject for partition in PARTITIONS for subject in first[partition]]
    assert len(assigned) == len(set(assigned)) == 80


def test_multi_record_subject_burden_keeps_records_together() -> None:
    records = tuple(
        DatasetRecord(
            "ltstdb",
            "1.0.0",
            record_id,
            "ltstdb:s2027",
            250.0,
            ("I", "II"),
            ("I", "II"),
            2,
            10_000,
            40.0,
            "SYNTHETIC",
        )
        for record_id in ("s20271", "s20272")
    )
    parsed = (ParsedAnnotations((), (), ()), ParsedAnnotations((), (), ()))
    burdens = build_subject_burdens(records, parsed)
    assert len(burdens) == 1
    assert burdens[0].records == ("s20271", "s20272")


def test_verified_cross_dataset_registry_contains_all_official_header_pairs() -> None:
    registry = verified_overlap_registry()
    pairs = {
        (item.record, item.verified_corresponding_record) for item in registry
    }
    assert len(registry) == 10
    assert ("s20021", "e0113") in pairs
    assert all(item.confidence == "verified" for item in registry)
    assert all(item.evidence_source.endswith(f"{item.record}.hea") for item in registry)
    exclusions = conservative_edb_overlap_exclusions()
    assert len(exclusions) == 15
    assert {"e0123", "e0124", "e0163", "e0133"} <= set(exclusions)


def test_committed_v1_split_hash_is_stable() -> None:
    path = Path(__file__).resolve().parents[2] / "protocols/splits/ltstdb_v1.json"
    manifest = load_split_manifest(path)
    validate_split_manifest(manifest, expected_hash=LTSTDB_V1_SPLIT_SHA256)
    assert manifest["split_sha256"] == LTSTDB_V1_SPLIT_SHA256
    assert [
        len(manifest["partitions"][partition]["subjects"])
        for partition in PARTITIONS
    ] == [56, 12, 12]


def test_manifest_hash_detects_assignment_drift() -> None:
    records = tuple(
        DatasetRecord(
            "ltstdb",
            "1.0.0",
            f"s{index:04d}1",
            f"ltstdb:s{index:04d}",
            250.0,
            ("I",),
            ("I",),
            1,
            10_000,
            40.0,
            "SYNTHETIC",
        )
        for index in range(6)
    )
    parsed = tuple(ParsedAnnotations((), (), ()) for _ in records)
    manifest = generate_split_manifest(records, parsed, "synthetic-sha")
    train_subjects = manifest["partitions"]["train"]["subjects"]
    validation_subjects = manifest["partitions"]["validation"]["subjects"]
    train_subjects[0], validation_subjects[0] = (
        validation_subjects[0],
        train_subjects[0],
    )
    with pytest.raises(ValueError, match="hash"):
        validate_split_manifest(manifest)
