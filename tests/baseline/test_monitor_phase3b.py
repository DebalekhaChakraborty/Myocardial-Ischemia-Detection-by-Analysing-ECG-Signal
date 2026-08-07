"""Offline tests for the read-only Phase 3B materialization monitor."""

import json
from pathlib import Path

from scripts.monitor_phase3b import inspect_feature_root, render_monitor_report


def write_manifest(
    root: Path, records: list[dict[str, object]], **extra: object
) -> Path:
    root.mkdir()
    path = root / "manifest.json"
    path.write_text(json.dumps({"records": records, **extra}, indent=2) + "\n")
    return path


def completed_record(
    record_id: str,
    partition: str,
    row_count: int,
    cache_sha256: str = "a" * 64,
    morphology_quality: dict[str, int] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "record_id": record_id,
        "partition": partition,
        "row_count": row_count,
        "cache_sha256": cache_sha256,
        "status": "complete",
    }
    if morphology_quality is not None:
        record["morphology_quality"] = morphology_quality
    return record


def test_monitor_handles_missing_feature_root(tmp_path: Path) -> None:
    report = inspect_feature_root(tmp_path / "missing")

    assert report.status == "feature root does not exist"
    assert "Status: feature root does not exist" in render_monitor_report(report)


def test_monitor_handles_missing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "features"
    root.mkdir()

    report = inspect_feature_root(root)

    assert report.status == "manifest.json does not exist"
    assert "Status: manifest.json does not exist" in render_monitor_report(report)


def test_monitor_reports_partially_completed_manifest_and_cache_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    records = [
        completed_record("r1", "train", 12, cache_sha256="1" * 64),
        {"record_id": "r2", "partition": "validation", "status": "pending"},
        completed_record("r3", "test", 8, cache_sha256="3" * 64),
    ]
    write_manifest(root, records)
    (root / "train").mkdir()
    (root / "train" / "r1.npz").write_bytes(b"cache")

    report = inspect_feature_root(root)
    rendered = render_monitor_report(report)

    assert report.completed_records == 2
    assert report.total_windows == 20
    assert report.partitions == (("train", 1), ("validation", 0), ("test", 1))
    assert report.cache_file_count == 1
    assert report.finalized is False
    assert "Completed records: 2 / 86 (2.3%)" in rendered
    assert "r1  train  12  111111111111" in rendered
    assert "r3  test  8  333333333333" in rendered


def test_monitor_reports_finalized_corpus_sha256(tmp_path: Path) -> None:
    root = tmp_path / "features"
    corpus_sha256 = "f" * 64
    write_manifest(
        root,
        [completed_record("r1", "train", 12)],
        feature_corpus_sha256=corpus_sha256,
    )

    report = inspect_feature_root(root)

    assert report.finalized is True
    assert report.feature_corpus_sha256 == corpus_sha256
    assert f"Feature corpus SHA-256: {corpus_sha256}" in render_monitor_report(report)


def test_monitor_aggregates_morphology_quality(tmp_path: Path) -> None:
    root = tmp_path / "features"
    write_manifest(
        root,
        [
            completed_record(
                "r1",
                "train",
                10,
                morphology_quality={
                    "morphology_valid_windows": 8,
                    "morphology_invalid_windows": 2,
                },
            ),
            completed_record(
                "r2",
                "validation",
                5,
                morphology_quality={
                    "morphology_valid_windows": 3,
                    "morphology_invalid_windows": 2,
                },
            ),
        ],
    )

    report = inspect_feature_root(root)
    rendered = render_monitor_report(report)

    assert report.morphology_valid_windows == 11
    assert report.morphology_invalid_windows == 4
    assert "Validity: 73.3%" in rendered


def test_monitor_does_not_mutate_manifest(tmp_path: Path) -> None:
    root = tmp_path / "features"
    manifest_path = write_manifest(root, [completed_record("r1", "train", 12)])
    before = manifest_path.read_bytes()

    inspect_feature_root(root)

    assert manifest_path.read_bytes() == before
