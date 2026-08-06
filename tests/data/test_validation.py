"""SYNTHETIC leakage and manifest validation fixtures only."""

from pathlib import Path

import pytest

from cardiosentinel.data.manifest import destination_is_git_ignored, write_manifest
from cardiosentinel.data.provenance import verify_sha256_manifest
from cardiosentinel.data.validation import validate_subject_partitions


def test_subject_leakage_across_records_fails() -> None:
    with pytest.raises(ValueError, match="Leakage risk"):
        validate_subject_partitions(
            {"subject-a": "train"},
            {"record-1": "subject-a", "record-2": "subject-a"},
            {"record-1": "train", "record-2": "test"},
        )


def test_subject_partition_requires_subject_assignment() -> None:
    with pytest.raises(ValueError, match="no partition assignment"):
        validate_subject_partitions(
            {}, {"record-1": "subject-a"}, {"record-1": "train"}
        )


def test_git_tracked_download_destination_is_rejected() -> None:
    assert destination_is_git_ignored(Path("data/raw/edb")) is True
    assert destination_is_git_ignored(Path("downloads/edb")) is False


def test_checksum_manifest_rejects_missing_or_modified_files(tmp_path: Path) -> None:
    data_file = tmp_path / "record.hea"
    data_file.write_text("synthetic metadata\n", encoding="utf-8")
    checksum_file = tmp_path / "SHA256SUMS.txt"
    checksum_file.write_text(
        "60352a5299d2b088d2c74e14681e20ba706810c296829db89ef2928e305e3115"
        "  record.hea\n",
        encoding="utf-8",
    )
    assert verify_sha256_manifest(checksum_file, tmp_path)["record.hea"].startswith(
        "60352"
    )
    data_file.write_text("modified synthetic metadata\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256_manifest(checksum_file, tmp_path)


def test_manifest_json_is_deterministic(tmp_path: Path) -> None:
    manifest = {"z": [2, 1], "a": {"name": "synthetic"}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_manifest(manifest, first)
    write_manifest(manifest, second)
    assert first.read_bytes() == second.read_bytes()
