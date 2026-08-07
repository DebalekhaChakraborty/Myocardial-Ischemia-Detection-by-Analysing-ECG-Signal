"""Offline Git-backed storage policy tests for external experiment artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import cardiosentinel.baseline.cache as cache
import cardiosentinel.baseline.workflow as workflow
from cardiosentinel.baseline.cache import (
    FeatureTable,
    finalize_feature_corpus,
    require_nonversioned_path,
    validate_feature_corpus,
    write_feature_table_atomic,
    write_json_atomic,
)
from cardiosentinel.baseline.source import (
    OFFICIAL_MANIFEST_NAME,
    RECORDS_NAME,
    _verify_source_against_manifest,
    required_source_filenames,
)
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.evaluation.protocol import LTSTDB_V1_SPLIT_SHA256
from cardiosentinel.features.schema import COMBINED_V1


def configure_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, ignored_roots: bool = True
) -> Path:
    """Create a temporary Git repository with the local artifact policy."""
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    ignore_lines = ["/.vscode/"]
    if ignored_roots:
        ignore_lines.extend(
            (
                "/cardiosentinel-data/",
                "/cardiosentinel-features/",
                "/cardiosentinel-runs/",
            )
        )
    (repository / ".gitignore").write_text("\n".join(ignore_lines) + "\n")
    resolved_repository = repository.resolve()
    monkeypatch.setattr(cache, "REPOSITORY_ROOT", resolved_repository)
    monkeypatch.setattr(
        cache,
        "LOCAL_NONVERSIONED_ROOTS",
        tuple(
            (resolved_repository / name).resolve()
            for name in (
                "cardiosentinel-data",
                "cardiosentinel-features",
                "cardiosentinel-runs",
            )
        ),
    )
    return resolved_repository


@pytest.mark.parametrize(
    "root_name",
    ("cardiosentinel-data", "cardiosentinel-features", "cardiosentinel-runs"),
)
def test_approved_ignored_repository_roots_are_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_name: str
) -> None:
    repository = configure_repository(tmp_path, monkeypatch)
    path = repository / root_name / "nested" / "artifact"

    assert require_nonversioned_path(path, "Artifact") == path.resolve()


def test_outside_repository_path_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = configure_repository(tmp_path, monkeypatch)
    path = repository.parent / "external" / "artifact"

    def unavailable(*_: object, **__: object) -> None:
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(cache.subprocess, "run", unavailable)

    assert require_nonversioned_path(path, "Artifact") == path.resolve()


def test_arbitrary_or_ignored_nonapproved_repository_paths_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = configure_repository(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="approved ignored"):
        require_nonversioned_path(repository / "ordinary-output", "Artifact")
    with pytest.raises(ValueError, match="approved ignored"):
        require_nonversioned_path(repository / ".vscode" / "settings.json", "Artifact")


def test_approved_root_must_be_git_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = configure_repository(tmp_path, monkeypatch, ignored_roots=False)

    with pytest.raises(ValueError, match="must be Git-ignored"):
        require_nonversioned_path(
            repository / "cardiosentinel-features" / "features", "Feature root"
        )


def test_git_unavailable_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = configure_repository(tmp_path, monkeypatch)

    def unavailable(*_: object, **__: object) -> None:
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(cache.subprocess, "run", unavailable)
    with pytest.raises(ValueError, match="cannot establish"):
        require_nonversioned_path(
            repository / "cardiosentinel-data" / "source", "Waveform source"
        )


def test_traversal_and_symlink_cannot_escape_approved_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = configure_repository(tmp_path, monkeypatch)
    approved = repository / "cardiosentinel-data"
    nonapproved = repository / "secrets"
    approved.mkdir()
    nonapproved.mkdir()
    (approved / "escape").symlink_to(nonapproved, target_is_directory=True)

    with pytest.raises(ValueError, match="approved ignored"):
        require_nonversioned_path(approved / ".." / "secrets" / "item", "Artifact")
    with pytest.raises(ValueError, match="approved ignored"):
        require_nonversioned_path(approved / "escape" / "item", "Artifact")


def write_source_fixture(root: Path, records: tuple[str, ...]) -> Path:
    root.mkdir(parents=True)
    (root / RECORDS_NAME).write_text("\n".join(records) + "\n", encoding="utf-8")
    required = required_source_filenames(records)
    for filename in required:
        if filename != RECORDS_NAME:
            (root / filename).write_bytes(filename.encode("ascii"))
    manifest = root / OFFICIAL_MANIFEST_NAME
    manifest.write_text(
        "\n".join(f"{sha256_file(root / name)} {name}" for name in required) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_source_verification_allows_repository_local_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = configure_repository(tmp_path, monkeypatch)
    records = ("s20001",)
    source = repository / "cardiosentinel-data" / "fixture"
    manifest = write_source_fixture(source, records)

    receipt = _verify_source_against_manifest(
        source,
        records,
        official_manifest_sha256=sha256_file(manifest),
        write_receipt=True,
    )

    assert receipt["verification_result"] == "passed"


def write_feature_fixture(root: Path) -> tuple[set[str], str]:
    cache_path = root / "train" / "r1.npz"
    table = FeatureTable(
        features=np.zeros((1, len(COMBINED_V1.names))),
        stable_ids=np.asarray(["ltstdb:r1:0:0:1"]),
        record_ids=np.asarray(["r1"]),
        subject_ids=np.asarray(["subject-1"]),
        channel_indices=np.asarray([0], dtype=np.int64),
        lead_names=np.asarray(["I"]),
        window_start_samples=np.asarray([0], dtype=np.int64),
        window_end_samples=np.asarray([1], dtype=np.int64),
        partitions=np.asarray(["train"]),
        target_families=np.asarray(["background_negative"]),
        context_flags=np.asarray([""]),
    )
    metadata = {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "record_id": "r1",
        "subject_id": "subject-1",
        "partition": "train",
        "source_sha256": "a" * 64,
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "feature_schema_sha256": COMBINED_V1.sha256,
        "processing_profile": "raw",
        "window_seconds": 10.0,
        "stride_seconds": 5.0,
        "annotation_definition": "ltstdb.stb",
        "row_count": 1,
        "target_counts": {"background_negative": 1},
    }
    write_feature_table_atomic(cache_path, table, metadata)
    manifest = {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "feature_schemas": {"combined_v1": COMBINED_V1.as_dict()},
        "processing_profile": "raw",
        "window_seconds": 10.0,
        "stride_seconds": 5.0,
        "annotation_definition": "ltstdb.stb",
        "records": [
            {
                "record_id": "r1",
                "subject_id": "subject-1",
                "partition": "train",
                "cache_path": "train/r1.npz",
                "status": "complete",
                "row_count": 1,
                "target_counts": {"background_negative": 1},
                "source_sha256": "a" * 64,
                "cache_sha256": sha256_file(cache_path),
            }
        ],
    }
    write_json_atomic(root / "manifest.json", manifest)
    expected = {"r1"}
    return expected, finalize_feature_corpus(root, expected)


def test_feature_validation_allows_repository_local_feature_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = configure_repository(tmp_path, monkeypatch)
    feature_root = repository / "cardiosentinel-features" / "fixture"
    expected, corpus_sha256 = write_feature_fixture(feature_root)

    assert validate_feature_corpus(feature_root, expected) == corpus_sha256


def test_fit_accepts_repository_local_run_artifacts_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = configure_repository(tmp_path, monkeypatch)
    feature_root = repository / "cardiosentinel-features" / "fixture"
    run_root = repository / "cardiosentinel-runs"
    monkeypatch.setattr(
        workflow, "git_provenance", lambda _: {"git_dirty": True}
    )

    with pytest.raises(ValueError, match="requires a clean"):
        workflow.fit_and_lock(
            feature_root, run_root, "fixture", "B0_constant_prior"
        )
