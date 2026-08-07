"""Offline tests for the read-only Phase 3B corpus integrity audit."""

import json
from pathlib import Path

import pytest

from cardiosentinel.baseline.preflight import PRIMARY_COUNTS
from scripts import audit_phase3b_corpus as audit


def split_fixture() -> dict[str, object]:
    return {
        "records_by_subject": {
            "subject-train": ["r-train"],
            "subject-validation": ["r-validation"],
            "subject-test": ["r-test"],
        }
    }


def completed_record(
    record_id: str,
    partition: str,
    *,
    primary_override: dict[str, int] | None = None,
    morphology_valid: int = 10,
    morphology_invalid: int = 2,
) -> dict[str, object]:
    targets = dict(PRIMARY_COUNTS[partition])
    if primary_override is not None:
        targets.update(primary_override)
    targets["rate_related_confounder"] = 3
    return {
        "record_id": record_id,
        "partition": partition,
        "status": "complete",
        "row_count": sum(targets.values()),
        "target_counts": targets,
        "morphology_quality": {
            "morphology_valid_windows": morphology_valid,
            "morphology_invalid_windows": morphology_invalid,
        },
    }


def write_manifest(root: Path, records: list[dict[str, object]]) -> Path:
    root.mkdir()
    manifest = {
        "generation": {"git_sha": "frozen-implementation", "git_dirty": False},
        "records": records,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def configure_successful_audit(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> list[set[str]]:
    expected_calls: list[set[str]] = []
    monkeypatch.setattr(audit, "load_split_manifest", lambda _: split_fixture())

    def validate(feature_root: Path, expected: set[str]) -> str:
        assert feature_root == root
        expected_calls.append(expected)
        return "f" * 64

    monkeypatch.setattr(audit, "validate_feature_corpus", validate)
    return expected_calls


def test_successful_corpus_audit_derives_expected_records_and_reports_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "features"
    write_manifest(
        root,
        [
            completed_record("r-train", "train"),
            completed_record("r-validation", "validation"),
            completed_record("r-test", "test"),
        ],
    )
    expected_calls = configure_successful_audit(monkeypatch, root)

    summary = audit.audit_feature_corpus(root, tmp_path / "frozen-split.json")
    rendered = audit.render_audit(summary)

    assert expected_calls == [{"r-train", "r-validation", "r-test"}]
    assert summary["feature_corpus_sha256"] == "f" * 64
    assert summary["completed_records_by_partition"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert "rate_related_confounder=3" in rendered
    assert "Algorithmic morphology-feature validity" in rendered


def test_primary_count_mismatch_is_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "features"
    write_manifest(
        root,
        [
            completed_record(
                "r-train",
                "train",
                primary_override={"ischemic_positive": 1},
            ),
            completed_record("r-validation", "validation"),
            completed_record("r-test", "test"),
        ],
    )
    configure_successful_audit(monkeypatch, root)

    with pytest.raises(ValueError, match="Primary Benchmark V1 counts differ"):
        audit.audit_feature_corpus(root, tmp_path / "frozen-split.json")


def test_morphology_aggregation_reports_exact_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "features"
    write_manifest(
        root,
        [
            completed_record(
                "r-train", "train", morphology_valid=3, morphology_invalid=1
            ),
            completed_record(
                "r-validation", "validation", morphology_valid=4, morphology_invalid=2
            ),
            completed_record(
                "r-test", "test", morphology_valid=1, morphology_invalid=1
            ),
        ],
    )
    configure_successful_audit(monkeypatch, root)

    summary = audit.audit_feature_corpus(root, tmp_path / "frozen-split.json")

    assert summary["morphology_valid_windows"] == 8
    assert summary["morphology_invalid_windows"] == 4
    assert summary["morphology_validity_fraction"] == pytest.approx(2 / 3)


def test_absent_and_incomplete_corpora_fail_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setattr(audit, "load_split_manifest", lambda _: split_fixture())
    with pytest.raises(FileNotFoundError):
        audit.audit_feature_corpus(missing, tmp_path / "frozen-split.json")

    root = tmp_path / "features"
    manifest_path = write_manifest(root, [completed_record("r-train", "train")])
    before = manifest_path.read_bytes()
    monkeypatch.setattr(
        audit,
        "validate_feature_corpus",
        lambda *_: (_ for _ in ()).throw(ValueError("Feature corpus is incomplete.")),
    )
    with pytest.raises(ValueError, match="incomplete"):
        audit.audit_feature_corpus(root, tmp_path / "frozen-split.json")

    assert manifest_path.read_bytes() == before
