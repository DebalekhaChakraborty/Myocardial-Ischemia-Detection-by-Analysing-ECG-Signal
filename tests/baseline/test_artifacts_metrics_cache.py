"""SYNTHETIC cache, lock, pooled, macro, bootstrap, and reporting tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.baseline.artifacts import (
    validate_experiment_lock,
    write_experiment_lock,
)
from cardiosentinel.baseline.cache import (
    FeatureTable,
    compute_feature_corpus_sha256,
    finalize_feature_corpus,
    read_json,
    validate_feature_corpus,
    write_feature_table_atomic,
    write_json_atomic,
)
from cardiosentinel.baseline.metrics import (
    binary_metrics,
    challenge_bootstrap_confidence_intervals,
    challenge_metrics,
    positive_context_analysis,
    subject_bootstrap_confidence_intervals,
    subject_macro_metrics,
)
from cardiosentinel.baseline.workflow import validate_locked_feature_corpus
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.evaluation.protocol import LTSTDB_V1_SPLIT_SHA256
from cardiosentinel.features.schema import COMBINED_V1, SIGNAL_V1


def feature_table(stable_ids: tuple[str, ...]) -> FeatureTable:
    count = len(stable_ids)
    return FeatureTable(
        features=np.zeros((count, len(COMBINED_V1.names))),
        stable_ids=np.asarray(stable_ids),
        record_ids=np.asarray(["record"] * count),
        subject_ids=np.asarray(["subject"] * count),
        channel_indices=np.zeros(count, dtype=np.int64),
        lead_names=np.asarray(["I"] * count),
        window_start_samples=np.arange(count, dtype=np.int64),
        window_end_samples=np.arange(count, dtype=np.int64) + 10,
        partitions=np.asarray(["train"] * count),
        target_families=np.asarray(["background_negative"] * count),
        context_flags=np.asarray([""] * count),
    )


def test_duplicate_stable_ids_are_blocking() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        feature_table(("duplicate", "duplicate"))


def test_primary_metrics_and_subject_macro_keep_undefined_values() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    pooled = binary_metrics(labels, scores, 0.5)
    assert pooled["auprc"] == 1.0
    assert pooled["auroc"] == 1.0
    assert pooled["f1"] == 1.0

    subjects = np.asarray(["negative-only", "mixed", "mixed", "positive-only"])
    macro = subject_macro_metrics(labels, scores, subjects, 0.5)
    assert macro["auroc"]["contributing_subject_count"] == 1
    assert macro["auroc"]["non_contributing_subject_count"] == 2
    assert macro["auprc"]["contributing_subject_count"] == 1
    assert macro["auprc"]["non_contributing_subject_count"] == 2
    assert macro["specificity"]["contributing_subject_count"] == 2


@pytest.mark.parametrize(
    ("labels", "expected_auprc", "expected_auroc"),
    (
        ([1, 1, 1], None, None),
        ([0, 0, 0], None, None),
        ([0, 1, 0, 1], 1.0, 1.0),
    ),
)
def test_discrimination_metrics_require_both_classes(
    labels: list[int],
    expected_auprc: float | None,
    expected_auroc: float | None,
) -> None:
    scores = np.linspace(0.1, 0.9, len(labels))
    if labels == [0, 1, 0, 1]:
        scores = np.asarray([0.1, 0.8, 0.2, 0.9])
    metrics = binary_metrics(np.asarray(labels), scores, 0.5)
    assert metrics["auprc"] == expected_auprc
    assert metrics["auroc"] == expected_auroc


def test_single_class_bootstrap_replicates_are_undefined_for_auprc() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    subjects = np.asarray(["negative", "negative", "positive", "positive"])
    report = subject_bootstrap_confidence_intervals(
        labels, scores, subjects, 0.5, replicates=40, seed=7
    )
    assert report["auprc"]["undefined_replicates"] > 0
    assert (
        report["auprc"]["successful_replicates"]
        + report["auprc"]["undefined_replicates"]
        == 40
    )


def test_subject_bootstrap_reports_successful_and_degenerate_replicates() -> None:
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.1, 0.9, 0.2, 0.8])
    subjects = np.asarray(["a", "a", "b", "b"])
    report = subject_bootstrap_confidence_intervals(
        labels, scores, subjects, 0.5, replicates=20, seed=7
    )
    assert report["auprc"]["successful_replicates"] == 20
    assert report["auprc"]["requested_replicates"] == 20
    assert report["auprc"]["lower_95"] == 1.0


def test_challenge_hierarchy_and_positive_context_denominators() -> None:
    families = np.asarray(
        [
            "rate_related_confounder",
            "axis_shift_confounder",
            "axis_shift_confounder",
            "conduction_change_confounder",
            "ischemic_positive",
            "ischemic_positive",
        ]
    )
    scores = np.asarray([0.9, 0.9, 0.1, 0.9, 0.8, 0.2])
    subjects = np.asarray(["a", "a", "b", "c", "a", "b"])
    challenges = challenge_metrics(families, scores, subjects, 0.5)
    assert challenges["rate_related"]["evidence_level"] == "quantitative_secondary"
    assert challenges["axis_shift"]["false_positive_count"] == 1
    assert challenges["conduction_change"]["evidence_level"] == (
        "exploratory_descriptive"
    )
    assert challenges["conduction_change"]["bootstrap_permitted"] is False
    intervals = challenge_bootstrap_confidence_intervals(
        families, scores, subjects, 0.5, replicates=10, seed=7
    )
    assert set(intervals) == {"rate_related", "axis_shift"}
    assert intervals["rate_related"]["successful_replicates"] == 0

    labels = (families == "ischemic_positive").astype(np.int64)
    flags = np.asarray(["", "", "", "", "axis_shift_context", "point_noise_context"])
    contexts = positive_context_analysis(labels, scores, subjects, flags, 0.5)
    assert contexts["axis_shift_context"]["window_count"] == 1
    assert contexts["point_noise_context"]["contributing_subject_count"] == 1


def valid_lock_payload(run_dir: Path) -> dict[str, object]:
    model = run_dir / "model.joblib"
    transform = run_dir / "transform.json"
    model.write_bytes(b"SYNTHETIC MODEL")
    transform.write_text("{}\n", encoding="utf-8")
    return {
        "experiment_id": "synthetic",
        "baseline_name": "B1_signal_logreg",
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "feature_schema_sha256": SIGNAL_V1.sha256,
        "feature_corpus_sha256": "a" * 64,
        "validation_selected_threshold": 0.5,
        "model_artifact": model.name,
        "trained_model_artifact_sha256": sha256_file(model),
        "transform_artifact": transform.name,
        "transform_artifact_sha256": sha256_file(transform),
    }


def test_test_evaluation_lock_is_required_and_hash_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires"):
        validate_experiment_lock(tmp_path)
    locked = write_experiment_lock(tmp_path, valid_lock_payload(tmp_path))
    assert validate_experiment_lock(tmp_path) == locked

    lock_path = tmp_path / "experiment_lock.json"
    tampered = json.loads(lock_path.read_text(encoding="utf-8"))
    tampered["validation_selected_threshold"] = 0.9
    lock_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_experiment_lock(tmp_path)


def test_lock_rejects_non_frozen_split(tmp_path: Path) -> None:
    payload = valid_lock_payload(tmp_path)
    payload["split_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen"):
        write_experiment_lock(tmp_path, payload)


def test_lock_hash_cryptographically_binds_feature_corpus(tmp_path: Path) -> None:
    locked = write_experiment_lock(tmp_path, valid_lock_payload(tmp_path))
    lock_path = tmp_path / "experiment_lock.json"
    locked["feature_corpus_sha256"] = "b" * 64
    lock_path.write_text(json.dumps(locked), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_experiment_lock(tmp_path)


def test_lock_requires_a_canonical_feature_corpus_hash(tmp_path: Path) -> None:
    payload = valid_lock_payload(tmp_path)
    payload["feature_corpus_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="feature_corpus"):
        write_experiment_lock(tmp_path, payload)


def _write_synthetic_corpus(root: Path) -> tuple[set[str], str]:
    records = (
        ("r2", "s2", "validation", ("background_negative",)),
        ("r1", "s1", "train", ("ischemic_positive", "background_negative")),
    )
    entries = []
    for record_id, subject_id, partition, families in records:
        count = len(families)
        cache_path = root / partition / f"{record_id}.npz"
        table = FeatureTable(
            features=np.arange(
                count * len(COMBINED_V1.names), dtype=np.float64
            ).reshape(count, len(COMBINED_V1.names)),
            stable_ids=np.asarray(
                [f"ltstdb:{record_id}:0:{i}:{i + 1}" for i in range(count)]
            ),
            record_ids=np.asarray([record_id] * count),
            subject_ids=np.asarray([subject_id] * count),
            channel_indices=np.zeros(count, dtype=np.int64),
            lead_names=np.asarray(["I"] * count),
            window_start_samples=np.arange(count, dtype=np.int64),
            window_end_samples=np.arange(count, dtype=np.int64) + 1,
            partitions=np.asarray([partition] * count),
            target_families=np.asarray(families),
            context_flags=np.asarray([""] * count),
        )
        target_counts = {
            family: families.count(family) for family in sorted(set(families))
        }
        source_sha256 = record_id * 32
        metadata = {
            "dataset": "ltstdb",
            "dataset_version": "1.0.0",
            "record_id": record_id,
            "subject_id": subject_id,
            "partition": partition,
            "source_sha256": source_sha256,
            "split_sha256": LTSTDB_V1_SPLIT_SHA256,
            "feature_schema_sha256": COMBINED_V1.sha256,
            "processing_profile": "raw",
            "window_seconds": 10.0,
            "stride_seconds": 5.0,
            "annotation_definition": "ltstdb.stb",
            "row_count": count,
            "target_counts": target_counts,
        }
        write_feature_table_atomic(cache_path, table, metadata)
        entries.append(
            {
                "record_id": record_id,
                "subject_id": subject_id,
                "partition": partition,
                "cache_path": f"{partition}/{record_id}.npz",
                "status": "complete",
                "resumed": False,
                "row_count": count,
                "target_counts": target_counts,
                "source_sha256": source_sha256,
                "cache_sha256": sha256_file(cache_path),
            }
        )
    manifest = {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "feature_schemas": {"combined_v1": COMBINED_V1.as_dict()},
        "processing_profile": "raw",
        "window_seconds": 10.0,
        "stride_seconds": 5.0,
        "annotation_definition": "ltstdb.stb",
        "generation": {"elapsed_seconds": 12.0},
        "records": entries,
    }
    write_json_atomic(root / "manifest.json", manifest)
    expected = {"r1", "r2"}
    return expected, finalize_feature_corpus(root, expected)


def test_feature_corpus_hash_is_canonical_and_ignores_mutable_bookkeeping(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    expected, corpus_sha256 = _write_synthetic_corpus(root)
    manifest = read_json(root / "manifest.json")
    manifest["records"].reverse()
    manifest["records"][0]["resumed"] = True
    manifest["generation"]["elapsed_seconds"] = 999.0
    assert compute_feature_corpus_sha256(manifest) == corpus_sha256
    write_json_atomic(root / "manifest.json", manifest)
    assert validate_feature_corpus(root, expected) == corpus_sha256


def test_feature_cache_hash_and_corpus_tamper_are_blocking(tmp_path: Path) -> None:
    root = tmp_path / "features"
    expected, _ = _write_synthetic_corpus(root)
    cache_path = root / "train" / "r1.npz"
    cache_path.write_bytes(cache_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="digest"):
        validate_feature_corpus(root, expected)


def test_locked_test_rejects_changed_feature_corpus(tmp_path: Path) -> None:
    root = tmp_path / "features"
    expected, corpus_sha256 = _write_synthetic_corpus(root)
    split = {"records_by_subject": {"s1": ["r1"], "s2": ["r2"]}}
    lock = {"feature_corpus_sha256": corpus_sha256}
    assert validate_locked_feature_corpus(root, lock, split) == corpus_sha256
    cache_path = root / "validation" / "r2.npz"
    cache_path.write_bytes(cache_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="digest"):
        validate_locked_feature_corpus(root, lock, split)
