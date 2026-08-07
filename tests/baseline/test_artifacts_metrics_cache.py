"""SYNTHETIC cache, lock, pooled, macro, bootstrap, and reporting tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.baseline.artifacts import (
    validate_experiment_lock,
    write_experiment_lock,
)
from cardiosentinel.baseline.cache import FeatureTable
from cardiosentinel.baseline.metrics import (
    binary_metrics,
    challenge_bootstrap_confidence_intervals,
    challenge_metrics,
    positive_context_analysis,
    subject_bootstrap_confidence_intervals,
    subject_macro_metrics,
)
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
    assert macro["specificity"]["contributing_subject_count"] == 2


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
