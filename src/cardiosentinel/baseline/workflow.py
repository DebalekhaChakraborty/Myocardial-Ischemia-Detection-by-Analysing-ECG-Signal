"""Separated fit/validation-lock and sealed-test baseline experiment stages."""

from __future__ import annotations

import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.baseline.artifacts import (
    validate_experiment_lock,
    write_experiment_lock,
    write_predictions_atomic,
)
from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    FeatureTable,
    load_partition,
    read_json,
    require_external_path,
    validate_feature_corpus,
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
from cardiosentinel.baseline.sampling import (
    build_training_selection_plan,
    load_selected_training_table,
)
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.evaluation.metrics import select_validation_f1_threshold
from cardiosentinel.evaluation.protocol import (
    DEFAULT_SEED,
    LTSTDB_V1_SPLIT_SHA256,
    TRAIN_NEGATIVE_RATIO,
)
from cardiosentinel.evaluation.splits import (
    load_split_manifest,
    validate_split_manifest,
)
from cardiosentinel.features.schema import SIGNAL_V1
from cardiosentinel.models.baselines import (
    BASELINE_NAMES,
    build_estimator,
    feature_schema_for_baseline,
    model_parameters,
    positive_scores,
    transform_snapshot,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_PATH = REPOSITORY_ROOT / "protocols/splits/ltstdb_v1.json"
PRIMARY_FAMILIES = ("ischemic_positive", "background_negative")
FROZEN_PRIMARY_COUNTS = {
    "train": {"ischemic_positive": 93_613, "background_negative": 2_049_986},
    "validation": {"ischemic_positive": 21_628, "background_negative": 452_269},
    "test": {"ischemic_positive": 20_899, "background_negative": 432_905},
}


def _labels(table: FeatureTable) -> NDArray[np.int64]:
    return (table.target_families == "ischemic_positive").astype(np.int64)


def _primary(table: FeatureTable) -> NDArray[np.bool_]:
    return np.isin(table.target_families, PRIMARY_FAMILIES)


def _expected_partition_identity(
    split: dict[str, Any], partition: str
) -> tuple[set[str], set[str]]:
    subjects = set(split["partitions"][partition]["subjects"])
    records = {
        record
        for subject in subjects
        for record in split["records_by_subject"][subject]
    }
    return subjects, records


def _expected_record_ids(split: dict[str, Any]) -> set[str]:
    return {
        record
        for subject_records in split["records_by_subject"].values()
        for record in subject_records
    }


def validate_locked_feature_corpus(
    feature_root: Path, lock: dict[str, Any], split: dict[str, Any]
) -> str:
    """Reject any corpus that differs from the fitted experiment lock."""
    corpus_sha256 = validate_feature_corpus(feature_root, _expected_record_ids(split))
    if corpus_sha256 != lock["feature_corpus_sha256"]:
        raise ValueError("Feature corpus differs from the frozen experiment lock.")
    return corpus_sha256


def _validate_partition(
    table: FeatureTable, split: dict[str, Any], partition: str
) -> None:
    expected_subjects, expected_records = _expected_partition_identity(split, partition)
    if set(table.subject_ids.tolist()) != expected_subjects:
        raise ValueError(f"{partition} subjects do not match the frozen split.")
    if set(table.record_ids.tolist()) != expected_records:
        raise ValueError(f"{partition} records do not match the frozen split.")
    if set(table.partitions.tolist()) != {partition}:
        raise ValueError(f"{partition} table has inconsistent partition metadata.")
    regenerated = {
        family: int(np.sum(table.target_families == family))
        for family in PRIMARY_FAMILIES
    }
    if regenerated != FROZEN_PRIMARY_COUNTS[partition]:
        raise ValueError(
            f"{partition} regenerated primary counts differ from Benchmark V1: "
            f"{regenerated}."
        )


def _validate_manifest_completeness(
    feature_root: Path, split: dict[str, Any], partition: str
) -> None:
    manifest = read_json(feature_root / FEATURE_MANIFEST_NAME)
    expected_subjects, expected_records = _expected_partition_identity(split, partition)
    completed_entries = [
        item
        for item in manifest["records"]
        if item["partition"] == partition and item["status"] == "complete"
    ]
    completed_records = {item["record_id"] for item in completed_entries}
    completed_subjects = {item["subject_id"] for item in completed_entries}
    if completed_records != expected_records or completed_subjects != expected_subjects:
        raise ValueError(f"Feature manifest is incomplete for {partition}.")
    for entry in completed_entries:
        if entry["record_id"] not in split["records_by_subject"][entry["subject_id"]]:
            raise ValueError(f"Feature manifest identity is invalid for {partition}.")


def _model_matrix(table: FeatureTable, baseline_name: str) -> NDArray[np.float64]:
    schema = feature_schema_for_baseline(baseline_name)
    if schema.version == SIGNAL_V1.version:
        return table.features[:, : len(SIGNAL_V1.names)]
    return table.features


def _dump_joblib_atomic(path: Path, value: Any) -> None:
    import joblib

    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _environment() -> dict[str, str]:
    import scipy
    import sklearn
    import wfdb

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "wfdb_version": wfdb.__version__,
    }


def _prediction_payload(
    table: FeatureTable, scores: NDArray[np.float64]
) -> dict[str, Any]:
    return {
        "stable_ids": table.stable_ids,
        "record_ids": table.record_ids,
        "subject_ids": table.subject_ids,
        "target_families": table.target_families,
        "context_flags": table.context_flags,
        "scores": scores,
    }


def fit_and_lock(
    feature_root: Path,
    run_root: Path,
    experiment_id: str,
    baseline_name: str,
    *,
    split_path: Path = DEFAULT_SPLIT_PATH,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Fit on train, select threshold on validation, and freeze an experiment."""
    if baseline_name not in BASELINE_NAMES:
        raise ValueError(f"Baseline must be one of {BASELINE_NAMES}.")
    feature_root = require_external_path(feature_root, "Feature root")
    run_root = require_external_path(run_root, "Run root")
    run_dir = run_root / experiment_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Experiment directory already contains artifacts: {run_dir}")
    provenance = git_provenance(REPOSITORY_ROOT)
    if require_clean and provenance["git_dirty"]:
        raise ValueError("Baseline fitting requires a clean implementation commit.")
    split = load_split_manifest(split_path)
    validate_split_manifest(
        split,
        expected_hash=LTSTDB_V1_SPLIT_SHA256,
        expected_subject_count=80,
        expected_record_count=86,
    )
    feature_corpus_sha256 = validate_feature_corpus(
        feature_root, _expected_record_ids(split)
    )
    _validate_manifest_completeness(feature_root, split, "train")
    _validate_manifest_completeness(feature_root, split, "validation")
    sampled_training = baseline_name != "B0_constant_prior"
    selection_plan = build_training_selection_plan(
        feature_root, sampled=sampled_training, seed=DEFAULT_SEED
    )
    expected_train = FROZEN_PRIMARY_COUNTS["train"]
    if (
        selection_plan.unsampled_positive_count != expected_train["ischemic_positive"]
        or selection_plan.unsampled_negative_count
        != expected_train["background_negative"]
    ):
        raise ValueError("Training metadata counts differ from Benchmark V1.")
    validation = load_partition(feature_root, "validation")
    _validate_partition(validation, split, "validation")
    estimator = build_estimator(baseline_name)
    training_started = time.monotonic()
    if baseline_name == "B0_constant_prior":
        estimator.fit_from_counts(
            selection_plan.unsampled_positive_count,
            selection_plan.unsampled_primary_count,
        )
    else:
        train = load_selected_training_table(
            feature_root, selection_plan.selected_stable_ids
        )
        train_labels = _labels(train)
        train_matrix = _model_matrix(train, baseline_name)
        estimator.fit(train_matrix, train_labels)
    training_seconds = time.monotonic() - training_started
    validation_started = time.monotonic()
    validation_scores = positive_scores(
        estimator, _model_matrix(validation, baseline_name)
    )
    validation_scoring_seconds = time.monotonic() - validation_started
    primary = _primary(validation)
    validation_labels = _labels(validation)
    threshold = select_validation_f1_threshold(
        validation_labels[primary].tolist(),
        validation_scores[primary].tolist(),
        partition="validation",
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = run_dir / "model.joblib"
    transform_path = run_dir / "transform.json"
    _dump_joblib_atomic(model_path, estimator)
    write_json_atomic(transform_path, transform_snapshot(estimator))
    write_predictions_atomic(
        run_dir / "validation_predictions.npz",
        **_prediction_payload(validation, validation_scores),
    )
    pooled = binary_metrics(
        validation_labels[primary], validation_scores[primary], threshold
    )
    macro = subject_macro_metrics(
        validation_labels[primary],
        validation_scores[primary],
        validation.subject_ids[primary],
        threshold,
    )
    validation_metrics = {"pooled_window": pooled, "subject_macro": macro}
    validation_challenges = challenge_metrics(
        validation.target_families,
        validation_scores,
        validation.subject_ids,
        threshold,
    )
    validation_context = positive_context_analysis(
        validation_labels,
        validation_scores,
        validation.subject_ids,
        validation.context_flags,
        threshold,
    )
    feature_manifest = read_json(feature_root / FEATURE_MANIFEST_NAME)
    write_json_atomic(run_dir / "feature_manifest.json", feature_manifest)
    training_manifest = {
        "sampling_seed": DEFAULT_SEED,
        "negative_ratio": TRAIN_NEGATIVE_RATIO,
        "unsampled_primary_rows": selection_plan.unsampled_primary_count,
        "selected_rows": selection_plan.selected_count,
        "positive_rows": selection_plan.selected_positive_count,
        "negative_rows": selection_plan.selected_negative_count,
        "subject_count": selection_plan.selected_subject_count,
    }
    write_json_atomic(run_dir / "training_manifest.json", training_manifest)
    write_json_atomic(run_dir / "metrics_validation.json", validation_metrics)
    write_json_atomic(
        run_dir / "challenge_metrics_validation.json", validation_challenges
    )
    write_json_atomic(run_dir / "context_analysis_validation.json", validation_context)
    write_json_atomic(run_dir / "environment.json", _environment())
    schema = feature_schema_for_baseline(baseline_name)
    run_config = {
        "experiment_id": experiment_id,
        "baseline_name": baseline_name,
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "feature_schema_version": schema.version,
        "feature_schema_sha256": schema.sha256,
        "feature_corpus_sha256": feature_corpus_sha256,
        "model_parameters": model_parameters(baseline_name),
        "processing_profile": "raw",
    }
    write_json_atomic(run_dir / "run_config.json", run_config)
    lock = write_experiment_lock(
        run_dir,
        {
            **run_config,
            "git_sha": provenance["git_sha"],
            "git_dirty": provenance["git_dirty"],
            "training_sampling_policy": training_manifest,
            "validation_selected_threshold": threshold,
            "model_artifact": model_path.name,
            "trained_model_artifact_sha256": sha256_file(model_path),
            "transform_artifact": transform_path.name,
            "transform_artifact_sha256": sha256_file(transform_path),
        },
    )
    summary = {
        **run_config,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "train_sample_counts": training_manifest,
        "validation_sample_counts": {
            "primary_rows": int(np.sum(primary)),
            "positive_rows": int(np.sum(validation_labels[primary] == 1)),
            "negative_rows": int(np.sum(validation_labels[primary] == 0)),
        },
        "validation_threshold": threshold,
        "validation": validation_metrics,
        "challenges_validation": validation_challenges,
        "positive_context_validation": validation_context,
        "timing": {
            "training_seconds": training_seconds,
            "validation_scoring_seconds": validation_scoring_seconds,
        },
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "test": None,
    }
    write_json_atomic(run_dir / "RESULTS_SUMMARY.json", summary)
    return summary


def evaluate_test(
    feature_root: Path,
    run_dir: Path,
    *,
    split_path: Path = DEFAULT_SPLIT_PATH,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Evaluate one immutable lock on the sealed test partition exactly once."""
    feature_root = require_external_path(feature_root, "Feature root")
    run_dir = require_external_path(run_dir, "Run directory")
    lock = validate_experiment_lock(run_dir)
    provenance = git_provenance(REPOSITORY_ROOT)
    if require_clean and provenance["git_dirty"]:
        raise ValueError("Test evaluation requires a clean checkout.")
    if provenance["git_sha"] != lock["git_sha"]:
        raise ValueError("Test evaluation checkout differs from the experiment lock.")
    result_path = run_dir / "metrics_test.json"
    attempt_path = run_dir / "test_evaluation_attempt.json"
    if result_path.exists() or attempt_path.exists():
        raise ValueError("This experiment lock has already been evaluated on test.")
    split = load_split_manifest(split_path)
    validate_split_manifest(
        split,
        expected_hash=LTSTDB_V1_SPLIT_SHA256,
        expected_subject_count=80,
        expected_record_count=86,
    )
    validate_locked_feature_corpus(feature_root, lock, split)
    _validate_manifest_completeness(feature_root, split, "test")
    test_started_at = datetime.now(timezone.utc).isoformat()
    write_json_atomic(
        attempt_path,
        {
            "test_evaluation_started_at": test_started_at,
            "experiment_lock_hash": lock["experiment_lock_sha256"],
            "status": "started_before_test_cache_access",
        },
    )
    test = load_partition(feature_root, "test")
    _validate_partition(test, split, "test")
    train_subjects, _ = _expected_partition_identity(split, "train")
    validation_subjects, _ = _expected_partition_identity(split, "validation")
    if set(test.subject_ids.tolist()) & (train_subjects | validation_subjects):
        raise ValueError("Test subjects overlap train or validation.")
    import joblib

    estimator = joblib.load(run_dir / lock["model_artifact"])
    scoring_started = time.monotonic()
    scores = positive_scores(estimator, _model_matrix(test, lock["baseline_name"]))
    test_scoring_seconds = time.monotonic() - scoring_started
    if scores.size != test.row_count or not np.isfinite(scores).all():
        raise ValueError("Test prediction count or finiteness validation failed.")
    threshold = float(lock["validation_selected_threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Validation-selected threshold must be finite.")
    labels = _labels(test)
    primary = _primary(test)
    pooled = binary_metrics(labels[primary], scores[primary], threshold)
    for metric in ("auprc", "auroc"):
        value = pooled[metric]
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"Test {metric} is outside [0, 1].")
    macro = subject_macro_metrics(
        labels[primary], scores[primary], test.subject_ids[primary], threshold
    )
    bootstrap_started = time.monotonic()
    bootstrap = subject_bootstrap_confidence_intervals(
        labels[primary], scores[primary], test.subject_ids[primary], threshold
    )
    challenge = challenge_metrics(
        test.target_families, scores, test.subject_ids, threshold
    )
    challenge_ci = challenge_bootstrap_confidence_intervals(
        test.target_families, scores, test.subject_ids, threshold
    )
    bootstrap_seconds = time.monotonic() - bootstrap_started
    context = positive_context_analysis(
        labels, scores, test.subject_ids, test.context_flags, threshold
    )
    if lock["baseline_name"] == "B0_constant_prior":
        if not np.all(scores == scores[0]):
            raise ValueError("B0 predictions are not constant.")
        if not np.isclose(pooled["auprc"], pooled["positive_prevalence"], atol=1e-12):
            raise ValueError("B0 AUPRC differs unexpectedly from test prevalence.")
    write_predictions_atomic(
        run_dir / "test_predictions.npz", **_prediction_payload(test, scores)
    )
    metrics_test = {
        "pooled_window": pooled,
        "subject_macro": macro,
        "subject_bootstrap_95_ci": bootstrap,
    }
    write_json_atomic(result_path, metrics_test)
    write_json_atomic(
        run_dir / "challenge_metrics.json",
        {"reports": challenge, "subject_bootstrap_95_ci": challenge_ci},
    )
    write_json_atomic(run_dir / "context_analysis.json", context)
    evaluated_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "test_evaluation_started_at": test_started_at,
        "test_evaluation_timestamp": evaluated_at,
        "experiment_lock_hash": lock["experiment_lock_sha256"],
        "status": "complete",
    }
    write_json_atomic(run_dir / "test_evaluation_audit.json", audit)
    summary = read_json(run_dir / "RESULTS_SUMMARY.json")
    summary.update(
        {
            "test_sample_counts": {
                "all_scored_rows": test.row_count,
                "primary_rows": int(np.sum(primary)),
                "positive_rows": int(np.sum(labels[primary] == 1)),
                "negative_rows": int(np.sum(labels[primary] == 0)),
            },
            "test": metrics_test,
            "challenges": {
                "reports": challenge,
                "subject_bootstrap_95_ci": challenge_ci,
            },
            "positive_context": context,
            "test_evaluation_timestamp": evaluated_at,
            "experiment_lock_hash": lock["experiment_lock_sha256"],
        }
    )
    summary["timing"].update(
        {
            "test_scoring_seconds": test_scoring_seconds,
            "bootstrap_seconds": bootstrap_seconds,
        }
    )
    write_json_atomic(run_dir / "RESULTS_SUMMARY.json", summary)
    return summary
