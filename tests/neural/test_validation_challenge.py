"""Synthetic tests for the official B4 validation challenge evidence suite.

Every fixture here is synthetic. No test in this module reads a real B4 run, a
real prediction artifact, a waveform cache, a dataset or the sealed test.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.baseline.metrics import challenge_metrics
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural import resource_benchmark as benchmark
from cardiosentinel.neural import validation_challenge as challenge
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.protocol import B4_PROTOCOL_SHA256
from cardiosentinel.neural.resource_benchmark import (
    ARCHITECTURE_PROTOCOL_SHA256,
    B4A_DEPENDENCY_DIGEST,
)

SPLIT_SHA = "66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7"
CORPUS_SHA = "f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5"
SELECTION_SHA = "318da148da5d638af44e73c06c00cc4df2815017d4ce8bb1a1b864e53eda8009"
FEATURE_INTEGRITY_SHA = (
    "8a7977dc4f0ac7308fa0a5ad439bb5961f806f049ddcb27fd6de461a05d690fd"
)
THRESHOLD = 0.5


def _rows() -> dict[str, list]:
    """A tiny, fully hand-checkable validation population.

    Scores are chosen so the expected counts at threshold 0.5 can be read off
    directly rather than recomputed by the code under test.
    """
    return {
        # family, subject, label, score, context
        "target_family": [
            "ischemic_positive",
            "ischemic_positive",
            "background_negative",
            "background_negative",
            "rate_related_confounder",  # >= thr -> false positive
            "rate_related_confounder",  # <  thr
            "rate_related_confounder",  # >= thr -> false positive
            "axis_shift_confounder",  # >= thr -> false positive
            "axis_shift_confounder",  # <  thr
            "conduction_change_confounder",  # <  thr
        ],
        "subject_id": [
            "s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1", "s2", "s1",
        ],
        "label": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "score": [0.9, 0.2, 0.1, 0.4, 0.7, 0.3, 0.8, 0.6, 0.2, 0.1],
        "context_flags": [
            "axis_shift_context",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "conduction_change_context",
        ],
    }


def _write_predictions(directory: Path, rows: dict[str, list]) -> Path:
    path = directory / challenge.VALIDATION_PREDICTIONS_NAME
    count = len(rows["label"])
    np.savez_compressed(
        path,
        stable_id=np.asarray([f"w{i}" for i in range(count)], dtype=np.str_),
        subject_id=np.asarray(rows["subject_id"], dtype=np.str_),
        record_id=np.asarray(["r0"] * count, dtype=np.str_),
        channel_index=np.zeros(count, dtype=np.int64),
        target_family=np.asarray(rows["target_family"], dtype=np.str_),
        context_flags=np.asarray(rows["context_flags"], dtype=np.str_),
        label=np.asarray(rows["label"], dtype=np.int64),
        score=np.asarray(rows["score"], dtype=np.float64),
    )
    return path


def _locked_candidate(
    directory: Path,
    model_key: str,
    *,
    rows: dict[str, list] | None = None,
    threshold: float = THRESHOLD,
    **lock_overrides,
) -> Path:
    """Build a synthetic locked run satisfying the frozen official mapping."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = rows or _rows()
    specification = benchmark.OFFICIAL_MODELS[model_key]
    factory = benchmark.SUPPORTED_ARCHITECTURES[specification["architecture"]]
    checkpoint = directory / "model_selected.pt"
    torch.save(factory().state_dict(), checkpoint)
    predictions = _write_predictions(directory, rows)

    lock = {
        "experiment_id": specification["experiment_id"],
        "candidate_architecture": specification["architecture"],
        "status": "locked_for_one_shot_test",
        "git_dirty": False,
        "protocol_sha256": B4_PROTOCOL_SHA256,
        "model": {
            "architecture": specification["architecture"],
            "verified_against_constructed_model": True,
        },
        "locked_inference_model": "model_selected.pt",
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "trainable_parameter_count": specification["trainable_parameter_count"],
        "environment_dependency_digest": B4A_DEPENDENCY_DIGEST,
        "split_sha256": SPLIT_SHA,
        "feature_corpus_sha256": CORPUS_SHA,
        "training_selection_sha256": SELECTION_SHA,
        "development_feature_integrity_sha256": FEATURE_INTEGRITY_SHA,
        "validation_predictions_sha256": sha256_file(predictions),
        "validation_threshold": threshold,
        "validation_rows": {
            "partition": "validation",
            "total": len(rows["label"]),
            "subjects": len(set(rows["subject_id"])),
        },
        "test": None,
    }
    if specification["requires_architecture_protocol"]:
        lock["architecture_protocol_sha256"] = ARCHITECTURE_PROTOCOL_SHA256
    lock.update(lock_overrides)
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    return directory


@pytest.fixture
def official_runs(tmp_path):
    return {
        key: _locked_candidate(tmp_path / key, key)
        for key in benchmark.OFFICIAL_ORDER
    }


# --------------------------------------------------------------------------
# Protocol identity
# --------------------------------------------------------------------------


def test_challenge_protocol_sha_is_enforced() -> None:
    assert (
        challenge.validate_validation_challenge_protocol()
        == challenge.VALIDATION_CHALLENGE_PROTOCOL_SHA256
    )


def test_tampered_challenge_protocol_is_refused(tmp_path) -> None:
    forged = tmp_path / "protocol.md"
    forged.write_text("not the frozen procedure\n", encoding="utf-8")
    with pytest.raises(challenge.ValidationChallengeError, match="frozen SHA-256"):
        challenge.validate_validation_challenge_protocol(forged)


# --------------------------------------------------------------------------
# Challenge arithmetic
# --------------------------------------------------------------------------


def test_rate_related_false_positive_fraction(official_runs) -> None:
    result = challenge.evaluate_candidate_validation_challenge(
        official_runs["B4-B"], official_model="B4-B"
    )
    rate = result["challenges"]["rate_related"]
    assert rate["denominator_negative_window_count"] == 3
    assert rate["false_positive_count"] == 2
    assert rate["false_positive_fraction"] == pytest.approx(2 / 3)
    assert rate["supporting_subject_count"] == 2
    assert rate["evidence_status"] == "quantitative_secondary"


def test_axis_shift_false_positive_fraction(official_runs) -> None:
    result = challenge.evaluate_candidate_validation_challenge(
        official_runs["B4-B"], official_model="B4-B"
    )
    axis = result["challenges"]["axis_shift"]
    assert axis["denominator_negative_window_count"] == 2
    assert axis["false_positive_count"] == 1
    assert axis["false_positive_fraction"] == pytest.approx(0.5)
    assert axis["supporting_subject_count"] == 2
    assert axis["evidence_status"] == "quantitative_secondary"


def test_conduction_is_exploratory_and_never_bootstrapped(official_runs) -> None:
    result = challenge.evaluate_candidate_validation_challenge(
        official_runs["B4-C"], official_model="B4-C"
    )
    conduction = result["challenges"]["conduction_change"]
    assert conduction["evidence_status"] == "exploratory_descriptive"
    assert conduction["bootstrap_permitted"] is False
    assert conduction["is_headline_metric"] is False
    assert conduction["denominator_negative_window_count"] == 1
    assert conduction["false_positive_count"] == 0


def test_matches_the_frozen_production_metric_exactly(official_runs) -> None:
    """The suite must not introduce a second, divergent definition."""
    rows = _rows()
    expected = challenge_metrics(
        np.asarray(rows["target_family"], dtype=np.str_),
        np.asarray(rows["score"], dtype=np.float64),
        np.asarray(rows["subject_id"], dtype=np.str_),
        THRESHOLD,
    )
    result = challenge.evaluate_candidate_validation_challenge(
        official_runs["B4-A"], official_model="B4-A"
    )
    for name, measured in expected.items():
        assert result["challenges"][name]["frozen_metric"] == measured


def test_positive_ischemic_context_is_not_converted_to_a_negative(
    official_runs,
) -> None:
    result = challenge.evaluate_candidate_validation_challenge(
        official_runs["B4-B"], official_model="B4-B"
    )
    # The axis-context positive stays positive and stays out of every
    # challenge denominator.
    assert result["validation_positive_count"] == 2
    assert result["challenges"]["axis_shift"]["denominator_negative_window_count"] == 2
    assert result["positive_context"]["evidence_status"] == "descriptive_error_analysis"
    assert result["positive_context"]["strata"]


def test_empty_challenge_slice_reports_null_not_zero(tmp_path) -> None:
    rows = _rows()
    rows["target_family"] = [
        "background_negative" if f == "conduction_change_confounder" else f
        for f in rows["target_family"]
    ]
    directory = _locked_candidate(tmp_path / "B4-A", "B4-A", rows=rows)
    result = challenge.evaluate_candidate_validation_challenge(
        directory, official_model="B4-A"
    )
    conduction = result["challenges"]["conduction_change"]
    assert conduction["denominator_negative_window_count"] == 0
    assert conduction["false_positive_fraction"] is None
    assert conduction["supporting_subject_count"] == 0


def test_single_subject_slice_reports_support_and_blocks_bootstrap(
    tmp_path,
) -> None:
    rows = _rows()
    rows["subject_id"] = ["s1"] * len(rows["subject_id"])
    directory = _locked_candidate(tmp_path / "B4-A", "B4-A", rows=rows)
    result = challenge.evaluate_candidate_validation_challenge(
        directory, official_model="B4-A"
    )
    rate = result["challenges"]["rate_related"]
    assert rate["supporting_subject_count"] == 1
    assert rate["bootstrap_permitted"] is False


# --------------------------------------------------------------------------
# Threshold provenance
# --------------------------------------------------------------------------


def test_threshold_comes_from_the_lock_and_is_never_selected(tmp_path) -> None:
    directory = _locked_candidate(tmp_path / "B4-B", "B4-B", threshold=0.75)
    result = challenge.evaluate_candidate_validation_challenge(
        directory, official_model="B4-B"
    )
    assert result["locked_validation_threshold"] == 0.75
    assert result["threshold_source"] == "locked_experiment_lock.validation_threshold"
    assert result["threshold_selected_by_evaluator"] is False
    # At 0.75 only the 0.8 rate window is a false positive.
    assert result["challenges"]["rate_related"]["false_positive_count"] == 1


def test_no_threshold_selection_function_is_reachable() -> None:
    source = inspect.getsource(challenge)
    for forbidden in (
        "validation_f1_threshold",
        "select_threshold",
        "optimize_threshold",
        "threshold_sweep",
    ):
        assert forbidden not in source


def test_non_finite_threshold_is_refused(tmp_path) -> None:
    directory = _locked_candidate(
        tmp_path / "B4-A", "B4-A", validation_threshold=float("nan")
    )
    with pytest.raises(challenge.ValidationChallengeError, match="finite"):
        challenge.evaluate_candidate_validation_challenge(
            directory, official_model="B4-A"
        )


# --------------------------------------------------------------------------
# Artifact integrity refusals
# --------------------------------------------------------------------------


def test_prediction_digest_mismatch_is_refused(official_runs) -> None:
    path = official_runs["B4-B"] / challenge.VALIDATION_PREDICTIONS_NAME
    rows = _rows()
    rows["score"][0] = 0.55
    _write_predictions(official_runs["B4-B"], rows)
    assert path.is_file()
    with pytest.raises(challenge.ValidationChallengeError, match="does not match"):
        challenge.evaluate_candidate_validation_challenge(
            official_runs["B4-B"], official_model="B4-B"
        )


def test_experiment_lock_digest_mismatch_is_refused(official_runs) -> None:
    path = official_runs["B4-C"] / "EXPERIMENT_LOCK.json"
    lock = json.loads(path.read_text())
    lock["validation_threshold"] = 0.123
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(benchmark.LockedModelError, match="hash validation failed"):
        challenge.evaluate_candidate_validation_challenge(
            official_runs["B4-C"], official_model="B4-C"
        )


def test_wrong_candidate_identity_is_refused(official_runs) -> None:
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="requires experiment_id"
    ):
        challenge.evaluate_candidate_validation_challenge(
            official_runs["B4-B"], official_model="B4-C"
        )


def test_unknown_official_model_is_refused(official_runs) -> None:
    with pytest.raises(challenge.ValidationChallengeError, match="Unknown official"):
        challenge.evaluate_candidate_validation_challenge(
            official_runs["B4-A"], official_model="B4-D"
        )


def test_missing_prediction_artifact_is_refused(official_runs) -> None:
    (official_runs["B4-A"] / challenge.VALIDATION_PREDICTIONS_NAME).unlink()
    with pytest.raises(
        challenge.ValidationChallengeError, match="No validation_pred"
    ):
        challenge.evaluate_candidate_validation_challenge(
            official_runs["B4-A"], official_model="B4-A"
        )


def test_malformed_prediction_columns_are_refused(tmp_path) -> None:
    directory = tmp_path / "B4-A"
    _locked_candidate(directory, "B4-A")
    lock = json.loads((directory / "EXPERIMENT_LOCK.json").read_text())
    path = directory / challenge.VALIDATION_PREDICTIONS_NAME
    np.savez_compressed(path, score=np.zeros(3), label=np.zeros(3, dtype=np.int64))
    lock["validation_predictions_sha256"] = sha256_file(path)
    lock.pop("experiment_lock_sha256")
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(json.dumps(lock))
    with pytest.raises(challenge.ValidationChallengeError, match="lacks columns"):
        challenge.evaluate_candidate_validation_challenge(
            directory, official_model="B4-A"
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_scores_are_refused(tmp_path, bad) -> None:
    rows = _rows()
    rows["score"][3] = bad
    directory = _locked_candidate(tmp_path / "B4-A", "B4-A", rows=rows)
    with pytest.raises(challenge.ValidationChallengeError, match="non-finite score"):
        challenge.evaluate_candidate_validation_challenge(
            directory, official_model="B4-A"
        )


def test_duplicate_stable_ids_are_refused(tmp_path) -> None:
    directory = tmp_path / "B4-A"
    _locked_candidate(directory, "B4-A")
    rows = _rows()
    count = len(rows["label"])
    path = directory / challenge.VALIDATION_PREDICTIONS_NAME
    np.savez_compressed(
        path,
        stable_id=np.asarray(["dup"] * count, dtype=np.str_),
        subject_id=np.asarray(rows["subject_id"], dtype=np.str_),
        record_id=np.asarray(["r0"] * count, dtype=np.str_),
        channel_index=np.zeros(count, dtype=np.int64),
        target_family=np.asarray(rows["target_family"], dtype=np.str_),
        context_flags=np.asarray(rows["context_flags"], dtype=np.str_),
        label=np.asarray(rows["label"], dtype=np.int64),
        score=np.asarray(rows["score"], dtype=np.float64),
    )
    lock = json.loads((directory / "EXPERIMENT_LOCK.json").read_text())
    lock["validation_predictions_sha256"] = sha256_file(path)
    lock.pop("experiment_lock_sha256")
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(json.dumps(lock))
    with pytest.raises(challenge.ValidationChallengeError, match="duplicate stable"):
        challenge.evaluate_candidate_validation_challenge(
            directory, official_model="B4-A"
        )


def test_wrong_validation_population_is_refused(tmp_path) -> None:
    directory = _locked_candidate(
        tmp_path / "B4-A",
        "B4-A",
        validation_rows={"partition": "validation", "total": 999, "subjects": 2},
    )
    with pytest.raises(challenge.ValidationChallengeError, match="rows but the lock"):
        challenge.evaluate_candidate_validation_challenge(
            directory, official_model="B4-A"
        )


def test_missing_mandatory_provenance_is_refused(tmp_path) -> None:
    directory = _locked_candidate(tmp_path / "B4-A", "B4-A", split_sha256=None)
    with pytest.raises(
        challenge.ValidationChallengeError, match="mandatory provenance"
    ):
        challenge.evaluate_candidate_validation_challenge(
            directory, official_model="B4-A"
        )


def test_b4a_historical_binding_is_recorded_without_fabrication(
    tmp_path,
) -> None:
    """B4-A may lack newer fields, but B4-B/B4-C are not weakened to match."""
    directory = tmp_path / "B4-A"
    _locked_candidate(directory, "B4-A")
    lock = json.loads((directory / "EXPERIMENT_LOCK.json").read_text())
    lock.pop("candidate_architecture")
    lock.pop("experiment_lock_sha256")
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(json.dumps(lock))
    result = challenge.evaluate_candidate_validation_challenge(
        directory, official_model="B4-A"
    )
    binding = result["provenance_binding"]
    assert "candidate_architecture" in binding["optional_fields_absent"]
    assert "split_sha256" in binding["required_fields_present"]
    assert result["architecture"] == "B4CompactCNN"


# --------------------------------------------------------------------------
# Suite semantics
# --------------------------------------------------------------------------


def test_official_suite_produces_one_combined_result(official_runs, tmp_path) -> None:
    root = tmp_path / "runs"
    suite = challenge.run_official_validation_challenge_suite(official_runs, root)
    assert suite["candidate_order"] == ["B4-A", "B4-B", "B4-C"]
    assert suite["dataset_accessed"] is False
    assert suite["test_accessed"] is False
    assert suite["model_inference_performed"] is False
    assert suite["architecture_selection_performed"] is False
    directory = root / challenge.SUITE_DIR_NAME
    assert (directory / challenge.SUITE_ATTEMPT_NAME).is_file()
    assert (directory / challenge.SUITE_RESULTS_NAME).is_file()
    attempt = json.loads((directory / challenge.SUITE_ATTEMPT_NAME).read_text())
    assert attempt["attempt_status"] == "COMPLETE"
    assert attempt["repeat_attempt_permitted"] is False
    assert attempt["selective_candidate_retry_permitted"] is False


def test_combined_and_child_digests_re_derive(official_runs, tmp_path) -> None:
    root = tmp_path / "runs"
    suite = challenge.run_official_validation_challenge_suite(official_runs, root)
    body = {
        k: v
        for k, v in suite.items()
        if k != "validation_challenge_suite_sha256"
    }
    assert canonical_sha256(body) == suite["validation_challenge_suite_sha256"]
    for name in benchmark.OFFICIAL_ORDER:
        result = suite["candidate_results"][name]
        child = {
            k: v for k, v in result.items() if k != "challenge_result_sha256"
        }
        assert canonical_sha256(child) == result["challenge_result_sha256"]
    assert challenge.read_official_validation_challenge_results(root) == suite


def test_repeat_official_attempt_is_refused(official_runs, tmp_path) -> None:
    root = tmp_path / "runs"
    challenge.run_official_validation_challenge_suite(official_runs, root)
    with pytest.raises(challenge.ValidationChallengeError, match="already exists"):
        challenge.run_official_validation_challenge_suite(official_runs, root)


def test_partial_prior_attempt_blocks_rerun(official_runs, tmp_path) -> None:
    root = tmp_path / "runs"
    directory = root / challenge.SUITE_DIR_NAME
    directory.mkdir(parents=True)
    (directory / challenge.SUITE_ATTEMPT_NAME).write_text("{corrupt", encoding="utf-8")
    with pytest.raises(challenge.ValidationChallengeError, match="already exists"):
        challenge.run_official_validation_challenge_suite(official_runs, root)


def test_missing_candidate_is_refused(official_runs, tmp_path) -> None:
    partial = {k: v for k, v in official_runs.items() if k != "B4-C"}
    with pytest.raises(challenge.ValidationChallengeError, match="requires exactly"):
        challenge.run_official_validation_challenge_suite(partial, tmp_path / "runs")


def test_fourth_candidate_is_refused(official_runs, tmp_path) -> None:
    extra = dict(official_runs)
    extra["B4-D"] = official_runs["B4-A"]
    with pytest.raises(challenge.ValidationChallengeError, match="requires exactly"):
        challenge.run_official_validation_challenge_suite(extra, tmp_path / "runs")


def test_failed_suite_leaves_the_attempt_claimed(official_runs, tmp_path) -> None:
    """A refusal after the claim must not silently free the attempt."""
    (official_runs["B4-C"] / challenge.VALIDATION_PREDICTIONS_NAME).unlink()
    root = tmp_path / "runs"
    with pytest.raises(challenge.ValidationChallengeError):
        challenge.run_official_validation_challenge_suite(official_runs, root)
    assert (root / challenge.SUITE_DIR_NAME / challenge.SUITE_ATTEMPT_NAME).is_file()
    with pytest.raises(challenge.ValidationChallengeError, match="already exists"):
        challenge.run_official_validation_challenge_suite(official_runs, root)


def test_suite_exposes_no_evaluator_or_retry_injection() -> None:
    parameters = inspect.signature(
        challenge.run_official_validation_challenge_suite
    ).parameters
    assert set(parameters) == {"run_directories", "run_root", "command"}
    for forbidden in ("runner", "backend", "evaluator", "metric", "retry", "force"):
        assert forbidden not in parameters


# --------------------------------------------------------------------------
# Structural firewall
# --------------------------------------------------------------------------


def test_test_partition_cannot_enter_the_evaluator() -> None:
    with pytest.raises(challenge.ValidationChallengeError, match="never access"):
        challenge.require_evaluated_partition("test")
    assert challenge.require_evaluated_partition("validation") == "validation"


def test_module_never_imports_the_sealed_test_path() -> None:
    tree = ast.parse(Path(challenge.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("sealed_test" in name for name in imported)
    assert not any(name.endswith("wfdb") or name == "wfdb" for name in imported)


def test_module_performs_no_model_inference_or_dataset_access() -> None:
    source = Path(challenge.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "evaluate_locked_test",
        "SealedTestAccess",
        "TEST_ATTEMPT",
        "load_locked_model",
        "torch.load",
        ".forward(",
        ".backward(",
        "B4CachedWaveformDataset",
        "build_development_indexes",
        "validate_waveform_cache",
        "wfdb",
    ):
        assert forbidden not in source, forbidden


def test_evidence_records_no_dataset_test_or_inference_access(
    official_runs,
) -> None:
    for name in benchmark.OFFICIAL_ORDER:
        result = challenge.evaluate_candidate_validation_challenge(
            official_runs[name], official_model=name
        )
        assert result["dataset_accessed"] is False
        assert result["test_accessed"] is False
        assert result["model_inference_performed"] is False
        assert result["waveform_accessed"] is False
        assert result["partition"] == "validation"


def test_module_declares_no_selection_or_scoring_helper() -> None:
    source = inspect.getsource(challenge)
    for forbidden in ("def select_", "def rank_", "winner", "pareto", "weighted_score"):
        assert forbidden not in source.lower()
