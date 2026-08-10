"""Synthetic tests for the official B4 validation challenge evidence suite.

Fixtures deliberately mirror the REAL artifact shape:

* locked `validation_predictions.npz` contains PRIMARY families only;
* challenge confounder rows live only in validation metadata and are scored by
  locked-model inference over validated waveforms.

An earlier draft of this suite derived challenge evidence from the prediction
artifact. `test_prediction_only_design_cannot_satisfy_the_gate` pins why that is
impossible on the real schema so the mistake cannot silently return.

No test here reads a real B4 run, a real waveform source, or the sealed test.
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
from cardiosentinel.neural.metadata import B4WindowReference
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    B4_SPLIT_SHA256,
    FEATURE_CORPUS_SHA256,
    WINDOW_SAMPLES,
)
from cardiosentinel.neural.resource_benchmark import (
    ARCHITECTURE_PROTOCOL_SHA256,
    B4A_DEPENDENCY_DIGEST,
)

SELECTION_SHA = "318da148da5d638af44e73c06c00cc4df2815017d4ce8bb1a1b864e53eda8009"
FEATURE_INTEGRITY_SHA = (
    "8a7977dc4f0ac7308fa0a5ad439bb5961f806f049ddcb27fd6de461a05d690fd"
)
THRESHOLD = 0.5

# A tiny challenge population mirroring the real family/subject structure:
# rate over two subjects, axis over two, conduction over exactly one.
CHALLENGE_ROWS = [
    ("rate_related_confounder", "s1", 0.9),
    ("rate_related_confounder", "s1", 0.2),
    ("rate_related_confounder", "s2", 0.8),
    ("axis_shift_confounder", "s1", 0.7),
    ("axis_shift_confounder", "s2", 0.1),
    ("conduction_change_confounder", "s3", 0.3),
]
PRIMARY_ROWS = [
    ("ischemic_positive", "s1", 1, 0.9, "axis_shift_context"),
    ("ischemic_positive", "s2", 1, 0.2, ""),
    ("background_negative", "s1", 0, 0.1, ""),
    ("background_negative", "s2", 0, 0.4, ""),
]


class _StubReader:
    """Stands in for the validated waveform reader; returns a constant window."""

    fill = 0.0

    def __init__(self, references, source, **kwargs) -> None:
        self._count = 0

    def read_waveform(self, reference) -> torch.Tensor:
        self._count += 1
        return torch.full((1, WINDOW_SAMPLES), self.fill)

    @property
    def stats(self):
        return type("Stats", (), {"source_reads": self._count})()


def _stub_reader_factory(fill: float):
    return type("_FilledStubReader", (_StubReader,), {"fill": fill})


def _reference(index: int, family: str, subject: str) -> B4WindowReference:
    start = index * WINDOW_SAMPLES
    end = start + WINDOW_SAMPLES
    return B4WindowReference(
        stable_id=f"ltstdb:r0:0:{start}:{end}",
        record_id="r0",
        subject_id=subject,
        channel_index=0,
        start_sample=start,
        end_sample=start + WINDOW_SAMPLES,
        partition="validation",
        target_family=family,
        context_flags=(),
    )


@pytest.fixture
def challenge_index():
    references = tuple(
        _reference(i, family, subject)
        for i, (family, subject, _) in enumerate(CHALLENGE_ROWS)
    )
    return challenge.ValidationChallengeIndex(
        references=references,
        selection_sha256=challenge.challenge_selection_digest(references),
        counts={
            "rate_related_confounder": {"windows": 3, "subjects": 2},
            "axis_shift_confounder": {"windows": 2, "subjects": 2},
            "conduction_change_confounder": {"windows": 1, "subjects": 1},
        },
    )


def _write_primary_predictions(directory: Path) -> Path:
    path = directory / challenge.VALIDATION_PREDICTIONS_NAME
    count = len(PRIMARY_ROWS)
    np.savez_compressed(
        path,
        stable_id=np.asarray([f"p{i}" for i in range(count)], dtype=np.str_),
        subject_id=np.asarray([r[1] for r in PRIMARY_ROWS], dtype=np.str_),
        record_id=np.asarray(["r0"] * count, dtype=np.str_),
        channel_index=np.zeros(count, dtype=np.int64),
        target_family=np.asarray([r[0] for r in PRIMARY_ROWS], dtype=np.str_),
        context_flags=np.asarray([r[4] for r in PRIMARY_ROWS], dtype=np.str_),
        label=np.asarray([r[2] for r in PRIMARY_ROWS], dtype=np.int64),
        score=np.asarray([r[3] for r in PRIMARY_ROWS], dtype=np.float64),
    )
    return path


def _locked_candidate(
    directory: Path, model_key: str, *, threshold: float = THRESHOLD, **overrides
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    specification = benchmark.OFFICIAL_MODELS[model_key]
    factory = benchmark.SUPPORTED_ARCHITECTURES[specification["architecture"]]
    checkpoint = directory / "model_selected.pt"
    torch.save(factory().state_dict(), checkpoint)
    predictions = _write_primary_predictions(directory)
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
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "training_selection_sha256": SELECTION_SHA,
        "development_feature_integrity_sha256": FEATURE_INTEGRITY_SHA,
        "validation_predictions_sha256": sha256_file(predictions),
        "validation_threshold": threshold,
        "validation_rows": {
            "partition": "validation",
            "total": len(PRIMARY_ROWS),
            "positive": sum(r[2] for r in PRIMARY_ROWS),
            "negative": sum(1 - r[2] for r in PRIMARY_ROWS),
        },
        "test": None,
    }
    if specification["requires_architecture_protocol"]:
        lock["architecture_protocol_sha256"] = ARCHITECTURE_PROTOCOL_SHA256
    lock.update(overrides)
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(json.dumps(lock))
    return directory


@pytest.fixture
def official_runs(tmp_path):
    return {
        key: _locked_candidate(tmp_path / key, key)
        for key in benchmark.OFFICIAL_ORDER
    }


@pytest.fixture
def scored(monkeypatch, challenge_index):
    """Deterministic stand-in for locked-model inference over waveforms."""
    scores = np.asarray([row[2] for row in CHALLENGE_ROWS], dtype=np.float64)

    def fake(run_dir, lock, index, source, *, batch_size=256):
        assert index.selection_sha256 == challenge_index.selection_sha256
        return scores, {
            "model_state_unchanged": True,
            "gradients_enabled": False,
            "optimizer_used": False,
            "training_checkpoint_used": False,
            "windows_scored": int(scores.shape[0]),
        }

    monkeypatch.setattr(challenge, "score_challenge_windows", fake)
    return scores


@pytest.fixture
def evaluated(monkeypatch, official_runs, challenge_index, scored, tmp_path):
    monkeypatch.setattr(
        challenge, "validate_development_feature_integrity", lambda root: {}
    )
    monkeypatch.setattr(
        challenge,
        "validate_development_source_integrity",
        lambda source, receipt: {"development_source_integrity_sha256": "src-sha"},
    )
    return official_runs


# --------------------------------------------------------------------------
# The blocker this revision exists to fix
# --------------------------------------------------------------------------


def test_locked_predictions_contain_primary_families_only(official_runs) -> None:
    """Mirror of the real schema: no confounder row is ever in the artifact."""
    path = official_runs["B4-B"] / challenge.VALIDATION_PREDICTIONS_NAME
    with np.load(path, allow_pickle=False) as archive:
        families = set(np.unique(archive["target_family"]).tolist())
    assert families == {"ischemic_positive", "background_negative"}
    assert not families & set(challenge.CHALLENGE_FAMILIES)


def test_prediction_only_design_cannot_satisfy_the_gate(official_runs) -> None:
    """A prediction-only evaluator would report every stratum empty.

    This is the regression that the original PR #14 design failed. Deriving
    challenge_metrics from the primary artifact yields zero denominators, i.e.
    no evidence at all, rather than a loud failure.
    """
    path = official_runs["B4-A"] / challenge.VALIDATION_PREDICTIONS_NAME
    with np.load(path, allow_pickle=False) as archive:
        measured = challenge_metrics(
            archive["target_family"],
            archive["score"].astype(np.float64),
            archive["subject_id"],
            THRESHOLD,
        )
    for name in challenge.CHALLENGE_NAMES:
        assert measured[name]["challenge_window_count"] == 0
        assert measured[name]["false_positive_fraction"] is None


def test_primary_artifact_carrying_challenge_rows_is_refused(tmp_path) -> None:
    directory = _locked_candidate(tmp_path / "B4-A", "B4-A")
    path = directory / challenge.VALIDATION_PREDICTIONS_NAME
    np.savez_compressed(
        path,
        stable_id=np.asarray(["x0"], dtype=np.str_),
        subject_id=np.asarray(["s1"], dtype=np.str_),
        record_id=np.asarray(["r0"], dtype=np.str_),
        channel_index=np.zeros(1, dtype=np.int64),
        target_family=np.asarray(["rate_related_confounder"], dtype=np.str_),
        context_flags=np.asarray([""], dtype=np.str_),
        label=np.zeros(1, dtype=np.int64),
        score=np.asarray([0.5], dtype=np.float64),
    )
    lock = json.loads((directory / "EXPERIMENT_LOCK.json").read_text())
    lock["validation_predictions_sha256"] = sha256_file(path)
    lock["validation_rows"] = {"partition": "validation", "total": 1}
    lock.pop("experiment_lock_sha256")
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(json.dumps(lock))
    with pytest.raises(
        challenge.ValidationChallengeError, match="primary families"
    ):
        challenge.load_primary_validation_predictions(directory, lock)


# --------------------------------------------------------------------------
# Frozen challenge population
# --------------------------------------------------------------------------


def test_frozen_challenge_counts_and_digest_are_pinned() -> None:
    assert challenge.CHALLENGE_EXPECTED_COUNTS == {
        "rate_related_confounder": {"windows": 4973, "subjects": 4},
        "axis_shift_confounder": {"windows": 3000, "subjects": 8},
        "conduction_change_confounder": {"windows": 164, "subjects": 1},
    }
    assert challenge.CHALLENGE_TOTAL_WINDOWS == 8137
    assert len(challenge.CHALLENGE_SELECTION_SHA256) == 64


def test_selection_digest_is_order_independent_and_rejects_duplicates(
    challenge_index,
) -> None:
    shuffled = tuple(reversed(challenge_index.references))
    assert (
        challenge.challenge_selection_digest(shuffled)
        == challenge_index.selection_sha256
    )
    duplicated = challenge_index.references + (challenge_index.references[0],)
    with pytest.raises(challenge.ValidationChallengeError, match="duplicate"):
        challenge.challenge_selection_digest(duplicated)


def test_challenge_index_excludes_non_challenge_families(challenge_index) -> None:
    families = set(challenge_index.target_families.tolist())
    assert families == set(challenge.CHALLENGE_FAMILIES)
    for excluded in (
        "quality_excluded",
        "boundary_ambiguous",
        "source_censored_or_unknown",
        "ischemic_positive",
        "background_negative",
    ):
        assert excluded not in families


# --------------------------------------------------------------------------
# Challenge arithmetic over locked-model scores
# --------------------------------------------------------------------------


def _evaluate(runs, key, index, **kwargs):
    return challenge.evaluate_candidate_validation_challenge(
        runs[key],
        official_model=key,
        feature_root=Path("unused"),
        source=Path("unused"),
        challenge_index=index,
        **kwargs,
    )


def test_rate_axis_and_conduction_evidence(evaluated, challenge_index) -> None:
    result = _evaluate(evaluated, "B4-B", challenge_index)
    rate = result["challenges"]["rate_related"]
    assert rate["challenge_window_count"] == 3
    assert rate["false_positive_count"] == 2
    assert rate["false_positive_fraction"] == pytest.approx(2 / 3)
    assert rate["supporting_subject_count"] == 2
    assert rate["evidence_status"] == "quantitative_secondary"

    axis = result["challenges"]["axis_shift"]
    assert axis["challenge_window_count"] == 2
    assert axis["false_positive_count"] == 1
    assert axis["false_positive_fraction"] == pytest.approx(0.5)

    conduction = result["challenges"]["conduction_change"]
    assert conduction["evidence_status"] == "exploratory_descriptive"
    assert conduction["bootstrap_permitted"] is False
    assert conduction["challenge_window_count"] == 1
    assert conduction["false_positive_count"] == 0


def test_matches_the_frozen_production_metric_exactly(
    evaluated, challenge_index, scored
) -> None:
    expected = challenge_metrics(
        challenge_index.target_families,
        scored,
        challenge_index.subject_ids,
        THRESHOLD,
    )
    result = _evaluate(evaluated, "B4-A", challenge_index)
    for name, measured in expected.items():
        assert result["challenges"][name]["frozen_metric"] == measured


def test_positive_context_comes_from_primary_predictions(
    evaluated, challenge_index
) -> None:
    result = _evaluate(evaluated, "B4-B", challenge_index)
    context = result["positive_context"]
    assert context["evidence_source"] == "locked_primary_validation_predictions"
    assert context["evidence_status"] == "descriptive_error_analysis"
    assert context["primary_window_count"] == len(PRIMARY_ROWS)
    assert result["challenge_evidence_source"] == (
        "locked_model_inference_on_frozen_challenge_rows"
    )


def test_threshold_comes_from_the_lock_only(
    monkeypatch, tmp_path, challenge_index, scored
) -> None:
    monkeypatch.setattr(
        challenge, "validate_development_feature_integrity", lambda root: {}
    )
    monkeypatch.setattr(
        challenge,
        "validate_development_source_integrity",
        lambda source, receipt: {"development_source_integrity_sha256": "src-sha"},
    )
    runs = {"B4-B": _locked_candidate(tmp_path / "B4-B", "B4-B", threshold=0.75)}
    result = _evaluate(runs, "B4-B", challenge_index)
    assert result["locked_validation_threshold"] == 0.75
    assert result["threshold_selected_by_evaluator"] is False
    assert result["threshold_search_performed"] is False
    # At 0.75 only the 0.9 and 0.8 rate windows remain false positives.
    assert result["challenges"]["rate_related"]["false_positive_count"] == 2
    assert result["challenges"]["axis_shift"]["false_positive_count"] == 0


def test_no_threshold_selection_helper_is_reachable() -> None:
    source = inspect.getsource(challenge)
    for forbidden in (
        "validation_f1_threshold",
        "select_threshold",
        "optimize_threshold",
        "threshold_sweep",
    ):
        assert forbidden not in source


# --------------------------------------------------------------------------
# Inference firewall
# --------------------------------------------------------------------------


def test_inference_is_eval_no_grad_and_leaves_the_model_unchanged(
    official_runs, challenge_index, monkeypatch
) -> None:
    lock = json.loads(
        (official_runs["B4-A"] / "EXPERIMENT_LOCK.json").read_text()
    )
    monkeypatch.setattr(challenge, "B4WaveformDataset", _StubReader)
    scores, receipt = challenge.score_challenge_windows(
        official_runs["B4-A"], lock, challenge_index, Path("unused")
    )
    assert scores.shape == (len(CHALLENGE_ROWS),)
    assert np.all(np.isfinite(scores))
    assert receipt["model_state_unchanged"] is True
    assert receipt["model_state_sha256_before"] == receipt["model_state_sha256_after"]
    assert receipt["gradients_enabled"] is False
    assert receipt["optimizer_used"] is False
    assert receipt["training_checkpoint_used"] is False
    assert receipt["waveform_contract"]["samples"] == WINDOW_SAMPLES
    assert receipt["waveform_contract"]["physical_unit"] == "mV"


def _code_without_prose(module) -> str:
    """Module source with docstrings and comments stripped.

    The firewall must assert on executable code; the module's own prose
    legitimately names the things it promises never to do.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and (
            ast.get_docstring(node) is not None
        ):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_module_never_trains_or_touches_the_sealed_test() -> None:
    source = _code_without_prose(challenge)
    for forbidden in (
        "evaluate_locked_test",
        "SealedTestAccess",
        "TEST_ATTEMPT",
        "'training_checkpoint.pt'",
        ".backward(",
        "torch.optim",
        "AdamW",
        ".train()",
        "wfdb",
        "loss",
    ):
        assert forbidden not in source, forbidden
    tree = ast.parse(Path(challenge.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("sealed_test" in name for name in imported)


def test_test_partition_cannot_enter_the_evaluator() -> None:
    with pytest.raises(challenge.ValidationChallengeError, match="never access"):
        challenge.require_evaluated_partition("test")
    assert challenge.require_evaluated_partition("validation") == "validation"


def test_evidence_flags_are_honest(evaluated, challenge_index) -> None:
    result = _evaluate(evaluated, "B4-C", challenge_index)
    assert result["test_accessed"] is False
    assert result["training_performed"] is False
    assert result["threshold_search_performed"] is False
    # Honest about what this path really does.
    assert result["dataset_accessed"] is True
    assert result["waveform_accessed"] is True
    assert result["model_inference_performed"] is True


def test_module_declares_no_selection_or_scoring_helper() -> None:
    source = _code_without_prose(challenge).lower()
    for forbidden in ("def select_", "def rank_", "winner", "pareto", "weighted_score"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# Integrity refusals
# --------------------------------------------------------------------------


def test_experiment_lock_tampering_is_refused(evaluated, challenge_index) -> None:
    path = evaluated["B4-C"] / "EXPERIMENT_LOCK.json"
    lock = json.loads(path.read_text())
    lock["validation_threshold"] = 0.123
    path.write_text(json.dumps(lock))
    with pytest.raises(benchmark.LockedModelError, match="hash validation failed"):
        _evaluate(evaluated, "B4-C", challenge_index)


def test_checkpoint_tampering_is_refused(evaluated, challenge_index) -> None:
    (evaluated["B4-B"] / "model_selected.pt").write_bytes(b"corrupt")
    with pytest.raises(benchmark.LockedModelError, match="SHA-256 does not match"):
        _evaluate(evaluated, "B4-B", challenge_index)


def test_wrong_candidate_identity_is_refused(evaluated, challenge_index) -> None:
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="requires experiment_id"
    ):
        challenge.evaluate_candidate_validation_challenge(
            evaluated["B4-B"],
            official_model="B4-C",
            feature_root=Path("unused"),
            source=Path("unused"),
            challenge_index=challenge_index,
        )


@pytest.mark.parametrize(
    "field,value",
    [("split_sha256", "0" * 64), ("feature_corpus_sha256", "1" * 64)],
)
def test_wrong_frozen_provenance_is_refused(
    monkeypatch, tmp_path, challenge_index, scored, field, value
) -> None:
    monkeypatch.setattr(
        challenge, "validate_development_feature_integrity", lambda root: {}
    )
    runs = {"B4-A": _locked_candidate(tmp_path / "B4-A", "B4-A", **{field: value})}
    with pytest.raises(challenge.ValidationChallengeError, match=field):
        _evaluate(runs, "B4-A", challenge_index)


def test_non_finite_threshold_is_refused(
    monkeypatch, tmp_path, challenge_index, scored
) -> None:
    runs = {
        "B4-A": _locked_candidate(
            tmp_path / "B4-A", "B4-A", validation_threshold=float("nan")
        )
    }
    with pytest.raises(challenge.ValidationChallengeError, match="finite"):
        _evaluate(runs, "B4-A", challenge_index)


def test_non_finite_waveform_is_refused(
    official_runs, challenge_index, monkeypatch
) -> None:
    lock = json.loads(
        (official_runs["B4-A"] / "EXPERIMENT_LOCK.json").read_text()
    )
    monkeypatch.setattr(
        challenge, "B4WaveformDataset", _stub_reader_factory(float("nan"))
    )
    with pytest.raises(
        challenge.ValidationChallengeError, match="non-finite sample"
    ):
        challenge.score_challenge_windows(
            official_runs["B4-A"], lock, challenge_index, Path("unused")
        )


def test_malformed_primary_labels_are_refused_not_coerced(tmp_path) -> None:
    directory = _locked_candidate(tmp_path / "B4-A", "B4-A")
    path = directory / challenge.VALIDATION_PREDICTIONS_NAME
    count = len(PRIMARY_ROWS)
    np.savez_compressed(
        path,
        stable_id=np.asarray([f"p{i}" for i in range(count)], dtype=np.str_),
        subject_id=np.asarray([r[1] for r in PRIMARY_ROWS], dtype=np.str_),
        record_id=np.asarray(["r0"] * count, dtype=np.str_),
        channel_index=np.zeros(count, dtype=np.int64),
        target_family=np.asarray([r[0] for r in PRIMARY_ROWS], dtype=np.str_),
        context_flags=np.asarray([r[4] for r in PRIMARY_ROWS], dtype=np.str_),
        # A float label column must be refused, not silently cast to int.
        label=np.asarray([float(r[2]) for r in PRIMARY_ROWS], dtype=np.float64),
        score=np.asarray([r[3] for r in PRIMARY_ROWS], dtype=np.float64),
    )
    lock = json.loads((directory / "EXPERIMENT_LOCK.json").read_text())
    lock["validation_predictions_sha256"] = sha256_file(path)
    lock.pop("experiment_lock_sha256")
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    with pytest.raises(challenge.ValidationChallengeError, match="integer 0/1"):
        challenge.load_primary_validation_predictions(directory, lock)


def test_primary_population_mismatch_is_refused(tmp_path) -> None:
    directory = _locked_candidate(
        tmp_path / "B4-A",
        "B4-A",
        validation_rows={"partition": "validation", "total": 999, "positive": 2},
    )
    lock = json.loads((directory / "EXPERIMENT_LOCK.json").read_text())
    with pytest.raises(
        challenge.ValidationChallengeError, match="but the lock records"
    ):
        challenge.load_primary_validation_predictions(directory, lock)


# --------------------------------------------------------------------------
# Suite semantics
# --------------------------------------------------------------------------


@pytest.fixture
def suite_runner(monkeypatch, evaluated, challenge_index, tmp_path):
    monkeypatch.setattr(
        challenge, "build_validation_challenge_index", lambda root: challenge_index
    )
    monkeypatch.setattr(
        challenge,
        "git_provenance",
        lambda root: {"git_sha": "a" * 40, "git_dirty": False},
    )
    monkeypatch.setattr(
        challenge, "require_nonversioned_path", lambda path, purpose: path
    )

    def run(root=None, runs=None):
        return challenge.run_official_validation_challenge_suite(
            runs or evaluated,
            root or (tmp_path / "runs"),
            Path("unused"),
            Path("unused"),
        )

    return run


def test_official_suite_produces_one_combined_result(suite_runner, tmp_path) -> None:
    suite = suite_runner()
    assert suite["candidate_order"] == ["B4-A", "B4-B", "B4-C"]
    assert suite["test_accessed"] is False
    assert suite["training_performed"] is False
    assert suite["architecture_selection_performed"] is False
    assert suite["challenge_selection_sha256"]
    assert len(suite["metric_implementation_sha256"]) == 64
    directory = tmp_path / "runs" / challenge.SUITE_DIR_NAME
    attempt = json.loads((directory / challenge.SUITE_ATTEMPT_NAME).read_text())
    assert attempt["attempt_status"] == "COMPLETE"


def test_combined_and_child_digests_re_derive(suite_runner, tmp_path) -> None:
    suite = suite_runner()
    body = {
        k: v for k, v in suite.items() if k != "validation_challenge_suite_sha256"
    }
    assert canonical_sha256(body) == suite["validation_challenge_suite_sha256"]
    for name in benchmark.OFFICIAL_ORDER:
        result = suite["candidate_results"][name]
        child = {k: v for k, v in result.items() if k != "challenge_result_sha256"}
        assert canonical_sha256(child) == result["challenge_result_sha256"]
    assert challenge.read_official_validation_challenge_results(
        tmp_path / "runs"
    ) == suite


def test_repeat_attempt_is_refused(suite_runner) -> None:
    suite_runner()
    with pytest.raises(challenge.ValidationChallengeError, match="already exists"):
        suite_runner()


def test_existing_result_without_attempt_is_refused(suite_runner, tmp_path) -> None:
    suite_runner()
    directory = tmp_path / "runs" / challenge.SUITE_DIR_NAME
    (directory / challenge.SUITE_ATTEMPT_NAME).unlink()
    with pytest.raises(
        challenge.ValidationChallengeError, match="result already exists"
    ):
        suite_runner()


def test_failure_marks_the_attempt_and_never_releases_it(
    suite_runner, evaluated, tmp_path
) -> None:
    (evaluated["B4-C"] / "model_selected.pt").write_bytes(b"corrupt")
    with pytest.raises(benchmark.LockedModelError):
        suite_runner()
    suite_dir = tmp_path / "runs" / challenge.SUITE_DIR_NAME
    path = suite_dir / challenge.SUITE_ATTEMPT_NAME
    attempt = json.loads(path.read_text())
    assert attempt["attempt_status"] == "FAILED_OR_INTERRUPTED"
    assert attempt["human_review_required"] is True
    assert attempt["repeat_attempt_permitted"] is False
    assert attempt["selective_candidate_retry_permitted"] is False
    assert attempt["automatic_retry_performed"] is False
    assert attempt["error_type"]
    assert attempt["traceback"]
    with pytest.raises(challenge.ValidationChallengeError, match="already exists"):
        suite_runner()


def test_dirty_checkout_is_refused_before_any_claim(
    monkeypatch, evaluated, tmp_path
) -> None:
    monkeypatch.setattr(
        challenge,
        "git_provenance",
        lambda root: {"git_sha": "a" * 40, "git_dirty": True},
    )
    with pytest.raises(challenge.ValidationChallengeError, match="clean Git"):
        challenge.run_official_validation_challenge_suite(
            evaluated, tmp_path / "runs", Path("unused"), Path("unused")
        )
    assert not (tmp_path / "runs" / challenge.SUITE_DIR_NAME).exists()


def test_versioned_run_root_is_refused(monkeypatch, evaluated) -> None:
    monkeypatch.setattr(
        challenge,
        "git_provenance",
        lambda root: {"git_sha": "a" * 40, "git_dirty": False},
    )
    with pytest.raises(ValueError):
        challenge.run_official_validation_challenge_suite(
            evaluated,
            Path(challenge.REPOSITORY_ROOT) / "docs",
            Path("unused"),
            Path("unused"),
        )


def test_missing_and_fourth_candidate_are_refused(suite_runner, evaluated) -> None:
    with pytest.raises(challenge.ValidationChallengeError, match="requires exactly"):
        suite_runner(runs={k: v for k, v in evaluated.items() if k != "B4-C"})
    with pytest.raises(challenge.ValidationChallengeError, match="requires exactly"):
        suite_runner(runs={**evaluated, "B4-D": evaluated["B4-A"]})


def test_suite_exposes_no_evaluator_or_retry_injection() -> None:
    parameters = inspect.signature(
        challenge.run_official_validation_challenge_suite
    ).parameters
    assert set(parameters) == {
        "run_directories",
        "run_root",
        "feature_root",
        "source",
        "command",
    }
    for forbidden in ("runner", "backend", "evaluator", "metric", "retry", "force"):
        assert forbidden not in parameters
