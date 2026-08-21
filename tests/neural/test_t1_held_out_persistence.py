"""Held-out evaluation evidence must survive the process that produced it.

Specification section 17 asks for the held-out state trace **and evaluation
evidence** to be persisted once. The canonical attempt persisted the trace, by
widening it into the stage-23 OOF store, and kept the evaluation evidence in
`run.held_out_traces` -- an in-process mapping that died with the interpreter.
Twelve folds of completed measurement were lost to a defect in the twenty-fourth
stage, and that, rather than the defect itself, is why the attempt could not be
finished.

These tests hold the fix to four properties: the evidence is written per fold
immediately after the evaluation returns, it carries every field the
specification names, the evaluation still happens exactly once, and a fold that
finished stays on disk when a later stage fails.

Nothing here runs the science. Every claimed run lives in `tmp_path`; the
consumed canonical attempt is never read, written or re-created, and no
evaluation is performed by any test in this file.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_development_run as R
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.p1_experiment import FROZEN_DEPENDENCY_DIGEST
from cardiosentinel.neural.provenance import dependency_environment
from cardiosentinel.neural.runtime_sentinel import RuntimeIntegrityRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Promotion runs the pre-promotion runtime check, which only passes on the
# frozen scientific interpreter. CI installs a different set and is refused
# there, which is correct behaviour; the refusal tests below carry no marker and
# run everywhere. Same convention as the sibling T1 suites.
ON_FROZEN_INTERPRETER = str(
    dependency_environment()["installed_packages_sha256"]
) == str(FROZEN_DEPENDENCY_DIGEST)
requires_frozen_runtime = pytest.mark.skipif(
    not ON_FROZEN_INTERPRETER,
    reason=(
        "promoting evidence runs the pre-promotion runtime check; this "
        "environment reports a different installed-package digest"
    ),
)


# ---------------------------------------------------------------------------
# Fixtures: a claimed run, a fold, and an evaluation that already happened
# ---------------------------------------------------------------------------


class _Fold:
    def __init__(self, index: int) -> None:
        self.fold_index = index
        self.held_out_subject = f"ltstdb:s{2000 + index}"
        self.fit_subjects = tuple(f"ltstdb:s{3000 + n}" for n in range(11))


def _claimed(tmp_path: Path, *, frozen: bool = True) -> PERSIST.T1ClaimedRun:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runtime = (
        RuntimeIntegrityRecord()
        if frozen
        else RuntimeIntegrityRecord(expected_digest="0" * 64)
    )
    return PERSIST.T1ClaimedRun(
        run_dir=run_dir,
        attempt_id=SPEC.T1_DEVELOPMENT_ATTEMPT_ID,
        started_at="2026-08-21T19:47:24Z",
        authorized_git_sha="c" * 40,
        runtime=runtime,
        stages=PERSIST.T1StageRecorder(),
    )


def _evaluated(index: int = 0) -> dict:
    """Shaped exactly as `T1CanonicalFoldEvaluator.evaluate_held_out` returns."""
    return {
        "fold_index": index,
        "held_out_subject": f"ltstdb:s{2000 + index}",
        "selected_policy_id": "qw0.9_qe0.99_FAST",
        "policy_runs": 1,
        "row_positions": (0, 1, 2, 3),
        "trace_columns": {"emitted_state": ("NORMAL", "WATCH", "EVENT", "RECOVERY")},
        "episode_evidence": {
            "reference_episodes": 7,
            "predicted_event_runs": 4,
            "matched_episodes": 3,
            "unmatched_predicted_runs": 1,
        },
        "onset_latency_seconds": (12.0, 30.0),
        "primary_confusion": {"tp": 3, "fp": 2, "tn": 90, "fn": 5},
        "evaluation_scope": {"scope": "held_out_subject_only", "test_accessed": False},
        "test_accessed": False,
    }


def _selection(index: int = 0) -> dict:
    return {
        "fold_index": index,
        "held_out_subject": f"ltstdb:s{2000 + index}",
        "selected_policy_id": "qw0.9_qe0.99_FAST",
        "q_watch": 0.9,
        "q_event": 0.99,
        "persistence_profile": "FAST",
        "p_watch": 0.0968,
        "s_watch": 0.0818,
        "p_event": 0.4822,
        "s_event": 0.9367,
        "held_out_labels_opened": False,
        "test_accessed": False,
    }


def _evidence(claimed, index: int = 0, digest: str = "a" * 64) -> dict:
    return R._held_out_evidence(
        claimed=claimed,
        fold=_Fold(index),
        evaluated=_evaluated(index),
        selection=_selection(index),
        selection_sha256=digest,
    )


def _stage_folds_tree() -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(R.T1DevelopmentRun.stage_folds)))


def _fold_loop() -> ast.For:
    """The per-fold loop, identified by what it iterates.

    `stage_folds` opens with a loop over stage names; picking the first `For`
    in the tree finds that one instead, which is how this probe first reported
    a body with no evaluation in it.
    """
    for node in ast.walk(_stage_folds_tree()):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "t1_folds"
        ):
            return node
    raise AssertionError("stage_folds no longer loops over t1_folds()")


# ---------------------------------------------------------------------------
# A. Persistence happens after the evaluation, and not only at the end
# ---------------------------------------------------------------------------


def test_the_evidence_is_promoted_inside_the_fold_loop():
    """Structural: the promotion is a statement of the per-fold body."""
    loop = _fold_loop()
    promoted = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "promote_held_out_evaluation"
    ]
    assert len(promoted) == 1, (
        "the held-out evidence is promoted once per fold, inside the loop; "
        "promoting after the loop is what the consumed attempt did"
    )


def test_the_evidence_is_promoted_after_the_evaluation_returns():
    """Ordering: evaluate, then persist -- never the reverse."""
    loop = _fold_loop()
    positions = {}
    for index, statement in enumerate(loop.body):
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                positions.setdefault(node.func.attr, index)
    assert "evaluate_held_out" in positions, "the loop no longer evaluates"
    assert "promote_held_out_evaluation" in positions
    assert positions["evaluate_held_out"] < positions["promote_held_out_evaluation"], (
        "evidence is persisted before the evaluation that produces it"
    )
    assert positions["promote_fold_selection"] < positions["evaluate_held_out"], (
        "the selection barrier no longer precedes the held-out evaluation"
    )


@requires_frozen_runtime
def test_promotion_writes_a_readable_artifact(tmp_path):
    claimed = _claimed(tmp_path)
    digest = PERSIST.promote_held_out_evaluation(claimed, 0, _evidence(claimed))
    path = claimed.held_out_dir / "T1_FOLD_00_HELD_OUT.json"
    assert path.is_file()
    assert len(digest) == 64
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fold_index"] == 0
    assert payload["artifact_class"] == PERSIST.HELD_OUT_EVALUATION_CLASS


# ---------------------------------------------------------------------------
# B. Every field the specification names
# ---------------------------------------------------------------------------


def test_the_evidence_carries_every_required_field(tmp_path):
    evidence = _evidence(_claimed(tmp_path, frozen=False))
    for field in PERSIST.HELD_OUT_EVALUATION_REQUIRED_FIELDS:
        assert field in evidence, f"the evidence omits {field!r}"


@pytest.mark.parametrize(
    "dropped",
    [
        "primary_confusion",
        "episode_evidence",
        "onset_latency_seconds",
        "policy",
        "thresholds",
        "policy_runs",
        "selected_policy_id",
        "fold_selection_sha256",
    ],
)
def test_incomplete_evidence_is_refused_rather_than_promoted(tmp_path, dropped):
    """An artifact that exists but cannot answer the specification is worse
    than an artifact that is absent."""
    claimed = _claimed(tmp_path, frozen=False)
    evidence = _evidence(claimed)
    evidence.pop(dropped)
    with pytest.raises(PERSIST.T1PersistenceError, match=dropped):
        PERSIST.promote_held_out_evaluation(claimed, 0, evidence)
    assert not claimed.held_out_dir.exists(), (
        "a refused promotion left a directory behind"
    )


def test_the_evidence_records_what_it_is_and_is_not(tmp_path):
    evidence = _evidence(_claimed(tmp_path, frozen=False))
    assert evidence["generated_during_canonical_execution"] is True
    assert evidence["is_recovery_artifact"] is False
    assert evidence["is_continuation_artifact"] is False
    assert evidence["test_accessed"] is False


def test_the_evidence_binds_its_provenance(tmp_path):
    claimed = _claimed(tmp_path, frozen=False)
    evidence = _evidence(claimed, digest="b" * 64)
    assert evidence["attempt_id"] == SPEC.T1_DEVELOPMENT_ATTEMPT_ID
    assert evidence["experiment_identity"] == SPEC.T1_EXPERIMENT_IDENTITY
    assert evidence["authorized_git_sha"] == claimed.authorized_git_sha
    assert evidence["protocol_sha256"] == PERSIST.T1_PROTOCOL_SHA256
    assert evidence["execution_spec_sha256"] == PERSIST.T1_EXECUTION_SPEC_SHA256
    assert evidence["fold_selection_sha256"] == "b" * 64
    assert evidence["held_out_subject"] == "ltstdb:s2000"


def test_the_policy_and_thresholds_come_from_the_promoted_selection(tmp_path):
    """The promoted artifact is the decision; the evaluation describes it."""
    claimed = _claimed(tmp_path, frozen=False)
    selection = _selection()
    evidence = R._held_out_evidence(
        claimed=claimed,
        fold=_Fold(0),
        evaluated=_evaluated(),
        selection=selection,
        selection_sha256="c" * 64,
    )
    assert evidence["thresholds"] == {
        "p_watch": selection["p_watch"],
        "s_watch": selection["s_watch"],
        "p_event": selection["p_event"],
        "s_event": selection["s_event"],
    }
    assert evidence["policy"]["persistence_profile"] == selection["persistence_profile"]


def test_the_per_row_trace_is_not_duplicated_here(tmp_path):
    """The trace belongs to the section 18 store; two records is two answers."""
    evidence = _evidence(_claimed(tmp_path, frozen=False))
    assert "trace_columns" not in evidence
    assert "row_positions" not in evidence
    assert evidence["evaluated_row_count"] == 4


# ---------------------------------------------------------------------------
# C. The evaluation still happens exactly once per fold
# ---------------------------------------------------------------------------


def test_the_loop_evaluates_each_held_out_subject_once():
    loop = _fold_loop()
    calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate_held_out"
    ]
    assert len(calls) == 1, "the fold body evaluates the held-out subject twice"
    inner = [node for node in ast.walk(loop) if isinstance(node, (ast.For, ast.While))]
    assert inner == [loop], "a loop inside the fold body could re-evaluate"


def test_persisting_evidence_calls_no_evaluator(tmp_path):
    """Building and promoting the record runs nothing scientific.

    The evaluation object is replaced with one that raises on any attribute
    access, so a persistence path that reached back into it would fail here.
    """

    class _Explodes(dict):
        def __getattr__(self, name):  # pragma: no cover - defensive
            raise AssertionError(f"persistence called the evaluator: {name}")

    claimed = _claimed(tmp_path, frozen=False)
    evidence = R._held_out_evidence(
        claimed=claimed,
        fold=_Fold(0),
        evaluated=_Explodes(_evaluated()),
        selection=_selection(),
        selection_sha256="d" * 64,
    )
    assert evidence["policy_runs"] == 1


def test_the_single_run_guard_is_unchanged():
    assert SPEC.T1_HELD_OUT_POLICY_RUNS_PER_FOLD == 1
    assert SPEC.T1_FOLD_RETRY_PERMITTED is False


@requires_frozen_runtime
def test_evidence_is_never_promoted_twice_for_one_fold(tmp_path):
    claimed = _claimed(tmp_path)
    PERSIST.promote_held_out_evaluation(claimed, 0, _evidence(claimed))
    with pytest.raises(PERSIST.T1PersistenceError, match="already"):
        PERSIST.promote_held_out_evaluation(claimed, 0, _evidence(claimed))


# ---------------------------------------------------------------------------
# D. A fold that finished stays finished
# ---------------------------------------------------------------------------


@requires_frozen_runtime
def test_earlier_fold_evidence_survives_a_later_failure(tmp_path):
    """The property the consumed attempt did not have."""
    claimed = _claimed(tmp_path)
    for index in (0, 1, 2):
        PERSIST.promote_held_out_evaluation(claimed, index, _evidence(claimed, index))

    # Fold 3 fails the way stage 24 failed: after earlier folds completed.
    PERSIST.write_failure_receipt(
        claimed,
        KeyError("true_positive"),
        state={"stage": SPEC.STAGE_OOF_RESULT, "current_fold": 3},
        repository_root=tmp_path,
    )

    survived = PERSIST.read_held_out_evaluations(claimed)
    assert sorted(survived) == [0, 1, 2]
    for index, payload in survived.items():
        assert payload["held_out_subject"] == f"ltstdb:s{2000 + index}"
        assert payload["primary_confusion"] == {"tp": 3, "fp": 2, "tn": 90, "fn": 5}
    assert (claimed.run_dir / PERSIST.FAILURE_RECEIPT_NAME).is_file()


@requires_frozen_runtime
def test_the_failure_receipt_does_not_disturb_promoted_evidence(tmp_path):
    claimed = _claimed(tmp_path)
    PERSIST.promote_held_out_evaluation(claimed, 0, _evidence(claimed))
    path = claimed.held_out_dir / "T1_FOLD_00_HELD_OUT.json"
    before = path.read_bytes()
    PERSIST.write_failure_receipt(
        claimed, RuntimeError("boom"), state={}, repository_root=tmp_path
    )
    assert path.read_bytes() == before


def test_reading_evidence_from_a_run_without_any_is_empty(tmp_path):
    assert PERSIST.read_held_out_evaluations(_claimed(tmp_path, frozen=False)) == {}


# ---------------------------------------------------------------------------
# Nothing was executed, consumed or created outside tmp_path
# ---------------------------------------------------------------------------


def test_the_consumed_attempt_is_untouched_by_these_tests():
    canonical = PERSIST.canonical_run_directory(REPOSITORY_ROOT)
    existed = canonical.exists()
    assert canonical.exists() is existed
    assert not (canonical / PERSIST.HELD_OUT_TRACE_DIR).exists(), (
        "these tests wrote into the consumed canonical attempt"
    )
    assert not (
        REPOSITORY_ROOT / "cardiosentinel-runs" / "phase9-t1-continuation-v1"
    ).exists()
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()
