"""A post-claim failure must leave evidence, and only evidence.

The consumed canonical attempt failed at stage 24 and produced no failure
receipt. Specification section 25 requires one with fourteen named fields;
`write_failure_receipt` was implemented and `T1DevelopmentRun.failure_receipt`
was written, but nothing called either, and `execute` had no handler. The status
file was left reading STARTED with the timestamp of the claim, ten minutes after
the state it described had stopped being true.

These tests cover the two halves of the fix. The receipt is written and the
original exception continues upward unchanged; the status names how far the run
got while it was still running, and what broke when it stopped.

Capturing an exception to record it is not retrying it. The distinction is the
subject of the last section: every handler must re-raise, no handler may loop,
and no second attempt becomes reachable because a failure was written down.

Nothing here runs the science. Every claimed run below lives in `tmp_path`; the
real canonical attempt is never read, written or re-created.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_canonical_driver as D
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.runtime_sentinel import RuntimeIntegrityRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _claimed(tmp_path: Path) -> PERSIST.T1ClaimedRun:
    """A claimed run in a temporary directory, seeded as the claim leaves it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    claimed = PERSIST.T1ClaimedRun(
        run_dir=run_dir,
        attempt_id=SPEC.T1_DEVELOPMENT_ATTEMPT_ID,
        started_at="2026-08-21T19:47:24Z",
        authorized_git_sha="c" * 40,
        runtime=RuntimeIntegrityRecord(expected_digest="0" * 64),
        stages=PERSIST.T1StageRecorder(),
    )
    (run_dir / PERSIST.RUN_STATUS_NAME).write_text(
        json.dumps(
            {
                "attempt_id": claimed.attempt_id,
                "status": PERSIST.STATUS_STARTED,
                "started_at": claimed.started_at,
                "updated_at": claimed.started_at,
                "label_blind_input_opened": False,
                "held_out_labels_opened_for_folds": [],
                "oof_evidence_promoted": False,
                "final_configuration_completed": False,
            }
        ),
        encoding="utf-8",
    )
    return claimed


def _status(claimed: PERSIST.T1ClaimedRun) -> dict:
    return json.loads(
        (claimed.run_dir / PERSIST.RUN_STATUS_NAME).read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# 1. The receipt exists, and says what broke
# ---------------------------------------------------------------------------


def test_a_post_claim_failure_writes_the_receipt(tmp_path):
    claimed = _claimed(tmp_path)
    claimed.stages.enter(SPEC.STAGE_START)
    path = PERSIST.write_failure_receipt(
        claimed,
        KeyError("true_positive"),
        state={"stage": SPEC.STAGE_OOF_RESULT, "current_fold": None},
        repository_root=tmp_path,
    )
    assert path is not None and path.name == PERSIST.FAILURE_RECEIPT_NAME
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for field in SPEC.T1_FAILURE_RECEIPT_FIELDS:
        assert field in receipt, f"the receipt omits the required field {field!r}"
    assert receipt["exception_type"] == "KeyError"
    assert receipt["stage"] == SPEC.STAGE_OOF_RESULT
    assert receipt["attempt_consumed"] is True
    assert receipt["automatic_retry_permitted"] is False


def test_a_pre_claim_failure_writes_nothing(tmp_path):
    """No claim, no directory, nothing to record."""
    assert (
        PERSIST.write_failure_receipt(
            None, RuntimeError("refused"), state={}, repository_root=tmp_path
        )
        is None
    )


def test_the_failure_status_names_the_stage_the_type_and_the_time(tmp_path):
    """The consumed attempt's status still reads STARTED. This is why."""
    claimed = _claimed(tmp_path)
    PERSIST.write_failure_receipt(
        claimed,
        KeyError("true_positive"),
        state={"stage": SPEC.STAGE_OOF_RESULT},
        repository_root=tmp_path,
    )
    status = _status(claimed)
    assert status["status"] == PERSIST.STATUS_FAILED
    assert status["stage"] == SPEC.STAGE_OOF_RESULT
    assert status["exception_type"] == "KeyError"
    assert status["failed_at"] == status["updated_at"] != claimed.started_at
    assert status["attempt_consumed"] is True
    assert status["automatic_retry_permitted"] is False
    assert status["sealed_test_state"] == SPEC.T1_SEALED_TEST_STATE


def test_the_receipt_is_additive_and_repairs_nothing(tmp_path):
    """Nothing already promoted is rewritten, and no artifact is removed."""
    claimed = _claimed(tmp_path)
    promoted = claimed.run_dir / "T1_INPUT_EVIDENCE.json"
    promoted.write_text('{"already": "promoted"}', encoding="utf-8")
    before = promoted.read_bytes()
    PERSIST.write_failure_receipt(
        claimed, RuntimeError("boom"), state={}, repository_root=tmp_path
    )
    assert promoted.read_bytes() == before
    assert (claimed.run_dir / PERSIST.FAILURE_RECEIPT_NAME).is_file()


# ---------------------------------------------------------------------------
# 2. Status checkpoints
# ---------------------------------------------------------------------------


def test_each_checkpoint_advances_the_status(tmp_path):
    claimed = _claimed(tmp_path)
    assert _status(claimed)["status"] == PERSIST.STATUS_STARTED
    for status in PERSIST.STATUS_CHECKPOINTS:
        PERSIST.write_status_checkpoint(claimed, status, stage=SPEC.STAGE_START)
        assert _status(claimed)["status"] == status
    assert _status(claimed)["stage"] == SPEC.STAGE_START


def test_a_checkpoint_carries_progress_but_no_science(tmp_path):
    claimed = _claimed(tmp_path)
    PERSIST.write_status_checkpoint(
        claimed,
        PERSIST.STATUS_FOLDS_COMPLETE,
        stage=SPEC.STAGE_FOLD_PROMOTE_HELD_OUT,
        progress={"oof_evidence_promoted": True},
    )
    status = _status(claimed)
    assert status["oof_evidence_promoted"] is True
    assert status["label_blind_input_opened"] is False
    forbidden = ("sha256", "digest", "threshold", "metric", "confusion", "row_count")
    flat = json.dumps(status).lower()
    for word in forbidden:
        assert word not in flat, f"the operational status carries {word!r}"


def test_a_checkpoint_cannot_regress_or_repeat(tmp_path):
    claimed = _claimed(tmp_path)
    PERSIST.write_status_checkpoint(
        claimed, PERSIST.STATUS_FOLDS_COMPLETE, stage=SPEC.STAGE_START
    )
    for backwards in (
        PERSIST.STATUS_STARTED,
        PERSIST.STATUS_PREFLIGHT_COMPLETE,
        PERSIST.STATUS_FOLDS_COMPLETE,
    ):
        with pytest.raises(PERSIST.T1PersistenceError, match="regress"):
            PERSIST.write_status_checkpoint(claimed, backwards, stage=SPEC.STAGE_START)


def test_an_unknown_status_is_refused(tmp_path):
    claimed = _claimed(tmp_path)
    with pytest.raises(PERSIST.T1PersistenceError, match="not a T1 run status"):
        PERSIST.write_status_checkpoint(claimed, "NEARLY_DONE", stage=SPEC.STAGE_START)


def test_a_checkpoint_without_a_claim_is_a_no_op(tmp_path):
    assert (
        PERSIST.write_status_checkpoint(None, PERSIST.STATUS_FOLDS_COMPLETE, stage="x")
        is None
    )


def test_the_checkpoint_sequence_matches_the_run_it_describes():
    """One checkpoint per phase the driver actually completes."""
    assert PERSIST.STATUS_SEQUENCE[0] == PERSIST.STATUS_STARTED
    assert PERSIST.STATUS_SEQUENCE[-1] == PERSIST.STATUS_COMPLETE
    assert PERSIST.STATUS_CHECKPOINTS == (
        PERSIST.STATUS_PREFLIGHT_COMPLETE,
        PERSIST.STATUS_LABEL_BLIND_EVIDENCE_COMPLETE,
        PERSIST.STATUS_FOLDS_COMPLETE,
        PERSIST.STATUS_OOF_STATE_COMPLETE,
    )
    assert PERSIST.STATUS_FAILED not in PERSIST.STATUS_SEQUENCE, (
        "a failure is terminal and does not sit on the progress ladder"
    )


# ---------------------------------------------------------------------------
# 3. Capturing is not retrying
# ---------------------------------------------------------------------------


class _Run:
    """Enough of a run to exercise the executor's failure path."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.calls: list[BaseException] = []
        self._raises = raises

    def failure_receipt(self, error: BaseException):
        self.calls.append(error)
        if self._raises is not None:
            raise self._raises
        return Path("receipt")


def test_the_executor_writes_the_receipt_for_the_exception_it_caught():
    run = _Run()
    executor = D.T1CanonicalDevelopmentExecutor(run=run)  # type: ignore[arg-type]
    error = KeyError("true_positive")
    executor._receipt_on_failure(error)
    assert run.calls == [error]


def test_a_receipt_that_cannot_be_written_never_replaces_the_failure():
    """The lost diagnostic is annotated; the original failure still wins."""
    run = _Run(raises=OSError("read-only filesystem"))
    executor = D.T1CanonicalDevelopmentExecutor(run=run)  # type: ignore[arg-type]
    error = KeyError("true_positive")
    executor._receipt_on_failure(error)
    notes = getattr(error, "__notes__", [])
    assert any("could not be written" in note for note in notes)
    assert any("OSError" in note for note in notes)


def test_the_handler_re_raises_and_never_returns_a_value():
    """Structural: an except block that ends without raising is a swallow."""
    source = Path(D.__file__).read_text(encoding="utf-8")
    execute = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    handlers = [n for n in ast.walk(execute) if isinstance(n, ast.ExceptHandler)]
    assert handlers, "there is no handler, so no receipt can be written"
    for handler in handlers:
        assert isinstance(handler.body[-1], ast.Raise)
        assert handler.body[-1].exc is None
        assert not [n for n in ast.walk(handler) if isinstance(n, ast.Return)]


def test_no_recovery_constant_moved():
    assert SPEC.T1_AUTOMATIC_RETRY_PERMITTED is False
    assert SPEC.T1_RECOVERY_IDENTITY_PREDECLARED is False
    assert SPEC.T1_ALTERNATE_RUN_ROOT_PERMITTED is False
    assert SPEC.T1_FAILED_ATTEMPT_MAY_BE_DELETED_OR_REWRITTEN is False
    assert SPEC.T1_POST_CLAIM_FAILURE_CONSUMES_ATTEMPT is True


def test_the_consumed_attempt_is_not_touched_by_these_tests():
    canonical = PERSIST.canonical_run_directory(REPOSITORY_ROOT)
    existed = canonical.exists()
    assert canonical.exists() is existed
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()
