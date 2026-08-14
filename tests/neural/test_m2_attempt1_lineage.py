"""Attempt #1's frozen forensic lineage, and the recovery it authorizes.

`consumed_failed_pre_scoring` is a SCIENTIFIC claim: that THIS attempt failed,
that it failed before any row was scored, that no metric was produced and that
the sealed test stayed shut. A claim directory proves none of that, so the
recovery route verifies the preserved artifacts against frozen digests instead.

The fixtures below rebuild attempt #1's promoted bytes exactly -- the writer is
`json.dumps(payload, indent=2, sort_keys=True) + "\\n"`, so a dict round-trips
byte-identically and the frozen digests reproduce. The REAL preserved attempt is
never read or touched by these tests.
"""

from __future__ import annotations

import pytest

from cardiosentinel.neural import m2_development_run as R
from cardiosentinel.neural import m2_persistence as PS
from tests.neural.m2_attempt1_fixtures import (
    FROZEN_RECEIPT,
    FROZEN_STATUS,
    ORIGINAL,
    _plant_frozen_attempt1,
    _resigned,
    _write,
)

# --------------------------------------------------------------------------
# The frozen fixture really is attempt #1
# --------------------------------------------------------------------------


def test_the_fixture_reproduces_the_frozen_digests(tmp_path):
    """If this drifts, every negative test below is testing the wrong thing."""
    run_root = _plant_frozen_attempt1(tmp_path / "runs")
    assert PS.preserved_status_digests(run_root, ORIGINAL) == dict(
        R.ORIGINAL_STATUS_SHA256
    )
    lineage = PS.validate_original_attempt1_failure_lineage(run_root)
    assert lineage["verified_from_artifacts"] is True
    assert lineage["scoring_started"] is False
    assert lineage["metrics_computed"] is False
    assert lineage["test_accessed"] is False
    assert lineage["promoted_any_claim_bearing_artifact"] is False


def test_only_a_fully_verified_original_authorizes_recovery(tmp_path):
    """§2.J -- the one positive case."""
    run_root = _plant_frozen_attempt1(tmp_path / "runs")
    history = R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)
    assert history["original_attempt"]["state"] == R.STATE_CONSUMED_FAILED_PRE_SCORING
    assert history["original_attempt"]["lineage_verified"] is True
    assert history["recovery_attempt"]["state"] == R.STATE_UNCLAIMED
    assert history["original_attempt_lineage"]["verified_from_artifacts"] is True


# --------------------------------------------------------------------------
# §2.A-J -- directory existence alone proves nothing
# --------------------------------------------------------------------------


def test_claimed_directories_without_the_lineage_do_not_authorize_recovery(tmp_path):
    """§2.A -- the previous self-fulfilling behaviour, now refused."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        path = run_root / PS.arm_experiment_id(ORIGINAL, arm) / PS.RUN_STATUS_NAME
        _write(path, {"status": "STARTED"})
    with pytest.raises(PS.M2PersistenceError, match="could not be proven"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)
    # And history refuses to call it pre-scoring.
    history = R.canonical_execution_history(run_root)
    assert history["original_attempt"]["state"] == R.STATE_CLAIMED
    assert history["original_attempt"]["lineage_verified"] is False


def test_bare_empty_claim_directories_do_not_authorize_recovery(tmp_path):
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        (run_root / PS.arm_experiment_id(ORIGINAL, arm)).mkdir(parents=True)
    with pytest.raises(PS.M2PersistenceError, match="is absent"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


def test_one_original_arm_only_does_not_authorize_recovery(tmp_path):
    """§2.B."""
    run_root = _plant_frozen_attempt1(tmp_path / "runs", arms=("M2-0",))
    with pytest.raises(PS.M2PersistenceError, match="M2-G claim directory"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


def test_a_wrong_original_status_digest_refuses_recovery(tmp_path):
    """§2.C -- a mutated historical status file blocks the recovery."""
    mutated = {**FROZEN_STATUS["M2-0"], "status": "COMPLETE"}
    run_root = _plant_frozen_attempt1(tmp_path / "runs", status={"M2-0": mutated})
    with pytest.raises(PS.M2PersistenceError, match="digests to"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


def test_a_missing_failure_receipt_refuses_recovery(tmp_path):
    """§2.D."""
    run_root = _plant_frozen_attempt1(tmp_path / "runs", receipt=False)
    with pytest.raises(PS.M2PersistenceError, match="failure receipt .* is absent"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


def test_a_wrong_failure_receipt_digest_refuses_recovery(tmp_path):
    """§2.E -- re-signed, so only the FROZEN digest can catch it."""
    run_root = _plant_frozen_attempt1(
        tmp_path / "runs", receipt=_resigned(recorded_at="2027-01-01T00:00:00Z")
    )
    with pytest.raises(PS.M2PersistenceError, match="not the frozen"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


def test_a_receipt_with_a_broken_self_digest_refuses_recovery(tmp_path):
    run_root = _plant_frozen_attempt1(
        tmp_path / "runs", receipt={**FROZEN_RECEIPT, "failed_stage": "elsewhere"}
    )
    with pytest.raises(PS.M2PersistenceError, match="digests to|canonical digest"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


@pytest.mark.parametrize(
    ("field", "value", "clause"),
    [
        ("scoring_started", True, "scoring_started"),
        ("metrics_computed", True, "metrics_computed"),
        ("test_accessed", True, "test_accessed"),
        ("sealed_test_state", "opened", "sealed test"),
        ("validation_opened", False, "validation_opened"),
    ],
)
def test_a_receipt_claiming_greater_exposure_refuses_recovery(
    tmp_path, field, value, clause
):
    """§2.F/G/H -- a receipt admitting scoring, metrics or TEST blocks recovery.

    Re-signed so the receipt is internally valid: what refuses it is the frozen
    determination about attempt #1, not a broken hash.
    """
    run_root = _plant_frozen_attempt1(
        tmp_path / "runs", receipt=_resigned(**{field: value})
    )
    with pytest.raises(PS.M2PersistenceError):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


@pytest.mark.parametrize("artifact", ["M2_ARM_RESULT.json", "M2_EXPERIMENT_LOCK.json"])
def test_any_original_arm_artifact_refuses_the_pre_scoring_classification(
    tmp_path, artifact
):
    """§2.I -- a promoted arm artifact contradicts 'promoted nothing'."""
    run_root = _plant_frozen_attempt1(tmp_path / "runs")
    (run_root / PS.arm_experiment_id(ORIGINAL, "M2-0") / artifact).write_text("{}")
    with pytest.raises(PS.M2PersistenceError, match="promoted nothing"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)
    assert (
        R.canonical_execution_history(run_root)["original_attempt"]["state"]
        != R.STATE_CONSUMED_FAILED_PRE_SCORING
    )


def test_an_original_suite_result_refuses_the_pre_scoring_classification(tmp_path):
    """§2.I -- and so does a completed original suite."""
    run_root = _plant_frozen_attempt1(tmp_path / "runs")
    _write(PS.suite_directory(run_root, ORIGINAL) / PS.SUITE_RESULT_NAME, {"suite": 1})
    with pytest.raises(PS.M2PersistenceError, match="never having completed"):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


def test_a_receipt_for_another_suite_refuses_recovery(tmp_path):
    run_root = _plant_frozen_attempt1(
        tmp_path / "runs", receipt=_resigned(suite_id="some-other-suite")
    )
    with pytest.raises(PS.M2PersistenceError):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


def test_a_receipt_describing_a_different_failure_refuses_recovery(tmp_path):
    """A later failure is not attempt #1's frozen pre-scoring failure."""
    run_root = _plant_frozen_attempt1(
        tmp_path / "runs",
        receipt=_resigned(failed_stage="post_replay_frozen_evidence"),
    )
    with pytest.raises(PS.M2PersistenceError):
        R.require_recovery_preconditions(run_root, R.CANONICAL_SUITE_ID)


def test_the_validator_writes_nothing(tmp_path):
    """§10 -- verification is read-only over the preserved attempt."""
    run_root = _plant_frozen_attempt1(tmp_path / "runs")
    before = {
        path: path.read_bytes()
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    }
    PS.validate_original_attempt1_failure_lineage(run_root)
    after = {
        path: path.read_bytes()
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    }
    assert before == after


# --------------------------------------------------------------------------
# §12/§13.20 -- no real scientific data is touched by these tests
# --------------------------------------------------------------------------


def test_these_tests_open_no_real_development_data():
    import ast
    from pathlib import Path

    forbidden = {
        "load_p1_embedding_cache",
        "build_validation_challenge_index",
        "load_stream_store",
        "iter_timeline_streams",
        "read_annotations",
        "read_record",
        "execute_canonical_development",
        "canonical_roots",
        "_run",
    }
    for module in ("test_m2_attempt1_lineage.py", "m2_attempt1_fixtures.py"):
        tree = ast.parse((Path(__file__).parent / module).read_text())
        called = {
            getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert not (called & forbidden), (module, sorted(called & forbidden))


def test_every_fixture_root_is_temporary():
    """The real preserved attempt is never read, written or resolved.

    Checked structurally: every test here takes `tmp_path`, and neither module
    calls `canonical_roots()` (asserted above), which is the only way to reach
    the real run root.
    """
    import inspect
    import sys

    module = sys.modules[__name__]
    for name, function in vars(module).items():
        if not name.startswith("test_") or not callable(function):
            continue
        parameters = set(inspect.signature(function).parameters)
        if parameters:
            assert "tmp_path" in parameters, name
