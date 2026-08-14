"""Post-claim failure accounting, the recovery identity and its lineage.

M2 development attempt #1 ended with both `M2_RUN_STATUS.json` files still
reading STARTED, because the exception escaped outside a promotion gate and
nothing recorded it. Once any arm claim exists, an uncaught canonical-run
exception must now leave deterministic non-claim-bearing evidence -- without
deleting, cleaning, renaming, retrying or making a failed attempt look
COMPLETE.

Synthetic fixtures only. No VALIDATION access, no scoring, no metric, no TEST.
"""

from __future__ import annotations

import inspect
import json

import pytest

from cardiosentinel.neural import m2_development_run as R
from cardiosentinel.neural import m2_persistence as PS
from cardiosentinel.neural import runtime_sentinel as S

FROZEN_DIGEST = "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
SUITE = R.CANONICAL_SUITE_ID


def _frozen_check(point, detail="test"):
    return S.RuntimeCheck(
        enforcement_point=S.EnforcementPoint(point).value,
        observed_digest=FROZEN_DIGEST,
        expected_digest=FROZEN_DIGEST,
        matches=True,
        package_count=335,
        observed_at="2026-01-01T00:00:00Z",
        detail=detail,
    )


def _runtime():
    record = S.RuntimeIntegrityRecord()
    record.record(_frozen_check(S.EnforcementPoint.START.value))
    return record


def _claim(run_root, arm, started="2026-01-01T00:00:00Z"):
    run_dir = run_root / PS.arm_experiment_id(SUITE, arm)
    run_dir.mkdir(parents=True)
    (run_dir / PS.RUN_STATUS_NAME).write_text(
        json.dumps(
            {
                "experiment_id": PS.arm_experiment_id(SUITE, arm),
                "arm": arm,
                "status": PS.STATUS_STARTED,
                "claim_bearing_result_promoted": False,
                "started_at": started,
                "updated_at": started,
            }
        )
    )
    return run_dir


# --------------------------------------------------------------------------
# §10 -- deterministic accounting at every post-claim stage
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage",
    [
        "start_and_claim_both_arms",
        "development_source_integrity",
        "full_label_blind_replay_both_arms",
        "post_replay_population_construction",
        "persist_and_promote_per_arm:M2-0",
    ],
)
def test_a_failure_at_any_post_claim_stage_is_recorded(tmp_path, stage):
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)

    receipt = PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("synthetic"),
        stage=stage,
        claimed_arms=list(R.CANONICAL_ARM_ORDER),
        validation_opened="replay" in stage,
        runtime_records={arm: _runtime() for arm in R.CANONICAL_ARM_ORDER},
    )
    assert receipt["failed_stage"] == stage
    assert receipt["exception_type"] == "RuntimeError"
    assert receipt["claim_bearing"] is False
    assert receipt["canonical"] is False
    assert receipt["automatic_retry_performed"] is False
    assert receipt["automatic_cleanup_performed"] is False
    assert receipt["alternate_suite_id_used"] is False
    assert receipt["staged_evidence_preserved"] is True
    assert receipt["human_review_required"] is True
    assert receipt["test_accessed"] is False
    assert receipt["sealed_test_state"] == "unopened"
    assert receipt["validation_opened"] == ("replay" in stage)
    assert set(receipt["runtime_identity_checks"]) == set(R.CANONICAL_ARM_ORDER)

    path = (
        PS.failure_review_directory(run_root, SUITE) / PS.ATTEMPT_FAILURE_RECEIPT_NAME
    )
    assert path.is_file()
    # Every claim now says FAILED, never COMPLETE.
    for arm in R.CANONICAL_ARM_ORDER:
        status = json.loads(
            (
                run_root / PS.arm_experiment_id(SUITE, arm) / PS.RUN_STATUS_NAME
            ).read_text()
        )
        assert status["status"] == PS.STATUS_FAILED
        assert status["canonical"] is False
        assert status["repeat_attempt_permitted"] is False
        assert status["automatic_retry_performed"] is False
        assert stage in status["reason"]


def test_a_failure_before_any_claim_records_nothing(tmp_path):
    """No claim means no attempt consumed and no directory to annotate."""
    run_root = tmp_path / "runs"
    tracker = R._AttemptTracker(run_root=run_root, suite_id=SUITE)
    assert tracker.record_failure(RuntimeError("before any claim")) is None
    assert not run_root.exists()


def test_failure_accounting_preserves_the_started_timestamp(tmp_path):
    run_root = tmp_path / "runs"
    _claim(run_root, "M2-0", started="2026-08-13T21:58:49Z")
    PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=ValueError("x"),
        stage="full_label_blind_replay_both_arms",
        claimed_arms=["M2-0"],
        validation_opened=True,
    )
    status = json.loads(
        (
            run_root / PS.arm_experiment_id(SUITE, "M2-0") / PS.RUN_STATUS_NAME
        ).read_text()
    )
    assert status["started_at"] == "2026-08-13T21:58:49Z"


def test_failure_accounting_never_deletes_or_renames(tmp_path):
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    workspace = PS.evidence_workspace(run_root, SUITE)
    (workspace / "M2-0").mkdir(parents=True)
    (workspace / "M2-0" / "partial.npz").write_bytes(b"staged")

    PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("x"),
        stage="full_label_blind_replay_both_arms",
        claimed_arms=list(R.CANONICAL_ARM_ORDER),
        validation_opened=True,
    )
    for arm in R.CANONICAL_ARM_ORDER:
        assert (run_root / PS.arm_experiment_id(SUITE, arm)).is_dir()
    assert (workspace / "M2-0" / "partial.npz").read_bytes() == b"staged"


def test_the_receipt_lives_outside_the_immutable_arm_directories(tmp_path):
    run_root = tmp_path / "runs"
    _claim(run_root, "M2-0")
    PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("x"),
        stage="development_source_integrity",
        claimed_arms=["M2-0"],
        validation_opened=False,
    )
    review = PS.failure_review_directory(run_root, SUITE)
    assert review.name.endswith(PS.FAILURE_REVIEW_SUFFIX)
    for arm in R.CANONICAL_ARM_ORDER:
        assert review != run_root / PS.arm_experiment_id(SUITE, arm)


def test_the_additive_writer_never_touches_the_claim_files(tmp_path):
    """§4 -- the historical status files are evidence, not scratch space."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    before = PS.preserved_status_digests(run_root, SUITE)
    PS.write_forensic_failure_receipt(
        run_root, SUITE, {"artifact_class": "m2_attempt_failure_receipt"}
    )
    assert PS.preserved_status_digests(run_root, SUITE) == before


def test_the_run_wraps_failures_in_the_tracker():
    source = inspect.getsource(R.execute_canonical_development)
    assert "_AttemptTracker(" in source
    assert "tracker.record_failure(error)" in source
    assert "raise" in source


# --------------------------------------------------------------------------
# §11 -- exactly one recovery suite
# --------------------------------------------------------------------------


def test_the_original_suite_is_permanently_consumed():
    assert R.ORIGINAL_SUITE_ID == "m2-v1-development-two-arm"
    assert R.CANONICAL_SUITE_ID == "m2-v1-development-two-arm-recovery1"
    assert R.CANONICAL_SUITE_ID != R.ORIGINAL_SUITE_ID


def test_no_public_suite_id_override_and_no_alternate_names():
    parameters = inspect.signature(R.execute_canonical_development).parameters
    assert "suite_id" not in parameters
    with pytest.raises(R.M2DevelopmentRunError, match="is refused"):
        R.require_canonical_suite_id(R.ORIGINAL_SUITE_ID)
    for name in ("m2-v1-development-two-arm-recovery2", "attempt3", "recovery1-2"):
        with pytest.raises(R.M2DevelopmentRunError, match="is refused"):
            R.require_canonical_suite_id(name)
    assert R.require_canonical_suite_id(R.CANONICAL_SUITE_ID) == R.CANONICAL_SUITE_ID


def test_the_cli_still_exposes_only_the_two_authorization_flags():
    options = {
        flag for action in R.build_parser()._actions for flag in action.option_strings
    }
    assert options == {"-h", "--help", R.EXECUTION_FLAG, R.EXPECTED_GIT_SHA_FLAG}


# --------------------------------------------------------------------------
# §12 -- recovery lineage is claim-bearing provenance
# --------------------------------------------------------------------------


def test_the_recovery_lineage_states_what_attempt_one_was():
    lineage = R.recovery_lineage()
    assert lineage["recovery_from_suite_id"] == R.ORIGINAL_SUITE_ID
    assert lineage["recovery_suite_id"] == R.CANONICAL_SUITE_ID
    assert lineage["recovery_reason_class"] == (
        "pre_scoring_partition_alignment_execution_defect"
    )
    assert lineage["prior_attempt_scoring_started"] is False
    assert lineage["prior_attempt_metrics_computed"] is False
    assert lineage["prior_attempt_test_accessed"] is False
    assert PS.validate_recovery_lineage(lineage)


def test_the_recovery_decision_digest_matches_the_committed_document():
    import hashlib

    from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

    path = REPOSITORY_ROOT / R.RECOVERY_DECISION_DOCUMENT
    assert hashlib.sha256(path.read_bytes()).hexdigest() == R.RECOVERY_DECISION_SHA256


@pytest.mark.parametrize(
    "mutation",
    [
        {"recovery_decision_sha256": "z" * 64},
        {"recovery_from_suite_id": "something-else"},
        {"recovery_suite_id": "m2-v1-development-two-arm"},
        {"recovery_reason_class": "scientific_redesign"},
        {"prior_attempt_scoring_started": True},
        {"prior_attempt_metrics_computed": True},
        {"prior_attempt_test_accessed": True},
    ],
)
def test_a_wrong_lineage_value_is_rejected(mutation):
    with pytest.raises(PS.M2PersistenceError):
        PS.validate_recovery_lineage({**R.recovery_lineage(), **mutation})


def test_missing_lineage_fields_are_rejected():
    with pytest.raises(PS.M2PersistenceError, match="must bind its lineage"):
        PS.validate_recovery_lineage({})


def test_every_claim_bearing_artifact_requires_the_lineage():
    for field in PS.RECOVERY_LINEAGE_FIELDS:
        assert field in PS.REQUIRED_RESULT_FIELDS, field
        assert field in PS.REQUIRED_PROVENANCE_FIELDS, field
    assert "recovery_lineage" in inspect.signature(PS.build_suite_body).parameters


# --------------------------------------------------------------------------
# §13 -- execution history distinguishes the two attempts
# --------------------------------------------------------------------------


def test_execution_history_reports_both_attempts(tmp_path):
    run_root = tmp_path / "runs"
    history = R.canonical_execution_history(run_root)
    assert history["original_attempt"]["suite_id"] == R.ORIGINAL_SUITE_ID
    assert history["original_attempt"]["state"] == R.STATE_UNCLAIMED
    assert history["recovery_attempt"]["suite_id"] == R.CANONICAL_SUITE_ID
    assert history["recovery_attempt"]["state"] == R.STATE_UNCLAIMED


def test_a_claimed_original_is_not_assumed_to_be_the_frozen_failure(tmp_path):
    """§2.A -- directory existence never grants `consumed_failed_pre_scoring`.

    The verified positive case is proved in `test_m2_attempt1_lineage.py`
    against attempt #1's frozen bytes.
    """
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        run_dir = run_root / PS.arm_experiment_id(R.ORIGINAL_SUITE_ID, arm)
        run_dir.mkdir(parents=True)
        (run_dir / PS.RUN_STATUS_NAME).write_text(json.dumps({"status": "STARTED"}))
    original = R.canonical_execution_history(run_root)["original_attempt"]
    assert original["state"] == R.STATE_CLAIMED
    assert original["lineage_verified"] is False
    assert "lineage_error" in original
    # And nothing asserts an exposure it has not proven.
    assert "scoring_started" not in original


def test_a_claimed_recovery_blocks_a_second_recovery_run(tmp_path):
    """Refused whether or not the original lineage would have verified."""
    run_root = tmp_path / "runs"
    _claim(run_root, "M2-0")
    with pytest.raises(
        (R.M2DevelopmentRunError, PS.M2PersistenceError), match="already|proven"
    ):
        R.require_recovery_preconditions(run_root, SUITE)


def test_a_failed_recovery_is_reported_as_failed(tmp_path):
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("x"),
        stage="full_label_blind_replay_both_arms",
        claimed_arms=list(R.CANONICAL_ARM_ORDER),
        validation_opened=True,
    )
    history = R.canonical_execution_history(run_root)
    assert history["recovery_attempt"]["state"] == R.STATE_FAILED
    assert history["recovery_attempt"]["failure_receipt_present"] is True
    # And no further recovery is implicitly authorized.
    with pytest.raises(
        (R.M2DevelopmentRunError, PS.M2PersistenceError), match="consumed|proven"
    ):
        R.require_recovery_preconditions(run_root, SUITE)


# --------------------------------------------------------------------------
# §14/§15 -- nothing scientific changed, nothing scientific was read
# --------------------------------------------------------------------------


def test_no_frozen_scientific_rule_changed():
    from cardiosentinel.neural import m2_gate as G
    from cardiosentinel.neural import m2_policy as P
    from cardiosentinel.neural import m2_scorer as SC

    assert SC.M1L_CLASSIFICATION_THRESHOLD == 0.7554003000259399
    assert SC.NORMAL_EVIDENCE_THRESHOLD == 0.0002997174742631614
    assert P.M2_ARMS == ("M2-0", "M2-G")
    assert G.validate_m2_protocol() == G.M2_PROTOCOL_SHA256
    assert G.validate_m2_gate_receipt() == G.M2_GATE_RECEIPT_SHA256
    SC.assert_thresholds_are_distinct()


def test_this_module_opens_no_real_development_data():
    import ast
    from pathlib import Path

    forbidden = {
        "load_p1_embedding_cache",
        "build_validation_challenge_index",
        "load_stream_store",
        "read_annotations",
        "read_record",
        "execute_canonical_development",
        "_run",
    }
    tree = ast.parse(Path(__file__).read_text())
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not (called & forbidden), sorted(called & forbidden)


# --------------------------------------------------------------------------
# §3-§6 -- a failure receipt reports REAL exposure, not a hard-coded optimism
# --------------------------------------------------------------------------


def _tracker(tmp_path, **fields):
    tracker = R._AttemptTracker(run_root=tmp_path / "runs", suite_id=SUITE)
    tracker.claimed_arms = list(R.CANONICAL_ARM_ORDER)
    for name, value in fields.items():
        setattr(tracker, name, value)
    return tracker


def test_a_failure_before_scoring_records_scoring_started_false(tmp_path):
    """§13.7."""
    tracker = _tracker(tmp_path, stage="development_source_integrity")
    exposure = tracker.exposure()
    assert exposure["scoring_started"] is False
    assert exposure["post_replay_evaluation_started"] is False
    assert exposure["metrics_computed_or_completed"] is False


def test_a_failure_after_the_first_scorer_call_records_scoring_started_true(tmp_path):
    """§13.8 -- the wrapper flags exposure the moment the scorer is invoked."""
    calls = []

    class _Scorer:
        def __call__(self, representation, d_long):
            calls.append((representation, d_long))
            return 0.1234567890123456

    tracker = _tracker(tmp_path, stage="full_label_blind_replay_both_arms")
    assert tracker.scoring_started is False
    wrapped = tracker.tracking_scorer(_Scorer())
    wrapped([1.0, 2.0], 0.5)
    assert tracker.scoring_started is True
    assert tracker.exposure()["scoring_started"] is True
    assert calls == [([1.0, 2.0], 0.5)]


def test_the_tracking_wrapper_changes_no_numeric(tmp_path):
    """§13.19 -- scorer parity: the wrapper returns the exact same object."""
    import numpy as np

    sentinel = object()

    class _Scorer:
        weights = "frozen"

        def __call__(self, representation, d_long):
            return sentinel

        def identity(self):
            return {"retained_lock_sha256": "x"}

    scorer = _Scorer()
    wrapped = _tracker(tmp_path).tracking_scorer(scorer)
    assert wrapped(np.zeros(3), 0.0) is sentinel
    # Attribute access passes straight through to the frozen scorer.
    assert wrapped.identity() == scorer.identity()
    assert wrapped.weights == "frozen"

    # And on real float values, byte-for-byte equality.
    class _Real:
        def __call__(self, representation, d_long):
            return float(np.sqrt(np.mean(np.asarray(representation) ** 2)) + d_long)

    real = _Real()
    tracked = _tracker(tmp_path).tracking_scorer(real)
    for vector in (np.array([0.1, 0.2, 0.3]), np.array([1e-17, np.pi, -1234.5])):
        assert tracked(vector, 0.75).hex() == real(vector, 0.75).hex()


def test_an_unfinished_replay_records_scoring_as_indeterminate(tmp_path):
    """A mid-replay exception cannot honestly claim scoring never started."""
    tracker = _tracker(
        tmp_path,
        stage="full_label_blind_replay_both_arms",
        validation_opened=True,
    )
    assert tracker.exposure()["scoring_started"] == PS.INDETERMINATE


def test_a_failure_during_post_replay_evidence_records_evaluation_started(tmp_path):
    """§13.9."""
    tracker = _tracker(
        tmp_path,
        stage="post_replay_population_construction",
        validation_opened=True,
        scoring_started=True,
        replay_completed=True,
        post_replay_evaluation_started=True,
    )
    exposure = tracker.exposure()
    assert exposure["post_replay_evaluation_started"] is True
    assert exposure["scoring_started"] is True
    # Evidence construction began but completed for no arm: not a confident no.
    assert exposure["metrics_computed_or_completed"] == PS.INDETERMINATE


def test_a_failure_after_metric_construction_never_claims_none_were_computed(tmp_path):
    """§13.10."""
    tracker = _tracker(
        tmp_path,
        stage="persist_and_promote_per_arm:M2-G",
        validation_opened=True,
        scoring_started=True,
        replay_completed=True,
        post_replay_evaluation_started=True,
        metrics_completed={"M2-0": True},
    )
    exposure = tracker.exposure()
    assert exposure["metrics_computed_or_completed"] is True
    assert exposure["metrics_completed_per_arm"] == {"M2-0": True}

    receipt = tracker.record_failure(RuntimeError("boom"))
    assert receipt["metrics_computed_or_completed"] is True
    assert receipt["scoring_started"] is True
    assert receipt["metrics_completed_per_arm"] == {"M2-0": True}


def test_the_receipt_carries_the_trackers_exposure(tmp_path):
    tracker = _tracker(
        tmp_path,
        stage="full_label_blind_replay_both_arms",
        validation_opened=True,
        scoring_started=True,
    )
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(tracker.run_root, arm)
    receipt = tracker.record_failure(RuntimeError("boom"))
    assert receipt["scoring_started"] is True
    assert receipt["validation_opened"] is True
    assert receipt["replay_completed"] is False
    assert receipt["exposure_source"] == "runtime execution tracker"


def test_attempt_one_style_optimism_is_not_hard_coded():
    """The old receipt asserted scoring_started=false for every future failure."""
    source = inspect.getsource(PS.record_attempt_failure)
    assert '"scoring_started": False' not in source
    assert '"metrics_computed": False' not in source
    assert "INDETERMINATE" in source


# --------------------------------------------------------------------------
# §4 -- promotion accounting is PER ARM
# --------------------------------------------------------------------------


def test_promotion_state_is_arm_specific(tmp_path):
    """§13.11."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    receipt = PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("x"),
        stage="persist_and_promote_per_arm:M2-G",
        claimed_arms=list(R.CANONICAL_ARM_ORDER),
        validation_opened=True,
        promotion_state={
            "arm_result_promoted": {"M2-0": True, "M2-G": False},
            "experiment_lock_promoted": {"M2-0": True, "M2-G": False},
            "suite_result_promoted": False,
        },
    )
    state = receipt["promotion_state"]
    assert state["arm_result_promoted"] == {"M2-0": True, "M2-G": False}
    assert state["experiment_lock_promoted"] == {"M2-0": True, "M2-G": False}
    assert state["suite_result_promoted"] is False


def test_m2_0_promoted_and_m2_g_failed_is_represented_exactly(tmp_path):
    """§13.12 -- one arm promoting never implies the other did."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    tracker = R._AttemptTracker(run_root=run_root, suite_id=SUITE)
    tracker.claimed_arms = list(R.CANONICAL_ARM_ORDER)
    tracker.arm_result_promoted["M2-0"] = True
    tracker.experiment_lock_promoted["M2-0"] = True
    tracker.stage = "persist_and_promote_per_arm:M2-G"

    receipt = tracker.record_failure(RuntimeError("M2-G failed"))
    state = receipt["promotion_state"]
    assert state["arm_result_promoted"]["M2-0"] is True
    assert state["arm_result_promoted"]["M2-G"] is False
    assert state["suite_result_promoted"] is False

    # The per-arm status files agree.
    statuses = {
        arm: json.loads(
            (
                run_root / PS.arm_experiment_id(SUITE, arm) / PS.RUN_STATUS_NAME
            ).read_text()
        )
        for arm in R.CANONICAL_ARM_ORDER
    }
    assert statuses["M2-0"]["claim_bearing_result_promoted"] is True
    assert statuses["M2-G"]["claim_bearing_result_promoted"] is False
    for status in statuses.values():
        assert status["status"] == PS.STATUS_FAILED
        assert status["canonical"] is False


def test_a_scalar_promotion_flag_is_normalised_per_arm(tmp_path):
    """A single boolean must never silently mean 'both arms'."""
    run_root = tmp_path / "runs"
    _claim(run_root, "M2-0")
    receipt = PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("x"),
        stage="two_arm_suite_without_selection",
        claimed_arms=["M2-0"],
        validation_opened=True,
        promotion_state={"arm_result_promoted": False},
    )
    assert receipt["promotion_state"]["arm_result_promoted"] == {
        "M2-0": False,
        "M2-G": False,
    }


def test_the_run_marks_metrics_and_promotion_per_arm():
    source = inspect.getsource(R._run)
    assert "track.metrics_completed[arm] = True" in source
    assert "track.arm_result_promoted[arm] = True" in source
    assert "track.experiment_lock_promoted[arm] = True" in source
    assert "track.tracking_scorer(scorer)" in source
