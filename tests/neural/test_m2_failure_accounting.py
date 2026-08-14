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


def test_both_prior_suites_are_permanently_consumed():
    assert R.ORIGINAL_SUITE_ID == "m2-v1-development-two-arm"
    assert R.RECOVERY1_SUITE_ID == "m2-v1-development-two-arm-recovery1"
    assert R.CANONICAL_SUITE_ID == "m2-v1-development-two-arm-recovery2"
    assert len({R.ORIGINAL_SUITE_ID, R.RECOVERY1_SUITE_ID, R.CANONICAL_SUITE_ID}) == 3


def test_no_public_suite_id_override_and_no_alternate_names():
    parameters = inspect.signature(R.execute_canonical_development).parameters
    assert "suite_id" not in parameters
    with pytest.raises(R.M2DevelopmentRunError, match="is refused"):
        R.require_canonical_suite_id(R.ORIGINAL_SUITE_ID)
    # Recovery1 is consumed and can never be reused.
    with pytest.raises(R.M2DevelopmentRunError, match="is refused"):
        R.require_canonical_suite_id(R.RECOVERY1_SUITE_ID)
    for name in ("m2-v1-development-two-arm-recovery3", "attempt4", "recovery2-2"):
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


def test_the_recovery_lineage_states_what_BOTH_prior_attempts_were():
    lineage = R.recovery_lineage()
    assert lineage["recovery_from_original_suite_id"] == R.ORIGINAL_SUITE_ID
    assert lineage["recovery1_suite_id"] == R.RECOVERY1_SUITE_ID
    assert lineage["recovery2_suite_id"] == R.CANONICAL_SUITE_ID
    assert lineage["attempt1_reason_class"] == (
        "pre_scoring_partition_alignment_execution_defect"
    )
    assert lineage["recovery1_reason_class"] == (
        "pre_scoring_source_null_join_sentinel_defect"
    )
    assert lineage["attempt1_scoring_started"] is False
    assert lineage["attempt1_metrics_computed"] is False
    assert lineage["attempt1_test_accessed"] is False
    # BOTH recovery1 scoring facts are preserved, neither replacing the other.
    assert lineage["recovery1_receipt_scoring_started"] == "indeterminate"
    assert lineage["recovery1_human_forensic_scorer_invocation_observed"] is False
    assert lineage["recovery1_replay_completed"] is False
    assert lineage["recovery1_metrics_computed"] is False
    assert lineage["recovery1_test_accessed"] is False
    assert PS.validate_recovery_lineage(lineage)


def test_both_recovery_decision_digests_match_their_committed_documents():
    import hashlib

    from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

    for document, digest in (
        (R.RECOVERY_DECISION_DOCUMENT, R.RECOVERY_DECISION_SHA256),
        (R.RECOVERY2_DECISION_DOCUMENT, R.RECOVERY2_DECISION_SHA256),
    ):
        path = REPOSITORY_ROOT / document
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, document


@pytest.mark.parametrize(
    "mutation",
    [
        {"recovery2_decision_sha256": "z" * 64},
        {"recovery_from_original_suite_id": "something-else"},
        {"recovery1_suite_id": "m2-v1-development-two-arm"},
        {"recovery2_suite_id": "m2-v1-development-two-arm-recovery1"},
        {"attempt1_reason_class": "scientific_redesign"},
        {"recovery1_reason_class": "scientific_redesign"},
        {"attempt1_scoring_started": True},
        {"attempt1_metrics_computed": True},
        {"attempt1_test_accessed": True},
        {"recovery1_receipt_scoring_started": False},
        {"recovery1_human_forensic_scorer_invocation_observed": True},
        {"recovery1_replay_completed": True},
        {"recovery1_metrics_computed": True},
        {"recovery1_test_accessed": True},
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


def test_execution_history_reports_all_three_attempts(tmp_path):
    run_root = tmp_path / "runs"
    history = R.canonical_execution_history(run_root)
    assert history["original_attempt"]["suite_id"] == R.ORIGINAL_SUITE_ID
    assert history["recovery1_attempt"]["suite_id"] == R.RECOVERY1_SUITE_ID
    assert history["recovery_attempt"]["suite_id"] == R.CANONICAL_SUITE_ID
    for key in ("original_attempt", "recovery1_attempt", "recovery_attempt"):
        assert history[key]["state"] == R.STATE_UNCLAIMED, key


def test_recovery2_requires_BOTH_prior_lineages(tmp_path):
    """§17.16 -- attempt #1 alone is not enough."""
    from tests.neural.m2_attempt1_fixtures import (
        _plant_both_prior_attempts,
        _plant_frozen_attempt1,
        _plant_frozen_recovery1,
    )

    only_attempt1 = _plant_frozen_attempt1(tmp_path / "a" / "runs")
    with pytest.raises(PS.M2PersistenceError, match="recovery1"):
        R.require_recovery_preconditions(only_attempt1, SUITE)

    only_recovery1 = _plant_frozen_recovery1(tmp_path / "b" / "runs")
    with pytest.raises(PS.M2PersistenceError, match="attempt #1"):
        R.require_recovery_preconditions(only_recovery1, SUITE)

    both = _plant_both_prior_attempts(tmp_path / "c" / "runs")
    history = R.require_recovery_preconditions(both, SUITE)
    assert history["original_attempt_lineage"]["verified_from_artifacts"] is True
    assert history["recovery1_attempt_lineage"]["verified_from_artifacts"] is True
    assert history["recovery_attempt"]["state"] == R.STATE_UNCLAIMED


def test_a_mutated_recovery1_artifact_blocks_recovery2(tmp_path):
    """§17.15 -- recovery1's lineage is proven, not assumed."""
    from tests.neural.m2_attempt1_fixtures import (
        FROZEN_RECOVERY1_STATUS,
        _plant_frozen_attempt1,
        _plant_frozen_recovery1,
        _resigned_recovery1,
    )

    root = _plant_frozen_attempt1(tmp_path / "runs")
    _plant_frozen_recovery1(
        root, status={"M2-0": {**FROZEN_RECOVERY1_STATUS["M2-0"], "status": "COMPLETE"}}
    )
    with pytest.raises(PS.M2PersistenceError, match="digests to"):
        R.require_recovery_preconditions(root, SUITE)

    # A receipt claiming greater exposure is refused. Re-signing it keeps the
    # canonical digest internally valid, so the FROZEN file digest is what
    # catches it -- the outer guard fires before any field is inspected.
    root2 = _plant_frozen_attempt1(tmp_path / "b" / "runs")
    _plant_frozen_recovery1(root2, receipt=_resigned_recovery1(scoring_started=True))
    with pytest.raises(PS.M2PersistenceError, match="receipt file digests to"):
        R.require_recovery_preconditions(root2, SUITE)


def test_a_recovery1_receipt_claiming_exposure_is_refused_on_its_fields(tmp_path):
    """With the file digest relaxed, the field checks still refuse it."""
    from tests.neural.m2_attempt1_fixtures import FROZEN_RECOVERY1_RECEIPT

    for field, value, clause in (
        ("scoring_started", True, "scoring_started"),
        ("replay_completed", True, "replay_completed"),
        ("test_accessed", True, "test_accessed"),
        ("sealed_test_state", "opened", "sealed test"),
    ):
        payload = {**FROZEN_RECOVERY1_RECEIPT, field: value}
        with pytest.raises(PS.M2PersistenceError, match=clause):
            PS._require_recovery1_receipt_fields(payload)


def test_the_recovery1_receipt_conservative_value_is_never_rewritten(tmp_path):
    """§3 -- `indeterminate` is preserved, alongside the forensic finding."""
    from tests.neural.m2_attempt1_fixtures import (
        FROZEN_RECOVERY1_RECEIPT,
        _plant_both_prior_attempts,
    )

    root = _plant_both_prior_attempts(tmp_path / "runs")
    assert FROZEN_RECOVERY1_RECEIPT["scoring_started"] == "indeterminate"
    proof = PS.validate_recovery1_failure_lineage(root)
    assert proof["recovery1_receipt_scoring_started"] == "indeterminate"
    assert proof["recovery1_human_forensic_scorer_invocation_observed"] is False


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


def _promote(run_root, arm, *, result=True, lock=False):
    """Place the ACTUAL immutable artifacts a promotion would have left."""
    run_dir = run_root / PS.arm_experiment_id(SUITE, arm)
    run_dir.mkdir(parents=True, exist_ok=True)
    if result:
        (run_dir / PS.ARM_RESULT_NAME).write_text("{}")
    if lock:
        (run_dir / PS.EXPERIMENT_LOCK_NAME).write_text("{}")


def test_promotion_state_is_read_from_the_actual_artifacts(tmp_path):
    """§10.1 -- the filesystem is the forensic authority."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    _promote(run_root, "M2-0", result=True, lock=True)
    _promote(run_root, "M2-G", result=True, lock=False)

    receipt = PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("lock promotion failed"),
        stage="persist_and_promote_per_arm:M2-G",
        claimed_arms=list(R.CANONICAL_ARM_ORDER),
        validation_opened=True,
    )
    state = receipt["promotion_state"]
    assert state["authority"] == "filesystem"
    assert state["arm_result_promoted"] == {"M2-0": True, "M2-G": True}
    assert state["experiment_lock_promoted"] == {"M2-0": True, "M2-G": False}
    assert state["suite_result_promoted"] is False


def test_the_tracker_cannot_overwrite_a_true_filesystem_promotion(tmp_path):
    """§10.3 -- a stale tracker never contradicts an artifact that exists."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    _promote(run_root, "M2-0", result=True, lock=False)

    receipt = PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("boom"),
        stage="persist_and_promote_per_arm:M2-0",
        claimed_arms=list(R.CANONICAL_ARM_ORDER),
        validation_opened=True,
        # The tracker believed nothing was promoted -- the finalizer raised
        # before it could record success.
        promotion_state={
            "arm_result_promoted": dict.fromkeys(R.CANONICAL_ARM_ORDER, False),
            "experiment_lock_promoted": dict.fromkeys(R.CANONICAL_ARM_ORDER, False),
            "suite_result_promoted": False,
        },
    )
    state = receipt["promotion_state"]
    assert state["arm_result_promoted"]["M2-0"] is True
    assert state["tracker_observation"]["arm_result_promoted"]["M2-0"] is False
    assert state["coherent"] is False
    # And the arm's own status agrees with the filesystem.
    status = json.loads(
        (
            run_root / PS.arm_experiment_id(SUITE, "M2-0") / PS.RUN_STATUS_NAME
        ).read_text()
    )
    assert status["claim_bearing_result_promoted"] is True
    assert status["experiment_lock_promoted"] is False
    assert status["canonical"] is False


def test_a_failure_before_any_promotion_stays_false(tmp_path):
    """§10.4."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    receipt = PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("x"),
        stage="full_label_blind_replay_both_arms",
        claimed_arms=list(R.CANONICAL_ARM_ORDER),
        validation_opened=True,
    )
    state = receipt["promotion_state"]
    assert state["arm_result_promoted"] == dict.fromkeys(R.CANONICAL_ARM_ORDER, False)
    assert state["experiment_lock_promoted"] == dict.fromkeys(
        R.CANONICAL_ARM_ORDER, False
    )
    assert state["suite_result_promoted"] is False
    assert state["coherent"] is True


def test_m2_0_complete_and_m2_g_partial_is_represented_per_arm(tmp_path):
    """§10.5 -- one arm's completion never implies the other's."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
    _promote(run_root, "M2-0", result=True, lock=True)

    tracker = R._AttemptTracker(run_root=run_root, suite_id=SUITE)
    tracker.claimed_arms = list(R.CANONICAL_ARM_ORDER)
    tracker.arm_result_promoted["M2-0"] = True
    tracker.experiment_lock_promoted["M2-0"] = True
    tracker.stage = "persist_and_promote_per_arm:M2-G"
    receipt = tracker.record_failure(RuntimeError("M2-G failed"))

    state = receipt["promotion_state"]
    assert state["arm_result_promoted"] == {"M2-0": True, "M2-G": False}
    assert state["experiment_lock_promoted"] == {"M2-0": True, "M2-G": False}
    assert state["coherent"] is True
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


def test_suite_promotion_state_comes_from_the_actual_suite_artifact(tmp_path):
    """§10.6."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        _claim(run_root, arm)
        _promote(run_root, arm, result=True, lock=True)
    suite_dir = PS.suite_directory(run_root, SUITE)
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / PS.SUITE_RESULT_NAME).write_text("{}")

    receipt = PS.record_attempt_failure(
        run_root,
        SUITE,
        exception=RuntimeError("bookkeeping after the suite landed"),
        stage="two_arm_suite_without_selection",
        claimed_arms=list(R.CANONICAL_ARM_ORDER),
        validation_opened=True,
        promotion_state={
            "arm_result_promoted": dict.fromkeys(R.CANONICAL_ARM_ORDER, True),
            "experiment_lock_promoted": dict.fromkeys(R.CANONICAL_ARM_ORDER, True),
            "suite_result_promoted": False,
        },
    )
    assert receipt["promotion_state"]["suite_result_promoted"] is True
    assert receipt["promotion_state"]["coherent"] is False


def test_a_scalar_promotion_flag_is_refused(tmp_path):
    """§5/§10.7 -- one boolean cannot say WHICH arm promoted."""
    run_root = tmp_path / "runs"
    _claim(run_root, "M2-0")
    with pytest.raises(PS.M2PersistenceError, match="cannot identify which arm"):
        PS.record_attempt_failure(
            run_root,
            SUITE,
            exception=RuntimeError("x"),
            stage="two_arm_suite_without_selection",
            claimed_arms=["M2-0"],
            validation_opened=True,
            promotion_state={"arm_result_promoted": True},
        )
    with pytest.raises(PS.M2PersistenceError, match="unknown arms"):
        PS._require_per_arm_map({"M2-X": True}, "arm_result_promoted")


def test_observed_promotion_state_ignores_unclaimed_arms(tmp_path):
    run_root = tmp_path / "runs"
    _claim(run_root, "M2-0")
    _promote(run_root, "M2-0", result=True, lock=True)
    state = PS.observed_promotion_state(run_root, SUITE, ["M2-0"])
    assert state["arm_result_promoted"] == {"M2-0": True, "M2-G": False}


def test_the_run_marks_metrics_and_promotion_per_arm():
    source = inspect.getsource(R._run)
    assert "track.metrics_completed[arm] = True" in source
    assert "track.arm_result_promoted[arm] = True" in source
    assert "track.experiment_lock_promoted[arm] = True" in source
    assert "track.tracking_scorer(scorer)" in source


# --------------------------------------------------------------------------
# §17.17-19 -- recovery2 is the only permitted new suite
# --------------------------------------------------------------------------


def test_recovery2_is_the_only_permitted_new_suite():
    """§17.17."""
    assert R.CANONICAL_SUITE_ID == "m2-v1-development-two-arm-recovery2"
    assert R.require_canonical_suite_id(R.CANONICAL_SUITE_ID) == R.CANONICAL_SUITE_ID


def test_recovery1_cannot_be_reused(tmp_path):
    """§17.18 -- the consumed recovery1 id is refused outright."""
    with pytest.raises(R.M2DevelopmentRunError, match="is refused"):
        R.require_canonical_suite_id(R.RECOVERY1_SUITE_ID)
    # And nothing was created by the refusal.
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "name",
    [
        "m2-v1-development-two-arm-recovery3",
        "m2-v1-development-two-arm-attempt4",
        "m2-v1-development-two-arm-recovery2-2026-08-14",
        "m2-v1-development-two-arm-recovery2-b7f1",
        "recovery2",
    ],
)
def test_recovery3_and_every_alternate_name_are_refused(name):
    """§17.19 -- no recovery3, timestamp, random suffix or bare alias."""
    with pytest.raises(R.M2DevelopmentRunError, match="is refused"):
        R.require_canonical_suite_id(name)


def test_a_claimed_recovery2_stops_rather_than_escalating(tmp_path):
    """Nothing authorizes a further attempt after recovery2 is claimed."""
    from tests.neural.m2_attempt1_fixtures import _plant_both_prior_attempts

    run_root = _plant_both_prior_attempts(tmp_path / "runs")
    _claim(run_root, "M2-0")
    with pytest.raises(R.M2DevelopmentRunError, match="already"):
        R.require_recovery_preconditions(run_root, SUITE)


def test_the_three_attempt_history_states_are_distinct(tmp_path):
    """§12 -- attempt #1, recovery1 and recovery2 each report their own state."""
    from tests.neural.m2_attempt1_fixtures import _plant_both_prior_attempts

    run_root = _plant_both_prior_attempts(tmp_path / "runs")
    history = R.canonical_execution_history(run_root)
    assert history["original_attempt"]["state"] == R.STATE_CONSUMED_FAILED_PRE_SCORING
    assert (
        history["recovery1_attempt"]["state"] == R.STATE_CONSUMED_FAILED_STREAM_ASSEMBLY
    )
    assert history["recovery_attempt"]["state"] == R.STATE_UNCLAIMED
    assert history["original_attempt"]["lineage_verified"] is True
    assert history["recovery1_attempt"]["lineage_verified"] is True
    # Recovery1's two scoring facts are both surfaced.
    assert history["recovery1_attempt"]["receipt_scoring_started"] == "indeterminate"
    assert (
        history["recovery1_attempt"]["human_forensic_scorer_invocation_observed"]
        is False
    )


def test_a_claimed_but_unproven_recovery1_is_not_called_consumed(tmp_path):
    """Recovery1's classification is proven from artifacts, never inferred."""
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        path = (
            run_root
            / PS.arm_experiment_id(R.RECOVERY1_SUITE_ID, arm)
            / PS.RUN_STATUS_NAME
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"status": "FAILED_OR_INTERRUPTED"}))
    entry = R.canonical_execution_history(run_root)["recovery1_attempt"]
    assert entry["state"] == R.STATE_CLAIMED
    assert entry["lineage_verified"] is False
    assert "lineage_error" in entry
