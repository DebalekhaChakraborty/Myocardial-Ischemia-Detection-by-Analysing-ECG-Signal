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


def test_a_claimed_original_reports_consumed_failed_pre_scoring(tmp_path):
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        run_dir = run_root / PS.arm_experiment_id(R.ORIGINAL_SUITE_ID, arm)
        run_dir.mkdir(parents=True)
        (run_dir / PS.RUN_STATUS_NAME).write_text(json.dumps({"status": "STARTED"}))
    history = R.canonical_execution_history(run_root)
    original = history["original_attempt"]
    assert original["state"] == R.STATE_CONSUMED_FAILED_PRE_SCORING
    assert original["scoring_started"] is False
    assert original["metrics_computed"] is False
    assert original["test_accessed"] is False


def test_a_claimed_recovery_blocks_a_second_recovery_run(tmp_path):
    run_root = tmp_path / "runs"
    _claim(run_root, "M2-0")
    with pytest.raises(R.M2DevelopmentRunError, match="already"):
        R.require_recovery_preconditions(run_root, SUITE)


def test_recovery_preconditions_pass_on_a_clean_recovery(tmp_path):
    run_root = tmp_path / "runs"
    for arm in R.CANONICAL_ARM_ORDER:
        run_dir = run_root / PS.arm_experiment_id(R.ORIGINAL_SUITE_ID, arm)
        run_dir.mkdir(parents=True)
        (run_dir / PS.RUN_STATUS_NAME).write_text(json.dumps({"status": "STARTED"}))
    history = R.require_recovery_preconditions(run_root, SUITE)
    assert history["recovery_attempt"]["state"] == R.STATE_UNCLAIMED


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
    with pytest.raises(R.M2DevelopmentRunError, match="attempt is consumed"):
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
