"""Tests for the canonical T1-v1 development harness.

The harness is implemented here but never executed: no canonical run is
started, no OOF evidence is generated, no VALIDATION row is read and TEST stays
sealed. These tests exercise the machinery on synthetic structures and on
temporary directories, and several of them exist specifically to prove that the
real one has not been touched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from _attempt_guard import assert_attempt_unconsumed

from cardiosentinel.neural import t1_development_run as R
from cardiosentinel.neural import t1_evidence_store as STORE
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural import t1_protocol as P
from cardiosentinel.neural.runtime_sentinel import RuntimeIntegrityRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULES = (R, STORE, PERSIST)


def _observed_dependency_digest() -> str:
    from cardiosentinel.neural.provenance import dependency_environment

    return str(dependency_environment()["installed_packages_sha256"])


def _frozen_dependency_digest() -> str:
    from cardiosentinel.neural.p1_experiment import FROZEN_DEPENDENCY_DIGEST

    return str(FROZEN_DEPENDENCY_DIGEST)


# A canonical claim requires the frozen scientific interpreter, by design: the
# runtime sentinel refuses to let a claim rest on an environment that is not the
# one the science was frozen against. CI installs a different set, so the tests
# that actually claim a directory skip there -- the same convention the M2/U1/T2
# selection suites use for artifacts that do not exist on that filesystem.
# `test_a_claim_is_refused_outside_the_frozen_interpreter` covers the refusal
# itself and runs everywhere.
ON_FROZEN_INTERPRETER = _observed_dependency_digest() == _frozen_dependency_digest()
requires_frozen_runtime = pytest.mark.skipif(
    not ON_FROZEN_INTERPRETER,
    reason=(
        "a canonical claim requires the frozen scientific interpreter; this "
        "environment reports a different installed-package digest"
    ),
)


def _started_runtime() -> RuntimeIntegrityRecord:
    """A record carrying the START observation a canonical claim requires."""
    from cardiosentinel.neural.runtime_sentinel import (
        EnforcementPoint,
        require_runtime_identity,
    )

    record = RuntimeIntegrityRecord()
    require_runtime_identity(EnforcementPoint.START, record=record, detail="test")
    return record


# ---------------------------------------------------------------------------
# Execution safety: nothing here starts anything
# ---------------------------------------------------------------------------


def test_importing_the_harness_creates_no_files_and_emits_nothing(tmp_path, capsys):
    import importlib

    before = {p for p in tmp_path.rglob("*")}
    for module in MODULES:
        importlib.reload(module)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert {p for p in tmp_path.rglob("*")} == before


def test_the_canonical_attempt_cannot_be_claimed_twice():
    """The one canonical attempt is spent, and a second claim is refused.

    This test used to assert the run directory did not exist. It did not, until
    2026-08-21; the attempt then ran and failed after the claim, and the
    directory is now permanent evidence. The property worth holding is not that
    no run has happened -- that can never be true again -- but that nothing
    claims the attempt a second time, which is what the guard proves for this
    suite and what the refusal below proves for the mechanism.
    """
    assert_attempt_unconsumed()
    with pytest.raises(PERSIST.T1PersistenceError) as caught:
        PERSIST.require_unclaimed_canonical_attempt(REPOSITORY_ROOT)
    message = str(caught.value).lower()
    assert "already claimed" in message
    assert "no automatic retry" in message
    assert "human review" in message


def test_the_harness_performs_no_work_at_module_scope():
    """Every statement at module level is a definition or a constant."""
    tree = ast.parse(Path(R.__file__).read_text())
    for node in tree.body:
        assert isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
                ast.If,
            ),
        ), type(node).__name__
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant), "module-level call at import"
        if isinstance(node, ast.If):
            assert "__main__" in ast.dump(node.test), "conditional work at import"


def test_constructing_the_run_object_touches_nothing(tmp_path):
    run = R.T1DevelopmentRun(authorized_git_sha="0" * 40, repository_root=tmp_path)
    assert run.claimed is None
    assert run.stages.entered == []
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_the_cli_registers_exactly_the_two_frozen_options():
    assert R.registered_options() == (
        SPEC.T1_CANONICAL_EXECUTION_FLAG,
        SPEC.T1_EXPECTED_GIT_SHA_FLAG,
    )


@pytest.mark.parametrize("forbidden", SPEC.T1_FORBIDDEN_CLI_OPTIONS)
def test_no_scientific_option_is_registered(forbidden):
    assert forbidden not in R.registered_options()
    with pytest.raises(SPEC.T1ExecutionSpecError):
        SPEC.require_cli_option_permitted(forbidden)


def test_every_registered_option_is_permitted_by_the_specification():
    for option in R.registered_options():
        assert SPEC.require_cli_option_permitted(option) == option


# ---------------------------------------------------------------------------
# TEST firewall
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["test", "TEST", " Test "])
def test_test_cannot_be_accessed(value):
    with pytest.raises(SPEC.T1ExecutionSpecError, match="sealed"):
        SPEC.require_no_test_access(value)
    with pytest.raises(SPEC.T1ExecutionSpecError, match="sealed"):
        PERSIST.require_no_test_path(value)


def test_no_module_mentions_a_test_artifact():
    for module in MODULES:
        source = Path(module.__file__).read_text()
        for marker in ("TEST_ATTEMPT", "evaluate-locked-test", "sealed_test_cache"):
            assert marker not in source, f"{module.__name__} mentions {marker}"


# ---------------------------------------------------------------------------
# The fold-scoped label firewall
# ---------------------------------------------------------------------------


def test_a_fit_authority_refuses_the_held_out_subject():
    fold = P.t1_folds()[0]
    authority = R.fit_authority(fold.fit_subjects)
    for subject in fold.fit_subjects:
        assert authority.require_authorized(subject) == subject
    with pytest.raises(R.T1DevelopmentError, match="closed"):
        authority.require_authorized(fold.held_out_subject)


def test_held_out_labels_are_unreachable_before_the_selection_is_promoted():
    fold = P.t1_folds()[3]
    with pytest.raises(SPEC.T1ExecutionSpecError, match="not cross-fitting"):
        R.held_out_authority(fold.held_out_subject, {"selection_promoted": False})


def test_held_out_labels_are_unreachable_before_the_digest_is_verified():
    fold = P.t1_folds()[3]
    with pytest.raises(SPEC.T1ExecutionSpecError, match="digest verified"):
        R.held_out_authority(
            fold.held_out_subject,
            {"selection_promoted": True, "selection_digest_verified": False},
        )


def test_an_authorized_held_out_authority_sees_exactly_one_subject():
    fold = P.t1_folds()[3]
    authority = R.held_out_authority(
        fold.held_out_subject,
        {
            "selection_promoted": True,
            "selection_digest_verified": True,
            SPEC.T1_HELD_OUT_ACCESS_FLAG: True,
        },
    )
    assert authority.authorized_subjects == (fold.held_out_subject,)
    with pytest.raises(R.T1DevelopmentError):
        authority.require_authorized(fold.fit_subjects[0])


def test_there_is_no_way_to_ask_an_authority_for_every_label():
    """The absence of a global accessor is the firewall."""
    members = dir(R.FoldScopedTargetAuthority)
    for banned in ("all_labels", "labels", "load_all", "everything"):
        assert banned not in members
    assert SPEC.T1_GLOBAL_LABEL_TABLE_PERMITTED is False
    assert SPEC.T1_T2_IDENTITY_NPZ_AS_LABEL_TABLE_PERMITTED is False


def test_only_one_policy_may_run_on_a_held_out_subject():
    assert SPEC.require_single_held_out_policy_run(1) == 1
    for wrong in (0, 2, 12):
        with pytest.raises(SPEC.T1ExecutionSpecError):
            SPEC.require_single_held_out_policy_run(wrong)


# ---------------------------------------------------------------------------
# Stage ordering and retry impossibility
# ---------------------------------------------------------------------------


def test_the_stage_recorder_enforces_the_frozen_order():
    recorder = PERSIST.T1StageRecorder()
    recorder.enter(SPEC.STAGE_START)
    recorder.enter(SPEC.STAGE_CLAIM)
    with pytest.raises(PERSIST.T1PersistenceError, match="not a suggestion"):
        recorder.enter(SPEC.STAGE_VERIFY_GIT)


def test_a_stage_cannot_be_re_entered():
    recorder = PERSIST.T1StageRecorder()
    recorder.enter(SPEC.STAGE_START)
    recorder.enter(SPEC.STAGE_CLAIM)
    with pytest.raises(PERSIST.T1PersistenceError, match="retry under another name"):
        recorder.enter(SPEC.STAGE_CLAIM)


def test_per_row_access_cannot_precede_the_claim():
    recorder = PERSIST.T1StageRecorder()
    recorder.enter(SPEC.STAGE_START)
    with pytest.raises(PERSIST.T1PersistenceError):
        recorder.enter(SPEC.STAGE_ASSEMBLE_LABEL_BLIND)
        recorder.enter(SPEC.STAGE_CLAIM)


def test_promotions_are_impossible_without_a_claim(tmp_path):
    run = R.T1DevelopmentRun(authorized_git_sha="0" * 40, repository_root=tmp_path)
    with pytest.raises(R.T1DevelopmentError, match="claimed"):
        run._require_claimed()


@pytest.mark.parametrize(
    "name",
    [
        "t1-v1-development-retry",
        "t1-v1-development-recovery2",
        "t1-v1-development-v2",
        "t1-v1-development-2026",
    ],
)
def test_no_alternate_or_recovery_attempt_identity_is_claimable(name):
    with pytest.raises(PERSIST.T1PersistenceError, match="exactly one canonical"):
        PERSIST.require_canonical_attempt_id(name)


def test_the_attempt_identity_is_deterministic():
    assert PERSIST.require_canonical_attempt_id(SPEC.T1_DEVELOPMENT_ATTEMPT_ID) == (
        "t1-v1-development"
    )
    # An AST import scan, not a substring scan: the module's prose says
    # "no timestamp, no uuid", and a naive grep would flag its own denylist.
    tree = ast.parse(Path(PERSIST.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for nondeterministic in ("uuid", "random", "secrets", "os"):
        assert nondeterministic not in imported, nondeterministic
    assert PERSIST.canonical_run_directory(REPOSITORY_ROOT).name == (
        SPEC.T1_DEVELOPMENT_ATTEMPT_ID
    )


@requires_frozen_runtime
def test_a_claimed_directory_cannot_be_claimed_twice(tmp_path):
    runtime = _started_runtime()
    stages = PERSIST.T1StageRecorder()
    claimed = PERSIST.claim_canonical_run(
        authorized_git_sha="0" * 40,
        runtime=runtime,
        stages=stages,
        repository_root=tmp_path,
    )
    assert claimed.run_dir.is_dir()
    with pytest.raises(PERSIST.T1PersistenceError, match="already claimed"):
        PERSIST.claim_canonical_run(
            authorized_git_sha="0" * 40,
            runtime=_started_runtime(),
            stages=PERSIST.T1StageRecorder(),
            repository_root=tmp_path,
        )
    with pytest.raises(PERSIST.T1PersistenceError, match="already claimed"):
        PERSIST.require_unclaimed_canonical_attempt(tmp_path)


@requires_frozen_runtime
def test_a_promoted_artifact_is_immutable(tmp_path):
    claimed = PERSIST.claim_canonical_run(
        authorized_git_sha="0" * 40,
        runtime=_started_runtime(),
        stages=PERSIST.T1StageRecorder(),
        repository_root=tmp_path,
    )
    PERSIST.promote(claimed, PERSIST.OOF_RESULT_NAME, {"a": 1})
    with pytest.raises(PERSIST.T1PersistenceError, match="immutable"):
        PERSIST.promote(claimed, PERSIST.OOF_RESULT_NAME, {"a": 2})


@requires_frozen_runtime
def test_an_unplanned_artifact_cannot_be_promoted(tmp_path):
    claimed = PERSIST.claim_canonical_run(
        authorized_git_sha="0" * 40,
        runtime=_started_runtime(),
        stages=PERSIST.T1StageRecorder(),
        repository_root=tmp_path,
    )
    with pytest.raises(PERSIST.T1PersistenceError, match="not a planned"):
        PERSIST.promote(claimed, "T1_SOMETHING_ELSE.json", {})


@requires_frozen_runtime
def test_a_failure_receipt_is_additive_and_admits_the_attempt_is_consumed(tmp_path):
    claimed = PERSIST.claim_canonical_run(
        authorized_git_sha="0" * 40,
        runtime=_started_runtime(),
        stages=PERSIST.T1StageRecorder(),
        repository_root=tmp_path,
    )
    path = PERSIST.write_failure_receipt(
        claimed, RuntimeError("boom"), state={"stage": "claim_run_directory"}
    )
    receipt = json.loads(path.read_text())
    assert receipt["automatic_retry_permitted"] is False
    assert receipt["attempt_consumed"] is True
    assert receipt["exception_message"] == "boom"
    status = PERSIST.read_artifact(claimed.run_dir, PERSIST.RUN_STATUS_NAME)
    assert status["status"] == PERSIST.STATUS_FAILED
    assert claimed.run_dir.is_dir(), "a failed attempt is not deleted"


# ---------------------------------------------------------------------------
# Frozen upstream identity and the evidence layer
# ---------------------------------------------------------------------------


def test_the_upstream_validators_the_harness_calls_are_the_canonical_ones():
    source = Path(R.__file__).read_text()
    for validator in (
        "validate_retained_m2_arm",
        "validate_retained_u1_calibration",
        "validate_retained_t2_arm",
    ):
        assert validator in source, validator
    assert SPEC.T1_PARALLEL_WEAKER_VERIFIER_PERMITTED is False


def test_the_harness_never_calls_a_fitting_routine():
    called = set()
    for module in MODULES:
        tree = ast.parse(Path(module.__file__).read_text())
        called |= {
            getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
    for forbidden in ("fit_calibrator", "select_calibrator_family", "minimize", "fit"):
        assert forbidden not in called, forbidden
    assert "apply_to_scores" in called, "the frozen calibrator must still be applied"


def test_the_binary_decision_of_the_temporal_arm_is_never_read():
    source = Path(STORE.__file__).read_text()
    assert "predicted_positive" in source, "it is named only to explain the refusal"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "predicted_positive":
            pytest.fail("predicted_positive is materialised somewhere")


@pytest.mark.parametrize("member", ["label", "target_family", "primary_mask"])
def test_a_forbidden_identity_member_cannot_be_requested(member, tmp_path):
    archive = tmp_path / "identity.npz"
    np.savez(archive, stable_id=np.asarray(["a"]), **{member: np.asarray([1])})
    with pytest.raises(SPEC.T1ExecutionSpecError):
        STORE.read_t2_identity_members(archive, members=("stable_id", member))


def test_the_m2_gate_outcome_cannot_be_requested(tmp_path):
    archive = tmp_path / "rows.npz"
    np.savez(archive, stable_id=np.asarray(["a"]), update_admitted=np.asarray([True]))
    with pytest.raises(STORE.T1EvidenceStoreError, match="gate outcome"):
        STORE.read_m2g_row_evidence(archive, columns=("stable_id", "update_admitted"))


def test_misaligned_stable_ids_are_a_hard_stop():
    with pytest.raises(STORE.T1EvidenceStoreError, match="diverge"):
        STORE.require_stable_id_alignment(
            np.asarray(["a", "b", "c"]), np.asarray(["a", "x", "c"])
        )
    with pytest.raises(STORE.T1EvidenceStoreError, match="same timeline"):
        STORE.require_stable_id_alignment(np.asarray(["a"]), np.asarray(["a", "b"]))


def test_disagreeing_availability_masks_are_a_hard_stop():
    with pytest.raises(STORE.T1EvidenceStoreError, match="disagree"):
        STORE.require_availability_alignment(
            np.asarray([True, True]), np.asarray([True, False])
        )
    census = STORE.require_availability_alignment(
        np.asarray([True, False]), np.asarray([True, False])
    )
    assert census == {"row_count": 2, "scored": 1, "unavailable": 1}


def test_the_frozen_row_census_must_close_exactly():
    good = {
        "row_count": SPEC.T1_TIMELINE_ROW_COUNT,
        "scored": SPEC.T1_EXPECTED_SCORED_ROWS,
        "unavailable": SPEC.T1_EXPECTED_UNAVAILABLE_ROWS,
    }
    assert STORE.require_expected_census(good) == good
    with pytest.raises(STORE.T1EvidenceStoreError, match="frozen"):
        STORE.require_expected_census({**good, "scored": good["scored"] - 1})


# ---------------------------------------------------------------------------
# Schemas exist, are deterministic, and carry no annotation
# ---------------------------------------------------------------------------


def _synthetic(columns) -> dict[str, np.ndarray]:
    rows = 4
    made: dict[str, np.ndarray] = {}
    for column in columns:
        if column in (
            "stable_id",
            "record_id",
            "subject_id",
            "selected_policy_id",
            "emitted_state",
            "transition_from",
            "transition_to",
        ):
            made[column] = np.asarray([f"{column}{i}" for i in range(rows)])
        elif column in (
            "score_present",
            "detector_decision_d_t",
            "transition_occurred",
        ):
            made[column] = np.asarray([True, False, True, False])
        elif column in ("channel_index", "start_sample", "fold_index"):
            made[column] = np.arange(rows)
        else:
            made[column] = np.linspace(0.0, 1.0, rows)
    return made


def test_every_planned_artifact_name_is_defined():
    assert len(PERSIST.planned_artifacts()) == 13
    for name in (
        "T1_PREFLIGHT.json",
        "T1_INPUT_LINEAGE.json",
        "T1_INPUT_EVIDENCE.json",
        "T1_FOLD_SELECTIONS.json",
        "T1_OOF_STATE_EVIDENCE.json",
        "T1_OOF_RESULT.json",
        "T1_SUBJECT_EVIDENCE.json",
        "T1_BOOTSTRAP.json",
        "T1_CHALLENGE_EVIDENCE.json",
        "T1_FINAL_CONFIGURATION.json",
        "T1_EXPERIMENT_LOCK.json",
    ):
        assert name in PERSIST.planned_artifacts(), name


def test_the_input_evidence_writer_is_deterministic(tmp_path):
    columns = _synthetic(SPEC.T1_INPUT_EVIDENCE_COLUMNS)
    first = STORE.write_input_evidence(tmp_path / "a", columns, lineage={"m2": "x"})
    second = STORE.write_input_evidence(tmp_path / "b", columns, lineage={"m2": "x"})
    assert first["content_sha256"] == second["content_sha256"]
    assert first["array_sha256"] == second["array_sha256"]
    assert first["row_count"] == 4


def test_the_oof_state_writer_is_deterministic(tmp_path):
    columns = _synthetic(SPEC.T1_OOF_STATE_EVIDENCE_COLUMNS)
    first = STORE.write_oof_state_evidence(
        tmp_path / "a", columns, fold_selection_sha256="0" * 64
    )
    second = STORE.write_oof_state_evidence(
        tmp_path / "b", columns, fold_selection_sha256="0" * 64
    )
    assert first["content_sha256"] == second["content_sha256"]
    assert first["is_unseen_generalization"] is False


def test_a_promoted_store_is_never_overwritten(tmp_path):
    columns = _synthetic(SPEC.T1_INPUT_EVIDENCE_COLUMNS)
    STORE.write_input_evidence(tmp_path, columns, lineage={})
    with pytest.raises(STORE.T1EvidenceStoreError, match="immutable"):
        STORE.write_input_evidence(tmp_path, columns, lineage={})


@pytest.mark.parametrize("contaminant", ["label", "target_family", "primary_mask"])
def test_an_evidence_store_refuses_evaluation_annotation(tmp_path, contaminant):
    columns = _synthetic(SPEC.T1_INPUT_EVIDENCE_COLUMNS)
    columns[contaminant] = np.zeros(4)
    with pytest.raises(STORE.T1EvidenceStoreError, match="unexpected columns"):
        STORE.write_input_evidence(tmp_path, columns, lineage={})


def test_a_store_reads_back_only_the_columns_asked_for(tmp_path):
    columns = _synthetic(SPEC.T1_INPUT_EVIDENCE_COLUMNS)
    STORE.write_input_evidence(tmp_path, columns, lineage={})
    partial = STORE.read_store(
        tmp_path, STORE.INPUT_EVIDENCE_MANIFEST_NAME, columns=("stable_id",)
    )
    assert set(partial) == {"stable_id"}


def test_a_tampered_store_is_refused(tmp_path):
    columns = _synthetic(SPEC.T1_INPUT_EVIDENCE_COLUMNS)
    STORE.write_input_evidence(tmp_path, columns, lineage={})
    (tmp_path / STORE.INPUT_EVIDENCE_ARRAY_NAME).write_bytes(b"tampered")
    with pytest.raises(STORE.T1EvidenceStoreError, match="changed on disk"):
        STORE.read_store(tmp_path, STORE.INPUT_EVIDENCE_MANIFEST_NAME)


# ---------------------------------------------------------------------------
# Metric semantics
# ---------------------------------------------------------------------------


def test_episode_f1_is_count_algebra_and_may_be_undefined():
    assert R.episode_f1(matched=2, predicted=2, reference=2) == 1.0
    assert R.episode_f1(matched=1, predicted=2, reference=2) == pytest.approx(0.5)
    assert R.episode_f1(matched=0, predicted=0, reference=0) is None


def test_an_undefined_metric_stops_selection_rather_than_becoming_zero():
    with pytest.raises(SPEC.T1ExecutionSpecError, match="human review"):
        SPEC.require_defined_metric("episode_f1", None)


def test_window_mcc_is_undefined_on_an_empty_margin():
    assert R.window_mcc(np.asarray([True, True]), np.asarray([True, True])) is None
    assert R.window_mcc(
        np.asarray([True, False]), np.asarray([True, False])
    ) == pytest.approx(1.0)


def test_physical_exposure_counts_every_position_including_unavailable_ones():
    assert R.physical_exposure_hours(720) == pytest.approx(1.0)
    assert R.false_event_onsets_per_hour(3, 720) == pytest.approx(3.0)
    assert SPEC.T1_EXPOSURE_INCLUDES_UNAVAILABLE_POSITIONS is True


def test_contiguous_runs_are_maximal():
    assert R.contiguous_runs([False, True, True, False, True]) == ((1, 3), (4, 5))
    assert R.contiguous_runs([]) == ()


def test_thresholds_come_from_the_frozen_quantiles_only():
    policy = P.candidate_policies()[0]
    values = [float(i) / 100 for i in range(100)]
    ids = [f"w{i}" for i in range(100)]
    thresholds = R.generate_thresholds(
        policy, background_p=values, background_s=values, stable_ids=ids
    )
    assert thresholds.p_event >= thresholds.p_watch
    bad = P.T1CandidatePolicy(q_watch=0.5, q_event=0.99, profile=policy.profile)
    with pytest.raises(R.T1DevelopmentError, match="outside the frozen"):
        R.generate_thresholds(
            bad, background_p=values, background_s=values, stable_ids=ids
        )


def test_the_bootstrap_is_deterministic_and_resamples_subjects():
    first = R.subject_bootstrap_indices(12)
    second = R.subject_bootstrap_indices(12)
    assert np.array_equal(first, second), "seed 2026 must give one answer"
    assert first.shape == (SPEC.T1_BOOTSTRAP_REPLICATES, 12)
    assert first.max() < 12 and first.min() >= 0


def test_row_quantities_use_the_frozen_detector_threshold():
    below = R.derive_row_quantities(0.70, 0.30)
    above = R.derive_row_quantities(0.80, 0.90)
    assert below["detector_decision_d_t"] is False
    assert above["detector_decision_d_t"] is True
    assert below["decision_error_uncertainty_u_t"] == pytest.approx(0.30)
    assert above["decision_error_uncertainty_u_t"] == pytest.approx(0.10)
    assert SPEC.T1_DETECTOR_THRESHOLD == P.T1_DETECTOR_THRESHOLD


def test_elapsed_time_is_physical_not_ordinal():
    assert R.elapsed_stream_seconds(1250, 0) == pytest.approx(5.0)
    assert R.elapsed_stream_seconds(2_500_000, 0) == pytest.approx(10000.0)
    assert SPEC.T1_ELAPSED_FROM_ROW_ORDINAL_PERMITTED is False


def test_unavailable_rows_carry_nothing(tmp_path):
    columns = {
        "stable_id": np.asarray(["a", "b"]),
        "score_present": np.asarray([True, False]),
        "detector_decision_d_t": np.asarray([True, False]),
        "oof_calibrated_probability_p_t": np.asarray([0.9, STORE.ABSENT]),
        "decision_error_uncertainty_u_t": np.asarray([0.1, STORE.ABSENT]),
        "s4d_temporal_evidence_s_t": np.asarray([0.8, STORE.ABSENT]),
        "elapsed_stream_seconds": np.asarray([0.0, 5.0]),
    }
    rows = R.build_rows(columns)
    assert rows[0].score_present and rows[0].calibrated_probability == pytest.approx(
        0.9
    )
    assert rows[1].score_present is False
    assert rows[1].calibrated_probability is None
    assert rows[1].decision_error_uncertainty is None
    assert rows[1].temporal_evidence is None


def test_a_claim_is_refused_outside_the_frozen_interpreter(tmp_path):
    """The guard that makes the four tests above skip in CI, tested directly.

    A canonical claim may only rest on the frozen scientific identity. A record
    that expects anything else is refused before a directory is created, so the
    refusal cannot leave a half-claimed attempt behind.
    """
    from cardiosentinel.neural.m2_persistence import M2PersistenceError

    impostor = RuntimeIntegrityRecord(expected_digest="0" * 64)
    with pytest.raises((M2PersistenceError, Exception)) as caught:
        PERSIST.claim_canonical_run(
            authorized_git_sha="0" * 40,
            runtime=impostor,
            stages=PERSIST.T1StageRecorder(),
            repository_root=tmp_path,
        )
    assert "frozen" in str(caught.value).lower()
    assert not (tmp_path / "cardiosentinel-runs").exists(), (
        "the refusal created a directory before refusing"
    )


def test_the_frozen_dependency_digest_is_the_one_the_program_records():
    """Bound so a silently changed frozen identity is visible immediately."""
    assert _frozen_dependency_digest() == (
        "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
    )
