"""The U1 protocol binding is structural: synthetic data only, no execution.

Nothing here fits a calibrator, scores a window, or opens a run artifact. The
tests prove the frozen design refuses what it must refuse.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import u1_protocol as U
from cardiosentinel.neural.u1_protocol import (
    U1ProtocolError,
    assign_calibration_folds,
    fold_assignment_digest,
    require_calibration_subjects,
    u1_protocol_identity,
    validate_against_frozen_split,
    validate_fold_assignment,
    validate_u1_protocol_document,
)

FROZEN_SPLIT = Path("protocols/splits/ltstdb_v1.json")


def _frozen_test_subjects():
    if not FROZEN_SPLIT.is_file():
        pytest.skip("frozen split manifest is not on this filesystem")
    manifest = json.loads(FROZEN_SPLIT.read_text())
    return sorted(manifest["partitions"]["test"]["subjects"])


# ---------------------------------------------------------------------------
# Protocol document
# ---------------------------------------------------------------------------


def test_protocol_document_is_frozen():
    assert validate_u1_protocol_document() == U.U1_PROTOCOL_SHA256


def test_protocol_document_states_its_hard_commitments():
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "FROZEN PROSPECTIVE SCIENTIFIC PROTOCOL" in text
    # the G4 admission score must never become "confidence"
    assert "must not calibrate, reinterpret, rescale or route on the G4" in text
    # true logits are genuinely unavailable; temperature scaling is not proper
    assert "True logits are not persisted" in text
    # routing threshold is not chosen in this task
    assert "No routing threshold is chosen in this task" in text
    # no automatic downstream transition
    assert "does **not** begin U2 conformal prediction" in text
    assert "Completion of U1 does not authorise TEST." in text


def test_protocol_refuses_a_mutated_document(tmp_path):
    forged = tmp_path / "forged.md"
    forged.write_text("not the protocol")
    with pytest.raises(U1ProtocolError, match="differs from the frozen"):
        validate_u1_protocol_document(forged)


def test_protocol_refuses_a_missing_document(tmp_path):
    with pytest.raises(U1ProtocolError, match="missing"):
        validate_u1_protocol_document(tmp_path / "absent.md")


# ---------------------------------------------------------------------------
# TEST firewall
# ---------------------------------------------------------------------------


def test_real_test_subjects_can_never_enter_calibration():
    """The decisive firewall test, using the real frozen TEST identities."""
    for subject in _frozen_test_subjects():
        with pytest.raises(U1ProtocolError, match="not in the frozen U1"):
            require_calibration_subjects([subject])


def test_a_single_test_subject_poisons_an_otherwise_valid_set():
    test_subject = _frozen_test_subjects()[0]
    contaminated = list(U.U1_CALIBRATION_SUBJECTS) + [test_subject]
    with pytest.raises(U1ProtocolError, match="not in the frozen U1"):
        assign_calibration_folds(contaminated)


def test_test_subjects_are_refused_not_silently_dropped():
    """A filter would return 12 folds; the protocol must raise instead."""
    test_subject = _frozen_test_subjects()[0]
    with pytest.raises(U1ProtocolError):
        assign_calibration_folds(list(U.U1_CALIBRATION_SUBJECTS) + [test_subject])


def test_frozen_split_proves_validation_and_test_are_disjoint():
    proof = validate_against_frozen_split(FROZEN_SPLIT)
    assert proof["calibration_test_intersection"] == []
    assert proof["sealed_test_partition"] is True
    assert proof["calibration_subject_count"] == 12
    assert proof["test_accessed"] is False
    assert proof["sealed_test_state"] == "unopened"


def test_split_validator_refuses_a_wrong_split(tmp_path):
    forged = tmp_path / "split.json"
    forged.write_text(
        json.dumps(
            {
                "split_sha256": "0" * 64,
                "sealed_test_partition": True,
                "partitions": {"validation": {"subjects": []}, "test": {}},
            }
        )
    )
    with pytest.raises(U1ProtocolError, match="not the frozen benchmark split"):
        validate_against_frozen_split(forged)


def test_split_validator_refuses_an_unsealed_test_partition(tmp_path):
    forged = tmp_path / "split.json"
    forged.write_text(
        json.dumps(
            {
                "split_sha256": U.U1_SPLIT_SHA256,
                "sealed_test_partition": False,
                "partitions": {
                    "validation": {"subjects": list(U.U1_CALIBRATION_SUBJECTS)},
                    "test": {"subjects": []},
                },
            }
        )
    )
    with pytest.raises(U1ProtocolError, match="does not seal the TEST"):
        validate_against_frozen_split(forged)


def test_split_validator_catches_a_leaked_test_subject(tmp_path):
    """If VALIDATION and TEST ever overlapped, the validator must refuse."""
    forged = tmp_path / "split.json"
    forged.write_text(
        json.dumps(
            {
                "split_sha256": U.U1_SPLIT_SHA256,
                "sealed_test_partition": True,
                "partitions": {
                    "validation": {"subjects": list(U.U1_CALIBRATION_SUBJECTS)},
                    "test": {"subjects": [U.U1_CALIBRATION_SUBJECTS[0]]},
                },
            }
        )
    )
    with pytest.raises(U1ProtocolError, match="appear in calibration"):
        validate_against_frozen_split(forged)


# ---------------------------------------------------------------------------
# Fold design
# ---------------------------------------------------------------------------


def test_folds_are_subject_disjoint_and_complete():
    folds = assign_calibration_folds()
    assert len(folds) == 12
    for fold in folds:
        assert fold.held_out_subject not in fold.fit_subjects
        assert len(fold.fit_subjects) == 11
        assert set(fold.fit_subjects) | {fold.held_out_subject} == set(
            U.U1_CALIBRATION_SUBJECTS
        )


def test_every_permitted_subject_is_held_out_exactly_once():
    folds = assign_calibration_folds()
    held_out = [fold.held_out_subject for fold in folds]
    assert sorted(held_out) == sorted(U.U1_CALIBRATION_SUBJECTS)
    assert len(set(held_out)) == 12


def test_fold_identity_is_deterministic_and_order_independent():
    a = assign_calibration_folds()
    b = assign_calibration_folds(list(reversed(U.U1_CALIBRATION_SUBJECTS)))
    assert fold_assignment_digest(a) == fold_assignment_digest(b)
    assert a == b


def test_fold_assignment_validator_accepts_the_frozen_design():
    proof = validate_fold_assignment(assign_calibration_folds())
    assert proof["fold_design"] == "leave_one_subject_out"
    assert proof["fold_count"] == 12
    assert proof["no_fold_fits_on_its_held_out_subject"] is True
    assert proof["fold_assignment_basis"] == ("frozen_subject_identity_ascending_only")


def test_fold_validator_refuses_a_fold_fitted_on_its_own_subject():
    """The core cross-fitting guarantee must be enforced, not assumed."""
    folds = list(assign_calibration_folds())
    first = folds[0]
    folds[0] = first._replace(
        fit_subjects=(first.held_out_subject,) + first.fit_subjects[1:]
    )
    with pytest.raises(U1ProtocolError, match="fits on its own held-out"):
        validate_fold_assignment(tuple(folds))


def test_fold_validator_refuses_an_incomplete_assignment():
    folds = assign_calibration_folds()[:-1]
    with pytest.raises(U1ProtocolError, match="Expected 12 folds"):
        validate_fold_assignment(folds)


def test_incomplete_subject_set_is_refused():
    with pytest.raises(U1ProtocolError, match="are absent"):
        assign_calibration_folds(U.U1_CALIBRATION_SUBJECTS[:6])


def test_duplicate_subjects_are_refused():
    doubled = list(U.U1_CALIBRATION_SUBJECTS) + [U.U1_CALIBRATION_SUBJECTS[0]]
    with pytest.raises(U1ProtocolError, match="Duplicate"):
        require_calibration_subjects(doubled)


def test_fold_assignment_cannot_depend_on_performance():
    """No score-derived quantity may be referenced as code by the assignment.

    Checked over AST identifiers rather than raw text, so prose that *forbids*
    such an input does not count as using one.
    """
    tree = ast.parse(Path(U.__file__).read_text())
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    referenced |= {
        k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords
    } - {None}
    forbidden = {
        "auprc",
        "auroc",
        "false_positive_rate",
        "prediction",
        "predictions",
        "error_rate",
        "calibration_error",
        "prevalence",
        "sensitivity",
        "specificity",
    }
    assert not (referenced & forbidden), referenced & forbidden


# ---------------------------------------------------------------------------
# Frozen science must not drift
# ---------------------------------------------------------------------------


def test_no_m2_scientific_constant_changes():
    from cardiosentinel.neural.m2_gate import (
        M1L_CLASSIFICATION_THRESHOLD,
        M2_GATE_RECEIPT_SHA256,
        M2_PROTOCOL_SHA256,
        M2_RETAINED_EXPERIMENT_ID,
        NORMAL_EVIDENCE_THRESHOLD,
    )

    assert U.U1_CLASSIFICATION_THRESHOLD == M1L_CLASSIFICATION_THRESHOLD
    assert U.U1_NORMAL_EVIDENCE_THRESHOLD == NORMAL_EVIDENCE_THRESHOLD
    assert M2_RETAINED_EXPERIMENT_ID == "M1L_long_memory_v2"
    assert M2_PROTOCOL_SHA256 == (
        "a8ba6fad038ed0ec01156b6959239f489426d55db8ad73a0c704fd527e7db91c"
    )
    assert M2_GATE_RECEIPT_SHA256 == (
        "5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24"
    )


def test_u1_binds_the_retained_m2g_arm_not_the_control():
    from cardiosentinel.neural import m2_selection as S

    assert U.U1_M2G_ARM_RESULT_SHA256 == S.M2_RETAINED_ARM_RESULT_SHA256
    assert U.U1_M2G_LOCK_SHA256 == S.M2_RETAINED_LOCK_SHA256
    assert U.U1_M2_SUITE_SHA256 == S.M2_SUITE_SHA256
    assert U.U1_M2_RETENTION_DECISION_SHA256 == S.M2_RETENTION_DECISION_SHA256
    # the naive control arm is never a U1 calibration input
    assert U.U1_M2G_ARM_RESULT_SHA256 != S.M2_CONTROL_ARM_RESULT_SHA256


def test_classification_threshold_is_not_changeable_by_u1():
    assert U.U1_MAY_CHANGE_CLASSIFICATION_THRESHOLD is False
    assert U.U1_CLASSIFICATION_THRESHOLD == 0.7554003000259399


def test_g4_admission_score_is_never_the_calibration_input():
    assert U.U1_CALIBRATION_INPUT_FIELD == "score"
    assert "normal_evidence" in U.U1_FORBIDDEN_CALIBRATION_INPUTS
    assert "g4_normal_evidence" in U.U1_FORBIDDEN_CALIBRATION_INPUTS
    # the two thresholds are distinct quantities and must stay distinct
    assert U.U1_CLASSIFICATION_THRESHOLD != U.U1_NORMAL_EVIDENCE_THRESHOLD


def test_full_replay_is_never_a_metric_denominator():
    assert U.U1_FULL_REPLAY_IS_METRIC_DENOMINATOR is False
    assert U.U1_CALIBRATION_POPULATION == "primary_metric"
    assert U.U1_PRIMARY_ROW_COUNT == 473_897
    assert U.U1_FULL_REPLAY_ROW_COUNT == 492_904


# ---------------------------------------------------------------------------
# The module cannot execute science
# ---------------------------------------------------------------------------


def test_protocol_module_imports_only_the_standard_library():
    """No scorer, replay, torch or run-artifact reader is reachable from here."""
    tree = ast.parse(Path(U.__file__).read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= {"hashlib", "json", "pathlib", "typing", "__future__"}, modules
    assert "torch" not in modules
    assert "cardiosentinel" not in modules


def test_protocol_module_cannot_fit_score_or_mutate():
    tree = ast.parse(Path(U.__file__).read_text())
    forbidden = {
        "fit",
        "train",
        "backward",
        "step",
        "score_batch",
        "write_text",
        "write_bytes",
        "unlink",
        "rmtree",
        "rename",
        "save",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden, name


def test_protocol_chooses_no_routing_threshold_and_no_retention():
    assert U.U1_ROUTING_THRESHOLD_CHOSEN_HERE is False
    assert U.U1_AUTOMATIC_RETENTION is False
    assert U.U1_AUTOMATIC_U2_TRANSITION is False
    assert U.U1_M2_RERUN_PERMITTED is False
    assert U.U1_EPISODE_PERSISTENCE_IMPLEMENTED is False
    assert U.U1_NEURAL_UNCERTAINTY_MODEL is False


def test_identity_record_is_complete_and_self_consistent():
    identity = u1_protocol_identity()
    assert identity["protocol_sha256"] == U.U1_PROTOCOL_SHA256
    assert identity["test_accessed"] is False
    assert identity["sealed_test_state"] == "unopened"
    assert identity["true_logits_persisted"] is False
    assert identity["primary_method"] == "platt_logistic_on_recovered_logit"
    assert identity["comparator_method"] == ("temperature_only_on_recovered_logit")
    assert identity["tie_break"] == "simpler_nested_model"
    assert identity["retained_coverage"] == 0.90
    assert identity["fold_assignment"]["fold_count"] == 12
    assert identity["uncertainty_definition"] == (
        "calibrated_probability_of_frozen_decision_error"
    )


def test_coverage_grid_is_frozen_sorted_and_includes_the_no_routing_reference():
    grid = U.U1_COVERAGE_GRID
    assert list(grid) == sorted(grid)
    assert grid[-1] == 1.00
    assert U.U1_RETAINED_COVERAGE in grid
    assert len(set(grid)) == len(grid)
