"""The M2 retention binding records a decision and cannot mutate anything."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import m2_selection as S
from cardiosentinel.neural.m2_selection import (
    M2SelectionError,
    validate_m2_retention_decision,
    validate_retained_m2_arm,
)

CANONICAL_RUN_ROOT = Path("cardiosentinel-runs/phase6-m2-development-v1")


def _suite_payload(**overrides):
    payload = {
        "m2_suite_sha256": S.M2_SUITE_SHA256,
        "suite_id": S.M2_RETAINED_SUITE_ID,
        "automatic_arm_preference_applied": False,
        "memory_selection_performed": False,
        "memory_selected": None,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    payload.update(overrides)
    return payload


def _write_suite(tmp_path, **overrides):
    suite_dir = tmp_path / S.M2_RETAINED_SUITE_ID
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "M2_SUITE_RESULT.json").write_text(
        json.dumps(_suite_payload(**overrides))
    )
    return tmp_path


def test_decision_document_is_frozen():
    assert validate_m2_retention_decision() == S.M2_RETENTION_DECISION_SHA256


def test_retained_arm_is_m2g_and_the_control_is_m2_0():
    assert S.M2_RETAINED_ARM == "M2-G"
    assert S.M2_CONTROL_ARM == "M2-0"
    assert S.M2_RETAINED_ARM != S.M2_CONTROL_ARM
    assert S.M2_RETAINED_ARM_RESULT_SHA256 != S.M2_CONTROL_ARM_RESULT_SHA256
    assert S.M2_RETAINED_LOCK_SHA256 != S.M2_CONTROL_LOCK_SHA256


def test_retained_suite_identity_is_recovery2():
    assert S.M2_RETAINED_SUITE_ID == "m2-v1-development-two-arm-recovery2"
    assert S.M2_SUITE_SHA256 == (
        "8a6b0a1c64da72fc0f4573c742bef491b01b4eb8179f0759da3c537a01939a02"
    )


def test_decision_records_its_own_limits():
    assert S.M2_SELECTION_BASIS == "development_evidence_only"
    assert S.M2_SELECTION_TEST_ACCESSED is False
    assert S.M2_SELECTION_WEIGHTED_SCORE_USED is False
    assert S.M2_SELECTION_STATISTICAL_SIGNIFICANCE_CLAIM is False
    assert S.M2_RERUN_PERMITTED is False


def test_binding_module_cannot_mutate_any_artifact():
    """A decision record must be incapable of touching scientific state."""
    tree = ast.parse(Path(S.__file__).read_text())
    forbidden_calls = {"write_json_atomic", "save", "unlink", "rmtree", "rename"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden_calls, name
            if name == "open":
                pytest.fail("the binding module must not open files for writing")


def test_binding_module_performs_no_scoring_or_training():
    tree = ast.parse(Path(S.__file__).read_text())
    imported = {
        alias.name
        for imp in ast.walk(tree)
        if isinstance(imp, ast.ImportFrom)
        for alias in imp.names
    }
    forbidden = {
        "execute_canonical_development",
        "evaluate_gate",
        "iter_timeline_streams",
        "M2Scorer",
        "build_m2_scorer",
        "replay_stream",
        "train_m1_arm",
    }
    assert not (imported & forbidden), imported & forbidden


def test_canonical_artifacts_prove_the_retained_policy():
    if not (CANONICAL_RUN_ROOT / S.M2_RETAINED_SUITE_ID).is_dir():
        pytest.skip("canonical recovery2 run directory is not on this filesystem")
    proof = validate_retained_m2_arm(CANONICAL_RUN_ROOT)
    assert proof["retained"] == {"M2-0": False, "M2-G": True}
    assert proof["retained_arm"] == "M2-G"
    assert proof["control_arm"] == "M2-0"
    assert proof["test_accessed"] is False
    assert proof["m2_rerun_permitted"] is False
    assert proof["weighted_score_used"] is False
    assert proof["statistical_significance_claim"] is False


def test_validator_refuses_a_wrong_suite(tmp_path):
    _write_suite(tmp_path, m2_suite_sha256="0" * 64)
    with pytest.raises(M2SelectionError, match="not the one the retention"):
        validate_retained_m2_arm(tmp_path)


def test_validator_refuses_a_mutated_suite_id(tmp_path):
    suite_dir = tmp_path / S.M2_RETAINED_SUITE_ID
    suite_dir.mkdir(parents=True)
    (suite_dir / "M2_SUITE_RESULT.json").write_text(
        json.dumps(_suite_payload(suite_id="m2-v1-development-two-arm-recovery3"))
    )
    with pytest.raises(M2SelectionError, match="not the retained"):
        validate_retained_m2_arm(tmp_path)


def test_validator_refuses_a_suite_that_claims_it_preferred_an_arm(tmp_path):
    _write_suite(tmp_path, automatic_arm_preference_applied=True)
    with pytest.raises(M2SelectionError, match="no\n?\\s*automatic arm preference"):
        validate_retained_m2_arm(tmp_path)


def test_validator_refuses_a_suite_that_claims_it_selected(tmp_path):
    _write_suite(tmp_path, memory_selection_performed=True)
    with pytest.raises(M2SelectionError, match="performed no"):
        validate_retained_m2_arm(tmp_path)


def test_validator_refuses_a_suite_recording_a_selected_arm(tmp_path):
    _write_suite(tmp_path, memory_selected="M2-G")
    with pytest.raises(M2SelectionError, match="records a selected arm"):
        validate_retained_m2_arm(tmp_path)


def test_validator_refuses_a_suite_recording_test_access(tmp_path):
    _write_suite(tmp_path, test_accessed=True)
    with pytest.raises(M2SelectionError, match="records test access"):
        validate_retained_m2_arm(tmp_path)


def test_validator_refuses_a_suite_whose_test_is_not_sealed(tmp_path):
    _write_suite(tmp_path, sealed_test_state="opened")
    with pytest.raises(M2SelectionError, match="TEST as unopened"):
        validate_retained_m2_arm(tmp_path)


def test_validator_refuses_a_missing_suite(tmp_path):
    with pytest.raises(M2SelectionError, match="No M2 suite result"):
        validate_retained_m2_arm(tmp_path)


def test_validator_refuses_a_mutated_retained_result(tmp_path):
    _write_suite(tmp_path)
    run_dir = tmp_path / f"{S.M2_RETAINED_SUITE_ID}__{S.M2_RETAINED_ARM}"
    run_dir.mkdir(parents=True)
    (run_dir / "M2_ARM_RESULT.json").write_text(json.dumps({"arm": "M2-G"}))
    (run_dir / "M2_EXPERIMENT_LOCK.json").write_text(
        json.dumps(
            {
                "arm": "M2-G",
                "experiment_lock_sha256": S.M2_RETAINED_LOCK_SHA256,
                "test_accessed": False,
            }
        )
    )
    with pytest.raises(M2SelectionError, match="result digests to"):
        validate_retained_m2_arm(tmp_path)


def _mirror_canonical_arm(tmp_path, arm):
    """Copy an arm's real promoted artifacts so a single field can be mutated."""
    source = CANONICAL_RUN_ROOT / f"{S.M2_RETAINED_SUITE_ID}__{arm}"
    if not source.is_dir():
        pytest.skip("canonical recovery2 run directory is not on this filesystem")
    target = tmp_path / f"{S.M2_RETAINED_SUITE_ID}__{arm}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("M2_ARM_RESULT.json", "M2_EXPERIMENT_LOCK.json"):
        (target / name).write_bytes((source / name).read_bytes())
    return target


def test_validator_refuses_a_mutated_control_lock(tmp_path):
    """The control arm must stay exactly as frozen, not merely be present."""
    _write_suite(tmp_path)
    _mirror_canonical_arm(tmp_path, S.M2_RETAINED_ARM)
    control_dir = _mirror_canonical_arm(tmp_path, S.M2_CONTROL_ARM)

    lock_path = control_dir / "M2_EXPERIMENT_LOCK.json"
    lock = json.loads(lock_path.read_text())
    lock["experiment_lock_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock))

    with pytest.raises(M2SelectionError, match="lock differs from the frozen"):
        validate_retained_m2_arm(tmp_path)


def test_validator_accepts_the_mirrored_canonical_arms(tmp_path):
    """The refusal tests are meaningful only if the unmutated mirror passes."""
    _write_suite(tmp_path)
    _mirror_canonical_arm(tmp_path, S.M2_RETAINED_ARM)
    _mirror_canonical_arm(tmp_path, S.M2_CONTROL_ARM)
    proof = validate_retained_m2_arm(tmp_path)
    assert proof["retained"] == {"M2-0": False, "M2-G": True}


def test_validator_refuses_a_missing_arm_artifact(tmp_path):
    _write_suite(tmp_path)
    with pytest.raises(M2SelectionError, match="is missing"):
        validate_retained_m2_arm(tmp_path)


def test_m2_science_is_unchanged_by_this_decision():
    """The retention decision binds evidence; it never reopens the science."""
    from cardiosentinel.neural.m2_gate import (
        M1L_CLASSIFICATION_THRESHOLD,
        M2_GATE_RECEIPT_SHA256,
        M2_PROTOCOL_SHA256,
        M2_RETAINED_EXPERIMENT_ID,
        NORMAL_EVIDENCE_THRESHOLD,
    )

    assert M2_PROTOCOL_SHA256 == (
        "a8ba6fad038ed0ec01156b6959239f489426d55db8ad73a0c704fd527e7db91c"
    )
    assert M2_GATE_RECEIPT_SHA256 == (
        "5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24"
    )
    assert M2_RETAINED_EXPERIMENT_ID == "M1L_long_memory_v2"
    assert M1L_CLASSIFICATION_THRESHOLD == 0.7554003000259399
    assert NORMAL_EVIDENCE_THRESHOLD == 0.0002997174742631614


def test_decision_document_records_its_governance_nature_and_non_claims():
    text = S.M2_RETENTION_DECISION_PATH.read_text()
    assert "THIS IS A HUMAN GOVERNANCE DECISION" in text
    assert "NOT A NEW SCIENTIFIC EXPERIMENT" in text
    # the false-alarm trade-off is recorded, never reversed into a benefit
    assert "M2-G DID NOT improve false-alarm behaviour." in text
    assert "No strict Pareto dominance is claimed." in text
    assert "No statistical significance is claimed." in text
    # gating that never updates would be a trivial pass
    assert "is NOT a trivial never-update policy" in text
    # cold start is carried forward, not explained away
    assert "M2 does not solve M1 cold start." in text
    # non-estimable families must not be read as zero drift
    assert "does NOT mean zero drift" in text
    assert "No M2 rerun is permitted." in text
