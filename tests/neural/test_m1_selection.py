"""The M1 retention binding records a decision and cannot mutate anything."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import m1_selection as S
from cardiosentinel.neural.m1_selection import (
    M1SelectionError,
    validate_m1_retention_decision,
    validate_retained_m1_arm,
)


def test_decision_document_is_frozen():
    assert validate_m1_retention_decision() == S.M1_RETENTION_DECISION_SHA256


def test_retained_arm_is_m1l_and_the_others_are_ablations():
    assert S.M1_RETAINED_EXPERIMENT_ID == "M1L_long_memory_v2"
    assert set(S.M1_ABLATION_LOCK_SHA256) == {
        "M1S_short_memory_v2",
        "M1D_dual_memory_v2",
    }
    assert S.M1_RETAINED_LOCK_SHA256 not in S.M1_ABLATION_LOCK_SHA256.values()


def test_decision_records_its_own_limits():
    assert S.M1_SELECTION_BASIS == "development_evidence_only"
    assert S.M1_SELECTION_TEST_ACCESSED is False
    assert S.M1_SELECTION_WEIGHTED_SCORE_USED is False
    assert S.M1_SELECTION_STATISTICAL_SIGNIFICANCE_CLAIM is False
    assert S.M1_RERUN_PERMITTED is False


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
    imported = {
        alias.name
        for imp in ast.walk(tree)
        if isinstance(imp, ast.ImportFrom)
        for alias in imp.names
    }
    assert "train_m1_arm" not in imported
    assert "execute_m1_stage1" not in imported


def test_validator_refuses_a_wrong_suite(tmp_path):
    (tmp_path / "M1_STAGE1_RESULTS.json").write_text(
        json.dumps(
            {
                "m1_stage1_suite_sha256": "0" * 64,
                "memory_selection_performed": False,
                "test_accessed": False,
            }
        )
    )
    with pytest.raises(M1SelectionError, match="not the one the retention"):
        validate_retained_m1_arm(tmp_path)


def test_validator_refuses_a_suite_that_claims_it_selected(tmp_path):
    (tmp_path / "M1_STAGE1_RESULTS.json").write_text(
        json.dumps(
            {
                "m1_stage1_suite_sha256": S.M1_STAGE1_SUITE_SHA256,
                "memory_selection_performed": True,
                "test_accessed": False,
            }
        )
    )
    with pytest.raises(M1SelectionError, match="performed no"):
        validate_retained_m1_arm(tmp_path)


def test_validator_refuses_a_missing_result(tmp_path):
    with pytest.raises(M1SelectionError, match="No M1 Stage-1 result"):
        validate_retained_m1_arm(tmp_path)


def test_m2_protocol_is_marked_proposed_and_defers_absent_capabilities():
    text = Path("docs/M2_CONTAMINATION_SAFE_MEMORY_PROTOCOL_V1.md").read_text()
    assert "PROPOSED — HUMAN REVIEW REQUIRED" in text
    # capabilities that provably do not exist must be deferred, not assumed
    assert "DEFERRED → U1/U2" in text
    assert "DEFERRED → T1" in text
    assert "must not use the phrase" in text
    # a challenge annotation is never a deployment warning
    assert "not a deployment-observable warning" in text
    # the classification threshold is not the memory-admission threshold
    assert "must NOT be reused as the" in text
