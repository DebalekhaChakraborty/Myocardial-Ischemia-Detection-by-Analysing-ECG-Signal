"""The T2 retention binding records a decision and cannot mutate anything.

Two kinds of test live here, and the split is deliberate.

* Constant and structural tests run everywhere. They prove what the decision
  says, and prove the binding module is *incapable* of computing: no model, no
  inference, no replay, no write.
* Tests that need the real one-shot outer evidence skip when the canonical run
  directory is not on this filesystem, exactly as the M2 and U1 retention
  bindings already do. `cardiosentinel-runs/` is gitignored, so CI has no
  artifacts to prove against; a developer machine that ran the canonical
  attempt does.

Nothing here executes science. No arm is scored, no metric is recomputed, no
threshold is refitted, no partition is opened and TEST is never touched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import t2_persistence as PS
from cardiosentinel.neural import t2_selection as S
from cardiosentinel.neural.t2_selection import (
    T2SelectionError,
    validate_retained_t2_arm,
    validate_t2_retention_decision,
)
from tests.neural.test_t2_canonical_training_route import attempt_content_snapshot

CANONICAL_RUN_ROOT = Path("cardiosentinel-runs/phase8-t2-development-v1")
_ABSENT = "canonical T2 outer-validation run directory is not on this filesystem"


def _require_canonical() -> Path:
    if not (CANONICAL_RUN_ROOT / S.T2_OUTER_VALIDATION_ATTEMPT_ID).is_dir():
        pytest.skip(_ABSENT)
    return CANONICAL_RUN_ROOT


# --- 1-3. the decision itself ----------------------------------------------


def test_the_retention_document_digest_is_exact():
    assert validate_t2_retention_decision() == S.T2_RETENTION_DECISION_SHA256
    assert S.T2_RETENTION_DECISION_PATH.is_file()


def test_a_mutated_retention_document_is_refused(tmp_path):
    forged = tmp_path / "forged.md"
    forged.write_text(S.T2_RETENTION_DECISION_PATH.read_text() + "\nappended\n")
    with pytest.raises(T2SelectionError, match="immutable"):
        validate_t2_retention_decision(forged)


def test_the_retained_arm_is_s4d_and_the_comparator_is_gru():
    assert S.T2_RETAINED_ARM == "causal_s4d_longitudinal_v1"
    assert S.T2_COMPARATOR_ARM == "causal_gru_longitudinal_v1"
    assert S.T2_RETAINED_ARM != S.T2_COMPARATOR_ARM
    assert S.T2_RETAINED_CHECKPOINT_SHA256 != S.T2_COMPARATOR_CHECKPOINT_SHA256
    assert (
        S.T2_RETAINED_CHECKPOINT_LOCK_SHA256 != S.T2_COMPARATOR_CHECKPOINT_LOCK_SHA256
    )
    assert S.T2_RETAINED_INTERNAL_DEV_THRESHOLD != (
        S.T2_COMPARATOR_INTERNAL_DEV_THRESHOLD
    )


# --- 4-10. the canonical evidence the decision rests on ---------------------


def test_canonical_artifacts_prove_the_retained_arm():
    proof = validate_retained_t2_arm(_require_canonical())
    assert proof["retained_arm"] == "causal_s4d_longitudinal_v1"
    assert proof["comparator_arm"] == "causal_gru_longitudinal_v1"
    assert proof["retained"] == {
        "causal_gru_longitudinal_v1": False,
        "causal_s4d_longitudinal_v1": True,
    }
    assert proof["retention_decision_sha256"] == S.T2_RETENTION_DECISION_SHA256
    assert proof["test_accessed"] is False
    assert proof["sealed_test_state"] == "unopened"


@pytest.mark.parametrize(
    "constant, replacement, match",
    [
        ("T2_OUTER_RESULT_SHA256", "0" * 64, "outer result digest"),
        ("T2_OUTER_EXPERIMENT_LOCK_SELF_SHA256", "1" * 64, "experiment-lock"),
        ("T2_OUTER_EXPERIMENT_LOCK_FILE_SHA256", "2" * 64, "experiment-lock file"),
        ("T2_ROW_EVIDENCE_CONTENT_SHA256", "3" * 64, "row-evidence store"),
        ("T2_ROW_EVIDENCE_MANIFEST_SHA256", "4" * 64, "row-evidence manifest"),
        ("T2_RETAINED_CHECKPOINT_SHA256", "5" * 64, "checkpoint digest"),
        ("T2_COMPARATOR_CHECKPOINT_SHA256", "6" * 64, "checkpoint digest"),
        ("T2_RETAINED_CHECKPOINT_LOCK_SHA256", "7" * 64, "checkpoint-lock digest"),
        ("T2_COMPARATOR_CHECKPOINT_LOCK_SHA256", "8" * 64, "checkpoint-lock digest"),
        ("T2_TRAINING_RESULT_SHA256", "9" * 64, "TRAIN result"),
        ("T2_TRAINING_EXPERIMENT_LOCK_SELF_SHA256", "a" * 64, "TRAIN experiment-lock"),
        ("T2_OUTER_AUTHORIZED_GIT_SHA", "b" * 40, "outer authorized commit"),
    ],
)
def test_any_drifted_bound_identity_is_refused(
    monkeypatch, constant, replacement, match
):
    """Every bound identity is load-bearing, not decoration.

    The evidence on disk is immutable, so the drift is injected on the decision
    side: a decision bound to a different digest must refuse the real evidence
    exactly as it would refuse mutated evidence bound to the real digest.
    """
    root = _require_canonical()
    monkeypatch.setattr(S, constant, replacement)
    with pytest.raises(T2SelectionError, match=match):
        validate_retained_t2_arm(root)


def test_a_canonical_selection_that_disagrees_with_retention_is_refused(monkeypatch):
    """Retention may agree with the frozen selector; it may not overrule it."""
    root = _require_canonical()
    monkeypatch.setattr(S, "T2_RETAINED_ARM", S.T2_COMPARATOR_ARM)
    with pytest.raises(T2SelectionError, match="canonical selected arm"):
        validate_retained_t2_arm(root)


def test_both_arms_remain_present_in_the_immutable_evidence():
    proof = validate_retained_t2_arm(_require_canonical())
    assert set(proof["retained"]) == {
        "causal_gru_longitudinal_v1",
        "causal_s4d_longitudinal_v1",
    }
    manifest = json.loads(
        (
            CANONICAL_RUN_ROOT
            / S.T2_OUTER_VALIDATION_ATTEMPT_ID
            / "row_evidence"
            / "T2_OUTER_ROW_EVIDENCE.json"
        ).read_text()
    )
    assert set(manifest["arms_persisted"]) == {
        "causal_gru_longitudinal_v1",
        "causal_s4d_longitudinal_v1",
    }


def test_a_row_evidence_store_missing_the_comparator_is_refused(monkeypatch):
    """Dropping the comparator would make the ablation unverifiable."""
    root = _require_canonical()
    monkeypatch.setattr(S, "T2_COMPARATOR_ARM", "causal_absent_longitudinal_v1")
    with pytest.raises(T2SelectionError, match="absent from the immutable"):
        validate_retained_t2_arm(root)


# --- 11-15. what the retained score is, and is not --------------------------


def test_the_retained_object_is_an_uncalibrated_temporal_model_score():
    proof = validate_retained_t2_arm(_require_canonical())
    assert proof["score_semantics"] == "uncalibrated_temporal_model_score"
    assert proof["score_definition"] == "sigmoid(current_window_t2_logit)"


def test_no_calibrated_probability_confidence_or_uncertainty_is_claimed():
    document = S.T2_RETENTION_DECISION_PATH.read_text().lower()
    normalised = " ".join(document.split())
    for forbidden in (
        "not a calibrated probability",
        "not a confidence",
        "not an uncertainty",
    ):
        assert forbidden in normalised, forbidden
    source = Path(S.__file__).read_text()
    assert "calibrated_probability" not in source.replace(
        "score_is_calibrated_probability", ""
    )


def test_the_threshold_is_not_t1_state_policy():
    assert S.T2_RETAINED_THRESHOLD_IS_T1_POLICY is False
    assert S.T2_RETAINED_THRESHOLD_MAY_SELECT_T1_STATE is False
    proof = validate_retained_t2_arm(_require_canonical())
    assert proof["threshold_is_t1_policy"] is False
    assert proof["threshold_may_select_t1_state"] is False
    assert proof["retained_internal_dev_threshold"] == 0.8972153067588806


def test_no_weighted_score_and_no_significance_claim():
    assert S.T2_SELECTION_WEIGHTED_SCORE_USED is False
    assert S.T2_RETENTION_STATISTICAL_SIGNIFICANCE_CLAIM is False
    assert S.T2_SELECTION_USED_TRAIN_EVIDENCE is False
    assert S.T2_SELECTION_USED_CHALLENGE_EVIDENCE is False
    assert S.T2_SELECTION_USED_LATENCY is False
    assert S.T2_SELECTION_BASIS == "pooled_primary_validation_auprc"
    assert S.T2_SELECTION_STAGE == "stage_1_pooled_primary_validation_auprc"
    assert S.T2_POOLED_AUPRC_DIFFERENCE > S.T2_SELECTION_TIE_TOLERANCE


# --- 16-17. no rerun, no extended training ----------------------------------


def test_no_t2_rerun_and_no_extended_training_are_permitted():
    assert S.T2_RERUN_PERMITTED is False
    assert S.T2_EXTENDED_TRAINING_PERMITTED is False
    normalised = " ".join(S.T2_RETENTION_DECISION_PATH.read_text().split())
    assert "T2_RERUN_PERMITTED              = False" in normalised or (
        "T2_RERUN_PERMITTED = False" in normalised
    )
    assert "no epoch 11+" in normalised.lower()


# --- 18-20. the binder records, it never computes ---------------------------


def test_no_train_or_validation_replay_path_exists():
    source = Path(S.__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = {
        "T2Timeline",
        "build_t2_model",
        "execute_canonical_training",
        "execute_canonical_outer_validation",
        "_outer_validation_worker",
        "_open_validation_timeline",
        "_load_validation_targets",
        "resolve_timeline_target_families",
        "train_t2_arm",
        "select_t2_arm",
        "read_t2_outer_row_group",
    }
    assert not (imported & forbidden), imported & forbidden
    # Scientific machinery must be absent as *code*. `sigmoid(...)` is named in
    # the module docstring because the retained score is defined by it, and a
    # substring scan would read that definition as an implementation of it.
    for module_name in ("torch", "numpy", "scipy", "sklearn"):
        assert module_name not in imported, module_name
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for computation in ("sigmoid", "forward", "load_state_dict", "state_dict"):
        assert computation not in called, computation


def test_no_model_construction_or_inference_exists():
    tree = ast.parse(Path(S.__file__).read_text())
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    forbidden = {
        "build_t2_model",
        "eval",
        "no_grad",
        "load_state_dict",
        "predict",
        "score",
        "fit",
        "train",
    }
    assert not (called & forbidden), called & forbidden


def test_binding_module_cannot_mutate_any_artifact():
    """A decision record must be incapable of touching scientific state."""
    tree = ast.parse(Path(S.__file__).read_text())
    forbidden_calls = {
        "write_json_atomic",
        "write_text",
        "write_bytes",
        "save",
        "unlink",
        "rmtree",
        "rename",
        "mkdir",
        "savez",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden_calls, name
            if name == "open":
                pytest.fail("the binding module must not open files for writing")


# --- 21-23. TEST stays sealed, and T1 can consume this ----------------------


def test_test_remains_unopened():
    assert S.T2_SEALED_TEST_STATE == "unopened"
    proof = validate_retained_t2_arm(_require_canonical())
    assert proof["test_accessed"] is False
    assert proof["sealed_test_state"] == "unopened"
    assert proof["outer_validation_is_development_evidence"] is True
    assert proof["outer_validation_is_unseen_generalization"] is False


def test_the_downstream_contract_names_s4d():
    proof = validate_retained_t2_arm(_require_canonical())
    assert proof["retained_arm"] == "causal_s4d_longitudinal_v1"
    assert proof["row_evidence_store_sha256"] == S.T2_ROW_EVIDENCE_CONTENT_SHA256
    assert proof["validation_row_count"] == 492_904
    assert proof["decision_class"] == "t2_longitudinal_temporal_retention_decision"


def test_t1_can_consume_row_evidence_without_rerunning_t2():
    proof = validate_retained_t2_arm(_require_canonical())
    assert proof["supports_t1_without_rerunning_outer_validation"] is True
    manifest = json.loads(
        (
            CANONICAL_RUN_ROOT
            / S.T2_OUTER_VALIDATION_ATTEMPT_ID
            / "row_evidence"
            / "T2_OUTER_ROW_EVIDENCE.json"
        ).read_text()
    )
    assert manifest["row_count"] == 492_904
    assert manifest["score_semantics"] == "uncalibrated_temporal_model_score"
    assert manifest["score_is_calibrated_probability"] is False
    assert manifest["score_is_confidence"] is False
    assert manifest["score_is_uncertainty"] is False


def test_t1_state_vocabulary_is_not_defined_here():
    """This decision closes T2; it must not smuggle in a T1 policy."""
    source = Path(S.__file__).read_text()
    for state in ("NORMAL", "WATCH", "EVENT", "RECOVERY"):
        assert state not in source, state
    for policy in ("hysteresis", "onset_confirmation", "recovery_confirmation"):
        assert policy not in source, policy


# --- A. the one-shot attempt guard is byte-safe -----------------------------
#
# Every case below runs against a synthetic temporary tree. The real canonical
# attempt is never mutated to prove a guard works.


def _synthetic_attempt(root):
    """A miniature attempt tree with the same shape as the real one."""
    attempt = root / "t2-v1-outer-validation"
    (attempt / "row_evidence").mkdir(parents=True)
    (attempt / "T2_OUTER_VALIDATION_STATUS.json").write_text('{"status": "COMPLETE"}')
    (attempt / "T2_OUTER_VALIDATION_RESULT.json").write_text('{"selected_arm": "s4d"}')
    (attempt / "T2_OUTER_VALIDATION_EXPERIMENT_LOCK.json").write_text('{"lock": 1}')
    (attempt / "row_evidence" / "T2_OUTER_ROW_EVIDENCE.json").write_text('{"rows": 3}')
    (attempt / "row_evidence" / "t2_outer_scores_s4d.npz").write_bytes(b"\x00scores")
    return attempt


def test_the_snapshot_detects_a_same_name_status_rewrite(tmp_path):
    attempt = _synthetic_attempt(tmp_path)
    before = attempt_content_snapshot(attempt)
    # Same filename, same length, different bytes: invisible to a name listing.
    (attempt / "T2_OUTER_VALIDATION_STATUS.json").write_text('{"status": "TAMPERED"}')
    assert attempt_content_snapshot(attempt) != before


def test_the_snapshot_detects_a_same_name_result_rewrite(tmp_path):
    attempt = _synthetic_attempt(tmp_path)
    before = attempt_content_snapshot(attempt)
    (attempt / "T2_OUTER_VALIDATION_RESULT.json").write_text('{"selected_arm": "gru"}')
    assert attempt_content_snapshot(attempt) != before


def test_the_snapshot_detects_a_nested_row_evidence_array_rewrite(tmp_path):
    attempt = _synthetic_attempt(tmp_path)
    before = attempt_content_snapshot(attempt)
    (attempt / "row_evidence" / "t2_outer_scores_s4d.npz").write_bytes(b"\x00forged")
    assert attempt_content_snapshot(attempt) != before


def test_the_snapshot_detects_a_nested_manifest_rewrite(tmp_path):
    attempt = _synthetic_attempt(tmp_path)
    before = attempt_content_snapshot(attempt)
    (attempt / "row_evidence" / "T2_OUTER_ROW_EVIDENCE.json").write_text('{"rows": 4}')
    assert attempt_content_snapshot(attempt) != before


def test_the_snapshot_detects_a_new_nested_file(tmp_path):
    attempt = _synthetic_attempt(tmp_path)
    before = attempt_content_snapshot(attempt)
    (attempt / "row_evidence" / "t2_outer_scores_gru.npz").write_bytes(b"added")
    assert attempt_content_snapshot(attempt) != before


def test_the_snapshot_detects_a_new_nested_directory(tmp_path):
    attempt = _synthetic_attempt(tmp_path)
    before = attempt_content_snapshot(attempt)
    (attempt / "recovery1").mkdir()
    assert attempt_content_snapshot(attempt) != before


def test_the_snapshot_detects_a_deletion(tmp_path):
    attempt = _synthetic_attempt(tmp_path)
    before = attempt_content_snapshot(attempt)
    (attempt / "row_evidence" / "t2_outer_scores_s4d.npz").unlink()
    assert attempt_content_snapshot(attempt) != before


def test_the_snapshot_accepts_an_unchanged_tree(tmp_path):
    attempt = _synthetic_attempt(tmp_path)
    assert attempt_content_snapshot(attempt) == attempt_content_snapshot(attempt)


def test_the_snapshot_treats_an_absent_attempt_as_none(tmp_path):
    absent = tmp_path / "t2-v1-outer-validation"
    assert attempt_content_snapshot(absent) is None
    assert attempt_content_snapshot(absent) == attempt_content_snapshot(absent)


def test_the_snapshot_binds_content_not_merely_names(tmp_path):
    """The explicit statement of the property the earlier guard lacked."""
    attempt = _synthetic_attempt(tmp_path)
    before = attempt_content_snapshot(attempt)
    names_before = sorted(p.name for p in attempt.iterdir())
    (attempt / "T2_OUTER_VALIDATION_STATUS.json").write_text('{"status": "OTHER"}')
    assert sorted(p.name for p in attempt.iterdir()) == names_before
    assert attempt_content_snapshot(attempt) != before


# --- C. the binder's explicit identities are load-bearing -------------------


@pytest.mark.parametrize(
    "constant, replacement, match",
    [
        (
            "T2_RETAINED_CHECKPOINT_LOCK_SELF_SHA256",
            "c" * 64,
            "checkpoint-lock self-digest",
        ),
        (
            "T2_COMPARATOR_CHECKPOINT_LOCK_SELF_SHA256",
            "d" * 64,
            "checkpoint-lock self-digest",
        ),
    ],
)
def test_a_drifted_checkpoint_lock_self_digest_is_refused(
    monkeypatch, constant, replacement, match
):
    root = _require_canonical()
    monkeypatch.setattr(S, constant, replacement)
    with pytest.raises(T2SelectionError, match=match):
        validate_retained_t2_arm(root)


def test_a_mutated_train_artifact_review_document_is_refused(monkeypatch):
    root = _require_canonical()
    monkeypatch.setattr(PS, "T2_TRAIN_ARTIFACT_REVIEW_SHA256", "e" * 64)
    with pytest.raises(PS.T2PersistenceError, match="review digest"):
        validate_retained_t2_arm(root)


def test_a_mutated_execution_spec_document_is_refused(monkeypatch):
    """The retention proof runs the spec's own validator, and it can refuse.

    The drift is injected on the *expected* digest rather than by rewriting the
    frozen document, for the same reason as everywhere else here: the real
    artifacts are never mutated to prove a guard works. `path` is a default
    argument bound at definition time, so patching it would not reach the
    validator at all -- patching what it compares against does.
    """
    root = _require_canonical()
    monkeypatch.setattr(PS, "T2_EXECUTION_SPEC_SHA256", "f" * 64)
    # The refusal arrives even earlier than the binder's own spec validation:
    # the canonical verifier's lock invariants already bind this identity, and
    # that transitive proof is exactly what the retention layer sits on top of.
    with pytest.raises(PS.T2PersistenceError, match="t2_execution_spec_sha256"):
        validate_retained_t2_arm(root)


def test_a_mutated_t2_protocol_document_is_refused(monkeypatch):
    from cardiosentinel.neural import t2_protocol as PR

    root = _require_canonical()
    monkeypatch.setattr(PR, "T2_PROTOCOL_SHA256", "0" * 64)
    with pytest.raises(PR.T2ProtocolError, match="protocol digest"):
        validate_retained_t2_arm(root)


def test_the_retention_proof_surfaces_the_documents_it_verified():
    proof = validate_retained_t2_arm(_require_canonical())
    assert proof["protocol_document_sha256"] == S.T2_PROTOCOL_SHA256
    assert proof["execution_spec_document_sha256"] == S.T2_EXECUTION_SPEC_SHA256
    assert proof["train_artifact_review_document_sha256"] == (
        S.T2_TRAIN_ARTIFACT_REVIEW_SHA256
    )
    assert proof["checkpoint_lock_self_sha256"] == {
        S.T2_RETAINED_ARM: S.T2_RETAINED_CHECKPOINT_LOCK_SELF_SHA256,
        S.T2_COMPARATOR_ARM: S.T2_COMPARATOR_CHECKPOINT_LOCK_SELF_SHA256,
    }


def test_the_binder_still_begins_from_the_canonical_outer_verifier():
    """The retention layer must not replace the transitive proof beneath it."""
    import inspect

    source = inspect.getsource(S.validate_retained_t2_arm)
    assert "validate_canonical_t2_outer_validation_attempt(" in source
