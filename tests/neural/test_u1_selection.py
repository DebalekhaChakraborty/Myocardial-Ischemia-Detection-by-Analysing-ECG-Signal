"""The U1 retention binding records a split decision and cannot mutate anything.

The decision retains calibration and rejects the symmetric window router, so
these tests prove both halves: that the retained calibrator family, the OOF
DEVELOPMENT input and the prospective calibrator are bound, and that the
rejected router is bound as *rejected* -- with the prespecified
`asymmetric_abstention` guard proven raised, since that is what the rejection
rests on.

The refusal tests drive the real validator against on-disk fixtures rather than
injecting past it: a mirror of the canonical attempt with exactly one field
mutated, so a check that lives in the wrong place fails visibly.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import u1_selection as S
from cardiosentinel.neural.u1_persistence import U1PersistenceError
from cardiosentinel.neural.u1_selection import (
    U1SelectionError,
    validate_retained_u1_calibration,
    validate_u1_retention_decision,
)

CANONICAL_RUN_ROOT = Path("cardiosentinel-runs/phase7-u1-development-v1")
ATTEMPT = "u1-v1-development"
EVIDENCE = f"{ATTEMPT}__evidence"


# ---------------------------------------------------------------------------
# The decision itself
# ---------------------------------------------------------------------------


def test_decision_document_is_frozen():
    assert validate_u1_retention_decision() == S.U1_RETENTION_DECISION_SHA256


def test_platt_calibration_is_retained():
    assert S.U1_CALIBRATION_RETAINED is True
    assert S.U1_RETAINED_CALIBRATOR_FAMILY == "platt_logistic_on_recovered_logit"
    assert S.U1_FAMILY_SELECTION_BASIS == "lower_pooled_out_of_fold_nll"


def test_oof_probabilities_are_the_downstream_development_input():
    assert S.U1_OOF_PROBABILITIES_RETAINED_FOR_DEVELOPMENT is True
    assert S.U1_DOWNSTREAM_DEVELOPMENT_INPUT == "u1_oof_development_calibration"


def test_final_calibrator_is_retained_for_unseen_subjects_only():
    assert S.U1_FINAL_CALIBRATOR_RETAINED_FOR_UNSEEN_SUBJECTS is True
    assert S.U1_DEPLOYMENT_CALIBRATOR_PURPOSE == (
        "unseen_subjects_and_separately_authorised_test_or_deployment_only"
    )
    # Parameterisation, never evaluation: its in-sample numbers are not evidence.
    assert S.U1_DEPLOYMENT_CALIBRATOR_IN_SAMPLE_PERFORMANCE_IS_EVIDENCE is False
    assert S.U1_DEPLOYMENT_CALIBRATOR_A == 0.3715906808641229
    assert S.U1_DEPLOYMENT_CALIBRATOR_B == -1.7662772879067046


def test_symmetric_window_router_is_not_retained():
    assert S.U1_SYMMETRIC_WINDOW_ROUTER_RETAINED is False


def test_neither_routing_threshold_is_retained_as_a_final_router():
    assert S.U1_U_STAR_DEV_RETAINED_AS_FINAL_ROUTER is False
    assert S.U1_U_STAR_DEPLOY_RETAINED_AS_FINAL_ROUTER is False
    # Rejected, not deleted: both remain immutable DEVELOPMENT evidence.
    assert S.U1_ROUTER_REMAINS_FROZEN_EVIDENCE is True
    assert S.U1_U_STAR_DEV == 0.12763774358328017
    assert S.U1_U_STAR_DEPLOY == 0.12914217081334087


def test_the_decision_introduces_no_new_routing_and_no_new_coverage_point():
    assert S.U1_NEW_ROUTING_RULE_INTRODUCED_HERE is False
    assert S.U1_ALTERNATIVE_COVERAGE_POINT_SELECTED_HERE is False
    assert S.U1_EVALUATED_COVERAGE == 0.90


def test_decision_records_its_own_limits():
    assert S.U1_SELECTION_BASIS == "development_evidence_only"
    assert S.U1_STATISTICAL_SIGNIFICANCE_CLAIM is False
    assert S.U1_TEST_ACCESSED is False
    assert S.U1_SEALED_TEST_STATE == "unopened"
    assert S.U1_AUTOMATIC_U2_TRANSITION is False
    assert S.U1_RERUN_PERMITTED is False


# ---------------------------------------------------------------------------
# Guard identities -- exact, because the rejection rests on them
# ---------------------------------------------------------------------------


def test_asymmetric_abstention_guard_identity_is_exact():
    assert S.U1_RAISED_GUARD == "asymmetric_abstention"
    assert S.U1_ASYMMETRIC_ABSTENTION_OBSERVED_RATIO == 6.453604523726777
    assert S.U1_ASYMMETRIC_ABSTENTION_BOUND == 3.0
    assert S.U1_ASYMMETRIC_ABSTENTION_OBSERVED_RATIO > S.U1_ASYMMETRIC_ABSTENTION_BOUND
    assert S.U1_TRUE_POSITIVE_ESCALATION_FRACTION == 0.5167375624190864
    assert S.U1_TRUE_NEGATIVE_ESCALATION_FRACTION == 0.0800696045937263


def test_calibration_agreement_guard_identity_is_exact_and_passed():
    assert S.U1_UNRAISED_GUARD == "routing_calibration_inadequacy"
    assert S.U1_ACCEPTED_RISK_AGREEMENT_ERROR == 0.006683691656635168
    assert S.U1_ACCEPTED_RISK_AGREEMENT_BOUND == 0.02
    assert S.U1_ACCEPTED_RISK_AGREEMENT_ERROR < S.U1_ACCEPTED_RISK_AGREEMENT_BOUND


def test_the_router_rejection_is_grounded_in_near_zero_accepted_sensitivity():
    """The low accepted risk must never be read without this number beside it."""
    assert S.U1_ACCEPTED_SENSITIVITY_AT_U_STAR_DEV == 0.0007654037504783774


def test_calibration_changed_no_frozen_decision():
    assert S.U1_CLASSIFICATION_DISAGREEMENTS == 0


# ---------------------------------------------------------------------------
# Bound canonical identities
# ---------------------------------------------------------------------------


def test_bound_u1_identities_are_exact():
    assert S.U1_RETAINED_ATTEMPT_ID == "u1-v1-development"
    assert S.U1_RETAINED_EXPERIMENT_IDENTITY == "U1_selective_v1"
    assert S.U1_EXECUTION_GIT_SHA == "233a474aca14dac4bad7d213eae46cd07836928a"
    assert S.U1_RESULT_SHA256 == (
        "649631cbf5188731d006f533997cfe28df4f5acb79e7693514e86ad0cef0cb12"
    )
    assert S.U1_EXPERIMENT_LOCK_SHA256 == (
        "7f4dd1505919e23a598773736dc57e2d1b4d360f496b45acdf2028ed0574b1b6"
    )
    assert S.U1_EXPERIMENT_LOCK_FILE_SHA256 == (
        "eca664ced24cdbc3f28b1ef339c99f0e37ec7185a034a7c7ed28b7f773d1ebfc"
    )
    assert S.U1_OOF_EVIDENCE_STORE_SHA256 == (
        "b95f484c9a7b08447f5a5d4330528136e040cf05acb9e2f7e54305e20bdffcba"
    )
    assert S.U1_FOLD_ASSIGNMENT_SHA256 == (
        "f0f5d8e93a757c0975f3613879d11f53970befa6c6bc57578b1a084c92c85b9a"
    )
    # The lock's file digest and its canonical self-digest are different things.
    assert S.U1_EXPERIMENT_LOCK_SHA256 != S.U1_EXPERIMENT_LOCK_FILE_SHA256


def test_every_promoted_component_is_bound():
    assert set(S.U1_COMPONENT_SHA256) == {
        "U1_SATURATION_CENSUS.json",
        "U1_FOLD_MANIFEST.json",
        "U1_OOF_CALIBRATION.json",
        "U1_FAMILY_SELECTION.json",
        "U1_OOF_RESULT.json",
        "U1_DEPLOYMENT_CALIBRATOR.json",
    }
    assert len(set(S.U1_COMPONENT_SHA256.values())) == 6


def test_u1_science_is_unchanged_by_this_decision():
    """The retention decision binds evidence; it never reopens the science."""
    from cardiosentinel.neural.u1_protocol import (
        U1_ASYMMETRIC_ABSTENTION_RATIO,
        U1_CLASSIFICATION_THRESHOLD,
        U1_PRIMARY_METHOD,
        U1_PROTOCOL_SHA256,
        U1_RETAINED_COVERAGE,
    )

    assert U1_PROTOCOL_SHA256 == (
        "d6235b477af278fe051822bdcccb54f985e4eceb0c6e92c1424f5e9d7d79b33b"
    )
    assert U1_CLASSIFICATION_THRESHOLD == 0.7554003000259399
    assert U1_PRIMARY_METHOD == "platt_logistic_on_recovered_logit"
    assert U1_RETAINED_COVERAGE == 0.90
    assert U1_ASYMMETRIC_ABSTENTION_RATIO == 3.0


# ---------------------------------------------------------------------------
# The binding module is a record, not a machine
# ---------------------------------------------------------------------------


def test_binding_module_cannot_mutate_any_artifact():
    tree = ast.parse(Path(S.__file__).read_text())
    forbidden_calls = {"write_json_atomic", "save", "unlink", "rmtree", "rename"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden_calls, name
            if name == "open":
                pytest.fail("the binding module must not open files for writing")


def test_binding_module_has_no_fit_scorer_or_replay_path():
    """A decision record must be incapable of refitting or re-routing anything."""
    tree = ast.parse(Path(S.__file__).read_text())
    imported = {
        alias.name
        for imp in ast.walk(tree)
        if isinstance(imp, ast.ImportFrom)
        for alias in imp.names
    }
    forbidden = {
        "execute_canonical_u1_development",
        "execute_canonical_development",
        "fit_calibrator",
        "fit_out_of_fold",
        "select_calibrator_family",
        "derive_routing_threshold",
        "derive_u_star_dev",
        "risk_coverage_curve",
        "routing_guards",
        "subject_bootstrap",
        "saturation_census",
        "load_m2g_score_table",
        "replay_stream",
        "M2Scorer",
    }
    assert not (imported & forbidden), imported & forbidden


def test_decision_document_records_its_governance_nature_and_non_claims():
    text = S.U1_RETENTION_DECISION_PATH.read_text()
    assert "THIS IS A HUMAN GOVERNANCE DECISION, NOT A NEW SCIENTIFIC EXPERIMENT." in (
        text
    )
    # the split, stated as a split
    assert "Calibration is retained." in text
    assert "symmetric selective router is **not** retained." in text
    # the imbalance caveat is stated, not implied
    assert "MUST NOT be interpreted in isolation." in text
    # the guard did its job; the response is to decline, not to retune
    assert "successfully exposed this behaviour" in text
    # the confounder benefit is kept, and kept subordinate
    assert "does NOT override the asymmetric-abstention failure" in text
    # no post-hoc coverage shopping
    assert "No post-hoc coverage point is chosen here." in text
    # the rejected router survives as evidence
    assert "immutable U1 DEVELOPMENT / ablation evidence" in text
    assert "No U1 rerun is permitted" in text
    assert "U2 conformal prediction does NOT automatically begin." in text


# ---------------------------------------------------------------------------
# The validator, driven against real and mirrored on-disk artifacts
# ---------------------------------------------------------------------------


def _mirror_canonical_attempt(tmp_path: Path) -> Path:
    """Mirror the canonical attempt so exactly one field can then be mutated.

    JSON is copied; the 134 MB per-row evidence is symlinked, which `sha256_file`
    follows, so the mirror is byte-identical without duplicating it.
    """
    source_run = CANONICAL_RUN_ROOT / ATTEMPT
    source_evidence = CANONICAL_RUN_ROOT / EVIDENCE
    if not source_run.is_dir() or not source_evidence.is_dir():
        pytest.skip("canonical U1 run directory is not on this filesystem")

    root = tmp_path / "phase7-u1-development-v1"
    for source, name in ((source_run, ATTEMPT), (source_evidence, EVIDENCE)):
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            if path.suffix == ".json":
                (target / path.name).write_bytes(path.read_bytes())
            else:
                (target / path.name).symlink_to(path.resolve())
    return root


def _rewrite(path: Path, **overrides) -> None:
    payload = json.loads(path.read_text())
    payload.update(overrides)
    path.write_text(json.dumps(payload))


def test_the_canonical_attempt_proves_the_split_decision():
    if not (CANONICAL_RUN_ROOT / ATTEMPT).is_dir():
        pytest.skip("canonical U1 run directory is not on this filesystem")
    proof = validate_retained_u1_calibration(CANONICAL_RUN_ROOT)
    assert proof["retained"] == {
        "calibration": True,
        "calibrator_family": "platt_logistic_on_recovered_logit",
        "oof_probabilities_for_development": True,
        "final_calibrator_for_unseen_subjects": True,
        "symmetric_window_router": False,
        "u_star_dev_as_final_router": False,
        "u_star_deploy_as_final_router": False,
    }
    assert proof["attempt_status"] == "COMPLETE"
    assert proof["canonical_verification"]["verified"] is True
    assert proof["selected_family"] == "platt_logistic_on_recovered_logit"
    assert proof["classification_disagreements"] == 0
    assert proof["raised_guard"] == "asymmetric_abstention"
    assert proof["calibration_agreement_guard_raised"] is False
    assert proof["test_accessed"] is False
    assert proof["sealed_test_state"] == "unopened"
    assert proof["u1_rerun_permitted"] is False
    assert proof["statistical_significance_claim"] is False
    assert proof["router_remains_frozen_evidence"] is True


def test_validator_accepts_the_mirrored_attempt(tmp_path):
    """The refusal tests below are meaningful only if the clean mirror passes."""
    root = _mirror_canonical_attempt(tmp_path)
    proof = validate_retained_u1_calibration(root)
    assert proof["u1_result_sha256"] == S.U1_RESULT_SHA256


def test_validator_refuses_a_missing_attempt(tmp_path):
    with pytest.raises(U1SelectionError, match="No canonical U1 result"):
        validate_retained_u1_calibration(tmp_path)


def test_validator_refuses_a_mutated_result(tmp_path):
    """Touching the result must break the binding, not merely a sub-check.

    Artifact integrity is delegated to the one canonical verifier rather than
    re-implemented here, so its refusal is what propagates. The binding must not
    catch, soften or re-wrap it.
    """
    root = _mirror_canonical_attempt(tmp_path)
    _rewrite(root / ATTEMPT / "U1_RESULT.json", experiment_identity="U1_selective_v2")
    with pytest.raises(U1PersistenceError, match="does not match its lock digest"):
        validate_retained_u1_calibration(root)


def test_validator_refuses_a_mutated_evidence_store(tmp_path):
    """The per-row evidence is part of the artifact, not an optional extra."""
    root = _mirror_canonical_attempt(tmp_path)
    store = root / EVIDENCE / "U1_OOF_EVIDENCE_STORE.json"
    _rewrite(store, content_sha256="0" * 64)
    with pytest.raises(U1PersistenceError, match="is not intact"):
        validate_retained_u1_calibration(root)


def test_validator_refuses_an_attempt_that_is_not_complete(tmp_path):
    root = _mirror_canonical_attempt(tmp_path)
    _rewrite(root / ATTEMPT / "U1_RUN_STATUS.json", status="STOPPED_FOR_HUMAN_REVIEW")
    with pytest.raises(U1SelectionError, match="binds a COMPLETE attempt"):
        validate_retained_u1_calibration(root)


def test_validator_refuses_a_run_that_retained_something_automatically(tmp_path):
    root = _mirror_canonical_attempt(tmp_path)
    _rewrite(root / ATTEMPT / "U1_RUN_STATUS.json", automatic_retention=True)
    with pytest.raises(U1SelectionError, match="retained nothing"):
        validate_retained_u1_calibration(root)


def test_validator_refuses_a_run_whose_guards_differ(tmp_path):
    """The rejection rests on exactly one raised guard; anything else is a lie."""
    root = _mirror_canonical_attempt(tmp_path)
    _rewrite(root / ATTEMPT / "U1_RUN_STATUS.json", routing_guard_flags_raised=[])
    with pytest.raises(U1SelectionError, match="grounded in exactly"):
        validate_retained_u1_calibration(root)


# ---------------------------------------------------------------------------
# The individual proofs, against synthetic on-disk fixtures
# ---------------------------------------------------------------------------


def _canonical(name: str) -> dict:
    path = CANONICAL_RUN_ROOT / ATTEMPT / name
    if not path.is_file():
        pytest.skip("canonical U1 run directory is not on this filesystem")
    return json.loads(path.read_text())


def _u_star_dev_point() -> dict:
    """The persisted c_star = 0.90 risk-coverage point, read-only.

    Nothing is recomputed from the per-row evidence: this reads the promoted
    artifact and checks only that its own recorded counts agree with each other.
    """
    oof = _canonical("U1_OOF_RESULT.json")
    points = [
        point
        for point in oof["risk_coverage"]["points"]
        if point["target_coverage"] == S.U1_EVALUATED_COVERAGE
    ]
    assert len(points) == 1
    return points[0]


def test_accepted_sensitivity_denominator_is_the_accepted_positive_windows():
    """8 / 10,452, never 8 / 21,628.

    The PRIMARY total is the population count. The accepted-sensitivity
    denominator is the *accepted* positive-label windows, which is a strictly
    smaller set because the router escalated the rest -- and the two ratios are
    genuinely different numbers, so conflating them misreports the result.
    """
    point = _u_star_dev_point()

    true_positives = point["accepted_true_positive_count"]
    false_negatives = point["accepted_false_negative_count"]
    accepted_positives = point["accepted_positive_count"]
    primary_positives = point["label_positive_count"]

    assert true_positives == 8
    assert false_negatives == 10_444
    assert accepted_positives == 10_452
    assert primary_positives == 21_628

    # The accepted positive-label windows are exactly TP + FN, and they are not
    # the population total: 11,176 positive-label windows were escalated.
    assert true_positives + false_negatives == accepted_positives
    assert accepted_positives < primary_positives
    assert primary_positives - accepted_positives == 11_176

    sensitivity = true_positives / accepted_positives
    assert sensitivity == point["accepted_sensitivity"]
    assert sensitivity == S.U1_ACCEPTED_SENSITIVITY_AT_U_STAR_DEV
    assert sensitivity == 0.0007654037504783774
    # The wrong denominator yields a different number; that is the whole point.
    assert true_positives / primary_positives != sensitivity


def test_decision_document_states_the_denominator_unambiguously():
    text = S.U1_RETENTION_DECISION_PATH.read_text()
    assert "8 / 10,452 = 0.0007654037504783774" in text
    assert "10,452 positive-label windows locally" in text
    assert "21,628 positive-label windows in the PRIMARY population" in text
    # positive-label window and true-positive detection are distinguished
    assert "true-positive detections" in text


def test_guard_proof_refuses_a_result_where_the_guard_did_not_fire():
    result = _canonical("U1_RESULT.json")
    result["routing_guards"]["flags"]["asymmetric_abstention"] = False
    result["routing_guards"]["flags_raised"] = []
    with pytest.raises(U1SelectionError, match="was NOT raised"):
        S._require_raised_guard(result)


def test_guard_proof_refuses_a_result_where_the_other_guard_fired():
    result = _canonical("U1_RESULT.json")
    result["routing_guards"]["flags"]["routing_calibration_inadequacy"] = True
    with pytest.raises(U1SelectionError, match="records it as passed"):
        S._require_raised_guard(result)


def test_guard_proof_refuses_a_different_abstention_ratio():
    result = _canonical("U1_RESULT.json")
    result["routing_guards"]["asymmetric_abstention_ratio"] = 3.5
    with pytest.raises(U1SelectionError, match="Guard field"):
        S._require_raised_guard(result)


def test_guard_proof_refuses_a_guard_that_refit_or_reselected():
    result = _canonical("U1_RESULT.json")
    result["routing_guards"]["threshold_reselected"] = True
    with pytest.raises(U1SelectionError, match="changes nothing"):
        S._require_raised_guard(result)


def _fixture_dir(tmp_path: Path, *names: str) -> Path:
    for name in names:
        (tmp_path / name).write_text(json.dumps(_canonical(name)))
    return tmp_path


def test_family_proof_refuses_a_different_selected_family(tmp_path):
    directory = _fixture_dir(tmp_path, "U1_FAMILY_SELECTION.json")
    _rewrite(
        directory / "U1_FAMILY_SELECTION.json",
        selected_family="temperature_only_on_recovered_logit",
    )
    with pytest.raises(U1SelectionError, match="this decision retains"):
        S._require_family_selection(directory)


def test_family_proof_refuses_selection_that_used_more_than_nll(tmp_path):
    directory = _fixture_dir(tmp_path, "U1_FAMILY_SELECTION.json")
    _rewrite(directory / "U1_FAMILY_SELECTION.json", ece_used=True)
    with pytest.raises(U1SelectionError, match="pooled out-of-fold NLL alone"):
        S._require_family_selection(directory)


def test_family_proof_refuses_an_artifact_claiming_to_be_the_human_decision(tmp_path):
    directory = _fixture_dir(tmp_path, "U1_FAMILY_SELECTION.json")
    _rewrite(directory / "U1_FAMILY_SELECTION.json", is_u1_retention_decision=True)
    with pytest.raises(U1SelectionError, match="must not claim to be the human"):
        S._require_family_selection(directory)


def test_equivalence_proof_refuses_any_induced_disagreement(tmp_path):
    directory = _fixture_dir(
        tmp_path, "U1_OOF_RESULT.json", "U1_DEPLOYMENT_CALIBRATOR.json"
    )
    payload = json.loads((directory / "U1_OOF_RESULT.json").read_text())
    payload["decision_equivalence_per_fold"][0]["disagreement_count"] = 1
    (directory / "U1_OOF_RESULT.json").write_text(json.dumps(payload))
    with pytest.raises(U1SelectionError, match="induced 1 classification"):
        S._require_zero_disagreements(directory)


def test_equivalence_proof_refuses_a_calibrated_boundary_sold_as_a_threshold(tmp_path):
    directory = _fixture_dir(
        tmp_path, "U1_OOF_RESULT.json", "U1_DEPLOYMENT_CALIBRATOR.json"
    )
    payload = json.loads((directory / "U1_DEPLOYMENT_CALIBRATOR.json").read_text())
    payload["decision_equivalence"]["calibrated_boundary_is_a_new_threshold"] = True
    (directory / "U1_DEPLOYMENT_CALIBRATOR.json").write_text(json.dumps(payload))
    with pytest.raises(U1SelectionError, match="probability transformation only"):
        S._require_zero_disagreements(directory)


def test_calibrator_proof_refuses_a_reported_in_sample_performance(tmp_path):
    directory = _fixture_dir(tmp_path, "U1_DEPLOYMENT_CALIBRATOR.json")
    _rewrite(
        directory / "U1_DEPLOYMENT_CALIBRATOR.json",
        in_sample_performance_reported=True,
    )
    with pytest.raises(U1SelectionError, match="never U1 DEVELOPMENT evidence"):
        S._require_retained_deployment_calibrator(directory)


def test_calibrator_proof_refuses_u_star_deploy_sold_as_evidence(tmp_path):
    directory = _fixture_dir(tmp_path, "U1_DEPLOYMENT_CALIBRATOR.json")
    _rewrite(
        directory / "U1_DEPLOYMENT_CALIBRATOR.json",
        u_star_deploy_is_scientific_evidence=True,
    )
    with pytest.raises(U1SelectionError, match="configuration provenance only"):
        S._require_retained_deployment_calibrator(directory)


def test_calibrator_proof_refuses_a_reselected_family(tmp_path):
    directory = _fixture_dir(tmp_path, "U1_DEPLOYMENT_CALIBRATOR.json")
    _rewrite(directory / "U1_DEPLOYMENT_CALIBRATOR.json", family_reselected=True)
    with pytest.raises(U1SelectionError, match="reselected the calibrator family"):
        S._require_retained_deployment_calibrator(directory)
