"""Immutable binding of the human U1 calibration / selective-routing decision.

This module records a *decision*, not a computation. The decision is a **split**
one, and that is the whole reason it needs binding: the calibration half of U1
is retained and the routing half is not, so a later phase must be able to prove
which half it is standing on.

Retained: the Platt calibrator family, the subject-disjoint OOF calibrated
probabilities as the downstream DEVELOPMENT input, and the final
all-VALIDATION calibrator as prospective parameterisation for genuinely unseen
subjects. Not retained: the symmetric window-level router at ``c_star = 0.90``,
``u_star_dev`` and ``u_star_deploy`` as operational routing thresholds.

The rejected router is *bound*, not erased. It stays immutable DEVELOPMENT /
ablation evidence, and this module proves the prespecified
``asymmetric_abstention`` guard was raised on it -- so no later phase can quietly
adopt the router as though the guard had passed.

Like the M2 binding, it contains no fitting, no scoring, no replay, no routing
recomputation and no mutation. Reading it can never alter an artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.u1_persistence import (
    DEPLOYMENT_CALIBRATOR_NAME,
    EXPERIMENT_LOCK_NAME,
    FAMILY_SELECTION_NAME,
    FOLD_MANIFEST_NAME,
    OOF_CALIBRATION_NAME,
    OOF_RESULT_NAME,
    RESULT_NAME,
    RUN_STATUS_NAME,
    SATURATION_CENSUS_NAME,
    STATUS_COMPLETE,
    u1_run_directory,
    validate_canonical_u1_attempt,
)
from cardiosentinel.neural.u1_protocol import (
    U1_ACCEPTED_RISK_AGREEMENT_TOLERANCE,
    U1_ASYMMETRIC_ABSTENTION_RATIO,
    U1_CLASSIFICATION_THRESHOLD,
    U1_PRIMARY_METHOD,
    U1_PROTOCOL_SHA256,
    U1_RETAINED_COVERAGE,
)

U1_RETENTION_DECISION_NAME: Final = "U1_CALIBRATION_ROUTING_RETENTION_DECISION_V1"
U1_RETENTION_DECISION_PATH: Final = (
    REPOSITORY_ROOT / "docs" / f"{U1_RETENTION_DECISION_NAME}.md"
)
U1_RETENTION_DECISION_SHA256: Final = (
    "9d8436f2b7d2c303aeeb03e438c60fb8110f7d06d0bbd589f5be65ea8f80cb7b"
)

# ---------------------------------------------------------------------------
# The one canonical U1 attempt this decision is about
# ---------------------------------------------------------------------------
U1_RETAINED_ATTEMPT_ID: Final = "u1-v1-development"
U1_RETAINED_EXPERIMENT_IDENTITY: Final = "U1_selective_v1"
U1_CANONICAL_RUN_ROOT: Final = Path("cardiosentinel-runs/phase7-u1-development-v1")
U1_EXECUTION_GIT_SHA: Final = "233a474aca14dac4bad7d213eae46cd07836928a"

U1_RESULT_SHA256: Final = (
    "649631cbf5188731d006f533997cfe28df4f5acb79e7693514e86ad0cef0cb12"
)
U1_EXPERIMENT_LOCK_SHA256: Final = (
    "7f4dd1505919e23a598773736dc57e2d1b4d360f496b45acdf2028ed0574b1b6"
)
U1_EXPERIMENT_LOCK_FILE_SHA256: Final = (
    "eca664ced24cdbc3f28b1ef339c99f0e37ec7185a034a7c7ed28b7f773d1ebfc"
)
U1_COMPONENT_SHA256: Final = {
    SATURATION_CENSUS_NAME: (
        "0ee3e80dc86d48d89dbb2e3a9f3d1ddb8263670a636335853e36bd91a710e5de"
    ),
    FOLD_MANIFEST_NAME: (
        "6de92e8d86f8fed03357a5daf6a5c33a5c97df06d5cefa846ee2d453e49ed82a"
    ),
    OOF_CALIBRATION_NAME: (
        "c6a48fcd5e14cbe9d543eaa1d81328a8eade41343cc9629c8f3f8b78eee47da2"
    ),
    FAMILY_SELECTION_NAME: (
        "cbf8dec21defa18143050cd74b5c08916a17f07279578541e234fac3cdce70d1"
    ),
    OOF_RESULT_NAME: (
        "dbe546ecb4da1b6a974ace6549803ac9a6894db321707da25cff39d9bca0e7e6"
    ),
    DEPLOYMENT_CALIBRATOR_NAME: (
        "acec97c1ebd3bed459ad2d75204b6c82f274b248edbb1d779b844bd46c62fdc1"
    ),
}
U1_OOF_EVIDENCE_STORE_SHA256: Final = (
    "b95f484c9a7b08447f5a5d4330528136e040cf05acb9e2f7e54305e20bdffcba"
)
U1_FOLD_ASSIGNMENT_SHA256: Final = (
    "f0f5d8e93a757c0975f3613879d11f53970befa6c6bc57578b1a084c92c85b9a"
)

# ---------------------------------------------------------------------------
# The split decision
# ---------------------------------------------------------------------------
U1_CALIBRATION_RETAINED: Final = True
U1_RETAINED_CALIBRATOR_FAMILY: Final = U1_PRIMARY_METHOD
U1_OOF_PROBABILITIES_RETAINED_FOR_DEVELOPMENT: Final = True
U1_FINAL_CALIBRATOR_RETAINED_FOR_UNSEEN_SUBJECTS: Final = True

U1_SYMMETRIC_WINDOW_ROUTER_RETAINED: Final = False
U1_U_STAR_DEV_RETAINED_AS_FINAL_ROUTER: Final = False
U1_U_STAR_DEPLOY_RETAINED_AS_FINAL_ROUTER: Final = False

U1_SELECTION_BASIS: Final = "development_evidence_only"
U1_FAMILY_SELECTION_BASIS: Final = "lower_pooled_out_of_fold_nll"
U1_STATISTICAL_SIGNIFICANCE_CLAIM: Final = False
U1_RERUN_PERMITTED: Final = False
U1_TEST_ACCESSED: Final = False
U1_SEALED_TEST_STATE: Final = "unopened"
U1_AUTOMATIC_U2_TRANSITION: Final = False
U1_NEW_ROUTING_RULE_INTRODUCED_HERE: Final = False
U1_ALTERNATIVE_COVERAGE_POINT_SELECTED_HERE: Final = False

# The rejected router is preserved as evidence, never deleted.
U1_ROUTER_REMAINS_FROZEN_EVIDENCE: Final = True

# Downstream DEVELOPMENT work on the 12 VALIDATION subjects reads the OOF
# probabilities; the all-validation calibrator is in-sample on those subjects.
U1_DOWNSTREAM_DEVELOPMENT_INPUT: Final = "u1_oof_development_calibration"
U1_DEPLOYMENT_CALIBRATOR_PURPOSE: Final = (
    "unseen_subjects_and_separately_authorised_test_or_deployment_only"
)
U1_DEPLOYMENT_CALIBRATOR_IN_SAMPLE_PERFORMANCE_IS_EVIDENCE: Final = False

# ---------------------------------------------------------------------------
# Guard identities -- the reason the router is not retained
# ---------------------------------------------------------------------------
U1_RAISED_GUARD: Final = "asymmetric_abstention"
U1_UNRAISED_GUARD: Final = "routing_calibration_inadequacy"
U1_ASYMMETRIC_ABSTENTION_OBSERVED_RATIO: Final = 6.453604523726777
U1_ASYMMETRIC_ABSTENTION_BOUND: Final = U1_ASYMMETRIC_ABSTENTION_RATIO
U1_TRUE_POSITIVE_ESCALATION_FRACTION: Final = 0.5167375624190864
U1_TRUE_NEGATIVE_ESCALATION_FRACTION: Final = 0.0800696045937263
U1_ACCEPTED_RISK_AGREEMENT_ERROR: Final = 0.006683691656635168
U1_ACCEPTED_RISK_AGREEMENT_BOUND: Final = U1_ACCEPTED_RISK_AGREEMENT_TOLERANCE

U1_EVALUATED_COVERAGE: Final = U1_RETAINED_COVERAGE
U1_U_STAR_DEV: Final = 0.12763774358328017
U1_U_STAR_DEPLOY: Final = 0.12914217081334087
U1_ACCEPTED_SENSITIVITY_AT_U_STAR_DEV: Final = 0.0007654037504783774

U1_DEPLOYMENT_CALIBRATOR_A: Final = 0.3715906808641229
U1_DEPLOYMENT_CALIBRATOR_B: Final = -1.7662772879067046

U1_CLASSIFICATION_DISAGREEMENTS: Final = 0


class U1SelectionError(RuntimeError):
    """Raised when the human U1 retention decision cannot be proven."""


def validate_u1_retention_decision(
    path: Path = U1_RETENTION_DECISION_PATH,
) -> str:
    """Verify the frozen U1 retention decision document byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise U1SelectionError(f"U1 retention decision is missing at {document}.")
    digest = sha256_file(document)
    if digest != U1_RETENTION_DECISION_SHA256:
        raise U1SelectionError(
            f"U1 retention decision digest {digest} differs from the frozen "
            f"{U1_RETENTION_DECISION_SHA256}. The decision is immutable."
        )
    return digest


def _require_completed_attempt(run_dir: Path) -> dict[str, Any]:
    """Prove the attempt this decision binds actually completed as canonical."""
    status_path = run_dir / RUN_STATUS_NAME
    if not status_path.is_file():
        raise U1SelectionError(f"No U1 run status at {status_path}.")
    status = json.loads(status_path.read_text())
    if status.get("experiment_id") != U1_RETAINED_ATTEMPT_ID:
        raise U1SelectionError(
            f"The run status records attempt {status.get('experiment_id')!r}, "
            f"not the retained {U1_RETAINED_ATTEMPT_ID!r}."
        )
    if status.get("status") != STATUS_COMPLETE:
        raise U1SelectionError(
            f"The retention decision binds a COMPLETE attempt; this one records "
            f"{status.get('status')!r}."
        )
    if status.get("canonical") is not True:
        raise U1SelectionError("The bound U1 attempt does not record itself canonical.")
    if status.get("automatic_retention") is not False:
        raise U1SelectionError(
            "The canonical run must continue to record that it retained nothing "
            "automatically; the decision lives in the governance document."
        )
    if status.get("human_review_required") is not True:
        raise U1SelectionError(
            "The canonical run must continue to require the human retention review."
        )
    if status.get("automatic_u2_transition") is not False:
        raise U1SelectionError("The canonical run records an automatic U2 transition.")
    if list(status.get("routing_guard_flags_raised", [])) != [U1_RAISED_GUARD]:
        raise U1SelectionError(
            f"The canonical run records raised guards "
            f"{status.get('routing_guard_flags_raised')!r}; this decision is "
            f"grounded in exactly [{U1_RAISED_GUARD!r}]."
        )
    return status


def _require_family_selection(run_dir: Path) -> dict[str, Any]:
    """Prove the retained family was chosen out-of-fold, on NLL alone."""
    selection = json.loads((run_dir / FAMILY_SELECTION_NAME).read_text())
    if selection.get("selected_family") != U1_RETAINED_CALIBRATOR_FAMILY:
        raise U1SelectionError(
            f"The canonical run selected {selection.get('selected_family')!r}, "
            f"but this decision retains {U1_RETAINED_CALIBRATOR_FAMILY!r}."
        )
    if selection.get("selection_basis") != U1_FAMILY_SELECTION_BASIS:
        raise U1SelectionError(
            f"Family selection basis is {selection.get('selection_basis')!r}, "
            f"not the frozen {U1_FAMILY_SELECTION_BASIS!r}."
        )
    if selection.get("evidence_source") != "out_of_fold_only":
        raise U1SelectionError("Family selection was not out-of-fold only.")
    for forbidden in (
        "auprc_used",
        "brier_used",
        "ece_used",
        "routing_risk_used",
        "challenge_evidence_used",
        "weighted_score_used",
    ):
        if selection.get(forbidden) is not False:
            raise U1SelectionError(
                f"Family selection records {forbidden}=true; the frozen basis is "
                "pooled out-of-fold NLL alone."
            )
    if selection.get("is_u1_retention_decision") is not False:
        raise U1SelectionError(
            "The calibrator-family artifact must not claim to be the human U1 "
            "retention decision."
        )
    return selection


def _require_raised_guard(result: dict[str, Any]) -> dict[str, Any]:
    """Prove the guard that grounds the router rejection, exactly as persisted."""
    guards = dict(result["routing_guards"])
    flags = dict(guards.get("flags", {}))
    if flags.get(U1_RAISED_GUARD) is not True:
        raise U1SelectionError(
            f"The {U1_RAISED_GUARD!r} guard was NOT raised in the bound result. "
            "The router rejection rests on that guard and cannot be proven."
        )
    if flags.get(U1_UNRAISED_GUARD) is not False:
        raise U1SelectionError(
            f"The {U1_UNRAISED_GUARD!r} guard was raised in the bound result, but "
            "the retention decision records it as passed."
        )
    if list(guards.get("flags_raised", [])) != [U1_RAISED_GUARD]:
        raise U1SelectionError(
            f"The bound result raises {guards.get('flags_raised')!r}, not exactly "
            f"[{U1_RAISED_GUARD!r}]."
        )
    for field_, expected in (
        ("asymmetric_abstention_ratio", U1_ASYMMETRIC_ABSTENTION_OBSERVED_RATIO),
        ("asymmetric_abstention_ratio_bound", U1_ASYMMETRIC_ABSTENTION_BOUND),
        ("true_positive_escalation_fraction", U1_TRUE_POSITIVE_ESCALATION_FRACTION),
        ("true_negative_escalation_fraction", U1_TRUE_NEGATIVE_ESCALATION_FRACTION),
        (
            "accepted_risk_absolute_agreement_error",
            U1_ACCEPTED_RISK_AGREEMENT_ERROR,
        ),
        ("accepted_risk_agreement_tolerance", U1_ACCEPTED_RISK_AGREEMENT_BOUND),
        ("evaluated_at_target_coverage", U1_EVALUATED_COVERAGE),
    ):
        if guards.get(field_) != expected:
            raise U1SelectionError(
                f"Guard field {field_} is {guards.get(field_)!r}, but the "
                f"retention decision binds {expected!r}."
            )
    for flag in ("refit_performed", "threshold_reselected", "automatic_retention"):
        if guards.get(flag) is not False:
            raise U1SelectionError(
                f"The raised guard must not have triggered {flag}; a guard reports "
                "a scientific outcome and changes nothing."
            )
    if guards.get("scientific_evidence_discarded") is not False:
        raise U1SelectionError("The bound result records discarded evidence.")
    return guards


def _require_zero_disagreements(run_dir: Path) -> int:
    """Prove calibration changed no frozen decision, per fold and in the final fit."""
    oof = json.loads((run_dir / OOF_RESULT_NAME).read_text())
    folds = list(oof["decision_equivalence_per_fold"])
    if not folds:
        raise U1SelectionError("The bound result proves no per-fold equivalence.")
    total = 0
    for proof in folds:
        if proof.get("classification_threshold") != U1_CLASSIFICATION_THRESHOLD:
            raise U1SelectionError(
                "A decision-equivalence proof does not use the frozen threshold."
            )
        if proof.get("threshold_selected_here") is not False:
            raise U1SelectionError("A fold claims it selected the threshold here.")
        total += int(proof.get("disagreement_count", -1))

    calibrator = json.loads((run_dir / DEPLOYMENT_CALIBRATOR_NAME).read_text())
    final_proof = dict(calibrator["decision_equivalence"])
    total += int(final_proof.get("disagreement_count", -1))
    if final_proof.get("row_for_row_identical") is not True:
        raise U1SelectionError(
            "The final calibrator's decisions are not row-for-row identical."
        )
    if final_proof.get("calibrated_boundary_is_a_new_threshold") is not False:
        raise U1SelectionError(
            "The calibrated boundary is recorded as a new threshold; calibration "
            "is retained as a probability transformation only."
        )
    if total != U1_CLASSIFICATION_DISAGREEMENTS:
        raise U1SelectionError(
            f"Calibration induced {total} classification disagreements; the "
            f"retention decision binds {U1_CLASSIFICATION_DISAGREEMENTS}."
        )
    return total


def _require_retained_deployment_calibrator(run_dir: Path) -> dict[str, Any]:
    """Prove the retained prospective calibrator is parameterisation, not evidence."""
    calibrator = json.loads((run_dir / DEPLOYMENT_CALIBRATOR_NAME).read_text())
    if calibrator.get("selected_family") != U1_RETAINED_CALIBRATOR_FAMILY:
        raise U1SelectionError("The final calibrator is not the retained family.")
    if calibrator.get("family_reselected") is not False:
        raise U1SelectionError("The final fit reselected the calibrator family.")
    if calibrator.get("is_parameterisation") is not True:
        raise U1SelectionError(
            "The final calibrator does not record itself as parameterisation."
        )
    if calibrator.get("is_evaluation") is not False:
        raise U1SelectionError("The final calibrator records itself as evaluation.")
    if calibrator.get("in_sample_performance_reported") is not False:
        raise U1SelectionError(
            "The final all-validation fit reports in-sample performance; that is "
            "never U1 DEVELOPMENT evidence."
        )
    if calibrator.get("in_sample_performance_claim_authorised") is not False:
        raise U1SelectionError(
            "The final all-validation fit claims an authorised in-sample result."
        )
    if calibrator.get("purpose") != U1_DEPLOYMENT_CALIBRATOR_PURPOSE:
        raise U1SelectionError(
            f"The final calibrator records purpose {calibrator.get('purpose')!r}."
        )
    parameters = dict(calibrator["calibrator"])
    for name, expected in (
        ("a", U1_DEPLOYMENT_CALIBRATOR_A),
        ("b", U1_DEPLOYMENT_CALIBRATOR_B),
    ):
        if parameters.get(name) != expected:
            raise U1SelectionError(
                f"The retained calibrator's {name} is {parameters.get(name)!r}, "
                f"not the frozen {expected!r}."
            )
    deploy = dict(calibrator["u_star_deploy"])
    if deploy.get("u_star") != U1_U_STAR_DEPLOY:
        raise U1SelectionError("The bound u_star_deploy is not the frozen value.")
    if calibrator.get("u_star_deploy_is_scientific_evidence") is not False:
        raise U1SelectionError(
            "u_star_deploy is recorded as scientific evidence; it is configuration "
            "provenance only, and it is NOT retained as a routing threshold."
        )
    return calibrator


def validate_retained_u1_calibration(
    run_root: Path = U1_CANONICAL_RUN_ROOT,
) -> dict[str, Any]:
    """Prove the split U1 retention decision against the frozen canonical attempt.

    Read-only. Nothing is fitted, scored, replayed or re-thresholded, and no
    artifact is written. The router half of the decision is proven by binding
    the raised guard, not by recomputing a routing curve.
    """
    root = Path(run_root)
    run_dir = u1_run_directory(root, U1_RETAINED_ATTEMPT_ID)
    result_path = run_dir / RESULT_NAME
    if not result_path.is_file():
        raise U1SelectionError(f"No canonical U1 result at {result_path}.")

    verification = validate_canonical_u1_attempt(root, U1_RETAINED_ATTEMPT_ID)
    if verification["result_sha256"] != U1_RESULT_SHA256:
        raise U1SelectionError(
            f"The canonical U1 result digests to {verification['result_sha256']}, "
            f"but the retention decision binds {U1_RESULT_SHA256}. The evidence "
            "is immutable."
        )
    if verification["experiment_lock_sha256"] != U1_EXPERIMENT_LOCK_SHA256:
        raise U1SelectionError(
            "The canonical U1 lock is not the one the retention decision binds."
        )
    if verification["oof_evidence_store_sha256"] != U1_OOF_EVIDENCE_STORE_SHA256:
        raise U1SelectionError(
            "The OOF evidence store is not the one the retention decision binds."
        )
    if dict(verification["component_sha256"]) != dict(U1_COMPONENT_SHA256):
        raise U1SelectionError(
            "The promoted component digests differ from the retention decision's."
        )

    lock_digest = sha256_file(run_dir / EXPERIMENT_LOCK_NAME)
    if lock_digest != U1_EXPERIMENT_LOCK_FILE_SHA256:
        raise U1SelectionError(
            f"The U1 experiment-lock file digests to {lock_digest}, not the frozen "
            f"{U1_EXPERIMENT_LOCK_FILE_SHA256}."
        )

    status = _require_completed_attempt(run_dir)
    result = json.loads(result_path.read_text())
    if result.get("experiment_identity") != U1_RETAINED_EXPERIMENT_IDENTITY:
        raise U1SelectionError(
            f"The bound result records identity "
            f"{result.get('experiment_identity')!r}, not "
            f"{U1_RETAINED_EXPERIMENT_IDENTITY!r}."
        )
    if result.get("test_accessed") is not False:
        raise U1SelectionError("The bound U1 result records test access.")
    if result.get("sealed_test_state") != U1_SEALED_TEST_STATE:
        raise U1SelectionError("The bound U1 result does not record TEST as unopened.")
    if result.get("automatic_retention") is not False:
        raise U1SelectionError("The bound U1 result records an automatic retention.")

    lock = json.loads((run_dir / EXPERIMENT_LOCK_NAME).read_text())
    if lock.get("git_sha") != U1_EXECUTION_GIT_SHA:
        raise U1SelectionError(
            f"The bound attempt executed at {lock.get('git_sha')!r}, not the "
            f"recorded {U1_EXECUTION_GIT_SHA!r}."
        )
    if lock.get("u1_protocol_sha256") != U1_PROTOCOL_SHA256:
        raise U1SelectionError("The bound attempt does not carry the U1 protocol.")
    if lock.get("repeat_attempt_permitted") is not False:
        raise U1SelectionError(
            "The bound attempt records that a repeat attempt is permitted; U1 "
            "cannot be rerun."
        )
    if lock.get("automatic_retry_performed") is not False:
        raise U1SelectionError("The bound attempt records an automatic retry.")
    if lock.get("fold_assignment_sha256") != U1_FOLD_ASSIGNMENT_SHA256:
        raise U1SelectionError(
            "The bound attempt's fold assignment is not the frozen one."
        )
    if dict(lock["u_star_dev"]).get("u_star") != U1_U_STAR_DEV:
        raise U1SelectionError("The bound u_star_dev is not the frozen value.")

    selection = _require_family_selection(run_dir)
    guards = _require_raised_guard(result)
    disagreements = _require_zero_disagreements(run_dir)
    calibrator = _require_retained_deployment_calibrator(run_dir)

    return {
        "decision_class": "u1_calibration_routing_retention_decision",
        "retention_decision_sha256": validate_u1_retention_decision(),
        "u1_protocol_sha256": U1_PROTOCOL_SHA256,
        "attempt_id": U1_RETAINED_ATTEMPT_ID,
        "experiment_identity": U1_RETAINED_EXPERIMENT_IDENTITY,
        "execution_git_sha": U1_EXECUTION_GIT_SHA,
        "attempt_status": status["status"],
        "canonical_verification": verification,
        "u1_result_sha256": U1_RESULT_SHA256,
        "u1_experiment_lock_sha256": U1_EXPERIMENT_LOCK_SHA256,
        "u1_experiment_lock_file_sha256": U1_EXPERIMENT_LOCK_FILE_SHA256,
        "oof_evidence_store_sha256": U1_OOF_EVIDENCE_STORE_SHA256,
        "fold_assignment_sha256": U1_FOLD_ASSIGNMENT_SHA256,
        "retained": {
            "calibration": U1_CALIBRATION_RETAINED,
            "calibrator_family": U1_RETAINED_CALIBRATOR_FAMILY,
            "oof_probabilities_for_development": (
                U1_OOF_PROBABILITIES_RETAINED_FOR_DEVELOPMENT
            ),
            "final_calibrator_for_unseen_subjects": (
                U1_FINAL_CALIBRATOR_RETAINED_FOR_UNSEEN_SUBJECTS
            ),
            "symmetric_window_router": U1_SYMMETRIC_WINDOW_ROUTER_RETAINED,
            "u_star_dev_as_final_router": U1_U_STAR_DEV_RETAINED_AS_FINAL_ROUTER,
            "u_star_deploy_as_final_router": (
                U1_U_STAR_DEPLOY_RETAINED_AS_FINAL_ROUTER
            ),
        },
        "selected_family": selection["selected_family"],
        "family_selection_basis": U1_FAMILY_SELECTION_BASIS,
        "classification_disagreements": disagreements,
        "raised_guard": U1_RAISED_GUARD,
        "asymmetric_abstention_ratio": guards["asymmetric_abstention_ratio"],
        "asymmetric_abstention_ratio_bound": U1_ASYMMETRIC_ABSTENTION_BOUND,
        "calibration_agreement_guard_raised": False,
        "accepted_sensitivity_at_u_star_dev": U1_ACCEPTED_SENSITIVITY_AT_U_STAR_DEV,
        "u_star_dev": U1_U_STAR_DEV,
        "u_star_deploy": U1_U_STAR_DEPLOY,
        "deployment_calibrator": {
            "a": calibrator["calibrator"]["a"],
            "b": calibrator["calibrator"]["b"],
            "purpose": U1_DEPLOYMENT_CALIBRATOR_PURPOSE,
            "in_sample_performance_is_evidence": (
                U1_DEPLOYMENT_CALIBRATOR_IN_SAMPLE_PERFORMANCE_IS_EVIDENCE
            ),
        },
        "downstream_development_input": U1_DOWNSTREAM_DEVELOPMENT_INPUT,
        "router_remains_frozen_evidence": U1_ROUTER_REMAINS_FROZEN_EVIDENCE,
        "alternative_coverage_point_selected_here": (
            U1_ALTERNATIVE_COVERAGE_POINT_SELECTED_HERE
        ),
        "new_routing_rule_introduced_here": U1_NEW_ROUTING_RULE_INTRODUCED_HERE,
        "selection_basis": U1_SELECTION_BASIS,
        "statistical_significance_claim": U1_STATISTICAL_SIGNIFICANCE_CLAIM,
        "test_accessed": U1_TEST_ACCESSED,
        "sealed_test_state": U1_SEALED_TEST_STATE,
        "automatic_u2_transition": U1_AUTOMATIC_U2_TRANSITION,
        "u1_rerun_permitted": U1_RERUN_PERMITTED,
    }
