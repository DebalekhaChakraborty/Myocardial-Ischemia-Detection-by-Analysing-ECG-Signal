"""Structural binding of the frozen U1 calibration / selective-routing protocol.

This module records a *protocol*, not a computation. It freezes the U1 design
constants, generates the deterministic subject-disjoint fold assignment, and
refuses any subject outside the permitted VALIDATION set -- which is how TEST
exclusion becomes provable rather than merely asserted.

It deliberately imports only the standard library. There is no scorer, no
replay, no model, no torch, no waveform reader and no run-artifact reader here,
so protocol validation cannot reach real development data even by accident.
Fitting a calibrator, producing a calibrated probability and choosing a routing
threshold all belong to the separate reviewed execution change set.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, NamedTuple, Sequence

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]

U1_PROTOCOL_NAME: Final = "U1_CALIBRATION_SELECTIVE_ROUTING_PROTOCOL_V1"
U1_PROTOCOL_PATH: Final = REPOSITORY_ROOT / "docs" / f"{U1_PROTOCOL_NAME}.md"
U1_PROTOCOL_SHA256: Final = (
    "d6235b477af278fe051822bdcccb54f985e4eceb0c6e92c1424f5e9d7d79b33b"
)

# ---------------------------------------------------------------------------
# Frozen upstream identities U1 consumes read-only
# ---------------------------------------------------------------------------
U1_M2_RETENTION_DECISION_SHA256: Final = (
    "da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47"
)
U1_M2_SUITE_SHA256: Final = (
    "8a6b0a1c64da72fc0f4573c742bef491b01b4eb8179f0759da3c537a01939a02"
)
U1_M2G_ARM_RESULT_SHA256: Final = (
    "a061d4d8c5211381c18baa228436bb9abc78b2f87f71fe4cab6ca71b2d15cf75"
)
U1_M2G_LOCK_SHA256: Final = (
    "5ac07d9f1ea3859e046c84fb91f22cee1bb20ef4857837b1c82fcd944dbf0fe8"
)
U1_SPLIT_SHA256: Final = (
    "66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7"
)

# ---------------------------------------------------------------------------
# Score identity: exactly one M2-G quantity may be calibrated
# ---------------------------------------------------------------------------
U1_CALIBRATION_INPUT_FIELD: Final = "score"
U1_CALIBRATION_INPUT_SEMANTICS: Final = (
    "uncalibrated sigmoid model score; not calibrated probability"
)
# The G4 memory-admission quantity is NOT confidence and is never calibrated.
U1_FORBIDDEN_CALIBRATION_INPUTS: Final = (
    "normal_evidence",
    "g4_normal_evidence",
    "memory_admission_threshold",
)
U1_CLASSIFICATION_THRESHOLD: Final = 0.7554003000259399
U1_NORMAL_EVIDENCE_THRESHOLD: Final = 0.0002997174742631614
U1_MAY_CHANGE_CLASSIFICATION_THRESHOLD: Final = False

# ---------------------------------------------------------------------------
# Partition and population
# ---------------------------------------------------------------------------
U1_PERMITTED_PARTITION: Final = "validation"
U1_CALIBRATION_POPULATION: Final = "primary_metric"
U1_PRIMARY_ROW_COUNT: Final = 473_897
U1_FULL_REPLAY_ROW_COUNT: Final = 492_904
U1_FULL_REPLAY_IS_METRIC_DENOMINATOR: Final = False

# Frozen VALIDATION subjects, canonical ascending order. Fold k holds out the
# k-th entry; the ordering is the entire fold-assignment rule.
U1_CALIBRATION_SUBJECTS: Final = (
    "ltstdb:s2004",
    "ltstdb:s2005",
    "ltstdb:s2019",
    "ltstdb:s2020",
    "ltstdb:s2023",
    "ltstdb:s2031",
    "ltstdb:s2057",
    "ltstdb:s2058",
    "ltstdb:s2059",
    "ltstdb:s3068",
    "ltstdb:s3072",
    "ltstdb:s3073",
)

U1_FOLD_DESIGN: Final = "leave_one_subject_out"
U1_FOLD_COUNT: Final = len(U1_CALIBRATION_SUBJECTS)
U1_FOLD_ASSIGNMENT_BASIS: Final = "frozen_subject_identity_ascending_only"

# ---------------------------------------------------------------------------
# Calibrator family and numerical safeguards
# ---------------------------------------------------------------------------
U1_PRIMARY_METHOD: Final = "platt_logistic_on_recovered_logit"
U1_COMPARATOR_METHOD: Final = "temperature_only_on_recovered_logit"
U1_SELECTION_CRITERION: Final = "pooled_out_of_fold_negative_log_likelihood"
U1_NLL_TIE_TOLERANCE: Final = 1e-4
U1_TIE_BREAK: Final = "simpler_nested_model"
U1_CLAMP_DELTA: Final = 1e-7
U1_SATURATED_FRACTION_REVIEW_BOUND: Final = 0.01
U1_TRUE_LOGITS_PERSISTED: Final = False
U1_NEURAL_UNCERTAINTY_MODEL: Final = False
# Family selection is out-of-fold only; the final all-subject fit may not
# reopen it, and a failed final fit stops rather than falling back.
U1_FAMILY_SELECTION_EVIDENCE: Final = "out_of_fold_only"
U1_FINAL_FIT_MAY_RESELECT_FAMILY: Final = False
U1_FINAL_FIT_FALLBACK_PERMITTED: Final = False

# ---------------------------------------------------------------------------
# Two distinct calibration artifacts (§7)
#   OOF evidence             = DEVELOPMENT evaluation
#   final all-validation fit = deployable configuration
# ---------------------------------------------------------------------------
U1_OOF_ARTIFACT: Final = "u1_oof_development_calibration"
U1_DEPLOY_ARTIFACT: Final = "u1_final_deployment_calibrator"
U1_CALIBRATION_ARTIFACTS: Final = (U1_OOF_ARTIFACT, U1_DEPLOY_ARTIFACT)

U1_OOF_CALIBRATOR_COUNT: Final = 12
U1_DEPLOY_CALIBRATOR_COUNT: Final = 1
U1_DEPLOY_FIT_SUBJECTS: Final = U1_CALIBRATION_SUBJECTS

# Every U1 DEVELOPMENT performance number comes from the OOF artifact.
U1_DEVELOPMENT_EVIDENCE_SOURCE: Final = U1_OOF_ARTIFACT
U1_DEPLOY_FIT_IS_EVALUATION: Final = False
U1_DEPLOY_IN_SAMPLE_PERFORMANCE_CLAIM_AUTHORISED: Final = False
U1_DEPLOY_ARTIFACT_PURPOSE: Final = (
    "unseen_subjects_and_separately_authorised_test_or_deployment_only"
)
# Later T1/T2 DEVELOPMENT work on VALIDATION subjects must use OOF (§14.1).
U1_DOWNSTREAM_DEVELOPMENT_CALIBRATION_SOURCE: Final = U1_OOF_ARTIFACT

# ---------------------------------------------------------------------------
# Uncertainty, risk and routing
# ---------------------------------------------------------------------------
U1_UNCERTAINTY_DEFINITION: Final = "calibrated_probability_of_frozen_decision_error"
U1_SECONDARY_UNCERTAINTY_DESCRIPTIVE_ONLY: Final = "calibrated_binary_entropy"
U1_RISK_DEFINITION: Final = "error_rate_of_frozen_decision_among_accepted"
U1_COVERAGE_GRID: Final = (
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
    1.00,
)
U1_RETAINED_COVERAGE: Final = 0.90
# c_star is an a-priori operational design assumption / reference operating
# point, NOT measured deployment capacity. Real cost is measured later in E1.
U1_RETAINED_COVERAGE_BASIS: Final = (
    "a_priori_operational_design_assumption_reference_operating_point"
)
U1_RETAINED_COVERAGE_IS_MEASURED_CAPACITY: Final = False

# The threshold is an explicit empirical order statistic, never a library
# quantile convention: k = ceil(c_star * N) guarantees achieved >= target.
U1_THRESHOLD_RULE: Final = "empirical_order_statistic_ceil_1_based"
U1_THRESHOLD_SORT_KEY: Final = ("uncertainty", "stable_id")
U1_THRESHOLD_ACCEPTANCE: Final = "u <= u_star"
U1_DEV_THRESHOLD_NAME: Final = "u_star_dev"
U1_DEPLOY_THRESHOLD_NAME: Final = "u_star_deploy"
U1_THRESHOLD_REPORT_FIELDS: Final = (
    "target_coverage",
    "achieved_coverage",
    "accepted_count",
    "threshold_tie_count",
)
U1_ASYMMETRIC_ABSTENTION_RATIO: Final = 3.0
U1_ACCEPTED_RISK_AGREEMENT_TOLERANCE: Final = 0.02
U1_ROUTING_THRESHOLD_CHOSEN_HERE: Final = False
U1_EPISODE_PERSISTENCE_IMPLEMENTED: Final = False

# ---------------------------------------------------------------------------
# Evidence conventions, reused from the frozen metrics protocol
# ---------------------------------------------------------------------------
U1_ECE_BIN_COUNT: Final = 15
U1_ECE_BINNINGS: Final = ("equal_width", "equal_mass")
# Equal-width: [b/15, (b+1)/15) with the FINAL bin closed so p == 1.0 is binned.
U1_ECE_EQUAL_WIDTH_FINAL_BIN_CLOSED: Final = True
# Equal-mass: contiguous groups from a stable (p, stable_id) sort, sizes
# differing by at most one. Never a library-default quantile.
U1_ECE_EQUAL_MASS_SORT_KEY: Final = ("calibrated_probability", "stable_id")
U1_ECE_LIBRARY_QUANTILE_PERMITTED: Final = False
U1_ECE_BIN_REPORT_FIELDS: Final = (
    "count",
    "minimum_probability",
    "maximum_probability",
    "mean_probability",
    "empirical_positive_fraction",
)

U1_BOOTSTRAP_REPLICATES: Final = 1000
U1_BOOTSTRAP_SEED: Final = 2026
U1_BOOTSTRAP_UNIT: Final = "subject"
# Intervals are conditional on the fitted OOF procedure unless a future
# protocol re-fits folds inside each replicate. Windows are not independent.
U1_BOOTSTRAP_CLAIM: Final = (
    "between_subject_variation_conditional_on_fitted_oof_calibration"
)
U1_BOOTSTRAP_REFITS_FOLDS_PER_REPLICATE: Final = False
U1_WINDOWS_ARE_INDEPENDENT_EVIDENCE: Final = False
U1_INFERENTIAL_UNIT: Final = "subject"
U1_INFERENTIAL_UNIT_COUNT: Final = 12

U1_COLD_START_STRATA: Final = ("0_5_minutes", "5_60_minutes", "over_60_minutes")

# ---------------------------------------------------------------------------
# Firewalls
# ---------------------------------------------------------------------------
U1_TEST_ACCESSED: Final = False
U1_SEALED_TEST_STATE: Final = "unopened"
U1_M2_RERUN_PERMITTED: Final = False
U1_AUTOMATIC_U2_TRANSITION: Final = False
U1_AUTOMATIC_RETENTION: Final = False


class U1ProtocolError(RuntimeError):
    """Raised when the frozen U1 protocol cannot be proven."""


class U1Fold(NamedTuple):
    """One leave-one-subject-out calibration fold."""

    fold_index: int
    held_out_subject: str
    fit_subjects: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_u1_protocol_document(path: Path = U1_PROTOCOL_PATH) -> str:
    """Verify the frozen U1 protocol document byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise U1ProtocolError(f"U1 protocol is missing at {document}.")
    digest = _sha256_file(document)
    if digest != U1_PROTOCOL_SHA256:
        raise U1ProtocolError(
            f"U1 protocol digest {digest} differs from the frozen "
            f"{U1_PROTOCOL_SHA256}. The protocol is immutable."
        )
    return digest


def require_calibration_subjects(subjects: object) -> tuple[str, ...]:
    """Refuse any subject outside the frozen VALIDATION calibration set.

    This is the TEST firewall in executable form: a TEST subject is not
    filtered out quietly, it raises.
    """
    if isinstance(subjects, str) or not hasattr(subjects, "__iter__"):
        raise U1ProtocolError("Calibration subjects must be an iterable of ids.")
    requested = tuple(str(subject) for subject in subjects)
    if not requested:
        raise U1ProtocolError("No calibration subjects were supplied.")

    permitted = set(U1_CALIBRATION_SUBJECTS)
    refused = sorted({s for s in requested if s not in permitted})
    if refused:
        raise U1ProtocolError(
            f"Subjects {refused} are not in the frozen U1 calibration set. "
            "Only the 12 VALIDATION subjects may enter calibration; TEST "
            "subjects are refused, never silently dropped."
        )
    duplicates = sorted({s for s in requested if requested.count(s) > 1})
    if duplicates:
        raise U1ProtocolError(f"Duplicate calibration subjects {duplicates}.")
    return requested


def assign_calibration_folds(
    subjects: object = U1_CALIBRATION_SUBJECTS,
) -> tuple[U1Fold, ...]:
    """Build the deterministic leave-one-subject-out fold assignment.

    The assignment depends only on frozen subject identity in ascending
    canonical order. No M2-G output, error, metric or prevalence is an input,
    and none is reachable from this module.
    """
    requested = require_calibration_subjects(subjects)
    if set(requested) != set(U1_CALIBRATION_SUBJECTS):
        missing = sorted(set(U1_CALIBRATION_SUBJECTS) - set(requested))
        raise U1ProtocolError(
            f"The fold assignment covers every permitted subject; {missing} are absent."
        )
    ordered = tuple(sorted(requested))
    if ordered != tuple(sorted(U1_CALIBRATION_SUBJECTS)):
        raise U1ProtocolError("The calibration subject set is not the frozen set.")

    return tuple(
        U1Fold(
            fold_index=index,
            held_out_subject=subject,
            fit_subjects=tuple(s for s in ordered if s != subject),
        )
        for index, subject in enumerate(ordered)
    )


def fold_assignment_digest(folds: tuple[U1Fold, ...]) -> str:
    """Canonical digest of a fold assignment, independent of any score."""
    payload = [
        {
            "fold_index": fold.fold_index,
            "held_out_subject": fold.held_out_subject,
            "fit_subjects": list(fold.fit_subjects),
        }
        for fold in folds
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_fold_assignment(folds: tuple[U1Fold, ...]) -> dict[str, Any]:
    """Prove the fold assignment is subject-disjoint and complete."""
    if len(folds) != U1_FOLD_COUNT:
        raise U1ProtocolError(f"Expected {U1_FOLD_COUNT} folds, received {len(folds)}.")
    held_out = [fold.held_out_subject for fold in folds]
    if sorted(held_out) != sorted(U1_CALIBRATION_SUBJECTS):
        raise U1ProtocolError("Every permitted subject must be held out exactly once.")
    for fold in folds:
        if fold.held_out_subject in fold.fit_subjects:
            raise U1ProtocolError(
                f"Fold {fold.fold_index} fits on its own held-out subject "
                f"{fold.held_out_subject!r}; the calibrator would evaluate a "
                "subject it was fitted on."
            )
        if set(fold.fit_subjects) | {fold.held_out_subject} != set(
            U1_CALIBRATION_SUBJECTS
        ):
            raise U1ProtocolError(
                f"Fold {fold.fold_index} does not partition the frozen set."
            )
    return {
        "fold_design": U1_FOLD_DESIGN,
        "fold_count": U1_FOLD_COUNT,
        "fold_assignment_basis": U1_FOLD_ASSIGNMENT_BASIS,
        "fold_assignment_sha256": fold_assignment_digest(folds),
        "every_subject_held_out_exactly_once": True,
        "no_fold_fits_on_its_held_out_subject": True,
    }


def validate_against_frozen_split(split_path: Path) -> dict[str, Any]:
    """Prove the permitted set is VALIDATION and disjoint from sealed TEST.

    Reads only the committed split manifest -- subject assignment, never a
    waveform, score, prediction or label.
    """
    manifest_path = Path(split_path)
    if not manifest_path.is_file():
        raise U1ProtocolError(f"No split manifest at {manifest_path}.")
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("split_sha256") != U1_SPLIT_SHA256:
        raise U1ProtocolError("The split manifest is not the frozen benchmark split.")
    if manifest.get("sealed_test_partition") is not True:
        raise U1ProtocolError("The split manifest does not seal the TEST partition.")

    partitions = manifest.get("partitions") or {}
    validation = set(partitions.get("validation", {}).get("subjects", []))
    test = set(partitions.get("test", {}).get("subjects", []))

    if set(U1_CALIBRATION_SUBJECTS) != validation:
        raise U1ProtocolError(
            "The frozen U1 calibration set is not exactly the VALIDATION set."
        )
    leaked = sorted(set(U1_CALIBRATION_SUBJECTS) & test)
    if leaked:
        raise U1ProtocolError(f"TEST subjects {leaked} appear in calibration.")

    return {
        "permitted_partition": U1_PERMITTED_PARTITION,
        "split_sha256": U1_SPLIT_SHA256,
        "sealed_test_partition": True,
        "calibration_subject_count": len(U1_CALIBRATION_SUBJECTS),
        "test_subject_count": len(test),
        "calibration_test_intersection": [],
        "test_accessed": U1_TEST_ACCESSED,
        "sealed_test_state": U1_SEALED_TEST_STATE,
    }


class U1RoutingThreshold(NamedTuple):
    """A routing threshold derived by the frozen empirical order statistic."""

    name: str
    u_star: float
    target_coverage: float
    achieved_coverage: float
    accepted_count: int
    threshold_tie_count: int
    eligible_count: int


def routing_threshold_rank(eligible_count: int, target_coverage: float) -> int:
    """The 1-based order-statistic rank `k = ceil(c_star * N)`.

    A `numpy`-style `'lower'` quantile can land one row short of the target --
    at the frozen PRIMARY size it yields 426507/473897 = 0.8999993669..., below
    0.90. Taking the ceiling makes achieved coverage `>= c_star` by
    construction.
    """
    if eligible_count <= 0:
        raise U1ProtocolError("A routing threshold needs at least one row.")
    if not 0.0 < target_coverage <= 1.0:
        raise U1ProtocolError(
            f"Target coverage {target_coverage!r} must lie in (0, 1]."
        )
    return max(1, math.ceil(target_coverage * eligible_count))


def select_routing_threshold(
    uncertainties: Sequence[float],
    stable_ids: Sequence[str],
    target_coverage: float,
    name: str = U1_DEV_THRESHOLD_NAME,
) -> U1RoutingThreshold:
    """Derive `u_star` by the frozen rule; acceptance is `u <= u_star`.

    Ties at `u_star` are all accepted, so achieved coverage may exceed the
    target -- never fall below it.
    """
    if name not in (U1_DEV_THRESHOLD_NAME, U1_DEPLOY_THRESHOLD_NAME):
        raise U1ProtocolError(f"Unknown routing threshold name {name!r}.")
    values = [float(value) for value in uncertainties]
    identities = [str(identity) for identity in stable_ids]
    if len(values) != len(identities):
        raise U1ProtocolError(
            "Every uncertainty needs its stable_id; the tie-break is not optional."
        )
    total = len(values)
    rank = routing_threshold_rank(total, target_coverage)

    ordered = sorted(zip(values, identities), key=lambda row: (row[0], row[1]))
    u_star = ordered[rank - 1][0]
    accepted = sum(1 for value in values if value <= u_star)
    ties = sum(1 for value in values if value == u_star)
    achieved = accepted / total
    if achieved < target_coverage:
        raise U1ProtocolError(
            f"Achieved coverage {achieved!r} fell below target "
            f"{target_coverage!r}; the order statistic is wrong."
        )
    return U1RoutingThreshold(
        name=name,
        u_star=u_star,
        target_coverage=float(target_coverage),
        achieved_coverage=achieved,
        accepted_count=accepted,
        threshold_tie_count=ties,
        eligible_count=total,
    )


def equal_width_bin_edges(
    bins: int = U1_ECE_BIN_COUNT,
) -> tuple[tuple[float, float], ...]:
    """Frozen equal-width ECE intervals over [0, 1].

    Lower edge inclusive, upper exclusive, except the final bin whose upper
    edge is inclusive so `p == 1.0` is always binned.
    """
    if bins <= 0:
        raise U1ProtocolError("ECE bin count must be positive.")
    return tuple((index / bins, (index + 1) / bins) for index in range(bins))


def equal_width_bin_index(probability: float, bins: int = U1_ECE_BIN_COUNT) -> int:
    """Frozen equal-width bin membership for one calibrated probability."""
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise U1ProtocolError(f"Calibrated probability {value!r} is outside [0, 1].")
    if value == 1.0:
        return bins - 1
    return min(bins - 1, int(value * bins))


def equal_mass_group_sizes(
    eligible_count: int, bins: int = U1_ECE_BIN_COUNT
) -> tuple[int, ...]:
    """Contiguous equal-mass group sizes differing by at most one.

    With `N = bins * q + r`, the first `r` groups hold `q + 1` rows and the
    rest hold `q`. Library quantile defaults are never used.
    """
    if bins <= 0:
        raise U1ProtocolError("ECE bin count must be positive.")
    if eligible_count < bins:
        raise U1ProtocolError(
            f"{eligible_count} rows cannot fill {bins} equal-mass groups."
        )
    quotient, remainder = divmod(eligible_count, bins)
    return tuple(
        quotient + 1 if index < remainder else quotient for index in range(bins)
    )


def equal_mass_group_boundaries(
    probabilities: Sequence[float],
    stable_ids: Sequence[str],
    bins: int = U1_ECE_BIN_COUNT,
) -> tuple[tuple[int, int], ...]:
    """Half-open [start, stop) index ranges over the stable (p, id) order."""
    if len(probabilities) != len(stable_ids):
        raise U1ProtocolError(
            "Every probability needs its stable_id; ties spanning a group "
            "boundary are resolved by stable_id."
        )
    sizes = equal_mass_group_sizes(len(probabilities), bins)
    boundaries: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        boundaries.append((start, start + size))
        start += size
    return tuple(boundaries)


def u1_protocol_identity() -> dict[str, Any]:
    """The frozen U1 design, as a machine-readable record."""
    folds = assign_calibration_folds()
    return {
        "protocol_sha256": validate_u1_protocol_document(),
        "protocol_name": U1_PROTOCOL_NAME,
        "m2_retention_decision_sha256": U1_M2_RETENTION_DECISION_SHA256,
        "m2_suite_sha256": U1_M2_SUITE_SHA256,
        "m2g_arm_result_sha256": U1_M2G_ARM_RESULT_SHA256,
        "m2g_lock_sha256": U1_M2G_LOCK_SHA256,
        "calibration_input_field": U1_CALIBRATION_INPUT_FIELD,
        "forbidden_calibration_inputs": list(U1_FORBIDDEN_CALIBRATION_INPUTS),
        "classification_threshold": U1_CLASSIFICATION_THRESHOLD,
        "may_change_classification_threshold": (U1_MAY_CHANGE_CLASSIFICATION_THRESHOLD),
        "permitted_partition": U1_PERMITTED_PARTITION,
        "calibration_population": U1_CALIBRATION_POPULATION,
        "primary_method": U1_PRIMARY_METHOD,
        "comparator_method": U1_COMPARATOR_METHOD,
        "selection_criterion": U1_SELECTION_CRITERION,
        "tie_break": U1_TIE_BREAK,
        "true_logits_persisted": U1_TRUE_LOGITS_PERSISTED,
        "clamp_delta": U1_CLAMP_DELTA,
        "saturated_fraction_review_bound": U1_SATURATED_FRACTION_REVIEW_BOUND,
        "uncertainty_definition": U1_UNCERTAINTY_DEFINITION,
        "risk_definition": U1_RISK_DEFINITION,
        "coverage_grid": list(U1_COVERAGE_GRID),
        "retained_coverage": U1_RETAINED_COVERAGE,
        "retained_coverage_basis": U1_RETAINED_COVERAGE_BASIS,
        "retained_coverage_is_measured_capacity": (
            U1_RETAINED_COVERAGE_IS_MEASURED_CAPACITY
        ),
        "threshold_rule": U1_THRESHOLD_RULE,
        "threshold_names": [U1_DEV_THRESHOLD_NAME, U1_DEPLOY_THRESHOLD_NAME],
        "routing_threshold_chosen_here": U1_ROUTING_THRESHOLD_CHOSEN_HERE,
        "calibration_artifacts": list(U1_CALIBRATION_ARTIFACTS),
        "development_evidence_source": U1_DEVELOPMENT_EVIDENCE_SOURCE,
        "downstream_development_calibration_source": (
            U1_DOWNSTREAM_DEVELOPMENT_CALIBRATION_SOURCE
        ),
        "deploy_fit_is_evaluation": U1_DEPLOY_FIT_IS_EVALUATION,
        "deploy_in_sample_performance_claim_authorised": (
            U1_DEPLOY_IN_SAMPLE_PERFORMANCE_CLAIM_AUTHORISED
        ),
        "family_selection_evidence": U1_FAMILY_SELECTION_EVIDENCE,
        "final_fit_may_reselect_family": U1_FINAL_FIT_MAY_RESELECT_FAMILY,
        "final_fit_fallback_permitted": U1_FINAL_FIT_FALLBACK_PERMITTED,
        "bootstrap_claim": U1_BOOTSTRAP_CLAIM,
        "windows_are_independent_evidence": U1_WINDOWS_ARE_INDEPENDENT_EVIDENCE,
        "inferential_unit": U1_INFERENTIAL_UNIT,
        "inferential_unit_count": U1_INFERENTIAL_UNIT_COUNT,
        "fold_assignment": validate_fold_assignment(folds),
        "test_accessed": U1_TEST_ACCESSED,
        "sealed_test_state": U1_SEALED_TEST_STATE,
        "m2_rerun_permitted": U1_M2_RERUN_PERMITTED,
        "automatic_u2_transition": U1_AUTOMATIC_U2_TRANSITION,
        "automatic_retention": U1_AUTOMATIC_RETENTION,
    }
