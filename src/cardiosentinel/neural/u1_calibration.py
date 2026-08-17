"""The frozen U1 numerics: calibration, uncertainty, routing and guards.

This module is the arithmetic of the U1 protocol and nothing else. It opens no
run artifact, resolves no partition path, reads no cache and writes no file, so
it can be exercised end to end on synthetic arrays. Claiming an attempt,
consuming the retained M2-G evidence and persisting results all belong to
`u1_development_run` and `u1_persistence`.

**The frozen rules live in `u1_protocol`, not here.** The order-statistic rank,
the equal-mass ordering, the equal-width bin edges, the coverage grid, the clamp
and the fold assignment are imported from the protocol binder so the execution
route cannot drift from the reviewed document. Where a stdlib protocol helper
would be quadratic or otherwise unexecutable at the frozen PRIMARY size, the
vectorised path here is proven equal to it by test, never substituted for it
silently.

**Recovered logits, not true logits.** The retained head emits
`single_raw_logit`, but only `sigmoid(logit)` is persisted and the head is
float32, so `z = log(p) - log1p(-p)` is a quantized reconstruction that
saturates. Every artifact this module produces calls the comparator
*approximate temperature-only*, never temperature scaling. U1 measures that
limitation (`saturation_census`); it never repairs it, and it never re-runs M2.

**Nothing here selects a threshold from results.** `tau` is frozen upstream,
`c_star` is frozen prospectively, and the two reporting guards raise flags for a
human -- they never re-derive, re-fit or re-select anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np

from cardiosentinel.neural.u1_protocol import (
    U1_ACCEPTED_RISK_AGREEMENT_TOLERANCE,
    U1_ASYMMETRIC_ABSTENTION_RATIO,
    U1_BOOTSTRAP_CLAIM,
    U1_BOOTSTRAP_REPLICATES,
    U1_BOOTSTRAP_SEED,
    U1_BOOTSTRAP_UNIT,
    U1_CLAMP_DELTA,
    U1_CLASSIFICATION_THRESHOLD,
    U1_COMPARATOR_METHOD,
    U1_COVERAGE_GRID,
    U1_DEV_THRESHOLD_NAME,
    U1_ECE_BIN_COUNT,
    U1_NLL_TIE_TOLERANCE,
    U1_PRIMARY_METHOD,
    U1_RETAINED_COVERAGE,
    U1_SATURATED_FRACTION_REVIEW_BOUND,
    U1_TIE_BREAK,
    equal_mass_groups,
    equal_width_bin_edges,
    equal_width_bin_index,
    routing_threshold_rank,
    select_routing_threshold,
)

FAMILY_PLATT: Final = U1_PRIMARY_METHOD
FAMILY_TEMPERATURE: Final = U1_COMPARATOR_METHOD
U1_FAMILIES: Final = (FAMILY_PLATT, FAMILY_TEMPERATURE)

U1_OPTIMIZER: Final = "L-BFGS-B"
U1_OPTIMIZER_MAXITER: Final = 500
U1_OPTIMIZER_GTOL: Final = 1e-10
U1_OPTIMIZER_INITIAL_A: Final = 1.0
U1_OPTIMIZER_INITIAL_B: Final = 0.0
U1_OPTIMIZER_RETRY_PERMITTED: Final = False
U1_OPTIMIZER_FALLBACK_PERMITTED: Final = False

U1_DTYPE: Final = "float64"


class U1CalibrationError(RuntimeError):
    """Raised when the frozen U1 numerics cannot proceed with integrity.

    Every raise on this path is a STOP: there is no retry, no alternate
    initialisation, no optimiser substitution and no fallback family.
    """


# --------------------------------------------------------------------------
# Input conditioning
# --------------------------------------------------------------------------


def _float64(values: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise U1CalibrationError(f"{label} must be one-dimensional.")
    if array.shape[0] == 0:
        raise U1CalibrationError(f"{label} is empty.")
    if not bool(np.all(np.isfinite(array))):
        raise U1CalibrationError(
            f"{label} contains a non-finite value. NaN is refused before any "
            "ordering or fit; it is never imputed."
        )
    return array


def _labels(values: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or array.shape[0] == 0:
        raise U1CalibrationError("Labels must be a non-empty one-dimensional array.")
    unexpected = sorted(set(array.tolist()) - {0, 1})
    if unexpected:
        raise U1CalibrationError(f"Labels are not binary: {unexpected}.")
    return array


def _require_aligned(**columns: np.ndarray) -> int:
    lengths = {
        name: int(np.asarray(values).shape[0]) for name, values in columns.items()
    }
    if len(set(lengths.values())) != 1:
        raise U1CalibrationError(f"Columns are not row-aligned: {lengths}.")
    return next(iter(lengths.values()))


def clamp_probabilities(
    values: Sequence[float] | np.ndarray, *, delta: float = U1_CLAMP_DELTA
) -> np.ndarray:
    """`clip(p, delta, 1 - delta)`, the one frozen clamp, in float64."""
    return np.clip(_float64(values, "probability"), delta, 1.0 - delta)


def recover_logits(
    scores: Sequence[float] | np.ndarray, *, delta: float = U1_CLAMP_DELTA
) -> np.ndarray:
    """`z = log(p_clamped) - log1p(-p_clamped)`.

    This is a RECOVERED logit, not the head's true logit: only the float32
    sigmoid was persisted, so `z` is quantized and saturates. No U1 output may
    describe a fit on `z` as true-logit temperature scaling.
    """
    clamped = clamp_probabilities(scores, delta=delta)
    return np.log(clamped) - np.log1p(-clamped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Overflow-free logistic. Identical to `1/(1+exp(-t))` where that is finite."""
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


# --------------------------------------------------------------------------
# §3.1 Saturation census -- a precondition, never a scientific result
# --------------------------------------------------------------------------


def saturation_census(
    scores: Sequence[float] | np.ndarray,
    *,
    delta: float = U1_CLAMP_DELTA,
    review_bound: float = U1_SATURATED_FRACTION_REVIEW_BOUND,
) -> dict[str, Any]:
    """Count exactly what the frozen protocol asks, before anything is fitted.

    `within_review_bound=False` means the run STOPS and fits nothing. It never
    widens the clamp, never substitutes a calibrator and never re-runs M2.
    """
    values = _float64(scores, "score")
    total = int(values.shape[0])
    zero = int(np.count_nonzero(values == 0.0))
    one = int(np.count_nonzero(values == 1.0))
    outside = int(np.count_nonzero((values < delta) | (values > 1.0 - delta)))
    distinct = int(np.unique(values).shape[0])
    outside_fraction = outside / total
    return {
        "census_class": "u1_saturation_census",
        "is_scientific_result": False,
        "population_row_count": total,
        "clamp_delta": float(delta),
        "score_equal_zero_count": zero,
        "score_equal_zero_fraction": zero / total,
        "score_equal_one_count": one,
        "score_equal_one_fraction": one / total,
        "score_outside_clamp_count": outside,
        "score_outside_clamp_fraction": outside_fraction,
        "distinct_persisted_score_count": distinct,
        "saturated_fraction_review_bound": float(review_bound),
        "within_review_bound": bool(outside_fraction <= review_bound),
        "true_logits_persisted": False,
        "clamp_widened": False,
        "m2_rerun_performed": False,
    }


# --------------------------------------------------------------------------
# §6 Calibrator families and the frozen deterministic fit
# --------------------------------------------------------------------------


def require_family(family: str) -> str:
    if family not in U1_FAMILIES:
        raise U1CalibrationError(
            f"Unknown calibrator family {family!r}; the frozen families are "
            f"{list(U1_FAMILIES)}."
        )
    return family


@dataclass(frozen=True, slots=True)
class U1Calibrator:
    """One fitted monotonic calibrator `g(s) = sigmoid(a * z(s) + b)`.

    `b` is structurally zero for the temperature-only family: the comparator is
    the nested special case, not a Platt fit whose intercept happened to land
    near zero.
    """

    family: str
    a: float
    b: float
    clamp_delta: float
    fit_row_count: int
    fit_subjects: tuple[str, ...]
    optimizer: dict[str, Any]

    def __post_init__(self) -> None:
        require_family(self.family)
        if self.family == FAMILY_TEMPERATURE and self.b != 0.0:
            raise U1CalibrationError(
                "The temperature-only family fixes b = 0; it is the nested "
                f"special case, but this record carries b = {self.b!r}."
            )
        if not math.isfinite(self.a) or not math.isfinite(self.b):
            raise U1CalibrationError("A fitted calibrator parameter is non-finite.")
        if self.a <= 0.0:
            raise U1CalibrationError(
                f"The fitted map is not strictly increasing (a = {self.a!r}). "
                "Decision equivalence depends on monotonicity, so this is a "
                "hard failure, never a re-initialisation or a substitution."
            )

    @property
    def parameter_count(self) -> int:
        return 2 if self.family == FAMILY_PLATT else 1

    def apply_to_logits(self, logits: Sequence[float] | np.ndarray) -> np.ndarray:
        """Calibrated probabilities for already-recovered logits."""
        return _sigmoid(self.a * _float64(logits, "recovered logit") + self.b)

    def apply_to_scores(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        """Calibrated probabilities straight from persisted M2-G scores."""
        return self.apply_to_logits(recover_logits(scores, delta=self.clamp_delta))

    def calibrated_boundary(
        self, threshold: float = U1_CLASSIFICATION_THRESHOLD
    ) -> float:
        """`pi = g(tau)`, persisted DESCRIPTIVELY to prove decision equivalence.

        This is not a newly selected classifier threshold: the decision remains
        `score >= tau` on the raw persisted score.
        """
        return float(self.apply_to_scores(np.asarray([float(threshold)]))[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "a": float(self.a),
            "b": float(self.b),
            "parameter_count": self.parameter_count,
            "intercept_fixed_at_zero": self.family == FAMILY_TEMPERATURE,
            "clamp_delta": float(self.clamp_delta),
            "fit_row_count": int(self.fit_row_count),
            "fit_subjects": list(self.fit_subjects),
            "fit_subject_count": len(self.fit_subjects),
            "optimizer": dict(self.optimizer),
            "monotonic_increasing": True,
            "is_true_logit_temperature_scaling": False,
            "dtype": U1_DTYPE,
        }


def _objective_and_gradient(
    parameters: np.ndarray,
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    free_intercept: bool,
    delta: float,
) -> tuple[float, np.ndarray]:
    """Mean clamped NLL and its exact gradient.

    The clamp is applied inside the objective exactly as §6.2 requires, so the
    gradient returned is the gradient of THAT objective: a row whose calibrated
    probability has been clamped contributes nothing, because the clamped
    objective is locally constant in the parameters there. Returning the
    unclamped gradient instead would hand L-BFGS-B a derivative for a function
    it is not minimising.
    """
    scale = float(parameters[0])
    intercept = float(parameters[1]) if free_intercept else 0.0
    activation = scale * logits + intercept
    probability = _sigmoid(activation)
    clamped = np.clip(probability, delta, 1.0 - delta)
    total = float(
        -np.mean(labels * np.log(clamped) + (1.0 - labels) * np.log1p(-clamped))
    )
    active = (probability > delta) & (probability < 1.0 - delta)
    residual = np.where(active, probability - labels, 0.0)
    rows = float(labels.shape[0])
    gradient = [float(np.sum(residual * logits) / rows)]
    if free_intercept:
        gradient.append(float(np.sum(residual) / rows))
    return total, np.asarray(gradient, dtype=np.float64)


def fit_calibrator(
    *,
    logits: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    family: str,
    fit_subjects: Sequence[str] = (),
    delta: float = U1_CLAMP_DELTA,
) -> U1Calibrator:
    """The frozen deterministic maximum-likelihood fit.

    L-BFGS-B, float64, initial `(a, b) = (1.0, 0.0)`, `maxiter = 500`,
    `gtol = 1e-10`, analytic gradient. Non-convergence, a non-finite parameter
    and a non-monotonic map (`a <= 0`) are hard failures.

    `a` is deliberately UNBOUNDED. Constraining `a > 0` in the optimiser would
    park a degenerate fit on the bound and hide exactly the non-monotonicity
    §6.2 requires be detected.
    """
    from scipy.optimize import minimize

    require_family(family)
    recovered = _float64(logits, "recovered logit")
    outcomes = _labels(labels)
    _require_aligned(logits=recovered, labels=outcomes)
    targets = outcomes.astype(np.float64)
    free_intercept = family == FAMILY_PLATT
    start = (
        np.asarray([U1_OPTIMIZER_INITIAL_A, U1_OPTIMIZER_INITIAL_B], dtype=np.float64)
        if free_intercept
        else np.asarray([U1_OPTIMIZER_INITIAL_A], dtype=np.float64)
    )

    result = minimize(
        lambda parameters: _objective_and_gradient(
            parameters,
            logits=recovered,
            labels=targets,
            free_intercept=free_intercept,
            delta=delta,
        ),
        start,
        method=U1_OPTIMIZER,
        jac=True,
        options={"maxiter": U1_OPTIMIZER_MAXITER, "gtol": U1_OPTIMIZER_GTOL},
    )

    optimizer = {
        "method": U1_OPTIMIZER,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "function_evaluations": int(getattr(result, "nfev", -1)),
        "objective": float(result.fun),
        "maxiter": U1_OPTIMIZER_MAXITER,
        "gtol": U1_OPTIMIZER_GTOL,
        "initial_a": U1_OPTIMIZER_INITIAL_A,
        "initial_b": U1_OPTIMIZER_INITIAL_B,
        "bounds_applied": False,
        "automatic_retry_performed": U1_OPTIMIZER_RETRY_PERMITTED,
        "automatic_fallback_performed": U1_OPTIMIZER_FALLBACK_PERMITTED,
        "dtype": U1_DTYPE,
    }
    if not bool(result.success):
        raise U1CalibrationError(
            f"The {family} fit did not converge: status {int(result.status)}, "
            f"{str(result.message)!r}. STOP FOR HUMAN REVIEW. There is no "
            "retry, no re-initialisation, no optimiser substitution and no "
            "fallback to the other family."
        )
    parameters = np.asarray(result.x, dtype=np.float64)
    if not bool(np.all(np.isfinite(parameters))):
        raise U1CalibrationError(
            f"The {family} fit produced a non-finite parameter {parameters!r}. "
            "STOP FOR HUMAN REVIEW."
        )
    return U1Calibrator(
        family=family,
        a=float(parameters[0]),
        b=float(parameters[1]) if free_intercept else 0.0,
        clamp_delta=float(delta),
        fit_row_count=int(outcomes.shape[0]),
        fit_subjects=tuple(str(subject) for subject in fit_subjects),
        optimizer=optimizer,
    )


# --------------------------------------------------------------------------
# §6.1 Family selection -- pooled OOF NLL and nothing else
# --------------------------------------------------------------------------


def select_calibrator_family(
    *,
    platt_pooled_oof_nll: float,
    temperature_pooled_oof_nll: float,
    tie_tolerance: float = U1_NLL_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Choose the retained family from pooled OOF NLL alone.

    The signature is the safeguard: ECE, Brier, AUPRC, routing risk, challenge
    evidence, TEST and any weighted combination are not parameters, so they
    cannot influence the decision even by mistake. Within `tie_tolerance` the
    simpler nested model wins -- simplicity breaks ties, never flexibility.
    """
    platt = float(platt_pooled_oof_nll)
    temperature = float(temperature_pooled_oof_nll)
    if not math.isfinite(platt) or not math.isfinite(temperature):
        raise U1CalibrationError(
            f"Pooled OOF NLL must be finite; received {platt!r} and {temperature!r}."
        )
    difference = platt - temperature
    tied = abs(difference) < float(tie_tolerance)
    if tied:
        selected = FAMILY_TEMPERATURE
        basis = "tie_within_tolerance_simpler_nested_model"
    elif platt < temperature:
        selected = FAMILY_PLATT
        basis = "lower_pooled_out_of_fold_nll"
    else:
        selected = FAMILY_TEMPERATURE
        basis = "lower_pooled_out_of_fold_nll"
    return {
        "decision_class": "u1_calibrator_family_selection",
        "criterion": "pooled_out_of_fold_negative_log_likelihood",
        "evidence_source": "out_of_fold_only",
        "pooled_oof_nll": {FAMILY_PLATT: platt, FAMILY_TEMPERATURE: temperature},
        "nll_difference_platt_minus_temperature": difference,
        "nll_tie_tolerance": float(tie_tolerance),
        "tie_within_tolerance": bool(tied),
        "tie_break": U1_TIE_BREAK,
        "selected_family": selected,
        "selection_basis": basis,
        "ece_used": False,
        "brier_used": False,
        "auprc_used": False,
        "routing_risk_used": False,
        "challenge_evidence_used": False,
        "weighted_score_used": False,
        "test_accessed": False,
        "is_u1_retention_decision": False,
        "selection_semantics": (
            "calibrator-family decision required by the frozen protocol; the "
            "human decision whether U1 itself is retained is separate"
        ),
    }


# --------------------------------------------------------------------------
# §8 / §14 The frozen decision, proven rather than asserted
# --------------------------------------------------------------------------


def frozen_decisions(
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float = U1_CLASSIFICATION_THRESHOLD,
) -> np.ndarray:
    """`y_hat = score >= tau`, on the RAW persisted score. Never recomputed."""
    if float(threshold) != U1_CLASSIFICATION_THRESHOLD:
        raise U1CalibrationError(
            f"The frozen classification threshold is "
            f"{U1_CLASSIFICATION_THRESHOLD!r}; U1 may not classify at "
            f"{threshold!r}."
        )
    return _float64(scores, "score") >= float(threshold)


def prove_decision_equivalence(
    *,
    scores: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    calibrated_boundary: float,
    threshold: float = U1_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Prove `score >= tau  <==>  g(score) >= g(tau)` row for row.

    Monotonicity makes this true in exact arithmetic; this checks it actually
    held in float64 on the real rows, and raises if it did not. No new
    threshold is optimised and no row is reclassified.
    """
    raw = frozen_decisions(scores, threshold=threshold)
    calibrated = _float64(probabilities, "calibrated probability") >= float(
        calibrated_boundary
    )
    _require_aligned(scores=raw, probabilities=calibrated)
    disagreements = int(np.count_nonzero(raw != calibrated))
    if disagreements:
        first = int(np.flatnonzero(raw != calibrated)[0])
        raise U1CalibrationError(
            f"The calibrated boundary disagrees with the frozen decision on "
            f"{disagreements} rows, first at position {first}. Calibration may "
            "never redefine the ischemia classifier."
        )
    return {
        "proof_class": "u1_frozen_decision_equivalence",
        "classification_threshold": float(threshold),
        "calibrated_boundary": float(calibrated_boundary),
        "calibrated_boundary_is_a_new_threshold": False,
        "threshold_selected_here": False,
        "row_count": int(raw.shape[0]),
        "disagreement_count": 0,
        "row_for_row_identical": True,
        "predicted_positive_count": int(np.count_nonzero(raw)),
    }


# --------------------------------------------------------------------------
# §9 Uncertainty
# --------------------------------------------------------------------------


def uncertainty_from_decision(
    *,
    probabilities: Sequence[float] | np.ndarray,
    decisions: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """`u = 1 - p` where the frozen decision is positive, else `u = p`.

    The calibrated probability that the FROZEN decision is wrong. A raw sigmoid
    output is never called uncertainty.
    """
    calibrated = _float64(probabilities, "calibrated probability")
    if np.any(calibrated < 0.0) or np.any(calibrated > 1.0):
        raise U1CalibrationError("A calibrated probability lies outside [0, 1].")
    positive = np.asarray(decisions, dtype=bool)
    _require_aligned(probabilities=calibrated, decisions=positive)
    return np.where(positive, 1.0 - calibrated, calibrated)


def binary_entropy(
    probabilities: Sequence[float] | np.ndarray, *, delta: float = U1_CLAMP_DELTA
) -> np.ndarray:
    """`H(p)` in nats. DESCRIPTIVE ONLY -- never the retained routing rule."""
    clamped = clamp_probabilities(probabilities, delta=delta)
    return -(clamped * np.log(clamped) + (1.0 - clamped) * np.log1p(-clamped))


# --------------------------------------------------------------------------
# §10.2 / §10.3 Calibration evidence
# --------------------------------------------------------------------------


def brier_score(
    labels: Sequence[int] | np.ndarray, probabilities: Sequence[float] | np.ndarray
) -> float:
    outcomes = _labels(labels).astype(np.float64)
    calibrated = _float64(probabilities, "probability")
    _require_aligned(labels=outcomes, probabilities=calibrated)
    return float(np.mean((calibrated - outcomes) ** 2))


def negative_log_likelihood(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    delta: float = U1_CLAMP_DELTA,
) -> float:
    """`-mean[y ln p + (1 - y) ln(1 - p)]` with the frozen clamp."""
    outcomes = _labels(labels).astype(np.float64)
    clamped = clamp_probabilities(probabilities, delta=delta)
    _require_aligned(labels=outcomes, probabilities=clamped)
    return float(
        -np.mean(outcomes * np.log(clamped) + (1.0 - outcomes) * np.log1p(-clamped))
    )


def _bin_report(
    labels: np.ndarray, probabilities: np.ndarray, members: np.ndarray, index: int
) -> dict[str, Any]:
    selected = probabilities[members]
    outcomes = labels[members]
    count = int(selected.shape[0])
    return {
        "bin_index": index,
        "count": count,
        "minimum_probability": float(np.min(selected)) if count else None,
        "maximum_probability": float(np.max(selected)) if count else None,
        "mean_probability": float(np.mean(selected)) if count else None,
        "empirical_positive_fraction": (
            float(np.mean(outcomes.astype(np.float64))) if count else None
        ),
    }


def _expected_calibration_error(bins: Sequence[dict[str, Any]], total: int) -> float:
    return float(
        sum(
            (entry["count"] / total)
            * abs(entry["empirical_positive_fraction"] - entry["mean_probability"])
            for entry in bins
            if entry["count"]
        )
    )


def _equal_width_assign(probabilities: np.ndarray, bins: int) -> np.ndarray:
    """Vectorised equal-width membership. Proven against the frozen scalar rule."""
    assigned = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    assigned[probabilities == 1.0] = bins - 1
    return assigned


def _assert_equal_width_parity(bins: int, probabilities: np.ndarray) -> None:
    """Prove the vectorised placement equals `equal_width_bin_index` exactly.

    The probes are every frozen bin edge -- where the inclusive/exclusive
    convention actually bites -- plus `1.0` and the extremes present in the
    data. The frozen scalar rule stays the authority; this only licenses the
    vectorised path used on 473,897 rows.
    """
    probes = np.asarray(
        [index / bins for index in range(bins + 1)]
        + [1.0, float(np.min(probabilities)), float(np.max(probabilities))],
        dtype=np.float64,
    )
    vectorised = _equal_width_assign(probes, bins)
    for position, probe in enumerate(probes.tolist()):
        frozen = equal_width_bin_index(probe, bins)
        if int(vectorised[position]) != frozen:
            raise U1CalibrationError(
                "Vectorised equal-width binning disagrees with the frozen "
                f"protocol rule at p = {probe!r}: {int(vectorised[position])} "
                f"vs {frozen}."
            )


def equal_width_reliability(
    *,
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    bins: int = U1_ECE_BIN_COUNT,
) -> dict[str, Any]:
    """Frozen equal-width ECE: 15 intervals, final bin closed so `p = 1.0` bins.

    Membership comes from the protocol binder's own `equal_width_bin_index`, so
    the endpoint convention cannot drift from the reviewed document.
    """
    outcomes = _labels(labels)
    calibrated = _float64(probabilities, "calibrated probability")
    total = _require_aligned(labels=outcomes, probabilities=calibrated)
    if np.any(calibrated < 0.0) or np.any(calibrated > 1.0):
        raise U1CalibrationError("A calibrated probability lies outside [0, 1].")

    edges = equal_width_bin_edges(bins)
    _assert_equal_width_parity(bins, calibrated)
    assigned = _equal_width_assign(calibrated, bins)
    reported = [
        _bin_report(outcomes, calibrated, np.flatnonzero(assigned == index), index)
        for index in range(bins)
    ]
    for index, entry in enumerate(reported):
        entry["lower_edge"] = edges[index][0]
        entry["upper_edge"] = edges[index][1]
        entry["upper_edge_inclusive"] = index == bins - 1
    return {
        "binning": "equal_width",
        "bin_count": bins,
        "row_count": total,
        "bins": reported,
        "expected_calibration_error": _expected_calibration_error(reported, total),
        "library_quantile_used": False,
    }


def equal_mass_reliability(
    *,
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    stable_ids: Sequence[str],
    bins: int = U1_ECE_BIN_COUNT,
) -> dict[str, Any]:
    """Frozen equal-mass ECE, built by the protocol binder's own grouping.

    `equal_mass_groups()` performs the frozen `(p, stable_id)` sort itself and
    returns explicit membership, so no ordering is invented here and unsorted
    rows cannot silently enter a bin claiming to be a frozen one.
    """
    outcomes = _labels(labels)
    calibrated = _float64(probabilities, "calibrated probability")
    identities = [str(value) for value in stable_ids]
    total = _require_aligned(
        labels=outcomes,
        probabilities=calibrated,
        stable_ids=np.asarray(identities, dtype=object),
    )

    groups = equal_mass_groups(calibrated.tolist(), identities, bins)
    reported: list[dict[str, Any]] = []
    for group in groups:
        members = np.asarray(group.member_indices, dtype=np.int64)
        entry = _bin_report(outcomes, calibrated, members, group.group_index)
        entry["minimum_probability"] = float(group.minimum_probability)
        entry["maximum_probability"] = float(group.maximum_probability)
        reported.append(entry)
    covered = sum(entry["count"] for entry in reported)
    if covered != total:
        raise U1CalibrationError(
            f"Equal-mass groups cover {covered} of {total} rows; the frozen "
            "grouping partitions every eligible row exactly once."
        )
    return {
        "binning": "equal_mass",
        "bin_count": bins,
        "row_count": total,
        "group_sizes": [entry["count"] for entry in reported],
        "bins": reported,
        "expected_calibration_error": _expected_calibration_error(reported, total),
        "sort_key": ["calibrated_probability", "stable_id"],
        "library_quantile_used": False,
    }


def calibration_evidence(
    *,
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    stable_ids: Sequence[str],
    name: str,
    is_out_of_fold: bool,
    delta: float = U1_CLAMP_DELTA,
) -> dict[str, Any]:
    """Brier, NLL, both ECEs and reliability evidence for one probability set.

    `is_out_of_fold` is carried into the artifact because §7 makes it the whole
    difference between DEVELOPMENT evidence and deployable configuration.
    """
    return {
        "evidence_class": "u1_calibration_evidence",
        "name": str(name),
        "out_of_fold": bool(is_out_of_fold),
        "development_evidence": bool(is_out_of_fold),
        "row_count": int(np.asarray(probabilities).shape[0]),
        "clamp_delta": float(delta),
        "brier": brier_score(labels, probabilities),
        "negative_log_likelihood": negative_log_likelihood(
            labels, probabilities, delta=delta
        ),
        "reliability_equal_width": equal_width_reliability(
            labels=labels, probabilities=probabilities
        ),
        "reliability_equal_mass": equal_mass_reliability(
            labels=labels, probabilities=probabilities, stable_ids=stable_ids
        ),
    }


# --------------------------------------------------------------------------
# §11 Routing thresholds -- the frozen empirical order statistic
# --------------------------------------------------------------------------


def routing_sort_order(
    uncertainties: Sequence[float] | np.ndarray, stable_ids: Sequence[str]
) -> np.ndarray:
    """Row positions in ascending frozen `(u, stable_id)` order.

    The vectorised equivalent of the protocol binder's ordering. It is proven
    equal to `select_routing_threshold()` at every derived threshold below, and
    by test across the whole coverage grid, so the frozen rule stays the
    authority while remaining executable at 473,897 rows.
    """
    values = _float64(uncertainties, "uncertainty")
    identities = np.asarray([str(value) for value in stable_ids], dtype=np.str_)
    _require_aligned(uncertainties=values, stable_ids=identities)
    return np.lexsort((identities, values))


def derive_routing_threshold(
    *,
    uncertainties: Sequence[float] | np.ndarray,
    stable_ids: Sequence[str],
    target_coverage: float,
    name: str = U1_DEV_THRESHOLD_NAME,
    order: np.ndarray | None = None,
) -> dict[str, Any]:
    """`k = ceil(c_star * N)` over the frozen sort; acceptance is `u <= u_star`.

    The rank comes from `routing_threshold_rank()` -- the frozen rule -- so no
    library quantile convention can enter. Ties at `u_star` are all accepted,
    which can only raise achieved coverage above target, never lower it.
    """
    values = _float64(uncertainties, "uncertainty")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise U1CalibrationError("An uncertainty lies outside [0, 1].")
    identities = [str(value) for value in stable_ids]
    total = _require_aligned(
        uncertainties=values, stable_ids=np.asarray(identities, dtype=object)
    )
    ordering = routing_sort_order(values, identities) if order is None else order
    rank = routing_threshold_rank(total, float(target_coverage))
    u_star = float(values[ordering[rank - 1]])
    accepted = int(np.count_nonzero(values <= u_star))
    ties = int(np.count_nonzero(values == u_star))
    achieved = accepted / total
    if achieved < float(target_coverage):
        raise U1CalibrationError(
            f"Achieved coverage {achieved!r} fell below target "
            f"{target_coverage!r}; the frozen order statistic is wrong."
        )
    return {
        "threshold_class": "u1_routing_threshold",
        "name": str(name),
        "rule": "empirical_order_statistic_ceil_1_based",
        "sort_key": ["uncertainty", "stable_id"],
        "acceptance": "u <= u_star",
        "rank": int(rank),
        "u_star": u_star,
        "target_coverage": float(target_coverage),
        "achieved_coverage": achieved,
        "accepted_count": accepted,
        "threshold_tie_count": ties,
        "eligible_count": total,
        "library_quantile_used": False,
    }


def derive_u_star_dev(
    *, uncertainties: Sequence[float] | np.ndarray, stable_ids: Sequence[str]
) -> dict[str, Any]:
    """`u_star_dev` by the protocol binder itself, cross-checked vectorised.

    The claim-bearing value is the one `select_routing_threshold()` returns.
    The vectorised derivation must agree with it exactly, or the run stops.
    """
    identities = [str(value) for value in stable_ids]
    values = _float64(uncertainties, "uncertainty")
    frozen = select_routing_threshold(
        values.tolist(),
        identities,
        U1_RETAINED_COVERAGE,
        name=U1_DEV_THRESHOLD_NAME,
    )
    derived = derive_routing_threshold(
        uncertainties=values,
        stable_ids=identities,
        target_coverage=U1_RETAINED_COVERAGE,
        name=U1_DEV_THRESHOLD_NAME,
    )
    mismatches = {
        field: (getattr(frozen, field), derived[field])
        for field in (
            "u_star",
            "target_coverage",
            "achieved_coverage",
            "accepted_count",
            "threshold_tie_count",
            "eligible_count",
        )
        if getattr(frozen, field) != derived[field]
    }
    if mismatches:
        raise U1CalibrationError(
            "The vectorised routing derivation disagrees with the frozen "
            f"protocol rule: {mismatches}. The frozen rule is the authority."
        )
    derived["derived_by"] = "u1_protocol.select_routing_threshold"
    derived["vectorised_cross_check_agreed"] = True
    return derived


# --------------------------------------------------------------------------
# §10 Risk, coverage and class-aware routing evidence
# --------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def accepted_evidence(
    *,
    labels: np.ndarray,
    decisions: np.ndarray,
    uncertainties: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    """Class-aware behaviour among accepted rows, undefined never imputed as 0.

    "true positive" and "true negative" here mean the frozen ground-truth label
    class, not the decision outcome: the point of §10.4 is to expose a system
    that looks safe only because it abstains disproportionately from difficult
    POSITIVES.
    """
    positive_label = labels == 1
    negative_label = labels == 0
    accepted_count = int(np.count_nonzero(accepted))
    total = int(labels.shape[0])

    true_positive = int(np.count_nonzero(accepted & positive_label & decisions))
    false_negative = int(np.count_nonzero(accepted & positive_label & ~decisions))
    true_negative = int(np.count_nonzero(accepted & negative_label & ~decisions))
    false_positive = int(np.count_nonzero(accepted & negative_label & decisions))
    errors = false_positive + false_negative
    observed_risk = _ratio(errors, accepted_count)
    predicted_risk = float(np.mean(uncertainties[accepted])) if accepted_count else None
    agreement = (
        None
        if observed_risk is None or predicted_risk is None
        else abs(predicted_risk - observed_risk)
    )
    label_positive = int(np.count_nonzero(positive_label))
    label_negative = int(np.count_nonzero(negative_label))
    return {
        "row_count": total,
        "accepted_count": accepted_count,
        "escalated_count": total - accepted_count,
        "coverage": accepted_count / total,
        "escalation_fraction": 1.0 - accepted_count / total,
        "accepted_risk": observed_risk,
        "observed_accepted_risk": observed_risk,
        "predicted_accepted_risk": predicted_risk,
        "accepted_risk_absolute_agreement_error": agreement,
        "accepted_error_count": errors,
        "accepted_true_positive_count": true_positive,
        "accepted_false_positive_count": false_positive,
        "accepted_true_negative_count": true_negative,
        "accepted_false_negative_count": false_negative,
        "accepted_positive_count": int(np.count_nonzero(accepted & positive_label)),
        "accepted_negative_count": int(np.count_nonzero(accepted & negative_label)),
        "accepted_sensitivity": _ratio(true_positive, true_positive + false_negative),
        "accepted_specificity": _ratio(true_negative, true_negative + false_positive),
        "accepted_ppv": _ratio(true_positive, true_positive + false_positive),
        "accepted_npv": _ratio(true_negative, true_negative + false_negative),
        "true_positive_escalation_fraction": _ratio(
            int(np.count_nonzero(~accepted & positive_label)), label_positive
        ),
        "true_negative_escalation_fraction": _ratio(
            int(np.count_nonzero(~accepted & negative_label)), label_negative
        ),
        "label_positive_count": label_positive,
        "label_negative_count": label_negative,
        "undefined_reported_as_null": True,
    }


def risk_coverage_curve(
    *,
    labels: Sequence[int] | np.ndarray,
    decisions: Sequence[bool] | np.ndarray,
    uncertainties: Sequence[float] | np.ndarray,
    stable_ids: Sequence[str],
    grid: Sequence[float] = U1_COVERAGE_GRID,
) -> dict[str, Any]:
    """The frozen coverage grid, one shared ordering, the frozen rank rule."""
    outcomes = _labels(labels)
    frozen = np.asarray(decisions, dtype=bool)
    values = _float64(uncertainties, "uncertainty")
    identities = [str(value) for value in stable_ids]
    _require_aligned(
        labels=outcomes,
        decisions=frozen,
        uncertainties=values,
        stable_ids=np.asarray(identities, dtype=object),
    )
    order = routing_sort_order(values, identities)

    points = []
    for target in grid:
        threshold = derive_routing_threshold(
            uncertainties=values,
            stable_ids=identities,
            target_coverage=float(target),
            name=U1_DEV_THRESHOLD_NAME,
            order=order,
        )
        accepted = values <= threshold["u_star"]
        point = {
            "target_coverage": float(target),
            "threshold": threshold,
            **accepted_evidence(
                labels=outcomes,
                decisions=frozen,
                uncertainties=values,
                accepted=accepted,
            ),
        }
        points.append(point)
    return {
        "evidence_class": "u1_risk_coverage_evidence",
        "out_of_fold": True,
        "coverage_grid": [float(value) for value in grid],
        "no_routing_reference_coverage": 1.0,
        "points": points,
        "risk_definition": "error_rate_of_frozen_decision_among_accepted",
        "threshold_rule": "empirical_order_statistic_ceil_1_based",
        "library_quantile_used": False,
    }


# --------------------------------------------------------------------------
# §11.6 Reporting guards -- flags for a human, never a re-selection
# --------------------------------------------------------------------------


def routing_guards(
    point: dict[str, Any],
    *,
    abstention_ratio_bound: float = U1_ASYMMETRIC_ABSTENTION_RATIO,
    risk_agreement_tolerance: float = U1_ACCEPTED_RISK_AGREEMENT_TOLERANCE,
) -> dict[str, Any]:
    """Evaluate both frozen guards at the retained operating point.

    These fire AFTER scientific evidence exists, so a firing guard is not an
    infrastructure failure: the complete U1 result is still persisted, with the
    flags beside it. Nothing here re-selects a threshold, re-fits, retries or
    begins downstream work.
    """
    positive = point.get("true_positive_escalation_fraction")
    negative = point.get("true_negative_escalation_fraction")
    if positive is None or negative is None:
        ratio: float | None = None
        asymmetric = False
        ratio_basis = "undefined_missing_class"
    elif negative > 0.0:
        ratio = float(positive) / float(negative)
        asymmetric = ratio > float(abstention_ratio_bound)
        ratio_basis = "positive_over_negative_escalation_fraction"
    elif positive > 0.0:
        # Positives are escalated and negatives never are: the ratio diverges,
        # which is the asymmetry the guard exists to surface.
        ratio = None
        asymmetric = True
        ratio_basis = "undefined_zero_negative_escalation_with_positive_escalation"
    else:
        ratio = None
        asymmetric = False
        ratio_basis = "undefined_no_escalation_in_either_class"

    disagreement = point.get("accepted_risk_absolute_agreement_error")
    inadequate = disagreement is not None and float(disagreement) > float(
        risk_agreement_tolerance
    )
    flags = {
        "asymmetric_abstention": bool(asymmetric),
        "routing_calibration_inadequacy": bool(inadequate),
    }
    fired = sorted(name for name, value in flags.items() if value)
    return {
        "guard_class": "u1_routing_guards",
        "evaluated_at_target_coverage": point.get("target_coverage"),
        "asymmetric_abstention_ratio_bound": float(abstention_ratio_bound),
        "asymmetric_abstention_ratio": ratio,
        "asymmetric_abstention_ratio_basis": ratio_basis,
        "true_positive_escalation_fraction": positive,
        "true_negative_escalation_fraction": negative,
        "accepted_risk_agreement_tolerance": float(risk_agreement_tolerance),
        "accepted_risk_absolute_agreement_error": disagreement,
        "predicted_accepted_risk": point.get("predicted_accepted_risk"),
        "observed_accepted_risk": point.get("observed_accepted_risk"),
        "flags": flags,
        "flags_raised": fired,
        "any_flag_raised": bool(fired),
        "human_review_required": True,
        "automatic_retention": False,
        "threshold_reselected": False,
        "refit_performed": False,
        "automatic_retry_performed": False,
        "scientific_evidence_discarded": False,
        "guard_semantics": (
            "a raised flag reports a scientific outcome that requires human "
            "review; the complete U1 evidence is persisted regardless and no "
            "threshold, calibrator or coverage target is changed"
        ),
    }


# --------------------------------------------------------------------------
# §10.5 Subject evidence and the frozen subject bootstrap
# --------------------------------------------------------------------------

_SUBJECT_METRICS: Final = (
    "coverage",
    "escalation_fraction",
    "accepted_risk",
    "accepted_sensitivity",
    "accepted_specificity",
)


def subject_level_evidence(
    *,
    labels: Sequence[int] | np.ndarray,
    decisions: Sequence[bool] | np.ndarray,
    uncertainties: Sequence[float] | np.ndarray,
    subject_ids: Sequence[str],
    u_star: float,
) -> dict[str, Any]:
    """Per-subject routing behaviour plus the subject-macro summary.

    A subject lacking a class leaves that quantity undefined and is excluded
    from the macro mean with its count reported -- it is never scored as zero.
    """
    outcomes = _labels(labels)
    frozen = np.asarray(decisions, dtype=bool)
    values = _float64(uncertainties, "uncertainty")
    subjects = np.asarray([str(value) for value in subject_ids], dtype=np.str_)
    _require_aligned(
        labels=outcomes, decisions=frozen, uncertainties=values, subjects=subjects
    )
    accepted = values <= float(u_star)

    per_subject: dict[str, dict[str, Any]] = {}
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        per_subject[subject] = accepted_evidence(
            labels=outcomes[mask],
            decisions=frozen[mask],
            uncertainties=values[mask],
            accepted=accepted[mask],
        )
    macro = {}
    for metric in _SUBJECT_METRICS:
        defined = [
            float(entry[metric])
            for entry in per_subject.values()
            if entry.get(metric) is not None
        ]
        macro[metric] = {
            "value": float(np.mean(defined)) if defined else None,
            "contributing_subject_count": len(defined),
            "non_contributing_subject_count": len(per_subject) - len(defined),
        }
    return {
        "evidence_class": "u1_subject_level_evidence",
        "out_of_fold": True,
        "u_star": float(u_star),
        "subject_count": len(per_subject),
        "inferential_unit": "subject",
        "per_subject": per_subject,
        "subject_macro": macro,
        "undefined_reported_as_null": True,
        "undefined_counted_as_zero": False,
    }


def subject_bootstrap(
    *,
    labels: Sequence[int] | np.ndarray,
    decisions: Sequence[bool] | np.ndarray,
    uncertainties: Sequence[float] | np.ndarray,
    subject_ids: Sequence[str],
    u_star: float,
    replicates: int = U1_BOOTSTRAP_REPLICATES,
    seed: int = U1_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """1000 subject-level replicates at seed 2026, on the fitted OOF predictions.

    The resampling unit is the SUBJECT and the plan comes from the repository's
    frozen `subject_bootstrap_plan`. Windows are never resampled, calibrators
    are never re-fitted inside a replicate, and `u_star` is held at the value
    derived once from the full OOF pool -- so the interval describes
    between-subject variation CONDITIONAL on the fitted OOF calibration, which
    is exactly the claim §10.5 permits and no more.
    """
    from cardiosentinel.evaluation.metrics import subject_bootstrap_plan

    outcomes = _labels(labels)
    frozen = np.asarray(decisions, dtype=bool)
    values = _float64(uncertainties, "uncertainty")
    subjects = np.asarray([str(value) for value in subject_ids], dtype=np.str_)
    _require_aligned(
        labels=outcomes, decisions=frozen, uncertainties=values, subjects=subjects
    )
    accepted = values <= float(u_star)

    index_by_subject = {
        subject: np.flatnonzero(subjects == subject)
        for subject in sorted(set(subjects.tolist()))
    }
    collected: dict[str, list[float]] = {metric: [] for metric in _SUBJECT_METRICS}
    undefined = dict.fromkeys(_SUBJECT_METRICS, 0)
    degenerate = 0
    plan = subject_bootstrap_plan(
        subjects.tolist(), replicates=int(replicates), seed=int(seed)
    )
    for sampled in plan:
        if len(set(sampled)) == 1:
            degenerate += 1
        indices = np.concatenate([index_by_subject[subject] for subject in sampled])
        replicate = accepted_evidence(
            labels=outcomes[indices],
            decisions=frozen[indices],
            uncertainties=values[indices],
            accepted=accepted[indices],
        )
        for metric in _SUBJECT_METRICS:
            value = replicate.get(metric)
            if value is None:
                undefined[metric] += 1
            else:
                collected[metric].append(float(value))

    return {
        "evidence_class": "u1_subject_bootstrap",
        "unit": U1_BOOTSTRAP_UNIT,
        "requested_replicates": int(replicates),
        "seed": int(seed),
        "u_star": float(u_star),
        "window_bootstrap_performed": False,
        "calibrators_refitted_per_replicate": False,
        "degenerate_single_subject_replicates": degenerate,
        "claim_scope": U1_BOOTSTRAP_CLAIM,
        "windows_are_independent_evidence": False,
        "intervals": {
            metric: {
                "lower_95": (
                    None if not values_ else float(np.percentile(values_, 2.5))
                ),
                "upper_95": (
                    None if not values_ else float(np.percentile(values_, 97.5))
                ),
                "successful_replicates": len(values_),
                "undefined_replicates": undefined[metric],
                "requested_replicates": int(replicates),
                "seed": int(seed),
            }
            for metric, values_ in collected.items()
        },
    }
