"""M2-v1 evidence aggregation, result schema and prototype-drift machinery.

This module provides the *shape* of the future M2 scientific result and the
arithmetic needed to compute prototype contamination evidence. It is
deliberately incapable of inventing a value: every field is either computed
from evidence handed to it or explicitly recorded as excluded with a reason.

**Nothing here is populated from the real DEVELOPMENT corpus in this
authorization.** No VALIDATION or TEST partition is referenced anywhere in
this module, and no retention decision is expressed.

Annotation-derived stress intervals may define *evaluation* windows **after**
replay has already happened. They never influence the runtime memory
trajectory: the drift functions below consume a prototype trajectory that was
produced by a label-blind replay, and cannot feed anything back into it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from cardiosentinel.neural.m2_policy import (
    M2_ARMS,
    M2PolicyError,
    M2RowEvidence,
    require_m2_arm,
)

# Frozen reporting offsets for post-stress residual drift, in seconds.
RESIDUAL_FOLLOW_UP_SECONDS: Final = (300.0, 1800.0)

EXCLUDED_NO_PRE_STRESS_PROTOTYPE: Final = "no_valid_pre_stress_prototype"
EXCLUDED_NO_ELIGIBLE_FOLLOW_UP: Final = "no_eligible_causal_follow_up"
EXCLUDED_NO_STRESS_ROWS: Final = "no_rows_inside_stress_interval"


# --------------------------------------------------------------------------
# Admission / refusal aggregation over already-replayed evidence
# --------------------------------------------------------------------------


def summarize_admission(evidence: Sequence[M2RowEvidence]) -> dict[str, Any]:
    """Coverage and refusal accounting over one already-replayed population.

    Refusal fractions are attributed to every condition that actually failed,
    so they overlap by construction and are reported as such rather than
    forced into a misleading single-cause partition.
    """
    total = len(evidence)
    if total == 0:
        raise M2PolicyError("Admission evidence is empty; nothing to summarize.")
    scored = [row for row in evidence if row.decision.score is not None]
    admitted = sum(1 for row in evidence if row.update_admitted)
    available = sum(1 for row in evidence if row.decision.g1_available)
    return {
        "rows": total,
        "available_rows": available,
        "scored_rows": len(scored),
        "update_admitted_count": admitted,
        "update_admission_fraction": admitted / total,
        "freeze_fraction": 1.0 - (admitted / total),
        "refusal_fractions": {
            "sqi": sum(1 for r in evidence if not r.decision.g3_sqi_admissible) / total,
            "normal_evidence": (
                sum(1 for r in evidence if not r.decision.g4_normal_evidence) / total
            ),
            "morphology": (
                sum(1 for r in evidence if not r.decision.g6_morphology_computable)
                / total
            ),
            "refractory": (
                sum(1 for r in evidence if not r.decision.g5_not_in_refractory) / total
            ),
        },
        "refusal_fraction_semantics": (
            "each fraction counts rows where that condition failed; a row "
            "failing several conditions is counted in each, so these overlap "
            "and do not sum to the freeze fraction"
        ),
        "time_since_last_admitted_update": {
            "defined_rows": sum(
                1 for r in evidence if r.time_since_last_admitted_update is not None
            ),
            "undefined_rows": sum(
                1 for r in evidence if r.time_since_last_admitted_update is None
            ),
            "undefined_semantics": (
                "no admitted update has occurred yet in this stream; the "
                "quantity is undefined and is never imputed as zero"
            ),
        },
    }


# --------------------------------------------------------------------------
# Prototype contamination / drift
# --------------------------------------------------------------------------


def prototype_drift(mu_long: np.ndarray, mu_ref: np.ndarray) -> float:
    """`sqrt(mean((mu_long(t) - mu_ref) ** 2))` in the standardized 146-d space."""
    current = np.asarray(mu_long, dtype=np.float64)
    reference = np.asarray(mu_ref, dtype=np.float64)
    if current.shape != reference.shape:
        raise M2PolicyError("Prototype drift compares misaligned prototypes.")
    return float(np.sqrt(np.mean((current - reference) ** 2)))


@dataclass(frozen=True, slots=True)
class PrototypeTrajectory:
    """A label-blind replay's prototype trajectory for one stream.

    `times[i]` is the availability time of row `i` and `prototypes[i]` is the
    long-memory prototype immediately AFTER that row's update decision.
    """

    times: np.ndarray
    prototypes: np.ndarray

    def __post_init__(self) -> None:
        if self.times.shape[0] != self.prototypes.shape[0]:
            raise M2PolicyError("Prototype trajectory times and rows are misaligned.")

    @classmethod
    def from_observer_records(
        cls, records: Iterable[tuple[float, np.ndarray]]
    ) -> PrototypeTrajectory:
        collected = list(records)
        if not collected:
            raise M2PolicyError("A prototype trajectory needs at least one row.")
        times = np.asarray([item[0] for item in collected], dtype=np.float64)
        prototypes = np.stack(
            [np.asarray(item[1], dtype=np.float64) for item in collected]
        )
        return cls(times=times, prototypes=prototypes)


def interval_drift_evidence(
    trajectory: PrototypeTrajectory,
    *,
    stress_start_time: float,
    stress_end_time: float,
) -> dict[str, Any]:
    """Prototype contamination evidence for one annotated stress interval.

    `mu_ref` is the long-memory prototype immediately BEFORE the first stress
    window, using only past causal history. Where a required statistic has no
    eligible causal support it is recorded as `None` with an explicit reason;
    follow-up is never fabricated and no tuned recovery threshold exists.
    """
    times = trajectory.times
    before = np.flatnonzero(times < float(stress_start_time))
    if before.size == 0:
        return {
            "mu_ref_available": False,
            "excluded_reason": EXCLUDED_NO_PRE_STRESS_PROTOTYPE,
            "peak_drift_during_stress": None,
            "mean_drift_during_stress": None,
            "drift_at_stress_end": None,
            "residual_drift": {
                f"at_least_{int(offset)}s": None
                for offset in RESIDUAL_FOLLOW_UP_SECONDS
            },
        }

    mu_ref = trajectory.prototypes[before[-1]]
    during = np.flatnonzero(
        (times >= float(stress_start_time)) & (times <= float(stress_end_time))
    )
    if during.size == 0:
        return {
            "mu_ref_available": True,
            "excluded_reason": EXCLUDED_NO_STRESS_ROWS,
            "peak_drift_during_stress": None,
            "mean_drift_during_stress": None,
            "drift_at_stress_end": None,
            "residual_drift": {
                f"at_least_{int(offset)}s": None
                for offset in RESIDUAL_FOLLOW_UP_SECONDS
            },
        }

    stress_drift = np.asarray(
        [prototype_drift(trajectory.prototypes[i], mu_ref) for i in during]
    )
    residual: dict[str, float | None] = {}
    residual_reasons: dict[str, str] = {}
    for offset in RESIDUAL_FOLLOW_UP_SECONDS:
        label = f"at_least_{int(offset)}s"
        eligible = np.flatnonzero(times >= float(stress_end_time) + float(offset))
        if eligible.size == 0:
            residual[label] = None
            residual_reasons[label] = EXCLUDED_NO_ELIGIBLE_FOLLOW_UP
            continue
        residual[label] = prototype_drift(trajectory.prototypes[eligible[0]], mu_ref)
    return {
        "mu_ref_available": True,
        "excluded_reason": None,
        "stress_rows": int(during.size),
        "peak_drift_during_stress": float(np.max(stress_drift)),
        "mean_drift_during_stress": float(np.mean(stress_drift)),
        "drift_at_stress_end": float(stress_drift[-1]),
        "residual_drift": residual,
        "residual_drift_excluded_reasons": residual_reasons,
    }


# --------------------------------------------------------------------------
# Future scientific result schema -- shape only, never populated here
# --------------------------------------------------------------------------

M2_RESULT_METRIC_FIELDS: Final = (
    "pooled_auprc",
    "pooled_auroc",
    "pooled_sensitivity",
    "pooled_specificity",
    "pooled_ppv",
    "pooled_mcc",
    "subject_macro_auprc",
    "subject_macro_auroc",
    "subject_macro_sensitivity",
    "subject_macro_specificity",
    "background_false_positive_rate",
    "subject_false_positive_distribution",
    "rate_challenge_false_positive_rate",
    "axis_challenge_false_positive_rate",
    "conduction_challenge_descriptive_false_positive_rate",
    "cold_start_evidence",
)

M2_RESULT_POLICY_FIELDS: Final = (
    "update_admission_fraction",
    "freeze_fraction",
    "sqi_refusal_fraction",
    "normal_evidence_refusal_fraction",
    "morphology_refusal_fraction",
    "refractory_refusal_fraction",
    "memory_update_count",
    "time_since_last_admitted_update",
    "prototype_drift_evidence",
)


def empty_m2_result_schema(arm: str) -> dict[str, Any]:
    """The unpopulated future-result shape for one arm.

    Every scientific field is `None`. This is a schema, not a result: it
    records that no M2 scientific execution has occurred and carries no
    fabricated value.
    """
    evaluated = require_m2_arm(arm)
    return {
        "schema": "m2_v1_arm_result",
        "arm": evaluated,
        "arms_in_core": list(M2_ARMS),
        "populated": False,
        "scientific_execution_performed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "classifier_retrained": False,
        "rollback_evaluated": False,
        "metrics": dict.fromkeys(M2_RESULT_METRIC_FIELDS),
        "policy_evidence": dict.fromkeys(M2_RESULT_POLICY_FIELDS),
        "exclusion_reasons": {},
        "conduction_change_evidence_status": "exploratory_descriptive_only",
        "cold_start_limitation": (
            "M1's zero sensitivity in the 0-5 minute bin at the frozen "
            "thresholds is inherited by every M2 arm and is not addressed by "
            "this protocol"
        ),
    }


def validate_unpopulated(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Refuse a result that silently claims scientific evidence."""
    if result.get("populated") is not False:
        raise M2PolicyError("This authorization produces no populated M2 result.")
    if result.get("scientific_execution_performed") is not False:
        raise M2PolicyError("No M2 scientific execution is authorized.")
    for field_name in ("validation_accessed", "test_accessed"):
        if result.get(field_name) is not False:
            raise M2PolicyError(f"M2 implementation must record {field_name}=false.")
    if any(value is not None for value in result.get("metrics", {}).values()):
        raise M2PolicyError("An unpopulated M2 result carries no metric value.")
    return result
