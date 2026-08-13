"""Post-replay M2 evaluation: the ONLY place annotations are permitted.

This module runs strictly AFTER a label-blind replay has already produced its
scores, decisions and prototype trajectory. Annotations enter here to *define
evaluation strata* and nothing else; they cannot reach the replay, because the
replay-side modules neither name an annotation quantity nor import this module
(`m2_execution.assert_label_firewall()` enforces that direction structurally).

Every metric delegates to the repository's already-frozen implementations --
`p1_validation_evidence`, `p1_challenge_evidence`, `cold_start_evidence`,
`subject_false_positive_evidence` and the frozen `challenge_metrics` -- rather
than restating them. A second implementation of a frozen metric would be a
second scientific truth.

**No threshold is selected here.** Thresholded metrics use the frozen retained
M1L classification threshold, which was fixed by the M1 retention decision. The
M2 normal-evidence margin is never used for classification and the
classification threshold is never used for memory admission.

**Nothing in this module is executed against VALIDATION under the current
authorization**, and no retention decision is expressed anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from cardiosentinel.neural.m2_evidence import (
    PrototypeTrajectory,
    interval_drift_evidence,
    summarize_admission,
)
from cardiosentinel.neural.m2_policy import M2RowEvidence, require_m2_arm
from cardiosentinel.neural.m2_scorer import (
    M1L_CLASSIFICATION_THRESHOLD,
    NORMAL_EVIDENCE_THRESHOLD,
)

# Conduction change has one-subject support in the frozen corpus and is
# therefore descriptive only, never a quantitative claim.
CONDUCTION_EVIDENCE_STATUS: Final = "exploratory_descriptive"
QUANTITATIVE_CHALLENGE_STATUS: Final = "quantitative_secondary"


class M2EvaluationError(RuntimeError):
    """Raised when post-replay evaluation cannot proceed with integrity."""


def _scored_rows(evidence: Sequence[M2RowEvidence]) -> list[M2RowEvidence]:
    return [row for row in evidence if row.decision.score is not None]


def window_evidence(
    evidence: Sequence[M2RowEvidence],
    *,
    labels: Sequence[int],
    subject_ids: Sequence[str],
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Pooled and subject-macro window discrimination for one arm.

    `labels` and `subject_ids` are joined here, after replay, and are aligned
    to the SCORED rows only -- a row that produced no score has no prediction
    and is never given an invented one.
    """
    from cardiosentinel.neural.p1_experiment import p1_validation_evidence

    if float(threshold) != M1L_CLASSIFICATION_THRESHOLD:
        raise M2EvaluationError(
            "M2 evaluation uses the frozen retained M1L classification "
            "threshold; no new threshold may be selected."
        )
    scored = _scored_rows(evidence)
    outcomes = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray([str(value) for value in subject_ids], dtype=np.str_)
    if not (len(scored) == outcomes.shape[0] == subjects.shape[0]):
        raise M2EvaluationError(
            f"Post-replay join is misaligned: {len(scored)} scored rows, "
            f"{outcomes.shape[0]} labels, {subjects.shape[0]} subjects."
        )
    scores = np.asarray([row.decision.score for row in scored], dtype=np.float64)
    payload = p1_validation_evidence(outcomes, scores, subjects, float(threshold))
    payload["evidence_class"] = "m2_window_evidence"
    payload["threshold_source"] = "frozen_retained_m1l_classification_threshold"
    payload["threshold_selected_here"] = False
    payload["scored_rows"] = len(scored)
    payload["unscored_rows_excluded"] = len(evidence) - len(scored)
    return payload


def false_alarm_evidence(
    evidence: Sequence[M2RowEvidence],
    *,
    labels: Sequence[int],
    target_families: Sequence[str],
    subject_ids: Sequence[str],
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Background FPR, its subject distribution, and challenge FPRs.

    Challenge-target precedence is preserved exactly as the frozen production
    metric defines it: an ischemic-positive row is never removed merely because
    a challenge context also applies to it.
    """
    from cardiosentinel.neural.m1_experiment import subject_false_positive_evidence
    from cardiosentinel.neural.p1_experiment import p1_challenge_evidence

    scored = _scored_rows(evidence)
    scores = np.asarray([row.decision.score for row in scored], dtype=np.float64)
    outcomes = np.asarray(labels, dtype=np.int64)
    families = np.asarray([str(value) for value in target_families], dtype=np.str_)
    subjects = np.asarray([str(value) for value in subject_ids], dtype=np.str_)
    if not (len(scored) == outcomes.shape[0] == families.shape[0] == subjects.shape[0]):
        raise M2EvaluationError("False-alarm evaluation inputs are not row-aligned.")

    subject_fpr = subject_false_positive_evidence(
        outcomes, scores, subjects, float(threshold)
    )
    challenge = p1_challenge_evidence(families, scores, subjects, float(threshold))
    return {
        "evidence_class": "m2_false_alarm_evidence",
        "threshold": float(threshold),
        "threshold_source": "frozen_retained_m1l_classification_threshold",
        "threshold_selected_here": False,
        "background_false_positive": subject_fpr,
        "challenge": challenge,
        "conduction_change_evidence_status": CONDUCTION_EVIDENCE_STATUS,
        "ischemic_positive_rows_removed_for_challenge_context": False,
    }


def cold_start_stratified_evidence(
    evidence: Sequence[M2RowEvidence],
    *,
    labels: Sequence[int],
    cold_start_bins: Sequence[str],
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Frozen recording-age strata, with the inherited limitation preserved."""
    from cardiosentinel.evaluation.metrics import binary_metrics
    from cardiosentinel.neural.patient_memory import COLD_START_BINS

    scored = _scored_rows(evidence)
    scores = np.asarray([row.decision.score for row in scored], dtype=np.float64)
    outcomes = np.asarray(labels, dtype=np.int64)
    bins = np.asarray([str(value) for value in cold_start_bins], dtype=np.str_)
    if not (len(scored) == outcomes.shape[0] == bins.shape[0]):
        raise M2EvaluationError("Cold-start evaluation inputs are not row-aligned.")

    strata: dict[str, Any] = {}
    for name, _low, _high in COLD_START_BINS:
        mask = bins == name
        count = int(np.sum(mask))
        entry: dict[str, Any] = {
            "window_count": count,
            "evidence_status": "supporting",
        }
        if count:
            entry["metrics"] = binary_metrics(
                outcomes[mask], scores[mask], float(threshold)
            )
        strata[name] = entry
    return {
        "evidence_class": "m2_cold_start_evidence",
        "threshold": float(threshold),
        "post_hoc_early_threshold_defined": False,
        "inherited_limitation": (
            "M1's zero sensitivity in the 0-5 minute bin at the frozen "
            "thresholds is inherited by every M2 arm and is not addressed by "
            "this protocol; gating can only make early adaptation more "
            "conservative"
        ),
        "strata": strata,
    }


def policy_evidence(evidence: Sequence[M2RowEvidence]) -> dict[str, Any]:
    """Update-admission and refusal accounting, with corrected denominators."""
    summary = summarize_admission(evidence)
    summary["evidence_class"] = "m2_policy_evidence"
    summary["memory_admission_threshold"] = NORMAL_EVIDENCE_THRESHOLD
    summary["classification_threshold_used_for_admission"] = False
    return summary


def contamination_evidence(
    trajectory: PrototypeTrajectory,
    *,
    stress_intervals: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Prototype drift for annotation-defined stress intervals.

    The trajectory was produced by a label-blind replay before any annotation
    was consulted. Intervals only *select* points on that fixed trajectory;
    they cannot alter it. Missing support is excluded with a reason and never
    fabricated, and no recovery threshold is defined.
    """
    results = []
    for interval in stress_intervals:
        family = str(interval["family"])
        entry = interval_drift_evidence(
            trajectory,
            stress_start_time=float(interval["start_time"]),
            stress_end_time=float(interval["end_time"]),
        )
        entry["family"] = family
        entry["evidence_status"] = (
            CONDUCTION_EVIDENCE_STATUS
            if family == "conduction_change"
            else QUANTITATIVE_CHALLENGE_STATUS
        )
        results.append(entry)
    return {
        "evidence_class": "m2_prototype_contamination_evidence",
        "trajectory_produced_label_blind": True,
        "annotations_applied_after_replay": True,
        "recovery_threshold_defined": False,
        "follow_up_fabricated": False,
        "intervals": results,
    }


def arm_evaluation(
    arm: str,
    evidence: Sequence[M2RowEvidence],
) -> dict[str, Any]:
    """The label-free half of an arm's evaluation, safe to compute anywhere."""
    return {
        "arm": require_m2_arm(arm),
        "policy_evidence": policy_evidence(evidence),
        "window_evidence": None,
        "false_alarm_evidence": None,
        "cold_start_evidence": None,
        "contamination_evidence": None,
        "label_joined_sections_populated": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
