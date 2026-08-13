"""Post-replay M2 evaluation: the ONLY place annotations are permitted.

This module runs strictly AFTER a label-blind replay has already produced its
scores, decisions and prototype trajectory. Annotations enter here to *define
evaluation strata* and nothing else; they cannot reach the replay, because the
replay-side modules neither name an annotation quantity nor import this module
(`m2_execution.assert_label_firewall()` enforces that direction structurally).

**Alignment is by immutable row identity, never by position.** Equal sequence
length proves nothing: a permuted annotation vector of the same length would be
silently accepted by a positional join. Every annotation therefore carries the
frozen `(record_id, channel_index, start_sample)` identity, which maps
one-to-one onto the frozen `stable_id`, and `M2EvaluationBundle` performs a
deterministic keyed join that rejects duplicates, missing identities, extra
identities and any mismatch.

**Unscored rows are refused, not dropped.** This reproduces the frozen M1
governance rule verbatim (`m1_experiment.require_available_rows`): a
score-bearing population containing a row with no prediction is a refusal, not
a silent exclusion -- "the row is never dropped from a metric, no prediction is
invented and no denominator is altered automatically."

**No threshold is selected here.** Every thresholded path is guarded by
`require_frozen_m1l_classification_threshold`, so no evaluation can silently
run at a non-frozen operating point.

**Nothing in this module is executed against VALIDATION under the current
authorization**, and no retention decision is expressed anywhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
from cardiosentinel.neural.protocol import DATASET, WINDOW_SAMPLES

CONDUCTION_EVIDENCE_STATUS: Final = "exploratory_descriptive"
QUANTITATIVE_CHALLENGE_STATUS: Final = "quantitative_secondary"

EvaluationKey = tuple[str, int, int]
"""`(record_id, channel_index, start_sample)` -- immutable frozen row identity."""


class M2EvaluationError(RuntimeError):
    """Raised when post-replay evaluation cannot proceed with integrity."""


# --------------------------------------------------------------------------
# Frozen threshold enforcement -- one helper, applied on every thresholded path
# --------------------------------------------------------------------------


def require_frozen_m1l_classification_threshold(threshold: float) -> float:
    """Refuse any operating point other than the frozen retained M1L threshold.

    Shared deliberately rather than restated per call site: subtly different
    per-function checks are exactly how a non-frozen operating point reaches a
    claim-bearing metric.
    """
    value = float(threshold)
    if value != M1L_CLASSIFICATION_THRESHOLD:
        raise M2EvaluationError(
            f"M2 evaluation requires the frozen retained M1L classification "
            f"threshold {M1L_CLASSIFICATION_THRESHOLD!r}; received {value!r}. "
            "No new threshold may be selected, and the M2 normal-evidence "
            "margin is never a classification threshold."
        )
    if value == NORMAL_EVIDENCE_THRESHOLD:
        raise M2EvaluationError(
            "The M2 normal-evidence margin must never be used as a "
            "classification threshold."
        )
    return value


# --------------------------------------------------------------------------
# Identity-keyed post-replay annotation join
# --------------------------------------------------------------------------


def evaluation_key(row: M2RowEvidence) -> EvaluationKey:
    """The immutable evaluation identity of one evidence row."""
    return (row.record_id, int(row.channel_index), int(row.start_sample))


def stable_id_for_key(key: EvaluationKey) -> str:
    """The frozen `stable_id` this evaluation key corresponds to.

    The correspondence is exact and one-to-one: the frozen identity is
    `dataset:record:channel:start:end` with `end = start + WINDOW_SAMPLES`, so
    the evaluation key determines it completely and vice versa.
    """
    record_id, channel_index, start_sample = key
    return (
        f"{DATASET}:{record_id}:{int(channel_index)}:{int(start_sample)}:"
        f"{int(start_sample) + WINDOW_SAMPLES}"
    )


@dataclass(frozen=True, slots=True)
class M2AnnotationRow:
    """One frozen annotation, bound to an immutable row identity.

    `subject_id` and `target_family` exist for evaluation only. Neither is ever
    a runtime predictive or gating input -- they are absent from
    `M2TimelineRow`, from `evaluate_gate` and from the whole replay path.
    """

    record_id: str
    channel_index: int
    start_sample: int
    label: int
    target_family: str
    subject_id: str
    cold_start_bin: str

    @property
    def key(self) -> EvaluationKey:
        return (self.record_id, int(self.channel_index), int(self.start_sample))

    @property
    def stable_id(self) -> str:
        return stable_id_for_key(self.key)


@dataclass(frozen=True, slots=True)
class M2EvaluationBundle:
    """Evidence joined to annotations by identity, in one deterministic order."""

    arm: str
    keys: tuple[EvaluationKey, ...]
    evidence: tuple[M2RowEvidence, ...]
    annotations: tuple[M2AnnotationRow, ...]

    @property
    def scores(self) -> np.ndarray:
        return np.asarray(
            [row.decision.score for row in self.evidence], dtype=np.float64
        )

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([int(a.label) for a in self.annotations], dtype=np.int64)

    @property
    def subject_ids(self) -> np.ndarray:
        return np.asarray([str(a.subject_id) for a in self.annotations], dtype=np.str_)

    @property
    def target_families(self) -> np.ndarray:
        return np.asarray(
            [str(a.target_family) for a in self.annotations], dtype=np.str_
        )

    @property
    def cold_start_bins(self) -> np.ndarray:
        return np.asarray(
            [str(a.cold_start_bin) for a in self.annotations], dtype=np.str_
        )

    @property
    def stable_ids(self) -> tuple[str, ...]:
        return tuple(stable_id_for_key(key) for key in self.keys)

    def population_identity(self) -> dict[str, Any]:
        """The exact evaluated population, bound by the frozen identity digest."""
        from cardiosentinel.neural.p1_experiment import ordered_stable_id_digest

        return {
            "evaluated_rows": len(self.keys),
            "evaluated_ordered_stable_id_sha256": ordered_stable_id_digest(
                self.stable_ids
            ),
            "identity_key": "(record_id, channel_index, start_sample)",
            "identity_corresponds_to_frozen_stable_id": True,
            "positional_join_used": False,
        }


def build_evaluation_bundle(
    arm: str,
    evidence: Sequence[M2RowEvidence],
    annotations: Iterable[M2AnnotationRow],
    *,
    require_full_population: bool = True,
) -> M2EvaluationBundle:
    """Join evidence to annotations by identity, refusing every ambiguity.

    Rejected: duplicate evidence identities, duplicate annotation identities,
    annotations with no matching evidence row, and -- when
    `require_full_population` is set -- evidence rows carrying no annotation.
    A permuted annotation ordering cannot be silently accepted: rows are
    realigned deterministically by identity, so order carries no meaning.

    An annotated row that produced no score is a REFUSAL, reproducing the
    frozen M1 rule exactly: it is never dropped from a metric, no prediction is
    invented for it, and no denominator is altered automatically.
    """
    evaluated_arm = require_m2_arm(arm)

    evidence_index: dict[EvaluationKey, M2RowEvidence] = {}
    for row in evidence:
        key = evaluation_key(row)
        if key in evidence_index:
            raise M2EvaluationError(
                f"Duplicate evidence identity {key}; the evaluated population "
                "would be ambiguous."
            )
        evidence_index[key] = row

    annotation_index: dict[EvaluationKey, M2AnnotationRow] = {}
    for annotation in annotations:
        if annotation.key in annotation_index:
            raise M2EvaluationError(f"Duplicate annotation identity {annotation.key}.")
        annotation_index[annotation.key] = annotation

    missing = sorted(set(annotation_index) - set(evidence_index))
    if missing:
        raise M2EvaluationError(
            f"{len(missing)} annotation identities have no evidence row, "
            f"beginning {missing[:3]}. Annotations are never dropped silently."
        )
    unannotated = sorted(set(evidence_index) - set(annotation_index))
    if unannotated and require_full_population:
        raise M2EvaluationError(
            f"{len(unannotated)} evidence rows carry no annotation, beginning "
            f"{unannotated[:3]}. Pass require_full_population=False only when "
            "the evaluated population is deliberately a subset."
        )

    keys = tuple(sorted(annotation_index))
    joined_evidence = tuple(evidence_index[key] for key in keys)
    unscored = [
        key
        for key, row in zip(keys, joined_evidence, strict=True)
        if row.decision.score is None
    ]
    if unscored:
        raise M2EvaluationError(
            f"Score-bearing M2 population contains {len(unscored)} rows with no "
            f"prediction, beginning {unscored[:3]}. STOP FOR HUMAN REVIEW. The "
            "row is never dropped from a metric, no prediction is invented and "
            "no denominator is altered automatically."
        )
    return M2EvaluationBundle(
        arm=evaluated_arm,
        keys=keys,
        evidence=joined_evidence,
        annotations=tuple(annotation_index[key] for key in keys),
    )


# --------------------------------------------------------------------------
# Thresholded evaluation, every path guarded
# --------------------------------------------------------------------------


def window_evidence(
    bundle: M2EvaluationBundle,
    *,
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Pooled and subject-macro window discrimination for one arm."""
    from cardiosentinel.neural.p1_experiment import p1_validation_evidence

    frozen = require_frozen_m1l_classification_threshold(threshold)
    payload = p1_validation_evidence(
        bundle.labels, bundle.scores, bundle.subject_ids, frozen
    )
    payload["evidence_class"] = "m2_window_evidence"
    payload["arm"] = bundle.arm
    payload["threshold_source"] = "frozen_retained_m1l_classification_threshold"
    payload["threshold_selected_here"] = False
    payload["population_identity"] = bundle.population_identity()
    return payload


def false_alarm_evidence(
    bundle: M2EvaluationBundle,
    *,
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Background FPR, its subject distribution, and challenge FPRs.

    Challenge-target precedence is preserved exactly as the frozen production
    metric defines it: an ischemic-positive row is never removed merely because
    a challenge context also applies to it.
    """
    from cardiosentinel.neural.m1_experiment import subject_false_positive_evidence
    from cardiosentinel.neural.p1_experiment import p1_challenge_evidence

    frozen = require_frozen_m1l_classification_threshold(threshold)
    subject_fpr = subject_false_positive_evidence(
        bundle.labels, bundle.scores, bundle.subject_ids, frozen
    )
    challenge = p1_challenge_evidence(
        bundle.target_families, bundle.scores, bundle.subject_ids, frozen
    )
    return {
        "evidence_class": "m2_false_alarm_evidence",
        "arm": bundle.arm,
        "threshold": frozen,
        "threshold_source": "frozen_retained_m1l_classification_threshold",
        "threshold_selected_here": False,
        "background_false_positive": subject_fpr,
        "challenge": challenge,
        "conduction_change_evidence_status": CONDUCTION_EVIDENCE_STATUS,
        "ischemic_positive_rows_removed_for_challenge_context": False,
        "population_identity": bundle.population_identity(),
    }


def cold_start_stratified_evidence(
    bundle: M2EvaluationBundle,
    *,
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Frozen recording-age strata, with the inherited limitation preserved."""
    from cardiosentinel.baseline.metrics import binary_metrics
    from cardiosentinel.neural.patient_memory import COLD_START_BINS

    frozen = require_frozen_m1l_classification_threshold(threshold)
    labels = bundle.labels
    scores = bundle.scores
    bins = bundle.cold_start_bins

    strata: dict[str, Any] = {}
    for name, _low, _high in COLD_START_BINS:
        mask = bins == name
        count = int(np.sum(mask))
        entry: dict[str, Any] = {"window_count": count, "evidence_status": "supporting"}
        if count:
            entry["metrics"] = binary_metrics(labels[mask], scores[mask], frozen)
        strata[name] = entry
    return {
        "evidence_class": "m2_cold_start_evidence",
        "arm": bundle.arm,
        "threshold": frozen,
        "threshold_source": "frozen_retained_m1l_classification_threshold",
        "threshold_selected_here": False,
        "post_hoc_early_threshold_defined": False,
        "inherited_limitation": (
            "M1's zero sensitivity in the 0-5 minute bin at the frozen "
            "thresholds is inherited by every M2 arm and is not addressed by "
            "this protocol; gating can only make early adaptation more "
            "conservative"
        ),
        "strata": strata,
        "population_identity": bundle.population_identity(),
    }


def policy_evidence(evidence: Sequence[M2RowEvidence]) -> dict[str, Any]:
    """Update-admission and refusal accounting. Label-free by construction."""
    summary = summarize_admission(evidence)
    summary["evidence_class"] = "m2_policy_evidence"
    summary["memory_admission_threshold"] = NORMAL_EVIDENCE_THRESHOLD
    summary["classification_threshold_used_for_admission"] = False
    return summary


# --------------------------------------------------------------------------
# Prototype contamination, bound to the correct stream
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2StressInterval:
    """An annotation-defined stress interval, bound to ONE stream.

    An interval belongs to a specific `(record_id, channel_index)` trajectory.
    Overlapping timestamps on another stream are a different patient-channel
    history and must never be evaluated against it.
    """

    record_id: str
    channel_index: int
    family: str
    start_time: float
    end_time: float

    @property
    def stream_key(self) -> tuple[str, int]:
        return (self.record_id, int(self.channel_index))


def contamination_evidence(
    trajectories: dict[tuple[str, int], PrototypeTrajectory],
    *,
    stress_intervals: Sequence[M2StressInterval],
) -> dict[str, Any]:
    """Prototype drift for annotation-defined intervals, per stream.

    Each interval is evaluated only against its own stream's trajectory. The
    trajectory itself was produced by a label-blind replay before any
    annotation was consulted; intervals only select points on a fixed
    trajectory and can never alter it. Missing support is excluded with a
    reason and never fabricated, and no recovery threshold is defined.
    """
    results = []
    for interval in stress_intervals:
        key = interval.stream_key
        if key not in trajectories:
            raise M2EvaluationError(
                f"Stress interval for stream {key} has no matching prototype "
                "trajectory; an interval may never be applied to another "
                "stream merely because its timestamps overlap."
            )
        entry = interval_drift_evidence(
            trajectories[key],
            stress_start_time=float(interval.start_time),
            stress_end_time=float(interval.end_time),
        )
        entry["family"] = interval.family
        entry["record_id"] = interval.record_id
        entry["channel_index"] = int(interval.channel_index)
        entry["evidence_status"] = (
            CONDUCTION_EVIDENCE_STATUS
            if interval.family == "conduction_change"
            else QUANTITATIVE_CHALLENGE_STATUS
        )
        results.append(entry)
    return {
        "evidence_class": "m2_prototype_contamination_evidence",
        "trajectory_produced_label_blind": True,
        "annotations_applied_after_replay": True,
        "intervals_bound_to_stream_identity": True,
        "recovery_threshold_defined": False,
        "follow_up_fabricated": False,
        "intervals": results,
    }


def arm_evaluation(arm: str, evidence: Sequence[M2RowEvidence]) -> dict[str, Any]:
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
