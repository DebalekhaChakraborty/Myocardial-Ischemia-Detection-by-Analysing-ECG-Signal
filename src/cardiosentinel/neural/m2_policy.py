"""The frozen M2-v1 contamination-safe memory update policy.

M2 changes **only the online memory update policy**. The B4-B encoder, the
128-d learned representation, the 18-d physiology block, the 146-d fused
representation, the physiology transform, the M1L head weights, the M1
distance standardizer, the long-memory alpha, the chronology key and the
classification threshold are all inherited unchanged from the frozen M1-v2
system. No classifier is retrained here and nothing in this module trains,
fits or tunes anything.

Two claim-bearing policies exist, and only two:

* **M2-0** -- the frozen M1L naive control. Every AVAILABLE finite observation
  updates the prototype, exactly reproducing `DualTimescaleMemory.observe`.
* **M2-G** -- the identical system with the frozen G1-G6 admission gate.

Prototype arithmetic is delegated to the frozen `DualTimescaleMemory` rather
than reimplemented, so the alpha semantics cannot silently drift: M2-0 is the
same call sequence M1 performs (deviations, then update), and M2-G differs
only in that the update is conditional. This module holds the admission
counters itself because M2-G must record an observation that was scored but
refused, which the M1 class has no vocabulary for; the prototype trajectory it
produces is still purely M1's.

**Label firewall.** No function in this module accepts an ischemia, challenge
or quality annotation, and `M2TimelineRow` has no field for one. Target labels
cannot admit, refuse, initialize, select, arm or alter anything here. Patient
identity selects a state namespace and is never a predictive or gating input.

**No rollback.** There is no snapshot, restore, oracle-correction or
delayed-label path, and no M2-GR arm.

This module is implementation only. It performs no canonical scientific
execution, touches no VALIDATION or TEST partition, and produces no retention
claim.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from cardiosentinel.neural import m2_gate as GATE
from cardiosentinel.neural.patient_memory import (
    OBSERVATION_AVAILABLE,
    OBSERVATION_UNAVAILABLE_EXACT_FLAT,
    REPRESENTATION_DIM,
    DualTimescaleMemory,
    M1DistanceStandardizer,
)
from cardiosentinel.neural.protocol import SAMPLING_FREQUENCY_HZ, WINDOW_SAMPLES

M2_ARM_NAIVE: Final = "M2-0"
M2_ARM_GATED: Final = "M2-G"
M2_ARMS: Final = (M2_ARM_NAIVE, M2_ARM_GATED)


class M2PolicyError(RuntimeError):
    """Raised when a causal M2 step cannot proceed with full integrity."""


def require_m2_arm(arm: str) -> str:
    if arm not in M2_ARMS:
        raise M2PolicyError(
            f"Unknown M2 arm {arm!r}. M2-v1 has exactly two claim-bearing "
            f"policies, {M2_ARMS}; rollback and additional arms are excluded."
        )
    return arm


def available_time_seconds(start_sample: int) -> float:
    """Real elapsed acquisition time of a window, per the frozen G5 rule.

    `(start_sample + 2500) / 250.0`. This is physical time, never a window
    count, update count, available-row count or episode state.
    """
    return (int(start_sample) + WINDOW_SAMPLES) / SAMPLING_FREQUENCY_HZ


# --------------------------------------------------------------------------
# One timeline row, carrying no label of any kind
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2TimelineRow:
    """One causal timeline row's gate-visible, deployment-observable inputs.

    Deliberately absent: `target_family`, any ischemia/challenge/quality
    annotation, and any subject identifier. `record_id` and `channel_index`
    select the causal state namespace and are never model or gate inputs.
    """

    record_id: str
    channel_index: int
    start_sample: int
    observation_state: int
    representation: np.ndarray | None
    finite_sample_fraction: float | None
    sqi: Mapping[str, float] | None
    morphology_valid: float | None

    @property
    def stream_key(self) -> tuple[str, int]:
        return (self.record_id, int(self.channel_index))

    @property
    def available_time(self) -> float:
        return available_time_seconds(self.start_sample)


# --------------------------------------------------------------------------
# The frozen G1-G6 admission gate
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2GateDecision:
    """Structured engineering evidence for one admission decision.

    Every gate result is recorded whether or not it was decisive, so a refusal
    can be audited without rerunning the replay. This is structured evidence,
    never a natural-language explanation.
    """

    g1_available: bool
    g2_finite_representation: bool
    g3_finite_sample_precondition: bool
    g3_feature_results: Mapping[str, bool]
    g3_sqi_admissible: bool
    g4_normal_evidence: bool
    g5_not_in_refractory: bool
    g6_morphology_computable: bool
    admitted: bool
    score: float | None
    refractory_until_before: float

    def refusal_reasons(self) -> tuple[str, ...]:
        """Every failing condition, in frozen gate order."""
        failures = []
        for name, passed in (
            ("G1", self.g1_available),
            ("G2", self.g2_finite_representation),
            ("G3", self.g3_sqi_admissible),
            ("G4", self.g4_normal_evidence),
            ("G5", self.g5_not_in_refractory),
            ("G6", self.g6_morphology_computable),
        ):
            if not passed:
                failures.append(name)
        return tuple(failures)


def evaluate_g3(
    finite_sample_fraction: float | None, sqi: Mapping[str, float] | None
) -> tuple[bool, bool, dict[str, bool]]:
    """The frozen waveform-SQI admission.

    Hard precondition `finite_sample_fraction == 1.0`, then each of the six
    declared `SIGNAL_V1` columns at or below its frozen TRAIN Q99 bound. No
    amplitude, rhythm, variance or near-flat criterion is consulted: those vary
    legitimately with patient physiology, and G3 screens artifact/noise rather
    than selecting a physiological phenotype.
    """
    precondition = finite_sample_fraction is not None and (
        float(finite_sample_fraction) == 1.0
    )
    results: dict[str, bool] = {}
    for column in GATE.G3_SQI_COLUMNS:
        if sqi is None or column not in sqi:
            results[column] = False
            continue
        value = float(sqi[column])
        # "at or below" -- the exact frozen bound passes.
        results[column] = np.isfinite(value) and value <= GATE.G3_UPPER_BOUNDS[column]
    admissible = precondition and all(results.values())
    return admissible, precondition, results


def evaluate_gate(
    *,
    observation_state: int,
    representation: np.ndarray | None,
    finite_sample_fraction: float | None,
    sqi: Mapping[str, float] | None,
    morphology_valid: float | None,
    score: float | None,
    available_time: float,
    refractory_until_before: float,
) -> M2GateDecision:
    """Evaluate the six frozen conditions independently.

    This signature accepts no label, no target family, no subject identifier
    and no patient identifier: admission is structurally incapable of
    consulting them.

    G5 is evaluated against the refractory state that existed **before** this
    row, so a row can never affect its own G5 decision.
    """
    g1 = int(observation_state) == OBSERVATION_AVAILABLE
    g2 = (
        representation is not None
        and np.asarray(representation).shape == (REPRESENTATION_DIM,)
        and bool(np.all(np.isfinite(np.asarray(representation, dtype=np.float64))))
    )
    g3, g3_precondition, g3_features = evaluate_g3(finite_sample_fraction, sqi)
    # The deterministic normal-evidence margin. Never a probability,
    # confidence, uncertainty, calibrated probability or conformal score.
    g4 = score is not None and float(score) <= GATE.NORMAL_EVIDENCE_THRESHOLD
    g5 = float(available_time) >= float(refractory_until_before)
    g6 = morphology_valid is not None and float(morphology_valid) == 1.0
    return M2GateDecision(
        g1_available=g1,
        g2_finite_representation=g2,
        g3_finite_sample_precondition=g3_precondition,
        g3_feature_results=dict(g3_features),
        g3_sqi_admissible=g3,
        g4_normal_evidence=g4,
        g5_not_in_refractory=g5,
        g6_morphology_computable=g6,
        admitted=bool(g1 and g2 and g3 and g4 and g5 and g6),
        score=None if score is None else float(score),
        refractory_until_before=float(refractory_until_before),
    )


# --------------------------------------------------------------------------
# Per-row evidence
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2RowEvidence:
    """Auditable record of why one row did or did not move the prototype."""

    record_id: str
    channel_index: int
    start_sample: int
    available_time: float
    observation_state: int
    arm: str
    decision: M2GateDecision
    d_long: float | None
    morphology_valid: float | None
    update_admitted: bool
    refractory_rearmed_after_decision: bool
    refractory_until_after: float
    past_observed_count_before: int
    past_update_count_before: int
    past_update_count_after: int
    time_since_last_admitted_update: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "channel_index": int(self.channel_index),
            "start_sample": int(self.start_sample),
            "available_time": self.available_time,
            "observation_state": int(self.observation_state),
            "arm": self.arm,
            "g1_available": self.decision.g1_available,
            "g2_finite_representation": self.decision.g2_finite_representation,
            "g3_finite_sample_precondition": (
                self.decision.g3_finite_sample_precondition
            ),
            "g3_feature_results": dict(self.decision.g3_feature_results),
            "g3_sqi_admissible": self.decision.g3_sqi_admissible,
            "score": self.decision.score,
            "g4_normal_evidence": self.decision.g4_normal_evidence,
            "refractory_until_before": self.decision.refractory_until_before,
            "g5_not_in_refractory": self.decision.g5_not_in_refractory,
            "morphology_valid": self.morphology_valid,
            "g6_morphology_computable": self.decision.g6_morphology_computable,
            "d_long": self.d_long,
            "update_admitted": self.update_admitted,
            "refractory_rearmed_after_current_decision": (
                self.refractory_rearmed_after_decision
            ),
            "refractory_until_after": self.refractory_until_after,
            "past_observed_count_before": self.past_observed_count_before,
            "past_update_count_before": self.past_update_count_before,
            "past_update_count_after": self.past_update_count_after,
            "time_since_last_admitted_update": self.time_since_last_admitted_update,
            "refusal_reasons": list(self.decision.refusal_reasons()),
        }


# --------------------------------------------------------------------------
# Causal per-stream state
# --------------------------------------------------------------------------


@dataclass(slots=True)
class M2StreamState:
    """Causal state for one `(record_id, channel_index)` stream.

    State never crosses channels, recordings or subjects: a new instance is
    constructed at every stream boundary, exactly as M1 does.

    The prototype itself lives in the frozen `DualTimescaleMemory`, so alpha
    and the EMA arithmetic are inherited rather than restated. The admission
    counters live here because M2-G must be able to record an observation that
    was scored and then refused.
    """

    memory: DualTimescaleMemory
    refractory_until: float = -np.inf
    past_observed_count: int = 0
    past_update_count: int = 0
    last_admitted_update_time: float | None = None
    _prior: np.ndarray = field(default_factory=lambda: np.empty(0))

    @classmethod
    def cold_start(cls, prior: np.ndarray) -> M2StreamState:
        vector = np.asarray(prior, dtype=np.float64)
        return cls(memory=DualTimescaleMemory(vector), _prior=vector.copy())

    @property
    def mu_long(self) -> np.ndarray:
        return self.memory.mu_long


# --------------------------------------------------------------------------
# Deterministic causal replay
# --------------------------------------------------------------------------

M2Scorer = Callable[[np.ndarray, float], float]
"""`(raw_146d_representation, d_long) -> frozen M1L score`.

Injected rather than imported so this module never reaches for a checkpoint on
its own. The real scorer runs the frozen retained M1L head on
`[z_t ; d_long(t)]`; no other input exists and no label is available to it.
"""


def _ordered_rows(rows: Iterable[M2TimelineRow]) -> list[M2TimelineRow]:
    """Causal order within one stream: strictly increasing `start_sample`."""
    ordered = sorted(rows, key=lambda row: int(row.start_sample))
    starts = [int(row.start_sample) for row in ordered]
    if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
        raise M2PolicyError(
            "An M2 stream does not have strictly increasing start samples; the "
            "causal order is ambiguous."
        )
    keys = {row.stream_key for row in ordered}
    if len(keys) > 1:
        raise M2PolicyError(f"An M2 stream replay mixes stream keys {sorted(keys)}.")
    return ordered


def replay_stream(
    rows: Sequence[M2TimelineRow],
    *,
    arm: str,
    standardizer: M1DistanceStandardizer,
    scorer: M2Scorer,
    prototype_observer: Callable[[int, np.ndarray], None] | None = None,
) -> list[M2RowEvidence]:
    """Replay one causal stream under the frozen M2 update order.

    The immutable order per row is: the row arrives; its physical observation
    state is determined; if AVAILABLE the frozen 146-d representation is taken;
    `d_long(t)` is computed against the EXISTING pre-update prototype; the
    frozen M1L score is computed from `[z_t ; d_long(t)]`; G1-G6 are evaluated
    independently using the PRIOR refractory state; the prototype is updated
    only when all six pass; and only then, if the score exceeds the
    normal-evidence threshold, the refractory is re-armed for future rows.

    A row therefore never alters its own prototype, its own `d_long`, its own
    score or its own G5 decision, and no future information is used.

    `prototype_observer` receives `(row_index, mu_long_after_this_row)` and
    exists so prototype-drift evidence can be collected later without forcing
    the whole trajectory to be retained.
    """
    evaluated_arm = require_m2_arm(arm)
    ordered = _ordered_rows(rows)
    state = M2StreamState.cold_start(standardizer.prior_vector())
    evidence: list[M2RowEvidence] = []

    for index, row in enumerate(ordered):
        available_time = row.available_time
        refractory_before = state.refractory_until
        observed_before = state.past_observed_count
        updates_before = state.past_update_count

        # B. An UNAVAILABLE row is not an observation. It receives no
        # representation, no score and no update; it arms no refractory; and it
        # leaves both counters untouched. Only real elapsed time advances, so an
        # existing refractory expires naturally across the gap.
        if int(row.observation_state) != OBSERVATION_AVAILABLE:
            if int(row.observation_state) != OBSERVATION_UNAVAILABLE_EXACT_FLAT:
                raise M2PolicyError(
                    f"Row {row.stream_key}/{row.start_sample} has unsupported "
                    f"observation state {row.observation_state!r}."
                )
            decision = evaluate_gate(
                observation_state=row.observation_state,
                representation=None,
                finite_sample_fraction=row.finite_sample_fraction,
                sqi=row.sqi,
                morphology_valid=row.morphology_valid,
                score=None,
                available_time=available_time,
                refractory_until_before=refractory_before,
            )
            evidence.append(
                M2RowEvidence(
                    record_id=row.record_id,
                    channel_index=int(row.channel_index),
                    start_sample=int(row.start_sample),
                    available_time=available_time,
                    observation_state=int(row.observation_state),
                    arm=evaluated_arm,
                    decision=decision,
                    d_long=None,
                    morphology_valid=row.morphology_valid,
                    update_admitted=False,
                    refractory_rearmed_after_decision=False,
                    refractory_until_after=state.refractory_until,
                    past_observed_count_before=observed_before,
                    past_update_count_before=updates_before,
                    past_update_count_after=state.past_update_count,
                    time_since_last_admitted_update=(
                        None
                        if state.last_admitted_update_time is None
                        else available_time - state.last_admitted_update_time
                    ),
                )
            )
            if prototype_observer is not None:
                prototype_observer(index, state.mu_long)
            continue

        # C. The frozen fused representation, standardized into the frozen
        # distance space. Scoring uses the raw 146-d vector; the prototype
        # lives in the standardized space, exactly as M1 does.
        raw = np.asarray(row.representation, dtype=np.float64)
        finite_representation = raw.shape == (REPRESENTATION_DIM,) and bool(
            np.all(np.isfinite(raw))
        )

        d_long: float | None = None
        score: float | None = None
        if finite_representation:
            standardized = standardizer.standardize(raw)[0]
            # D. Deviations against the EXISTING pre-update prototype.
            features = state.memory.deviations(standardized)
            d_long = float(features.d_long)
            # E. The frozen M1L score from [z_t ; d_long(t)].
            score = float(scorer(raw, d_long))
            if not np.isfinite(score):
                raise M2PolicyError(
                    "The frozen M1L scorer produced a non-finite score for "
                    f"{row.record_id}/{row.channel_index}/{row.start_sample}."
                )
        else:
            standardized = None

        # F. Independent evaluation against the PRIOR refractory state.
        decision = evaluate_gate(
            observation_state=row.observation_state,
            representation=raw if finite_representation else None,
            finite_sample_fraction=row.finite_sample_fraction,
            sqi=row.sqi,
            morphology_valid=row.morphology_valid,
            score=score,
            available_time=available_time,
            refractory_until_before=refractory_before,
        )

        # An AVAILABLE finite row is an observation under inherited M1
        # semantics whether or not the gate admits its update.
        if finite_representation:
            state.past_observed_count += 1

        # G. M2-0 reproduces the frozen naive control: every AVAILABLE finite
        # observation updates. M2-G updates only when all six conditions pass.
        if evaluated_arm == M2_ARM_NAIVE:
            admitted = finite_representation
        else:
            admitted = decision.admitted

        if admitted:
            if standardized is None:  # pragma: no cover - guarded above
                raise M2PolicyError("An admitted update has no standardized vector.")
            state.memory.update(standardized)
            state.past_update_count += 1
            state.last_admitted_update_time = available_time

        # H. Only AFTER the current row's decision is complete may the
        # refractory be re-armed, and only by a scored AVAILABLE finite row
        # whose score exceeds the normal-evidence margin. A row failing only
        # SQI, morphology or physical availability never arms it.
        rearmed = False
        if score is not None and score > GATE.NORMAL_EVIDENCE_THRESHOLD:
            state.refractory_until = max(
                state.refractory_until,
                available_time + GATE.REFRACTORY_DURATION_SECONDS,
            )
            rearmed = True

        evidence.append(
            M2RowEvidence(
                record_id=row.record_id,
                channel_index=int(row.channel_index),
                start_sample=int(row.start_sample),
                available_time=available_time,
                observation_state=int(row.observation_state),
                arm=evaluated_arm,
                decision=decision,
                d_long=d_long,
                morphology_valid=row.morphology_valid,
                update_admitted=admitted,
                refractory_rearmed_after_decision=rearmed,
                refractory_until_after=state.refractory_until,
                past_observed_count_before=observed_before,
                past_update_count_before=updates_before,
                past_update_count_after=state.past_update_count,
                time_since_last_admitted_update=(
                    None
                    if state.last_admitted_update_time is None
                    else available_time - state.last_admitted_update_time
                ),
            )
        )
        if prototype_observer is not None:
            prototype_observer(index, state.mu_long)

    return evidence


def replay_streams(
    rows: Iterable[M2TimelineRow],
    *,
    arm: str,
    standardizer: M1DistanceStandardizer,
    scorer: M2Scorer,
) -> dict[tuple[str, int], list[M2RowEvidence]]:
    """Replay every `(record_id, channel_index)` stream independently.

    Streams are fully isolated: no prototype, counter or refractory state
    crosses a channel, a recording or a subject.
    """
    grouped: dict[tuple[str, int], list[M2TimelineRow]] = {}
    for row in rows:
        grouped.setdefault(row.stream_key, []).append(row)
    return {
        key: replay_stream(
            grouped[key], arm=arm, standardizer=standardizer, scorer=scorer
        )
        for key in sorted(grouped)
    }


def m2_policy_identity(arm: str) -> dict[str, Any]:
    """The frozen policy identity, asserted against `m2_gate.py` on every call."""
    evaluated = require_m2_arm(arm)
    identity = GATE.m2_gate_identity()
    if tuple(identity["core_arms"]) != M2_ARMS:
        raise M2PolicyError("M2 arms differ from the frozen core arms.")
    if identity["rollback_in_core"] is not False:
        raise M2PolicyError("M2-v1 excludes rollback from the claim-bearing core.")
    return {
        "arm": evaluated,
        "policy": (
            "frozen_m1l_naive_control"
            if evaluated == M2_ARM_NAIVE
            else "frozen_g1_g6_admission_gate"
        ),
        "classifier_retrained": False,
        "encoder_changed": False,
        "representation_changed": False,
        "alpha_changed": False,
        "classification_threshold_used_for_admission": False,
        "rollback": False,
        "uncertainty_admission": False,
        "event_state_admission": False,
        "label_gated_update": False,
        "patient_identity_is_a_feature": False,
        **identity,
    }
