"""Model-agnostic window evidence, and its causal adaptation to protocol rows.

The T1 state machine consumes ``T1Row`` -- a deliberately narrow structure with
no label, no target family and no future field. Real producers emit something
wider: a window id, a subject, a model score, a quality flag, whatever context
annotation the evaluation harness happens to carry.

This module is the boundary between the two. It accepts the wide record, keeps
the reporting-only parts in a container the transition function never sees, and
emits the narrow row. Three properties are enforced here rather than trusted:

**Causality.** Windows are consumed in the order given and a window that is not
strictly after its predecessor is refused. The adapter never sorts, never
buffers and never looks ahead; sorting a batch would silently import ordering
information that a live stream cannot have.

**Availability.** Signal quality decides whether a window is evidence at all. An
unusable window becomes a row with ``score_present=False`` carrying no invented
probability, uncertainty or temporal score.

**The firewall.** Anything the protocol forbids the transition function is
refused structurally, not by convention.

Nothing here knows which model produced the score. A future B4 neural output,
a calibrated ensemble or a synthetic reviewer stream all arrive as the same
record, which is what makes the harness model-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Final

from cardiosentinel.neural.t1_config import T1EpisodeConfig
from cardiosentinel.neural.t1_protocol import (
    T1_FORBIDDEN_TRANSITION_INPUTS,
    T1ProtocolError,
    T1Row,
    decision_error_uncertainty,
)

# How close a supplied uncertainty must sit to the derived one before the
# adapter concludes the producer meant something else by the field.
UNCERTAINTY_AGREEMENT_TOLERANCE: Final = 1e-9

QUALITY_UNAVAILABLE: Final = "unavailable"


class T1StreamError(RuntimeError):
    """Raised when a window stream violates a causal or structural contract."""


@dataclass(frozen=True, slots=True)
class T1WindowEvidence:
    """One chronological window of model output, from any model.

    ``model_score`` is the placeholder every producer fills: a detector
    probability, a calibrated score, an ensemble mean. ``temporal_evidence`` is
    the continuous temporal-model score the protocol calls ``s_t``.

    ``context_flags`` is reporting-only annotation -- confounder families,
    acquisition notes, anything a report might stratify on. It is carried
    through to the outputs and is structurally barred from the transition
    function.
    """

    window_id: str
    subject_id: str
    record_id: str
    channel_index: int
    start_sample: int
    model_score: float | None = None
    calibrated_probability: float | None = None
    temporal_evidence: float | None = None
    calibrated_uncertainty: float | None = None
    signal_quality: str = "good"
    context_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.window_id:
            raise T1StreamError("A window needs a non-empty window_id.")
        if self.channel_index < 0:
            raise T1StreamError(f"channel_index must not be negative: {self!r}.")
        if self.start_sample < 0:
            raise T1StreamError(f"start_sample must not be negative: {self!r}.")

    @property
    def stream_key(self) -> tuple[str, int]:
        """State namespace. Never a predictive feature -- only a namespace."""
        return (self.record_id, int(self.channel_index))


@dataclass(frozen=True, slots=True)
class T1ReportingContext:
    """The parts of a window that may be reported but never reasoned from."""

    window_id: str
    subject_id: str
    record_id: str
    channel_index: int
    start_sample: int
    signal_quality: str
    context_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class T1AdaptedWindow:
    """One window, split into what the machine may see and what it may not."""

    row: T1Row
    context: T1ReportingContext
    stream_key: tuple[str, int]
    elapsed_stream_seconds: float
    gap_seconds: float
    stream_restarted: bool


def require_no_forbidden_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Refuse a transition payload carrying anything the protocol forbids.

    Used at the seam where an external producer hands the harness a mapping. It
    is cheaper to refuse the whole payload here than to discover a leaked label
    downstream in a state trace that already looks plausible.
    """
    present = sorted(set(payload) & set(T1_FORBIDDEN_TRANSITION_INPUTS))
    if present:
        raise T1StreamError(
            f"A transition payload may not carry {present}. A rule that reads a "
            "label, an evaluation annotation or a future value is not deployable: "
            "none of them exists on a live ECG stream."
        )
    return payload


def is_available(evidence: T1WindowEvidence, config: T1EpisodeConfig) -> bool:
    """A window is evidence only if its quality is accepted and it has a score."""
    if evidence.signal_quality not in config.signal_quality_accept:
        return False
    return evidence.model_score is not None


def _resolve_probability(evidence: T1WindowEvidence) -> float:
    """``p_t`` is the calibrated probability, which the producer must supply.

    The harness deliberately does not calibrate. Calibration is U1's frozen,
    already-fitted responsibility, and a harness that quietly fitted its own
    would be inventing evidence.
    """
    if evidence.calibrated_probability is None:
        raise T1StreamError(
            f"Window {evidence.window_id} is available but carries no calibrated "
            "probability. T1 consumes an already-fitted calibrator's output; it "
            "does not calibrate, and it will not substitute the raw model score."
        )
    probability = float(evidence.calibrated_probability)
    if not 0.0 <= probability <= 1.0:
        raise T1StreamError(
            f"Window {evidence.window_id} has calibrated_probability "
            f"{probability!r} outside [0, 1]."
        )
    return probability


def _resolve_temporal_evidence(evidence: T1WindowEvidence) -> float:
    if evidence.temporal_evidence is None:
        raise T1StreamError(
            f"Window {evidence.window_id} is available but carries no temporal "
            "evidence. Mature EVENT requires the temporal term; a stream without "
            "one would make mature EVENT unreachable by construction rather than "
            "by evidence."
        )
    value = float(evidence.temporal_evidence)
    if not 0.0 <= value <= 1.0:
        raise T1StreamError(
            f"Window {evidence.window_id} has temporal_evidence {value!r} outside "
            "[0, 1]."
        )
    return value


def _check_supplied_uncertainty(evidence: T1WindowEvidence, derived: float) -> None:
    """``u_t`` is defined by the protocol, so a supplied value is checked, not used."""
    if evidence.calibrated_uncertainty is None:
        return
    supplied = float(evidence.calibrated_uncertainty)
    if abs(supplied - derived) > UNCERTAINTY_AGREEMENT_TOLERANCE:
        raise T1StreamError(
            f"Window {evidence.window_id} supplies calibrated_uncertainty "
            f"{supplied!r}, but the protocol derives {derived!r} from the "
            "detector decision and the calibrated probability. The protocol "
            "definition is authoritative; a producer that means something else "
            "by 'uncertainty' must not silently redefine u_t."
        )


def adapt_window(
    evidence: T1WindowEvidence,
    config: T1EpisodeConfig,
    *,
    elapsed_stream_seconds: float,
) -> tuple[T1Row, T1ReportingContext]:
    """Split one window into its protocol row and its reporting context."""
    context = T1ReportingContext(
        window_id=evidence.window_id,
        subject_id=evidence.subject_id,
        record_id=evidence.record_id,
        channel_index=int(evidence.channel_index),
        start_sample=int(evidence.start_sample),
        signal_quality=evidence.signal_quality,
        context_flags=tuple(evidence.context_flags)
        if config.record_context_flags
        else (),
    )
    if not is_available(evidence, config):
        row = T1Row(
            stable_id=evidence.window_id,
            score_present=False,
            detector_decision=None,
            calibrated_probability=None,
            decision_error_uncertainty=None,
            temporal_evidence=None,
            elapsed_stream_seconds=float(elapsed_stream_seconds),
        )
        return row, context

    probability = _resolve_probability(evidence)
    temporal = _resolve_temporal_evidence(evidence)
    decision = float(evidence.model_score) >= config.detector_threshold
    uncertainty = decision_error_uncertainty(decision, probability)
    _check_supplied_uncertainty(evidence, uncertainty)
    row = T1Row(
        stable_id=evidence.window_id,
        score_present=True,
        detector_decision=decision,
        calibrated_probability=probability,
        decision_error_uncertainty=uncertainty,
        temporal_evidence=temporal,
        elapsed_stream_seconds=float(elapsed_stream_seconds),
    )
    return row, context


def iter_adapted_windows(
    windows: Iterable[T1WindowEvidence], config: T1EpisodeConfig
) -> Iterator[T1AdaptedWindow]:
    """Adapt a chronological stream one window at a time, strictly causally.

    The iterator pulls exactly one window per yielded result and holds no
    lookahead buffer, so a caller can hand it a generator that would fail if it
    were read ahead of the current position -- which is how the harness proves
    the causal guarantee rather than asserting it.

    Elapsed stream time is measured from the first window seen on that stream.
    A cadence gap does not invent windows to fill it: the gap is measured,
    reported, and left to the engine to apply the frozen unavailable-window
    semantics to.
    """
    first_sample: dict[tuple[str, int], int] = {}
    last_sample: dict[tuple[str, int], int] = {}
    frequency = float(config.sampling_frequency_hz)
    stride_seconds = config.stride_seconds
    reset_after = config.max_gap_seconds_before_stream_reset

    for evidence in windows:
        if not isinstance(evidence, T1WindowEvidence):
            raise T1StreamError(f"Expected T1WindowEvidence, got {type(evidence)!r}.")
        key = evidence.stream_key
        position = int(evidence.start_sample)
        gap_seconds = 0.0
        restarted = False

        if key in last_sample:
            previous = last_sample[key]
            if config.require_strictly_increasing and position <= previous:
                raise T1StreamError(
                    f"Window {evidence.window_id} at start_sample {position} does "
                    f"not follow {previous} on stream {key}. The stream is consumed "
                    "in the order given and is never re-sorted: reordering would "
                    "let a later window decide where an earlier one belongs, which "
                    "is exactly the future dependence T1 forbids."
                )
            elapsed_since_previous = (position - previous) / frequency
            gap_seconds = max(0.0, elapsed_since_previous - stride_seconds)
            if reset_after is not None and gap_seconds > reset_after:
                restarted = True
                first_sample[key] = position
        else:
            first_sample[key] = position

        last_sample[key] = position
        elapsed = (position - first_sample[key]) / frequency
        row, context = adapt_window(evidence, config, elapsed_stream_seconds=elapsed)
        yield T1AdaptedWindow(
            row=row,
            context=context,
            stream_key=key,
            elapsed_stream_seconds=elapsed,
            gap_seconds=gap_seconds,
            stream_restarted=restarted,
        )


def require_row_is_transition_safe(row: T1Row) -> T1Row:
    """Prove the row the engine is about to use carries only allowed fields."""
    forbidden = set(T1Row._fields) & set(T1_FORBIDDEN_TRANSITION_INPUTS)
    if forbidden:  # pragma: no cover - structural, T1Row is frozen upstream
        raise T1ProtocolError(
            f"T1Row gained forbidden fields {sorted(forbidden)}; the state machine "
            "must not be run against it."
        )
    return row
