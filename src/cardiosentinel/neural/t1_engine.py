"""The T1 episode-state execution engine.

This module *runs* the frozen state machine; it does not define one. Every
transition is computed by :func:`cardiosentinel.neural.t1_protocol.next_state`,
and the engine's job is to feed it correctly, one window at a time, and to
record what happened in a form a reviewer can audit.

Five outputs come out of a run:

``state_trace``
    One entry per window: the state before, the state after, the evidence the
    decision was made from, and the streak counters afterwards.
``episodes``
    Maximal EVENT runs per stream, with onset and offset.
``transitions``
    Only the windows where the state actually changed, each with the reason.
``alerts``
    Operational notifications, with refractory suppression applied.
``recovery_spans``
    Every RECOVERY period and how it ended -- cleared, re-escalated, or still
    open when the stream ran out.

**Where the refractory period lives.** The protocol has no refractory concept,
and adding one to the transition function would be a different protocol. So the
refractory period is applied to *alert emission only*. Run the same stream with
a refractory of zero and of an hour and the state trace, the episodes, the
transitions and the recovery spans are byte-identical; only the ``suppressed``
flag on alerts differs. Suppressed alerts are still recorded.

**Causality.** The engine consumes an iterator and never reads past the window
it is deciding on. It keeps per-stream state and streaks, and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Final

from cardiosentinel.neural.t1_config import T1EpisodeConfig
from cardiosentinel.neural.t1_protocol import (
    T1_INITIAL_STATE,
    T1_STATE_EVENT,
    T1_STATE_NORMAL,
    T1_STATE_RECOVERY,
    T1_STATE_WATCH,
    T1_ZERO_STREAKS,
    T1Row,
    T1Streaks,
    T1Thresholds,
    is_cold_start,
    is_event_evidence,
    is_normal_evidence,
    is_watch_evidence,
    next_state,
    required_event_confirm_windows,
)
from cardiosentinel.neural.t1_stream import (
    T1AdaptedWindow,
    T1WindowEvidence,
    iter_adapted_windows,
    require_row_is_transition_safe,
)

EVIDENCE_EVENT: Final = "event"
EVIDENCE_NORMAL: Final = "normal"
EVIDENCE_AMBIGUOUS: Final = "ambiguous"
EVIDENCE_UNAVAILABLE: Final = "unavailable"

RECOVERY_CLEARED: Final = "cleared_to_normal"
RECOVERY_RE_ESCALATED: Final = "re_escalated_to_event"
RECOVERY_OPEN: Final = "open_at_stream_end"


class T1EngineError(RuntimeError):
    """Raised when the engine is asked to run something it cannot run honestly."""


@dataclass(frozen=True, slots=True)
class T1TraceEntry:
    """One window's decision, with everything needed to explain it."""

    window_id: str
    subject_id: str
    record_id: str
    channel_index: int
    start_sample: int
    elapsed_stream_seconds: float
    elapsed_state_seconds: float
    gap_seconds: float
    stream_restarted: bool
    score_present: bool
    detector_decision: bool | None
    calibrated_probability: float | None
    decision_error_uncertainty: float | None
    temporal_evidence: float | None
    cold_start: bool
    required_event_confirm_windows: int | None
    evidence_level: str
    state_before: str
    state_after: str
    streaks_before: dict[str, int]
    streaks_after: dict[str, int]
    context_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class T1TransitionEntry:
    """A window where the state changed, and why."""

    window_id: str
    subject_id: str
    record_id: str
    channel_index: int
    start_sample: int
    elapsed_stream_seconds: float
    state_before: str
    state_after: str
    evidence_level: str
    reason: str


@dataclass(frozen=True, slots=True)
class T1Episode:
    """A maximal EVENT run on one stream."""

    subject_id: str
    record_id: str
    channel_index: int
    onset_window_id: str
    offset_window_id: str
    onset_start_sample: int
    offset_start_sample: int
    window_count: int
    duration_seconds: float
    onset_elapsed_stream_seconds: float
    cold_start_onset: bool
    closed: bool


@dataclass(frozen=True, slots=True)
class T1Alert:
    """One operational notification, possibly suppressed by the refractory."""

    window_id: str
    subject_id: str
    record_id: str
    channel_index: int
    start_sample: int
    elapsed_stream_seconds: float
    entered_state: str
    suppressed: bool
    seconds_since_previous_alert: float | None
    refractory_seconds: float


@dataclass(frozen=True, slots=True)
class T1RecoverySpan:
    """One RECOVERY period and its outcome."""

    subject_id: str
    record_id: str
    channel_index: int
    entered_window_id: str
    exited_window_id: str | None
    entered_start_sample: int
    window_count: int
    duration_seconds: float
    outcome: str


@dataclass(slots=True)
class _StreamState:
    """Per-stream mutable bookkeeping. Never shared across streams."""

    state: str = T1_INITIAL_STATE
    streaks: T1Streaks = T1_ZERO_STREAKS
    elapsed_state_seconds: float = 0.0
    last_start_sample: int | None = None
    episode_open: dict[str, Any] | None = None
    recovery_open: dict[str, Any] | None = None
    last_alert_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class T1RunOutputs:
    """Everything one execution produced."""

    state_trace: tuple[T1TraceEntry, ...]
    episodes: tuple[T1Episode, ...]
    transitions: tuple[T1TransitionEntry, ...]
    alerts: tuple[T1Alert, ...]
    recovery_spans: tuple[T1RecoverySpan, ...]
    window_count: int
    available_window_count: int
    unavailable_window_count: int
    stream_count: int
    subject_count: int

    def summary(self) -> dict[str, Any]:
        """Counts only. Deliberately no rate, score or performance claim."""
        return {
            "window_count": self.window_count,
            "available_window_count": self.available_window_count,
            "unavailable_window_count": self.unavailable_window_count,
            "stream_count": self.stream_count,
            "subject_count": self.subject_count,
            "state_trace_length": len(self.state_trace),
            "episode_count": len(self.episodes),
            "transition_count": len(self.transitions),
            "alert_count": len(self.alerts),
            "alert_emitted_count": sum(
                1 for alert in self.alerts if not alert.suppressed
            ),
            "alert_suppressed_count": sum(
                1 for alert in self.alerts if alert.suppressed
            ),
            "recovery_span_count": len(self.recovery_spans),
        }

    def as_json_payload(self) -> dict[str, Any]:
        """A deterministic, JSON-safe rendering of every output."""
        return {
            "state_trace": [_as_dict(entry) for entry in self.state_trace],
            "episodes": [_as_dict(entry) for entry in self.episodes],
            "transitions": [_as_dict(entry) for entry in self.transitions],
            "alerts": [_as_dict(entry) for entry in self.alerts],
            "recovery_spans": [_as_dict(entry) for entry in self.recovery_spans],
            "summary": self.summary(),
        }


def _as_dict(entry: Any) -> dict[str, Any]:
    payload = asdict(entry)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }


def resolve_thresholds(config: T1EpisodeConfig) -> T1Thresholds:
    """The thresholds this run will use, or a refusal explaining why it cannot.

    A derived run needs a FIT population and the frozen order-statistic rule.
    Assembling that population is the canonical development harness's job and
    needs its own authorization, so the engine refuses rather than inventing a
    default.
    """
    if config.literal_thresholds is not None:
        return config.literal_thresholds
    raise T1EngineError(
        "This config derives its thresholds from FIT-subject background "
        "negatives, which requires an authorized canonical development run to "
        "assemble that population. The engine will not substitute a default: a "
        "silently defaulted threshold is a hand-chosen threshold wearing a "
        "generated threshold's name."
    )


def classify_evidence(row: T1Row, thresholds: T1Thresholds) -> str:
    """Which of the four frozen evidence levels this window argues for."""
    if not row.score_present:
        return EVIDENCE_UNAVAILABLE
    if is_event_evidence(row, thresholds):
        return EVIDENCE_EVENT
    if is_normal_evidence(row, thresholds):
        return EVIDENCE_NORMAL
    if is_watch_evidence(row, thresholds):
        return EVIDENCE_AMBIGUOUS
    return EVIDENCE_AMBIGUOUS


def _transition_reason(
    state_before: str,
    state_after: str,
    evidence_level: str,
    streaks_before: T1Streaks,
    config: T1EpisodeConfig,
    row: T1Row,
) -> str:
    """A sentence a reviewer can check against the transition table."""
    profile = config.profile
    if evidence_level == EVIDENCE_UNAVAILABLE:
        return (
            "window unavailable: state held, every confirmation streak reset, no "
            "transition considered"
        )
    if state_after == T1_STATE_EVENT and state_before != T1_STATE_EVENT:
        needed = required_event_confirm_windows(row, profile)
        if state_before == T1_STATE_RECOVERY:
            return (
                f"re-escalation from RECOVERY: EVENT evidence on "
                f"{streaks_before.re_event_confirm + 1} consecutive available "
                f"windows, meeting re_event_confirm_windows="
                f"{profile.re_event_confirm_windows}"
            )
        budget = "cold-start" if is_cold_start(row) else "mature"
        return (
            f"escalation to EVENT: EVENT evidence on "
            f"{streaks_before.event_confirm + 1} consecutive available windows, "
            f"meeting the {budget} budget of {needed}"
        )
    if state_before == T1_STATE_EVENT and state_after == T1_STATE_RECOVERY:
        return (
            f"release from EVENT: NORMAL evidence on "
            f"{streaks_before.event_release + 1} consecutive available windows, "
            f"meeting event_release_windows="
            f"{profile.event_release_windows}"
        )
    if state_after == T1_STATE_NORMAL and state_before == T1_STATE_RECOVERY:
        return (
            f"full clear: NORMAL evidence on {streaks_before.recovery_clear + 1} "
            f"consecutive available windows, meeting recovery_clear_windows="
            f"{profile.recovery_clear_windows}"
        )
    if state_after == T1_STATE_NORMAL:
        return (
            f"de-escalation to NORMAL: NORMAL evidence on "
            f"{streaks_before.watch_clear + 1} consecutive available windows, "
            f"meeting watch_clear_windows={profile.watch_clear_windows}"
        )
    if state_after == T1_STATE_WATCH:
        return "WATCH entered immediately on one window of WATCH-level evidence"
    return f"{state_before} -> {state_after} on {evidence_level} evidence"


def _close_episode(
    stream: _StreamState, *, closed: bool, stride_seconds: float
) -> T1Episode | None:
    open_episode = stream.episode_open
    if open_episode is None:
        return None
    stream.episode_open = None
    return T1Episode(
        subject_id=open_episode["subject_id"],
        record_id=open_episode["record_id"],
        channel_index=open_episode["channel_index"],
        onset_window_id=open_episode["onset_window_id"],
        offset_window_id=open_episode["last_window_id"],
        onset_start_sample=open_episode["onset_start_sample"],
        offset_start_sample=open_episode["last_start_sample"],
        window_count=open_episode["window_count"],
        duration_seconds=round(open_episode["window_count"] * stride_seconds, 6),
        onset_elapsed_stream_seconds=open_episode["onset_elapsed_stream_seconds"],
        cold_start_onset=open_episode["cold_start_onset"],
        closed=closed,
    )


def _close_recovery(
    stream: _StreamState,
    *,
    outcome: str,
    exited_window_id: str | None,
    stride_seconds: float,
) -> T1RecoverySpan | None:
    open_recovery = stream.recovery_open
    if open_recovery is None:
        return None
    stream.recovery_open = None
    return T1RecoverySpan(
        subject_id=open_recovery["subject_id"],
        record_id=open_recovery["record_id"],
        channel_index=open_recovery["channel_index"],
        entered_window_id=open_recovery["entered_window_id"],
        exited_window_id=exited_window_id,
        entered_start_sample=open_recovery["entered_start_sample"],
        window_count=open_recovery["window_count"],
        duration_seconds=round(open_recovery["window_count"] * stride_seconds, 6),
        outcome=outcome,
    )


def run_t1_episode_state_machine(
    windows: Iterable[T1WindowEvidence],
    config: T1EpisodeConfig,
    *,
    thresholds: T1Thresholds | None = None,
) -> T1RunOutputs:
    """Run the frozen state machine over a chronological window stream.

    The same input stream and config always produce the same outputs: nothing
    here reads a clock, a random source, an environment variable or the
    filesystem. Run provenance is captured separately, by the run scaffold.
    """
    active = thresholds if thresholds is not None else resolve_thresholds(config)
    stride_seconds = config.stride_seconds
    streams: dict[tuple[str, int], _StreamState] = {}
    subjects: set[str] = set()

    trace: list[T1TraceEntry] = []
    episodes: list[T1Episode] = []
    transitions: list[T1TransitionEntry] = []
    alerts: list[T1Alert] = []
    recoveries: list[T1RecoverySpan] = []

    available = 0
    unavailable = 0

    adapted: T1AdaptedWindow
    for adapted in iter_adapted_windows(windows, config):
        row = require_row_is_transition_safe(adapted.row)
        context = adapted.context
        key = adapted.stream_key
        subjects.add(context.subject_id)
        stream = streams.setdefault(key, _StreamState())

        if adapted.stream_restarted:
            # A dropout longer than the configured tolerance ends the stream's
            # continuity. This is a harness-layer operational policy, disabled
            # by default; it never alters what the frozen transition does.
            finished = _close_episode(
                stream, closed=False, stride_seconds=stride_seconds
            )
            if finished is not None:
                episodes.append(finished)
            span = _close_recovery(
                stream,
                outcome=RECOVERY_OPEN,
                exited_window_id=None,
                stride_seconds=stride_seconds,
            )
            if span is not None:
                recoveries.append(span)
            streams[key] = stream = _StreamState()

        if stream.last_start_sample is None:
            stream.elapsed_state_seconds = 0.0
        else:
            stream.elapsed_state_seconds += (
                context.start_sample - stream.last_start_sample
            ) / float(config.sampling_frequency_hz)
        stream.last_start_sample = context.start_sample

        evidence_level = classify_evidence(row, active)
        if row.score_present:
            available += 1
        else:
            unavailable += 1

        state_before = stream.state
        streaks_before = stream.streaks
        state_after, streaks_after = next_state(
            state_before, streaks_before, row, active, config.profile
        )
        stream.state = state_after
        stream.streaks = streaks_after
        if state_after != state_before:
            stream.elapsed_state_seconds = 0.0

        trace.append(
            T1TraceEntry(
                window_id=context.window_id,
                subject_id=context.subject_id,
                record_id=context.record_id,
                channel_index=context.channel_index,
                start_sample=context.start_sample,
                elapsed_stream_seconds=adapted.elapsed_stream_seconds,
                elapsed_state_seconds=stream.elapsed_state_seconds,
                gap_seconds=adapted.gap_seconds,
                stream_restarted=adapted.stream_restarted,
                score_present=row.score_present,
                detector_decision=row.detector_decision,
                calibrated_probability=row.calibrated_probability,
                decision_error_uncertainty=row.decision_error_uncertainty,
                temporal_evidence=row.temporal_evidence,
                cold_start=is_cold_start(row),
                required_event_confirm_windows=(
                    required_event_confirm_windows(row, config.profile)
                    if row.score_present
                    else None
                ),
                evidence_level=evidence_level,
                state_before=state_before,
                state_after=state_after,
                streaks_before=streaks_before._asdict(),
                streaks_after=streaks_after._asdict(),
                context_flags=context.context_flags,
            )
        )

        if state_after != state_before:
            transitions.append(
                T1TransitionEntry(
                    window_id=context.window_id,
                    subject_id=context.subject_id,
                    record_id=context.record_id,
                    channel_index=context.channel_index,
                    start_sample=context.start_sample,
                    elapsed_stream_seconds=adapted.elapsed_stream_seconds,
                    state_before=state_before,
                    state_after=state_after,
                    evidence_level=evidence_level,
                    reason=_transition_reason(
                        state_before,
                        state_after,
                        evidence_level,
                        streaks_before,
                        config,
                        row,
                    ),
                )
            )
            if state_after in config.alert_on_entry_to:
                alerts.append(
                    _build_alert(
                        stream=stream,
                        context=context,
                        elapsed_stream_seconds=adapted.elapsed_stream_seconds,
                        entered_state=state_after,
                        config=config,
                    )
                )

        # Episode bookkeeping: a maximal EVENT run on this stream.
        if state_after == T1_STATE_EVENT:
            if stream.episode_open is None:
                stream.episode_open = {
                    "subject_id": context.subject_id,
                    "record_id": context.record_id,
                    "channel_index": context.channel_index,
                    "onset_window_id": context.window_id,
                    "onset_start_sample": context.start_sample,
                    "onset_elapsed_stream_seconds": adapted.elapsed_stream_seconds,
                    "cold_start_onset": is_cold_start(row),
                    "window_count": 0,
                    "last_window_id": context.window_id,
                    "last_start_sample": context.start_sample,
                }
            stream.episode_open["window_count"] += 1
            stream.episode_open["last_window_id"] = context.window_id
            stream.episode_open["last_start_sample"] = context.start_sample
        elif stream.episode_open is not None:
            finished = _close_episode(
                stream, closed=True, stride_seconds=stride_seconds
            )
            if finished is not None:
                episodes.append(finished)

        # Recovery bookkeeping.
        if state_after == T1_STATE_RECOVERY:
            if stream.recovery_open is None:
                stream.recovery_open = {
                    "subject_id": context.subject_id,
                    "record_id": context.record_id,
                    "channel_index": context.channel_index,
                    "entered_window_id": context.window_id,
                    "entered_start_sample": context.start_sample,
                    "window_count": 0,
                }
            stream.recovery_open["window_count"] += 1
        elif stream.recovery_open is not None:
            outcome = (
                RECOVERY_RE_ESCALATED
                if state_after == T1_STATE_EVENT
                else RECOVERY_CLEARED
            )
            span = _close_recovery(
                stream,
                outcome=outcome,
                exited_window_id=context.window_id,
                stride_seconds=stride_seconds,
            )
            if span is not None:
                recoveries.append(span)

    for key in sorted(streams):
        stream = streams[key]
        finished = _close_episode(stream, closed=False, stride_seconds=stride_seconds)
        if finished is not None:
            episodes.append(finished)
        span = _close_recovery(
            stream,
            outcome=RECOVERY_OPEN,
            exited_window_id=None,
            stride_seconds=stride_seconds,
        )
        if span is not None:
            recoveries.append(span)

    return T1RunOutputs(
        state_trace=tuple(trace),
        episodes=tuple(episodes),
        transitions=tuple(transitions),
        alerts=tuple(alerts),
        recovery_spans=tuple(recoveries),
        window_count=available + unavailable,
        available_window_count=available,
        unavailable_window_count=unavailable,
        stream_count=len(streams),
        subject_count=len(subjects),
    )


def _build_alert(
    *,
    stream: _StreamState,
    context: Any,
    elapsed_stream_seconds: float,
    entered_state: str,
    config: T1EpisodeConfig,
) -> T1Alert:
    """Apply the refractory to emission only; the state machine never sees it."""
    previous = stream.last_alert_seconds.get(entered_state)
    since = None if previous is None else elapsed_stream_seconds - previous
    suppressed = since is not None and since < config.refractory_seconds
    if not suppressed:
        stream.last_alert_seconds[entered_state] = elapsed_stream_seconds
    return T1Alert(
        window_id=context.window_id,
        subject_id=context.subject_id,
        record_id=context.record_id,
        channel_index=context.channel_index,
        start_sample=context.start_sample,
        elapsed_stream_seconds=elapsed_stream_seconds,
        entered_state=entered_state,
        suppressed=suppressed,
        seconds_since_previous_alert=since,
        refractory_seconds=config.refractory_seconds,
    )
