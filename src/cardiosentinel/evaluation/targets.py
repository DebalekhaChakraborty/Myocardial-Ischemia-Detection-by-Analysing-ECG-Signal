"""Annotation-after-window target assignment for benchmark protocol V1."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from cardiosentinel.data.models import (
    AnnotationMarker,
    DatasetRecord,
    SignalQualityInterval,
    SourceCensoredInterval,
    STEvent,
)
from cardiosentinel.evaluation.models import BenchmarkWindow, TargetFamily, WindowTarget
from cardiosentinel.evaluation.protocol import MARKER_VICINITY_SECONDS
from cardiosentinel.signal.windows import seconds_to_samples


def iter_causal_window_indices(
    record: DatasetRecord,
    length_seconds: float,
    stride_seconds: float,
) -> Iterator[BenchmarkWindow]:
    """Generate causal geometry from record metadata only, without annotations."""
    length = seconds_to_samples(
        length_seconds, record.sampling_frequency_hz, "window length"
    )
    stride = seconds_to_samples(
        stride_seconds, record.sampling_frequency_hz, "window stride"
    )
    for start in range(0, record.sample_count - length + 1, stride):
        end = start + length
        for channel in range(record.signal_count):
            yield BenchmarkWindow(
                dataset=record.dataset_id,
                record_id=record.record_id,
                subject_id=record.subject_id,
                channel_index=channel,
                lead_name=record.lead_names[channel],
                sampling_frequency_hz=record.sampling_frequency_hz,
                start_sample=start,
                end_sample=end,
                available_at_sample=end,
            )


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def _same_lead(window: BenchmarkWindow, lead: int | None) -> bool:
    return lead is None or lead == window.channel_index


def _event_id(event: STEvent) -> str:
    return (
        f"{event.annotation_definition}:{event.record_id}:{event.lead}:"
        f"{event.event_subtype}:{event.onset_sample}:{event.end_sample}"
    )


def _marker_id(marker: AnnotationMarker) -> str:
    return (
        f"{marker.annotation_source}:{marker.record_id}:{marker.lead}:"
        f"{marker.subtype}:{marker.sample}"
    )


def _target(
    window: BenchmarkWindow,
    family: TargetFamily,
    subtype: str | None,
    annotation_definition: str,
    overlapping_ids: Iterable[str],
    quality_state: str | None = None,
    exclusion_reason: str | None = None,
) -> WindowTarget:
    primary = family in {"ischemic_positive", "background_negative"}
    confounder = family in {
        "rate_related_confounder",
        "axis_shift_confounder",
        "conduction_change_confounder",
    }
    return WindowTarget(
        dataset=window.dataset,
        record_id=window.record_id,
        subject_id=window.subject_id,
        channel_index=window.channel_index,
        lead_name=window.lead_name,
        window_start_sample=window.start_sample,
        window_end_sample=window.end_sample,
        target_family=family,
        target_subtype=subtype,
        eligible_for_training=primary,
        eligible_for_primary_evaluation=primary,
        eligible_for_confounder_evaluation=confounder,
        annotation_definition=annotation_definition,
        overlapping_event_ids=tuple(sorted(set(overlapping_ids))),
        quality_state=quality_state,
        exclusion_reason=exclusion_reason,
    )


def assign_window_target(
    window: BenchmarkWindow,
    events: Iterable[STEvent],
    quality_intervals: Iterable[SignalQualityInterval],
    markers: Iterable[AnnotationMarker],
    source_censored_intervals: Iterable[SourceCensoredInterval],
    annotation_definition: str,
    marker_vicinity_seconds: float = MARKER_VICINITY_SECONDS,
    marker_vicinity_samples: int | None = None,
) -> WindowTarget:
    """Assign one semantic target after annotation-independent geometry exists."""
    events = tuple(
        event
        for event in events
        if event.record_id == window.record_id
        and _same_lead(window, event.lead)
        and _overlaps(
            window.start_sample,
            window.end_sample,
            event.onset_sample,
            event.end_sample,
        )
    )
    event_ids = tuple(_event_id(event) for event in events)
    quality = tuple(
        interval
        for interval in quality_intervals
        if interval.record_id == window.record_id
        and _same_lead(window, interval.lead)
        and _overlaps(
            window.start_sample,
            window.end_sample,
            interval.start_sample,
            interval.end_sample,
        )
    )
    quality_states = {interval.state for interval in quality}
    if "unreadable" in quality_states:
        return _target(
            window,
            "quality_excluded",
            "expert_unreadable_interval",
            annotation_definition,
            event_ids,
            quality_state="unreadable",
            exclusion_reason="overlaps expert unreadable reference interval",
        )

    censored = tuple(
        interval
        for interval in source_censored_intervals
        if interval.record_id == window.record_id
        and _same_lead(window, interval.lead)
        and _overlaps(
            window.start_sample,
            window.end_sample,
            interval.start_sample,
            interval.end_sample,
        )
    )
    if censored:
        reasons = tuple(sorted({interval.reason for interval in censored}))
        return _target(
            window,
            "source_censored_or_unknown",
            ",".join(reasons),
            annotation_definition,
            event_ids,
            quality_state="noisy" if "noisy" in quality_states else None,
            exclusion_reason="overlaps source-censored annotation region",
        )

    ischemic = tuple(
        event
        for event in events
        if event.event_subtype in {"ischemic", "reference_st_change"}
    )
    fully_contained = tuple(
        event
        for event in ischemic
        if window.start_sample >= event.onset_sample
        and window.end_sample <= event.end_sample
    )
    if fully_contained:
        return _target(
            window,
            "ischemic_positive",
            "fully_contained",
            annotation_definition,
            event_ids,
            quality_state="noisy" if "noisy" in quality_states else None,
        )
    if ischemic:
        return _target(
            window,
            "boundary_ambiguous",
            "partial_ischemic_overlap",
            annotation_definition,
            event_ids,
            quality_state="noisy" if "noisy" in quality_states else None,
            exclusion_reason="partially overlaps an ischemic episode boundary",
        )

    rate_related = tuple(
        event for event in events if event.event_subtype == "heart_rate_related"
    )
    if rate_related:
        fully_rate_related = all(
            window.start_sample >= event.onset_sample
            and window.end_sample <= event.end_sample
            for event in rate_related
        )
        return _target(
            window,
            "rate_related_confounder",
            "fully_contained" if fully_rate_related else "boundary_overlap",
            annotation_definition,
            event_ids,
            quality_state="noisy" if "noisy" in quality_states else None,
        )

    axis_events = tuple(
        event for event in events if event.event_subtype == "apparent_st_change"
    )
    if axis_events:
        return _target(
            window,
            "axis_shift_confounder",
            "duration_bounded_apparent_st_change",
            annotation_definition,
            event_ids,
            quality_state="noisy" if "noisy" in quality_states else None,
        )

    radius = marker_vicinity_samples
    if radius is None:
        radius = seconds_to_samples(
            marker_vicinity_seconds,
            window.sampling_frequency_hz,
            "marker challenge vicinity",
        )
    nearby_markers = tuple(
        marker
        for marker in markers
        if marker.record_id == window.record_id
        and _same_lead(window, marker.lead)
        and marker.subtype in {"axis_related", "conduction_related"}
        and _overlaps(
            window.start_sample,
            window.end_sample,
            max(0, marker.sample - radius),
            marker.sample + radius,
        )
    )
    marker_ids = tuple(_marker_id(marker) for marker in nearby_markers)
    if any(marker.subtype == "conduction_related" for marker in nearby_markers):
        return _target(
            window,
            "conduction_change_confounder",
            "marker_vicinity_30_seconds",
            annotation_definition,
            (*event_ids, *marker_ids),
            quality_state="noisy" if "noisy" in quality_states else None,
        )
    if nearby_markers:
        return _target(
            window,
            "axis_shift_confounder",
            "marker_vicinity_30_seconds",
            annotation_definition,
            (*event_ids, *marker_ids),
            quality_state="noisy" if "noisy" in quality_states else None,
        )
    return _target(
        window,
        "background_negative",
        "clean_by_v1_exclusions",
        annotation_definition,
        event_ids,
        quality_state="noisy" if "noisy" in quality_states else None,
    )
