"""Typed canonical records and annotations without reducing them to binary labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


class AnnotationValidationError(ValueError):
    """Raised when an annotation stream violates its documented protocol."""


@dataclass(frozen=True)
class AnnotationSample:
    """Minimal WFDB-like annotation used by adapters and synthetic tests."""

    sample: int
    symbol: str
    subtype: int = 0
    channel: int = 0
    aux_note: str = ""


@dataclass(frozen=True)
class DatasetRecord:
    """Metadata read from a WFDB header for one ECG record."""

    dataset_id: str
    dataset_version: str
    record_id: str
    subject_id: str
    sampling_frequency_hz: float
    signal_names: tuple[str, ...]
    lead_names: tuple[str | None, ...]
    signal_count: int
    sample_count: int
    duration_seconds: float
    source_path: str
    header_metadata: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class STEvent:
    """An expert ST episode with original semantics and timing retained."""

    dataset_id: str
    record_id: str
    subject_id: str
    lead: int
    event_family: str
    event_subtype: str
    onset_sample: int
    peak_sample: int | None
    end_sample: int
    onset_seconds: float
    peak_seconds: float | None
    end_seconds: float
    peak_deviation_uv: float | None
    direction: str | None
    annotation_source: str
    annotation_definition: str
    is_primary_definition: bool
    original_annotations: tuple[AnnotationSample, ...]


@dataclass(frozen=True)
class SignalQualityInterval:
    """A quality or unreadable interval, separated from ST-event semantics."""

    record_id: str
    subject_id: str
    lead: int | None
    start_sample: int
    end_sample: int
    state: str
    annotation_source: str
    original_annotations: tuple[AnnotationSample, ...]


@dataclass(frozen=True)
class AnnotationMarker:
    """A traceable point annotation that is not an onset-to-end episode."""

    record_id: str
    subject_id: str
    lead: int | None
    sample: int
    category: str
    subtype: str | None
    annotation_source: str
    original_annotation: AnnotationSample


@dataclass(frozen=True)
class ParsedAnnotations:
    """Events, quality intervals, and point markers from one annotation set."""

    events: tuple[STEvent, ...]
    quality_intervals: tuple[SignalQualityInterval, ...]
    markers: tuple[AnnotationMarker, ...]
