"""Typed canonical records and annotations without reducing them to binary labels."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Mapping


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


AnnotationDisposition = Literal[
    "recognized_relevant", "recognized_irrelevant", "unknown"
]


@dataclass(frozen=True)
class AnnotationClassification:
    """Explicit accounting decision for one raw WFDB annotation."""

    annotation: AnnotationSample
    disposition: AnnotationDisposition
    reason: str
    potentially_st_related: bool = False


def annotation_pattern(aux_note: str) -> str:
    """Return a non-identifying structural representation of an auxiliary note."""
    normalized = aux_note.strip(" \t\r\n\x00")
    if not normalized:
        return "<empty>"
    normalized = re.sub(r"[A-Za-z]+", "A", normalized)
    normalized = re.sub(r"\d+", "#", normalized)
    return normalized[:80]


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
    classifications: tuple[AnnotationClassification, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def recognized_annotation_count(self) -> int:
        """Count source annotations used as explicit, relevant semantics."""
        return sum(
            item.disposition == "recognized_relevant" for item in self.classifications
        )

    @property
    def ignored_known_annotation_count(self) -> int:
        """Count explicitly known annotations outside this phase's target schema."""
        return sum(
            item.disposition == "recognized_irrelevant"
            for item in self.classifications
        )

    @property
    def unknown_annotations(self) -> tuple[AnnotationClassification, ...]:
        """Return annotations with no documented classification."""
        return tuple(
            item for item in self.classifications if item.disposition == "unknown"
        )

    @property
    def unknown_relevant_annotations(self) -> tuple[AnnotationClassification, ...]:
        """Return unknown forms that might alter ST semantics."""
        return tuple(
            item for item in self.unknown_annotations if item.potentially_st_related
        )

    @property
    def unknown_aux_patterns(self) -> tuple[str, ...]:
        """Return bounded structural forms without exposing raw auxiliary text."""
        return tuple(
            sorted(
                {
                    annotation_pattern(item.annotation.aux_note)
                    for item in self.unknown_annotations
                }
            )[:20]
        )
