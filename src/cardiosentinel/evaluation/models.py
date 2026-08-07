"""Typed benchmark windows, targets, burdens, and provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetFamily = Literal[
    "ischemic_positive",
    "background_negative",
    "rate_related_confounder",
    "axis_shift_confounder",
    "conduction_change_confounder",
    "quality_excluded",
    "boundary_ambiguous",
    "source_censored_or_unknown",
]


@dataclass(frozen=True)
class BenchmarkWindow:
    """A causal index-only window constructed without annotations or waveform values."""

    dataset: str
    record_id: str
    subject_id: str
    channel_index: int
    lead_name: str | None
    sampling_frequency_hz: float
    start_sample: int
    end_sample: int
    available_at_sample: int

    def __post_init__(self) -> None:
        if self.sampling_frequency_hz <= 0:
            raise ValueError("Benchmark window sampling frequency must be positive.")
        if self.channel_index < 0:
            raise ValueError("Benchmark window channel index cannot be negative.")
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("Benchmark window interval must be non-empty.")
        if self.available_at_sample < self.end_sample:
            raise ValueError("A benchmark window cannot be available before its end.")

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sampling_frequency_hz

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sampling_frequency_hz


@dataclass(frozen=True)
class WindowTarget:
    """Traceable target semantics without reducing every window to a binary label."""

    dataset: str
    record_id: str
    subject_id: str
    channel_index: int
    lead_name: str | None
    window_start_sample: int
    window_end_sample: int
    target_family: TargetFamily
    target_subtype: str | None
    eligible_for_training: bool
    eligible_for_primary_evaluation: bool
    eligible_for_confounder_evaluation: bool
    annotation_definition: str
    overlapping_event_ids: tuple[str, ...]
    quality_state: str | None
    exclusion_reason: str | None

    @property
    def stable_id(self) -> str:
        return (
            f"{self.dataset}:{self.record_id}:{self.channel_index}:"
            f"{self.window_start_sample}:{self.window_end_sample}"
        )


@dataclass(frozen=True)
class SubjectBurden:
    """Subject-level annotation metadata used only for pre-model split balancing."""

    subject_id: str
    records: tuple[str, ...]
    recording_seconds: float
    ischemic_episode_count: int
    ischemic_episode_seconds: float
    rate_related_episode_count: int
    rate_related_episode_seconds: float
    signal_channel_count: int


@dataclass(frozen=True)
class CrossDatasetProvenance:
    """Authoritative cross-release overlap evidence for one record."""

    dataset: str
    record: str
    source_collection: str
    overlap_risk: str
    verified_corresponding_dataset: str | None
    verified_corresponding_record: str | None
    evidence_source: str
    confidence: Literal["verified", "collection-level-risk", "unknown"]
