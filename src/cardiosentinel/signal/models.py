"""Typed physical ECG structures with explicit sample and unit semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.signal.errors import SignalValidationError


def _readonly_float64(values: NDArray[np.floating] | object) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class WaveformSegment:
    """A physical ECG interval in canonical mV with shape [samples, channels]."""

    dataset_id: str
    dataset_version: str
    record_id: str
    subject_id: str
    sampling_frequency_hz: float
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    signal_names: tuple[str, ...]
    lead_names: tuple[str | None, ...]
    physical_units: tuple[str, ...]
    source_physical_units: tuple[str, ...]
    values: NDArray[np.float64]
    source: str
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _readonly_float64(self.values)
        object.__setattr__(self, "values", values)
        if values.ndim != 2:
            raise SignalValidationError(
                "Waveform values must have [samples, channels]."
            )
        if self.sampling_frequency_hz <= 0 or not np.isfinite(
            self.sampling_frequency_hz
        ):
            raise SignalValidationError(
                "Sampling frequency must be positive and finite."
            )
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise SignalValidationError("Waveform sample interval must be non-empty.")
        if values.shape[0] != self.end_sample - self.start_sample:
            raise SignalValidationError(
                "Waveform length does not match sample interval."
            )
        channel_count = values.shape[1]
        if channel_count == 0:
            raise SignalValidationError(
                "Waveform segment must have at least one channel."
            )
        metadata_lengths = {
            len(self.signal_names),
            len(self.lead_names),
            len(self.physical_units),
            len(self.source_physical_units),
        }
        if metadata_lengths != {channel_count}:
            raise SignalValidationError(
                "Channel metadata does not match waveform shape."
            )
        expected_start = self.start_sample / self.sampling_frequency_hz
        expected_end = self.end_sample / self.sampling_frequency_hz
        if not isclose(self.start_seconds, expected_start, abs_tol=1e-12):
            raise SignalValidationError("start_seconds does not match start_sample.")
        if not isclose(self.end_seconds, expected_end, abs_tol=1e-12):
            raise SignalValidationError("end_seconds does not match end_sample.")

    @property
    def sample_count(self) -> int:
        return self.values.shape[0]

    @property
    def channel_count(self) -> int:
        return self.values.shape[1]


@dataclass(frozen=True)
class ProcessedWaveformSegment:
    """Waveform output plus visible causal-filter startup state."""

    waveform: WaveformSegment
    profile_name: str
    is_warm: bool
    warmup_samples_in_segment: int
    warmup_samples_remaining: int
    processed_sample_count: int

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_samples_in_segment <= self.waveform.sample_count:
            raise SignalValidationError("Processed-segment warm-up count is invalid.")
        if self.warmup_samples_remaining < 0:
            raise SignalValidationError("Remaining warm-up cannot be negative.")
        if self.processed_sample_count < self.waveform.sample_count:
            raise SignalValidationError("Processed sample count is inconsistent.")


@dataclass(frozen=True)
class CausalWindow:
    """One completed single-channel window, emitted only after its final sample."""

    record_id: str
    subject_id: str
    channel_index: int
    signal_name: str
    lead_name: str | None
    physical_unit: str
    sampling_frequency_hz: float
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    available_at_sample: int
    contains_filter_warmup: bool
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        values = _readonly_float64(self.values)
        object.__setattr__(self, "values", values)
        if self.sampling_frequency_hz <= 0 or not np.isfinite(
            self.sampling_frequency_hz
        ):
            raise SignalValidationError("Window sampling frequency must be valid.")
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise SignalValidationError("Window sample interval must be non-empty.")
        if values.ndim != 1 or values.size != self.end_sample - self.start_sample:
            raise SignalValidationError("Window values must match its sample interval.")
        if not isclose(
            self.start_seconds,
            self.start_sample / self.sampling_frequency_hz,
            abs_tol=1e-12,
        ) or not isclose(
            self.end_seconds,
            self.end_sample / self.sampling_frequency_hz,
            abs_tol=1e-12,
        ):
            raise SignalValidationError("Window times do not match sample indices.")
        if self.available_at_sample < self.end_sample:
            raise SignalValidationError("A window cannot be available before its end.")


@dataclass(frozen=True)
class SignalQualityMetrics:
    """Label-free descriptive metrics; no aggregate clinical score is implied."""

    finite_sample_fraction: float
    flatline_fraction: float | None
    repeated_value_fraction: float | None
    robust_amplitude_range_mv: float | None
    robust_derivative_scale_mv: float | None
    derivative_outlier_fraction: float | None
    low_frequency_power_ratio: float | None
    high_frequency_power_ratio: float | None
    powerline_ratio_50hz: float | None
    powerline_ratio_60hz: float | None
