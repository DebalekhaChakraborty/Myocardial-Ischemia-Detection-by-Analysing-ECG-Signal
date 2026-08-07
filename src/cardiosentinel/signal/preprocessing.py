"""Stateful, strictly causal ECG preprocessing using SciPy SOS filters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import ceil, log

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from cardiosentinel.signal.config import FilterProfile
from cardiosentinel.signal.models import (
    ProcessedWaveformSegment,
    SignalValidationError,
    WaveformSegment,
)
from cardiosentinel.signal.validation import validate_waveform_segment

PROCESSING_SCHEMA_VERSION = "2.0"
WARMUP_RESIDUAL_TOLERANCE = 1e-3
AUDIT_FREQUENCIES_HZ = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 40.0, 50.0, 60.0)


@dataclass(frozen=True)
class DesignedFilter:
    sos: NDArray[np.float64]
    components: tuple[dict[str, object], ...]


def design_filter(
    profile: FilterProfile, sampling_frequency_hz: float
) -> DesignedFilter:
    """Design stable library-provided filter sections for an explicit profile."""
    if sampling_frequency_hz <= 0 or not np.isfinite(sampling_frequency_hz):
        raise SignalValidationError("Sampling frequency must be positive and finite.")
    nyquist = sampling_frequency_hz / 2.0
    sections: list[NDArray[np.float64]] = []
    components: list[dict[str, object]] = []
    if profile.highpass.enabled:
        cutoff = float(profile.highpass.cutoff_hz)
        _validate_below_nyquist("high-pass", cutoff, nyquist)
        sos = scipy_signal.butter(
            int(profile.highpass.order),
            cutoff,
            btype="highpass",
            fs=sampling_frequency_hz,
            output="sos",
        )
        sections.append(sos)
        components.append(
            {
                "filter_type": "butterworth_highpass",
                "order": profile.highpass.order,
                "cutoff_hz": cutoff,
            }
        )
    if profile.lowpass.enabled:
        cutoff = float(profile.lowpass.cutoff_hz)
        _validate_below_nyquist("low-pass", cutoff, nyquist)
        sos = scipy_signal.butter(
            int(profile.lowpass.order),
            cutoff,
            btype="lowpass",
            fs=sampling_frequency_hz,
            output="sos",
        )
        sections.append(sos)
        components.append(
            {
                "filter_type": "butterworth_lowpass",
                "order": profile.lowpass.order,
                "cutoff_hz": cutoff,
            }
        )
    if profile.notch.enabled:
        frequency = float(profile.notch.frequency_hz)
        _validate_below_nyquist("notch", frequency, nyquist)
        numerator, denominator = scipy_signal.iirnotch(
            frequency, float(profile.notch.quality_factor), fs=sampling_frequency_hz
        )
        sos = scipy_signal.tf2sos(numerator, denominator)
        sections.append(sos)
        components.append(
            {
                "filter_type": "iir_notch",
                "frequency_hz": frequency,
                "quality_factor": profile.notch.quality_factor,
            }
        )
    combined = np.vstack(sections) if sections else np.empty((0, 6), dtype=np.float64)
    return DesignedFilter(np.asarray(combined, dtype=np.float64), tuple(components))


def _validate_below_nyquist(name: str, frequency: float, nyquist: float) -> None:
    if frequency <= 0 or frequency >= nyquist:
        raise SignalValidationError(
            f"{name} frequency {frequency} must be below Nyquist {nyquist}."
        )


def warmup_samples_for_sos(
    sos: NDArray[np.float64], residual_tolerance: float = WARMUP_RESIDUAL_TOLERANCE
) -> int:
    """Estimate causal settling from the slowest pole and a declared tolerance."""
    if sos.size == 0:
        return 0
    if not 0 < residual_tolerance < 1:
        raise SignalValidationError(
            "Warm-up residual tolerance must lie between 0 and 1."
        )
    radii: list[float] = []
    for section in sos:
        denominator = section[3:]
        poles = np.roots(denominator)
        radii.extend(float(abs(pole)) for pole in poles)
    maximum_radius = max(radii, default=0.0)
    if maximum_radius >= 1.0:
        raise SignalValidationError("Designed causal filter is not stable.")
    if maximum_radius <= 0:
        return 0
    return max(0, ceil(log(residual_tolerance) / log(maximum_radius)))


class StreamingPreprocessor:
    """Persistent per-channel SOS state with exact sample continuity checks."""

    initial_state_strategy = "zero_state_no_assumed_preceding_history"

    def __init__(
        self,
        profile: FilterProfile,
        sampling_frequency_hz: float,
        channel_count: int,
    ) -> None:
        if channel_count <= 0:
            raise SignalValidationError("Streaming channel count must be positive.")
        self.profile = profile
        self.sampling_frequency_hz = float(sampling_frequency_hz)
        self.channel_count = channel_count
        self._design = design_filter(profile, sampling_frequency_hz)
        self.warmup_sample_count = warmup_samples_for_sos(self._design.sos)
        self._state = np.zeros(
            (self._design.sos.shape[0], 2, channel_count), dtype=np.float64
        )
        self.processed_sample_count = 0
        self._next_sample: int | None = None

    @property
    def sos_coefficients(self) -> NDArray[np.float64]:
        return self._design.sos.copy()

    @property
    def current_state(self) -> NDArray[np.float64]:
        return self._state.copy()

    @property
    def is_warm(self) -> bool:
        return self.processed_sample_count >= self.warmup_sample_count

    @property
    def warmup_samples_remaining(self) -> int:
        return max(0, self.warmup_sample_count - self.processed_sample_count)

    def reset(self) -> None:
        """Return to explicit zero history; subsequent samples begin a new stream."""
        self._state.fill(0.0)
        self.processed_sample_count = 0
        self._next_sample = None

    def process(self, waveform: WaveformSegment) -> ProcessedWaveformSegment:
        validate_waveform_segment(waveform, require_dynamic=False)
        if waveform.sampling_frequency_hz != self.sampling_frequency_hz:
            raise SignalValidationError(
                "Waveform sampling frequency changed within stream."
            )
        if waveform.channel_count != self.channel_count:
            raise SignalValidationError("Waveform channel count changed within stream.")
        if self._next_sample is not None and waveform.start_sample != self._next_sample:
            raise SignalValidationError("Waveform chunks are not sample-contiguous.")
        before = self.processed_sample_count
        if self.profile.is_raw:
            output = waveform.values.copy()
        else:
            output, self._state = scipy_signal.sosfilt(
                self._design.sos, waveform.values, axis=0, zi=self._state
            )
        self.processed_sample_count += waveform.sample_count
        self._next_sample = waveform.end_sample
        warmup_in_segment = min(
            waveform.sample_count, max(0, self.warmup_sample_count - before)
        )
        processed = WaveformSegment(
            dataset_id=waveform.dataset_id,
            dataset_version=waveform.dataset_version,
            record_id=waveform.record_id,
            subject_id=waveform.subject_id,
            sampling_frequency_hz=waveform.sampling_frequency_hz,
            start_sample=waveform.start_sample,
            end_sample=waveform.end_sample,
            start_seconds=waveform.start_seconds,
            end_seconds=waveform.end_seconds,
            signal_names=waveform.signal_names,
            lead_names=waveform.lead_names,
            physical_units=waveform.physical_units,
            source_physical_units=waveform.source_physical_units,
            values=output,
            source=waveform.source,
            provenance={
                **waveform.provenance,
                "processing_schema_version": PROCESSING_SCHEMA_VERSION,
                "filter_profile": self.profile.as_dict(),
                "initial_state_strategy": self.initial_state_strategy,
            },
        )
        return ProcessedWaveformSegment(
            waveform=processed,
            profile_name=self.profile.name,
            is_warm=self.is_warm,
            warmup_samples_in_segment=warmup_in_segment,
            warmup_samples_remaining=self.warmup_samples_remaining,
            processed_sample_count=self.processed_sample_count,
        )


def filter_audit(
    profile: FilterProfile, sampling_frequency_hz: float
) -> dict[str, object]:
    """Return reproducible response data; this is not clinical certification."""
    design = design_filter(profile, sampling_frequency_hz)
    frequencies = [
        frequency
        for frequency in AUDIT_FREQUENCIES_HZ
        if frequency < sampling_frequency_hz / 2.0
    ]
    if design.sos.size:
        _, response = scipy_signal.sosfreqz(
            design.sos, worN=np.asarray(frequencies), fs=sampling_frequency_hz
        )
    else:
        response = np.ones(len(frequencies), dtype=np.complex128)
    sos_list = design.sos.tolist()
    digest = hashlib.sha256(
        json.dumps(sos_list, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    from cardiosentinel.signal.provenance import runtime_provenance

    return {
        "processing_schema_version": PROCESSING_SCHEMA_VERSION,
        "research_only": True,
        "clinical_certification": False,
        "profile": profile.as_dict(),
        "dataset": None,
        "record": None,
        "sample_interval": None,
        "physical_units": None,
        "sampling_frequency_hz": sampling_frequency_hz,
        "initial_state_strategy": StreamingPreprocessor.initial_state_strategy,
        "warmup_residual_tolerance": WARMUP_RESIDUAL_TOLERANCE,
        "warmup_sample_count": warmup_samples_for_sos(design.sos),
        "components": list(design.components),
        "sos_coefficients": sos_list,
        "coefficient_sha256": digest,
        "runtime": runtime_provenance(),
        "response": [
            {
                "frequency_hz": frequency,
                "magnitude": float(abs(value)),
                "gain_db": float(
                    20.0 * np.log10(max(abs(value), np.finfo(float).tiny))
                ),
                "phase_degrees": float(np.degrees(np.angle(value))),
            }
            for frequency, value in zip(frequencies, response, strict=True)
        ],
    }
