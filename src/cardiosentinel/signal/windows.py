"""Stateful causal windows constructed without labels or annotation timing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.signal.models import (
    CausalWindow,
    ProcessedWaveformSegment,
    SignalValidationError,
    WaveformSegment,
)


def seconds_to_samples(seconds: float, sampling_frequency_hz: float, name: str) -> int:
    """Resolve a duration only when it maps exactly to a source sample count."""
    if seconds <= 0 or sampling_frequency_hz <= 0:
        raise SignalValidationError(f"{name} and sampling frequency must be positive.")
    exact = seconds * sampling_frequency_hz
    samples = round(exact)
    if samples <= 0 or not np.isclose(exact, samples, atol=1e-9):
        raise SignalValidationError(f"{name} must align exactly with source samples.")
    return samples


class CausalWindowGenerator:
    """Emit windows after all required samples arrive, preserving chunk continuity."""

    def __init__(
        self,
        sampling_frequency_hz: float,
        length_seconds: float,
        stride_seconds: float,
    ) -> None:
        self.sampling_frequency_hz = float(sampling_frequency_hz)
        self.length_samples = seconds_to_samples(
            length_seconds, sampling_frequency_hz, "window length"
        )
        self.stride_samples = seconds_to_samples(
            stride_seconds, sampling_frequency_hz, "window stride"
        )
        self._values: NDArray[np.float64] | None = None
        self._warmup: NDArray[np.bool_] | None = None
        self._buffer_start: int | None = None
        self._stream_end: int | None = None
        self._next_window_start: int | None = None
        self._identity: tuple[object, ...] | None = None

    def process(
        self, segment: ProcessedWaveformSegment | WaveformSegment
    ) -> tuple[CausalWindow, ...]:
        waveform = (
            segment.waveform
            if isinstance(segment, ProcessedWaveformSegment)
            else segment
        )
        if waveform.sampling_frequency_hz != self.sampling_frequency_hz:
            raise SignalValidationError("Window stream sampling frequency changed.")
        identity = (
            waveform.record_id,
            waveform.subject_id,
            waveform.signal_names,
            waveform.lead_names,
            waveform.physical_units,
        )
        if self._identity is None:
            self._identity = identity
            self._buffer_start = waveform.start_sample
            self._next_window_start = waveform.start_sample
            self._values = np.empty((0, waveform.channel_count), dtype=np.float64)
            self._warmup = np.empty(0, dtype=np.bool_)
        elif identity != self._identity:
            raise SignalValidationError("Window stream identity or channels changed.")
        if self._stream_end is not None and waveform.start_sample != self._stream_end:
            raise SignalValidationError(
                "Window input chunks are not sample-contiguous."
            )
        warmup = np.zeros(waveform.sample_count, dtype=np.bool_)
        if isinstance(segment, ProcessedWaveformSegment):
            warmup[: segment.warmup_samples_in_segment] = True
        self._values = np.vstack((self._values, waveform.values))
        self._warmup = np.concatenate((self._warmup, warmup))
        self._stream_end = waveform.end_sample
        windows: list[CausalWindow] = []
        while self._next_window_start + self.length_samples <= self._stream_end:
            start = self._next_window_start
            end = start + self.length_samples
            offset = start - self._buffer_start
            values = self._values[offset : offset + self.length_samples]
            warmup_slice = self._warmup[offset : offset + self.length_samples]
            for channel in range(waveform.channel_count):
                windows.append(
                    CausalWindow(
                        record_id=waveform.record_id,
                        subject_id=waveform.subject_id,
                        channel_index=channel,
                        signal_name=waveform.signal_names[channel],
                        lead_name=waveform.lead_names[channel],
                        physical_unit=waveform.physical_units[channel],
                        sampling_frequency_hz=self.sampling_frequency_hz,
                        start_sample=start,
                        end_sample=end,
                        start_seconds=start / self.sampling_frequency_hz,
                        end_seconds=end / self.sampling_frequency_hz,
                        available_at_sample=end,
                        contains_filter_warmup=bool(warmup_slice.any()),
                        values=values[:, channel],
                    )
                )
            self._next_window_start += self.stride_samples
        self._discard_consumed_prefix()
        return tuple(windows)

    def _discard_consumed_prefix(self) -> None:
        discard = min(
            max(0, self._next_window_start - self._buffer_start), len(self._values)
        )
        if discard:
            self._values = self._values[discard:]
            self._warmup = self._warmup[discard:]
            self._buffer_start += discard

    def reset(self) -> None:
        self._values = None
        self._warmup = None
        self._buffer_start = None
        self._stream_end = None
        self._next_window_start = None
        self._identity = None
