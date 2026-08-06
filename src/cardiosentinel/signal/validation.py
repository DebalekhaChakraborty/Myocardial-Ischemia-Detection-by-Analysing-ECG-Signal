"""Physical-unit conversion and hard waveform validity checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.signal.models import SignalValidationError, WaveformSegment

CANONICAL_PHYSICAL_UNIT = "mV"
_MILLIVOLT_FACTORS = {
    "V": 1000.0,
    "mV": 1.0,
    "uV": 0.001,
    "µV": 0.001,
    "μV": 0.001,
}


def millivolt_factor(unit: str) -> float:
    """Return an explicit physical conversion factor; never guess unknown units."""
    normalized = unit.strip()
    try:
        return _MILLIVOLT_FACTORS[normalized]
    except KeyError as error:
        raise SignalValidationError(
            f"Unsupported physical ECG unit {unit!r}; expected V, mV, uV, or µV."
        ) from error


def convert_to_millivolts(
    values: NDArray[np.floating] | object, units: tuple[str, ...]
) -> NDArray[np.float64]:
    """Convert each physical channel to mV while retaining channel order."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise SignalValidationError("Physical values must have [samples, channels].")
    if len(units) != array.shape[1]:
        raise SignalValidationError("Physical units do not match channel count.")
    factors = np.asarray([millivolt_factor(unit) for unit in units], dtype=np.float64)
    return array * factors[np.newaxis, :]


def validate_header_calibration(
    adc_gain: tuple[float | None, ...], units: tuple[str | None, ...]
) -> None:
    """Require usable WFDB calibration before trusting its physical signal output."""
    if len(adc_gain) != len(units) or not units:
        raise SignalValidationError("WFDB calibration metadata is incomplete.")
    for index, (gain, unit) in enumerate(zip(adc_gain, units, strict=True)):
        if gain is None or not np.isfinite(gain) or gain <= 0:
            raise SignalValidationError(f"Invalid WFDB adc_gain for channel {index}.")
        if unit is None:
            raise SignalValidationError(
                f"Missing WFDB physical unit for channel {index}."
            )
        millivolt_factor(unit)


def validate_waveform_segment(
    segment: WaveformSegment,
    *,
    expected_channel_count: int | None = None,
    require_finite: bool = True,
    require_dynamic: bool = True,
) -> None:
    """Apply hard physical checks separately from descriptive signal quality."""
    if segment.sample_count == 0:
        raise SignalValidationError("Waveform segment has no samples.")
    if (
        expected_channel_count is not None
        and segment.channel_count != expected_channel_count
    ):
        raise SignalValidationError(
            f"Expected {expected_channel_count} channels, "
            f"found {segment.channel_count}."
        )
    if any(unit != CANONICAL_PHYSICAL_UNIT for unit in segment.physical_units):
        raise SignalValidationError("Waveform values must use canonical mV units.")
    if require_finite and not np.isfinite(segment.values).all():
        raise SignalValidationError("Waveform segment contains NaN or infinite values.")
    sufficiently_long = segment.sample_count >= round(segment.sampling_frequency_hz)
    if require_dynamic and sufficiently_long:
        ranges = np.ptp(segment.values, axis=0)
        flat_channels = np.flatnonzero(ranges <= np.finfo(np.float64).eps)
        if flat_channels.size:
            raise SignalValidationError(
                "Waveform segment has no dynamic variation in channels "
                f"{flat_channels.tolist()}."
            )
