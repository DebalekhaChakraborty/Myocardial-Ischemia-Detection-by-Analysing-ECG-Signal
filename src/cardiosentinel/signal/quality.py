"""Deterministic, label-free descriptive signal-quality measurements."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from cardiosentinel.signal.models import (
    CausalWindow,
    SignalQualityMetrics,
    SignalValidationError,
)


def validate_frequency_band(
    sampling_frequency_hz: float, lower_hz: float, upper_hz: float
) -> None:
    """Reject bands that are empty, negative, or unsupported by Nyquist."""
    nyquist = sampling_frequency_hz / 2.0
    if lower_hz < 0 or upper_hz <= lower_hz or upper_hz >= nyquist:
        raise SignalValidationError(
            f"Frequency band [{lower_hz}, {upper_hz}] is invalid for "
            f"sampling frequency {sampling_frequency_hz}."
        )


def _band_ratio(
    frequencies: np.ndarray,
    power: np.ndarray,
    sampling_frequency_hz: float,
    lower_hz: float,
    upper_hz: float,
) -> float | None:
    try:
        validate_frequency_band(sampling_frequency_hz, lower_hz, upper_hz)
    except SignalValidationError:
        return None
    usable = np.isfinite(frequencies) & np.isfinite(power) & (power >= 0.0)
    selected = usable & (frequencies >= lower_hz) & (frequencies <= upper_hz)
    total = usable & (frequencies > 0.0)
    if np.count_nonzero(selected) < 2 or not total.any():
        return None
    denominator = float(np.sum(power[total]))
    if denominator <= np.finfo(np.float64).tiny:
        return None
    return float(np.sum(power[selected]) / denominator)


def compute_signal_quality(window: CausalWindow) -> SignalQualityMetrics:
    """Compute waveform-only metrics without expert quality annotations."""
    values = np.asarray(window.values, dtype=np.float64)
    finite = np.isfinite(values)
    finite_fraction = float(np.mean(finite)) if values.size else 0.0
    finite_values = values[finite]
    if finite_values.size < 2:
        return SignalQualityMetrics(
            finite_fraction, None, None, None, None, None, None, None, None, None
        )

    adjacent_finite = finite[:-1] & finite[1:]
    differences = np.diff(values)[adjacent_finite]
    derivative_mv_per_s = differences * window.sampling_frequency_hz
    scale_reference = max(float(np.max(np.abs(finite_values))), 1.0)
    flatline_tolerance = max(np.finfo(np.float64).eps * scale_reference * 16.0, 1e-12)
    flatline_fraction = (
        float(np.mean(np.abs(differences) <= flatline_tolerance))
        if differences.size
        else None
    )
    repeated_fraction = float(np.mean(differences == 0.0)) if differences.size else None
    robust_range = float(
        np.percentile(finite_values, 95) - np.percentile(finite_values, 5)
    )
    derivative_scale: float | None = None
    derivative_outliers: float | None = None
    if derivative_mv_per_s.size:
        median_derivative = float(np.median(derivative_mv_per_s))
        derivative_scale = float(
            1.4826 * np.median(np.abs(derivative_mv_per_s - median_derivative))
        )
        if derivative_scale > np.finfo(np.float64).eps:
            derivative_outliers = float(
                np.mean(
                    np.abs(derivative_mv_per_s - median_derivative)
                    > 6.0 * derivative_scale
                )
            )
        else:
            derivative_outliers = 0.0

    low_ratio = high_ratio = ratio_50 = ratio_60 = None
    if finite.all() and values.size >= 4:
        frequencies, power = scipy_signal.welch(
            values,
            fs=window.sampling_frequency_hz,
            nperseg=min(values.size, 1024),
            detrend="constant",
            scaling="density",
        )
        low_ratio = _band_ratio(
            frequencies, power, window.sampling_frequency_hz, 0.01, 0.5
        )
        high_upper = min(100.0, window.sampling_frequency_hz / 2.0 - 1e-9)
        if high_upper > 40.0:
            high_ratio = _band_ratio(
                frequencies, power, window.sampling_frequency_hz, 40.0, high_upper
            )
        ratio_50 = _band_ratio(
            frequencies, power, window.sampling_frequency_hz, 49.0, 51.0
        )
        ratio_60 = _band_ratio(
            frequencies, power, window.sampling_frequency_hz, 59.0, 61.0
        )
    return SignalQualityMetrics(
        finite_sample_fraction=finite_fraction,
        flatline_fraction=flatline_fraction,
        repeated_value_fraction=repeated_fraction,
        robust_amplitude_range_mv=robust_range,
        robust_derivative_scale_mv_per_s=derivative_scale,
        derivative_outlier_fraction=derivative_outliers,
        low_frequency_power_ratio=low_ratio,
        high_frequency_power_ratio=high_ratio,
        powerline_ratio_50hz=ratio_50,
        powerline_ratio_60hz=ratio_60,
    )
