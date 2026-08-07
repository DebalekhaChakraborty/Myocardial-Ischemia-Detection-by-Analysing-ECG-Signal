"""Deterministic physical waveform statistics for a completed causal window."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.features.schema import SIGNAL_V1
from cardiosentinel.signal.models import CausalWindow
from cardiosentinel.signal.quality import compute_signal_quality


def extract_signal_features(window: CausalWindow) -> NDArray[np.float64]:
    """Return signal_v1 in schema order without labels or identity inputs."""
    values = np.asarray(window.values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    quality = asdict(compute_signal_quality(window))
    if finite_values.size:
        percentiles = np.percentile(finite_values, [5, 25, 75, 95])
        raw = {
            "median_amplitude_mv": float(np.median(finite_values)),
            "mean_amplitude_mv": float(np.mean(finite_values)),
            "standard_deviation_mv": float(np.std(finite_values)),
            "rms_amplitude_mv": float(np.sqrt(np.mean(finite_values**2))),
            "percentile_05_mv": float(percentiles[0]),
            "percentile_25_mv": float(percentiles[1]),
            "percentile_75_mv": float(percentiles[2]),
            "percentile_95_mv": float(percentiles[3]),
            "interquartile_range_mv": float(percentiles[2] - percentiles[1]),
            "peak_to_peak_range_mv": float(np.ptp(finite_values)),
        }
    else:
        raw = {
            name: np.nan
            for name in (
                "median_amplitude_mv",
                "mean_amplitude_mv",
                "standard_deviation_mv",
                "rms_amplitude_mv",
                "percentile_05_mv",
                "percentile_25_mv",
                "percentile_75_mv",
                "percentile_95_mv",
                "interquartile_range_mv",
                "peak_to_peak_range_mv",
            )
        }
    adjacent_finite = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    derivative = np.diff(values)[adjacent_finite] * window.sampling_frequency_hz
    raw["median_absolute_derivative_mv_per_s"] = (
        float(np.median(np.abs(derivative))) if derivative.size else np.nan
    )
    raw["percentile_95_absolute_derivative_mv_per_s"] = (
        float(np.percentile(np.abs(derivative), 95)) if derivative.size else np.nan
    )
    values_by_name = {**quality, **raw}
    return np.asarray(
        [
            np.nan if values_by_name[name] is None else values_by_name[name]
            for name in SIGNAL_V1.names
        ],
        dtype=np.float64,
    )
