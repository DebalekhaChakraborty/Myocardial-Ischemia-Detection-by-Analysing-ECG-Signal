"""Waveform-only R-aligned morphology proxies for completed causal windows."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.features.schema import MORPHOLOGY_V1
from cardiosentinel.signal.models import CausalWindow

PRE_R_START_SECONDS = -0.200
PRE_R_END_SECONDS = -0.080
QRS_START_SECONDS = -0.080
QRS_END_SECONDS = 0.080
TEMPLATE_START_SECONDS = -0.200
TEMPLATE_END_SECONDS = 0.400
POST_R_SECONDS = (0.080, 0.120, 0.160, 0.200)
MINIMUM_USABLE_BEATS = 2


def _offset(seconds: float, sampling_frequency_hz: float) -> int:
    return int(round(seconds * sampling_frequency_hz))


def _detect_r_peaks(values: NDArray[np.float64], fs: float) -> NDArray[np.int64]:
    """Run WFDB XQRS on this window only; detection failure remains explicit."""
    if values.size < round(2.0 * fs) or not np.isfinite(values).all():
        return np.empty(0, dtype=np.int64)
    try:
        from wfdb.processing import xqrs_detect

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peaks = xqrs_detect(sig=values, fs=fs, verbose=False)
    except (ValueError, RuntimeError, IndexError, FloatingPointError):
        return np.empty(0, dtype=np.int64)
    peaks = np.asarray(peaks, dtype=np.int64)
    return peaks[(peaks >= 0) & (peaks < values.size)]


def _local_median(values: NDArray[np.float64], center: int, radius: int) -> float:
    return float(np.median(values[center - radius : center + radius + 1]))


def extract_morphology_features(window: CausalWindow) -> NDArray[np.float64]:
    """Return morphology_v1 from samples contained entirely in this window."""
    values = np.asarray(window.values, dtype=np.float64)
    fs = float(window.sampling_frequency_hz)
    peaks = _detect_r_peaks(values, fs)
    result = {name: np.nan for name in MORPHOLOGY_V1.names}
    result["detected_r_peak_count"] = float(peaks.size)

    template_start = _offset(TEMPLATE_START_SECONDS, fs)
    template_end = _offset(TEMPLATE_END_SECONDS, fs)
    usable = peaks[(peaks + template_start >= 0) & (peaks + template_end < values.size)]
    result["usable_beat_count"] = float(usable.size)
    result["morphology_valid"] = float(usable.size >= MINIMUM_USABLE_BEATS)
    if usable.size < MINIMUM_USABLE_BEATS:
        return np.asarray([result[name] for name in MORPHOLOGY_V1.names])

    rr_ms = np.diff(peaks).astype(np.float64) * 1000.0 / fs
    if rr_ms.size:
        rr_mean = float(np.mean(rr_ms))
        result.update(
            {
                "rr_mean_ms": rr_mean,
                "rr_median_ms": float(np.median(rr_ms)),
                "rr_std_ms": float(np.std(rr_ms)),
                "rr_cv": float(np.std(rr_ms) / rr_mean) if rr_mean > 0 else np.nan,
                "estimated_hr_bpm": 60_000.0 / rr_mean if rr_mean > 0 else np.nan,
            }
        )

    pre_start = _offset(PRE_R_START_SECONDS, fs)
    pre_end = _offset(PRE_R_END_SECONDS, fs)
    qrs_start = _offset(QRS_START_SECONDS, fs)
    qrs_end = _offset(QRS_END_SECONDS, fs)
    post_offsets = tuple(_offset(seconds, fs) for seconds in POST_R_SECONDS)
    local_radius = max(1, _offset(0.010, fs))
    baselines: list[float] = []
    qrs_ranges: list[float] = []
    post_values: list[tuple[float, float, float, float]] = []
    areas: list[float] = []
    templates: list[NDArray[np.float64]] = []
    for peak in usable:
        baseline = float(np.median(values[peak + pre_start : peak + pre_end]))
        baselines.append(baseline)
        qrs_ranges.append(float(np.ptp(values[peak + qrs_start : peak + qrs_end + 1])))
        post = tuple(
            _local_median(values, peak + offset, local_radius) - baseline
            for offset in post_offsets
        )
        post_values.append(post)
        area_values = (
            values[peak + post_offsets[0] : peak + post_offsets[-1] + 1] - baseline
        )
        areas.append(float(np.sum((area_values[:-1] + area_values[1:]) * (0.5 / fs))))
        templates.append(values[peak + template_start : peak + template_end + 1])

    post_matrix = np.asarray(post_values)
    result["pre_r_baseline_median_mv"] = float(np.median(baselines))
    result["qrs_proxy_peak_to_peak_mv"] = float(np.median(qrs_ranges))
    for column, name in enumerate(
        (
            "post_r_80ms_delta_mv",
            "post_r_120ms_delta_mv",
            "post_r_160ms_delta_mv",
            "post_r_200ms_delta_mv",
        )
    ):
        result[name] = float(np.median(post_matrix[:, column]))
    result["post_r_80_160_slope_mv_per_s"] = float(
        np.median((post_matrix[:, 2] - post_matrix[:, 0]) / 0.080)
    )
    result["post_r_80_200_area_mv_s"] = float(np.median(areas))

    template_matrix = np.vstack(templates)
    median_template = np.median(template_matrix, axis=0)
    correlations = []
    if np.std(median_template) > np.finfo(float).eps:
        for template in template_matrix:
            if np.std(template) > np.finfo(float).eps:
                correlations.append(float(np.corrcoef(template, median_template)[0, 1]))
    result["beat_template_correlation_median"] = (
        float(np.median(correlations)) if correlations else np.nan
    )
    result["beat_template_variability"] = float(
        np.median(np.abs(template_matrix - median_template))
    )
    return np.asarray([result[name] for name in MORPHOLOGY_V1.names])
