"""Frozen deterministic feature schemas with canonical SHA-256 identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    """One numeric predictive input with a physical unit and honest description."""

    name: str
    unit: str
    description: str


@dataclass(frozen=True)
class FeatureSchema:
    """An ordered feature contract whose digest changes with any definition."""

    version: str
    features: tuple[FeatureDefinition, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def sha256(self) -> str:
        payload = {
            "feature_schema_version": self.version,
            "features": [asdict(feature) for feature in self.features],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_schema_version": self.version,
            "feature_schema_sha256": self.sha256,
            "features": [asdict(feature) for feature in self.features],
        }


def _feature(name: str, unit: str, description: str) -> FeatureDefinition:
    return FeatureDefinition(name, unit, description)


SIGNAL_V1 = FeatureSchema(
    "signal_v1",
    (
        _feature("finite_sample_fraction", "fraction", "Finite waveform samples."),
        _feature("flatline_fraction", "fraction", "Near-zero first differences."),
        _feature("repeated_value_fraction", "fraction", "Exactly repeated samples."),
        _feature("robust_amplitude_range_mv", "mV", "95th minus 5th percentile."),
        _feature(
            "robust_derivative_scale_mv_per_s",
            "mV/s",
            "Scaled MAD of the physical first time derivative.",
        ),
        _feature(
            "derivative_outlier_fraction",
            "fraction",
            "Derivative samples beyond six robust scales.",
        ),
        _feature(
            "low_frequency_power_ratio", "fraction", "Welch power at 0.01-0.5 Hz."
        ),
        _feature("high_frequency_power_ratio", "fraction", "Welch power at 40-100 Hz."),
        _feature("powerline_ratio_50hz", "fraction", "Welch power at 49-51 Hz."),
        _feature("powerline_ratio_60hz", "fraction", "Welch power at 59-61 Hz."),
        _feature("median_amplitude_mv", "mV", "Window median amplitude."),
        _feature("mean_amplitude_mv", "mV", "Window mean amplitude."),
        _feature("standard_deviation_mv", "mV", "Window standard deviation."),
        _feature("rms_amplitude_mv", "mV", "Root-mean-square amplitude."),
        _feature("percentile_05_mv", "mV", "Window 5th percentile."),
        _feature("percentile_25_mv", "mV", "Window 25th percentile."),
        _feature("percentile_75_mv", "mV", "Window 75th percentile."),
        _feature("percentile_95_mv", "mV", "Window 95th percentile."),
        _feature("interquartile_range_mv", "mV", "75th minus 25th percentile."),
        _feature("peak_to_peak_range_mv", "mV", "Maximum minus minimum."),
        _feature(
            "median_absolute_derivative_mv_per_s",
            "mV/s",
            "Median absolute physical first time derivative.",
        ),
        _feature(
            "percentile_95_absolute_derivative_mv_per_s",
            "mV/s",
            "95th percentile absolute physical first time derivative.",
        ),
    ),
)

MORPHOLOGY_V1 = FeatureSchema(
    "morphology_v1",
    (
        _feature("detected_r_peak_count", "count", "Waveform-derived R peaks."),
        _feature(
            "usable_beat_count", "count", "Detected beats with all proxy samples."
        ),
        _feature(
            "morphology_valid",
            "binary",
            "One when at least two complete waveform-derived beats are usable.",
        ),
        _feature("rr_mean_ms", "ms", "Mean detected R-R interval."),
        _feature("rr_median_ms", "ms", "Median detected R-R interval."),
        _feature("rr_std_ms", "ms", "Detected R-R interval standard deviation."),
        _feature("rr_cv", "ratio", "R-R standard deviation divided by mean."),
        _feature("estimated_hr_bpm", "beats/min", "60,000 divided by mean R-R."),
        _feature(
            "pre_r_baseline_median_mv",
            "mV",
            "Median waveform from 200 to 80 ms before detected R.",
        ),
        _feature(
            "qrs_proxy_peak_to_peak_mv",
            "mV",
            "Peak-to-peak waveform from 80 ms before to 80 ms after R.",
        ),
        _feature(
            "post_r_80ms_delta_mv", "mV", "R+80 ms proxy relative to pre-R baseline."
        ),
        _feature(
            "post_r_120ms_delta_mv",
            "mV",
            "R+120 ms proxy relative to pre-R baseline.",
        ),
        _feature(
            "post_r_160ms_delta_mv",
            "mV",
            "R+160 ms proxy relative to pre-R baseline.",
        ),
        _feature(
            "post_r_200ms_delta_mv",
            "mV",
            "R+200 ms proxy relative to pre-R baseline.",
        ),
        _feature(
            "post_r_80_160_slope_mv_per_s",
            "mV/s",
            "R-aligned 80-to-160 ms proxy slope.",
        ),
        _feature(
            "post_r_80_200_area_mv_s",
            "mV*s",
            "Baseline-relative waveform area from R+80 to R+200 ms.",
        ),
        _feature(
            "beat_template_correlation_median",
            "correlation",
            "Median complete-beat correlation with the median template.",
        ),
        _feature(
            "beat_template_variability",
            "mV",
            "Median sample-wise MAD around the median complete-beat template.",
        ),
    ),
)

COMBINED_V1 = FeatureSchema(
    "combined_v1",
    SIGNAL_V1.features + MORPHOLOGY_V1.features,
)

if len(COMBINED_V1.names) != len(set(COMBINED_V1.names)):
    raise RuntimeError("Feature names must be unique across combined_v1.")
