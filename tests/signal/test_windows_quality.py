"""SYNTHETIC causal-window, SQI, provenance, and dependency-boundary tests."""

import ast
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.signal.config import raw_profile
from cardiosentinel.signal.models import (
    CausalWindow,
    ProcessedWaveformSegment,
    SignalValidationError,
    WaveformSegment,
)
from cardiosentinel.signal.provenance import processing_provenance
from cardiosentinel.signal.quality import (
    compute_signal_quality,
    validate_frequency_band,
)
from cardiosentinel.signal.windows import CausalWindowGenerator


def waveform(values: np.ndarray, fs: float = 100.0, start: int = 0) -> WaveformSegment:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    end = start + len(values)
    channels = values.shape[1]
    names = tuple(f"lead-{index}" for index in range(channels))
    return WaveformSegment(
        "synthetic",
        "1",
        "record",
        "subject",
        fs,
        start,
        end,
        start / fs,
        end / fs,
        names,
        names,
        ("mV",) * channels,
        ("mV",) * channels,
        values,
        "synthetic-engineering-fixture",
    )


def window(values: np.ndarray, fs: float = 250.0) -> CausalWindow:
    return CausalWindow(
        "record",
        "subject",
        0,
        "I",
        "I",
        "mV",
        fs,
        0,
        len(values),
        0.0,
        len(values) / fs,
        len(values),
        False,
        values,
    )


def test_causal_windows_preserve_absolute_indices_and_channel_identity() -> None:
    generator = CausalWindowGenerator(100.0, length_seconds=2.0, stride_seconds=1.0)
    values = np.column_stack((np.arange(300.0), -np.arange(300.0)))
    assert generator.process(waveform(values[:50], start=1_000)) == ()
    emitted = generator.process(waveform(values[50:200], start=1_050))
    assert len(emitted) == 2
    assert emitted[0].start_sample == 1_000
    assert emitted[0].end_sample == 1_200
    assert emitted[0].available_at_sample == 1_200
    assert emitted[1].channel_index == 1
    more = generator.process(waveform(values[200:], start=1_200))
    assert {item.start_sample for item in more} == {1_100}


def test_future_samples_do_not_change_an_emitted_window() -> None:
    prefix = np.sin(np.arange(200) / 10.0)
    first = CausalWindowGenerator(100.0, 2.0, 1.0)
    second = CausalWindowGenerator(100.0, 2.0, 1.0)
    from_prefix = first.process(waveform(prefix))[0]
    from_future = second.process(waveform(np.concatenate((prefix, np.ones(100)))))[0]
    assert np.array_equal(from_prefix.values, from_future.values)
    assert from_prefix.available_at_sample == from_prefix.end_sample


def test_window_reports_filter_warmup() -> None:
    source = waveform(np.arange(100.0))
    processed = ProcessedWaveformSegment(source, "custom", False, 60, 10, 100)
    emitted = CausalWindowGenerator(100.0, 1.0, 1.0).process(processed)
    assert emitted[0].contains_filter_warmup is True


def test_signal_quality_is_stable_and_label_free() -> None:
    fs = 250.0
    time = np.arange(2_500) / fs
    values = np.sin(2 * np.pi * 5 * time) + 0.05 * np.sin(2 * np.pi * 50 * time)
    metrics = compute_signal_quality(window(values, fs))
    assert metrics.finite_sample_fraction == 1.0
    assert metrics.robust_amplitude_range_mv is not None
    assert metrics.robust_derivative_scale_mv is not None
    assert metrics.powerline_ratio_50hz is not None
    assert 0 <= metrics.powerline_ratio_50hz <= 1


def test_flatline_and_nonfinite_quality_are_explicit() -> None:
    flat = compute_signal_quality(window(np.zeros(500)))
    assert flat.flatline_fraction == 1.0
    assert flat.repeated_value_fraction == 1.0
    assert flat.robust_amplitude_range_mv == 0.0
    assert flat.low_frequency_power_ratio is None
    nonfinite = compute_signal_quality(window(np.array([0.0, np.nan, np.inf, 1.0])))
    assert nonfinite.finite_sample_fraction == 0.5
    assert nonfinite.low_frequency_power_ratio is None


def test_frequency_bands_validate_sampling_support() -> None:
    with pytest.raises(SignalValidationError, match="invalid"):
        validate_frequency_band(100.0, 49.0, 51.0)
    metrics = compute_signal_quality(window(np.sin(np.arange(500)), fs=100.0))
    assert metrics.powerline_ratio_50hz is None
    assert metrics.powerline_ratio_60hz is None


def test_processing_provenance_has_required_schema() -> None:
    provenance = processing_provenance(waveform(np.arange(100.0)), raw_profile())
    expected = {
        "git_sha",
        "git_dirty",
        "cardiosentinel_version",
        "python_version",
        "numpy_version",
        "scipy_version",
        "wfdb_version",
        "dataset",
        "record",
        "sample_interval",
        "sampling_frequency_hz",
        "source_physical_units",
        "canonical_physical_units",
        "filter_configuration",
        "processing_schema_version",
    }
    assert expected <= provenance.keys()


def test_signal_layer_has_no_annotation_adapter_dependency() -> None:
    signal_root = Path(__file__).resolve().parents[2] / "src/cardiosentinel/signal"
    imports: set[str] = set()
    for path in signal_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert not any(module.startswith("cardiosentinel.data") for module in imports)
