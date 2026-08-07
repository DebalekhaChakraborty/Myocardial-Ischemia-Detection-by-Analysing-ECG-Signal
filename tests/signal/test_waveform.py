"""SYNTHETIC physical waveform and unit-integrity tests."""

from pathlib import Path

import numpy as np
import pytest
import wfdb

from cardiosentinel.signal.io import read_local_segment
from cardiosentinel.signal.models import SignalValidationError, WaveformSegment
from cardiosentinel.signal.validation import (
    canonical_source_unit,
    convert_to_millivolts,
    millivolt_factor,
    validate_header_calibration,
    validate_waveform_segment,
)


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


def test_physical_units_convert_explicitly_to_millivolts() -> None:
    converted = convert_to_millivolts(
        np.array([[1.0, 1.0, 1000.0, 1000.0]]),
        ("V", "mV", "uV", "µV"),
    )
    assert converted.tolist() == [[1000.0, 1.0, 1.0, 1.0]]


@pytest.mark.parametrize(
    ("source_unit", "canonical_unit", "factor"),
    (
        ("V", "V", 1000.0),
        ("mV", "mV", 1.0),
        ("mv", "mV", 1.0),
        ("uV", "uV", 0.001),
        ("µV", "uV", 0.001),
        ("μV", "uV", 0.001),
    ),
)
def test_source_unit_spellings_map_explicitly(
    source_unit: str, canonical_unit: str, factor: float
) -> None:
    assert canonical_source_unit(source_unit) == canonical_unit
    assert millivolt_factor(source_unit) == factor


def test_unknown_physical_unit_is_rejected() -> None:
    with pytest.raises(SignalValidationError, match="Unsupported physical ECG unit"):
        convert_to_millivolts(np.ones((2, 1)), ("ADC",))


@pytest.mark.parametrize("gain", (None, 0.0, -200.0, np.nan))
def test_invalid_header_calibration_is_rejected(gain: float | None) -> None:
    with pytest.raises(SignalValidationError, match="Invalid WFDB adc_gain"):
        validate_header_calibration((gain,), ("mV",))


def test_missing_header_unit_is_rejected() -> None:
    with pytest.raises(SignalValidationError, match="Missing WFDB physical unit"):
        validate_header_calibration((200.0,), (None,))


def test_waveform_preserves_absolute_sample_indices() -> None:
    segment = waveform(np.arange(20.0), fs=200.0, start=12_345)
    assert segment.values.shape == (20, 1)
    assert segment.start_sample == 12_345
    assert segment.end_sample == 12_365
    assert segment.start_seconds == 12_345 / 200.0
    assert segment.values.flags.writeable is False


def test_hard_validation_rejects_nonfinite_and_long_flatline() -> None:
    with pytest.raises(SignalValidationError, match="NaN or infinite"):
        validate_waveform_segment(waveform(np.array([0.0, np.nan])))
    with pytest.raises(SignalValidationError, match="no dynamic variation"):
        validate_waveform_segment(waveform(np.zeros(100), fs=100.0))


def test_hard_validation_checks_expected_channels_and_each_channel_variation() -> None:
    values = np.column_stack((np.arange(100.0), np.zeros(100)))
    segment = waveform(values, fs=100.0)
    with pytest.raises(SignalValidationError, match="Expected 3 channels"):
        validate_waveform_segment(segment, expected_channel_count=3)
    with pytest.raises(SignalValidationError, match=r"channels \[1\]"):
        validate_waveform_segment(segment)


def test_local_reader_is_bounded_and_preserves_selected_channel_order(
    tmp_path: Path,
) -> None:
    fs = 200
    samples = np.arange(400) / fs
    physical = np.column_stack(
        (0.5 * np.sin(2 * np.pi * 3 * samples), np.cos(2 * np.pi * samples))
    )
    wfdb.wrsamp(
        "s20011",
        fs=fs,
        units=["mV", "mV"],
        sig_name=["I", "II"],
        p_signal=physical,
        write_dir=str(tmp_path),
    )
    segment = read_local_segment(tmp_path, "ltstdb", "s20011", 25, 225, channels=(1, 0))
    assert segment.values.shape == (200, 2)
    assert segment.start_sample == 25
    assert segment.end_sample == 225
    assert segment.signal_names == ("II", "I")
    assert segment.source_physical_units == ("mV", "mV")
    assert np.allclose(segment.values[:, 0], physical[25:225, 1], atol=1e-4)


def test_local_reader_preserves_legacy_mv_source_spelling_without_rescaling(
    tmp_path: Path,
) -> None:
    fs = 200
    samples = np.arange(400) / fs
    physical = np.column_stack(
        (0.5 * np.sin(2 * np.pi * 3 * samples), np.cos(2 * np.pi * samples))
    )
    wfdb.wrsamp(
        "s20051",
        fs=fs,
        units=["mV", "mv"],
        sig_name=["I", "II"],
        p_signal=physical,
        write_dir=str(tmp_path),
    )
    wfdb_physical = wfdb.rdrecord(
        str(tmp_path / "s20051"), sampfrom=25, sampto=225, physical=True
    ).p_signal

    segment = read_local_segment(tmp_path, "ltstdb", "s20051", 25, 225)

    assert wfdb_physical is not None
    assert segment.source_physical_units == ("mV", "mv")
    assert segment.physical_units == ("mV", "mV")
    assert np.array_equal(segment.values, wfdb_physical)
