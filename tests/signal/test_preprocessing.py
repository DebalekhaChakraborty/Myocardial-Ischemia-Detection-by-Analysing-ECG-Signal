"""SYNTHETIC causal-filter state, stability, and morphology guardrail tests."""

import numpy as np
import pytest

from cardiosentinel.signal.config import (
    FilterProfile,
    HighPassConfig,
    LowPassConfig,
    NotchConfig,
    raw_profile,
)
from cardiosentinel.signal.models import SignalValidationError, WaveformSegment
from cardiosentinel.signal.preprocessing import StreamingPreprocessor, filter_audit


def waveform(values: np.ndarray, fs: float, start: int = 0) -> WaveformSegment:
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


def causal_profile(notch_frequency: float = 50.0) -> FilterProfile:
    return FilterProfile(
        "st_preserving",
        highpass=HighPassConfig(True, 0.05, 1),
        lowpass=LowPassConfig(True, 40.0, 4),
        notch=NotchConfig(True, notch_frequency, 30.0),
    )


def synthetic_morphology(fs: float, seconds: float = 40.0) -> np.ndarray:
    time = np.arange(round(fs * seconds)) / fs
    baseline = 0.2 * np.sin(2 * np.pi * 0.02 * time)
    st_plateau = 0.15 * ((time % 2.0 >= 0.25) & (time % 2.0 < 0.55))
    qrs_like = 0.8 * np.exp(-(((time % 1.0 - 0.1) / 0.015) ** 2))
    mains = 0.03 * np.sin(2 * np.pi * 50.0 * time)
    return baseline + st_plateau + qrs_like + mains


def process_chunks(
    values: np.ndarray, fs: float, sizes: list[int]
) -> tuple[np.ndarray, StreamingPreprocessor]:
    processor = StreamingPreprocessor(causal_profile(), fs, 1)
    outputs: list[np.ndarray] = []
    start = 0
    for size in sizes:
        if start >= len(values):
            break
        end = min(len(values), start + size)
        outputs.append(
            processor.process(waveform(values[start:end], fs, start)).waveform.values
        )
        start = end
    if start < len(values):
        outputs.append(
            processor.process(waveform(values[start:], fs, start)).waveform.values
        )
    return np.vstack(outputs), processor


def test_raw_profile_is_exact_identity() -> None:
    values = synthetic_morphology(200.0, 2.0)
    output = StreamingPreprocessor(raw_profile(), 200.0, 1).process(
        waveform(values, 200.0)
    )
    assert np.array_equal(output.waveform.values[:, 0], values)
    assert output.is_warm is True
    assert output.warmup_samples_remaining == 0


@pytest.mark.parametrize("fs", [200.0, 250.0, 360.0])
def test_filter_is_finite_stable_and_sampling_frequency_explicit(fs: float) -> None:
    values = synthetic_morphology(fs, 5.0)
    output = StreamingPreprocessor(causal_profile(), fs, 1).process(
        waveform(values, fs)
    )
    assert output.waveform.values.shape == (len(values), 1)
    assert np.isfinite(output.waveform.values).all()
    assert np.max(np.abs(output.waveform.values)) < 5.0


def test_one_shot_matches_one_second_chunks() -> None:
    fs = 250.0
    values = synthetic_morphology(fs)
    one_shot, _ = process_chunks(values, fs, [len(values)])
    chunked, _ = process_chunks(values, fs, [int(fs)] * int(len(values) / fs))
    assert np.allclose(one_shot, chunked, rtol=0.0, atol=1e-12)


def test_one_shot_matches_irregular_and_one_sample_chunks() -> None:
    fs = 250.0
    values = synthetic_morphology(fs, 4.0)
    one_shot, _ = process_chunks(values, fs, [len(values)])
    irregular, _ = process_chunks(values, fs, [1, 17, 251, 3, 89, 400])
    assert np.allclose(one_shot, irregular, rtol=0.0, atol=1e-12)


def test_future_independence_causality_invariant() -> None:
    fs = 250.0
    prefix = synthetic_morphology(fs, 3.0)
    future = np.linspace(-1.0, 1.0, 500)
    prefix_output, _ = process_chunks(prefix, fs, [len(prefix)])
    combined_output, _ = process_chunks(
        np.concatenate((prefix, future)), fs, [len(prefix) + len(future)]
    )
    assert np.allclose(prefix_output, combined_output[: len(prefix)], atol=1e-12)


def test_reset_restores_zero_state_and_sample_origin() -> None:
    fs = 250.0
    values = synthetic_morphology(fs, 2.0)
    processor = StreamingPreprocessor(causal_profile(), fs, 1)
    first = processor.process(waveform(values, fs)).waveform.values
    assert processor.processed_sample_count == len(values)
    processor.reset()
    second = processor.process(waveform(values, fs)).waveform.values
    assert np.array_equal(first, second)


def test_warmup_is_coefficient_derived_and_visible() -> None:
    fs = 250.0
    processor = StreamingPreprocessor(causal_profile(), fs, 1)
    assert processor.warmup_sample_count > 0
    first = processor.process(waveform(np.ones(100), fs))
    assert first.is_warm is False
    assert first.warmup_samples_in_segment == 100
    remaining = processor.warmup_samples_remaining
    final_values = np.ones(remaining)
    final = processor.process(waveform(final_values, fs, 100))
    assert final.is_warm is True
    assert final.warmup_samples_remaining == 0


@pytest.mark.parametrize("mains_frequency", [50.0, 60.0])
def test_notch_attenuates_only_configured_mains_region(mains_frequency: float) -> None:
    profile = FilterProfile("custom", notch=NotchConfig(True, mains_frequency, 30.0))
    audit = filter_audit(profile, 250.0)
    response = {item["frequency_hz"]: item["magnitude"] for item in audit["response"]}
    assert response[mains_frequency] < 1e-6
    assert response[10.0] > 0.99


def test_highpass_response_and_st_preserving_guardrail() -> None:
    profile = FilterProfile("st_preserving", highpass=HighPassConfig(True, 0.05, 1))
    response = {
        item["frequency_hz"]: item["magnitude"]
        for item in filter_audit(profile, 250.0)["response"]
    }
    assert response[0.01] < response[0.05] < response[0.5]
    assert max(response.values()) <= 1.000001
    with pytest.raises(SignalValidationError, match="cannot exceed 0.05"):
        FilterProfile("st_preserving", highpass=HighPassConfig(True, 0.5, 1))
