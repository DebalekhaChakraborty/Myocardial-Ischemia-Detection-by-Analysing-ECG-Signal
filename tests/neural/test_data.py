from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.neural.data import B4PredictiveSample, B4WaveformDataset
from cardiosentinel.neural.metadata import B4WindowReference
from cardiosentinel.signal.models import WaveformSegment


def reference(**changes) -> B4WindowReference:
    values = {
        "stable_id": "ltstdb:s20011:0:0:2500",
        "record_id": "s20011",
        "subject_id": "s20011",
        "channel_index": 0,
        "start_sample": 0,
        "end_sample": 2500,
        "partition": "train",
        "target_family": "ischemic_positive",
        "context_flags": (),
    }
    values.update(changes)
    return B4WindowReference(**values)


def segment(values=None, **changes) -> WaveformSegment:
    array = (
        np.linspace(-1.25, 2.5, 2500, dtype=np.float64).reshape(-1, 1)
        if values is None
        else values
    )
    fields = {
        "dataset_id": "ltstdb",
        "dataset_version": "1.0.0",
        "record_id": "s20011",
        "subject_id": "s20011",
        "sampling_frequency_hz": 250.0,
        "start_sample": 0,
        "end_sample": 2500,
        "start_seconds": 0.0,
        "end_seconds": 10.0,
        "signal_names": ("ECG",),
        "lead_names": ("ECG",),
        "physical_units": ("mV",),
        "source_physical_units": ("mV",),
        "values": array,
        "source": "fixture",
        "provenance": {"requested_channels": (0,)},
    }
    fields.update(changes)
    return WaveformSegment(**fields)


def dataset_for(waveform: WaveformSegment, *, cache_windows: int = 0):
    calls = []

    def reader(*args):
        calls.append(args)
        return waveform

    dataset = B4WaveformDataset(
        (reference(),),
        Path("/fixture/source"),
        cache_windows=cache_windows,
        _reader=reader,
        _source_verifier=lambda _: {},
    )
    return dataset, calls


def test_dataset_returns_exact_single_cast_without_normalization() -> None:
    source = segment()
    dataset, calls = dataset_for(source)
    sample = dataset[0]

    assert isinstance(sample, B4PredictiveSample)
    assert sample.waveform.shape == (1, 2500)
    assert sample.waveform.dtype == torch.float32
    np.testing.assert_array_equal(
        sample.waveform.numpy()[0], source.values[:, 0].astype(np.float32)
    )
    assert sample.label.item() == 1.0
    assert calls[0][2:] == ("s20011", 0, 2500, (0,))
    assert B4PredictiveSample._fields == ("waveform", "label")


def test_bounded_cache_is_optional_and_lossless() -> None:
    dataset, calls = dataset_for(segment(), cache_windows=1)
    assert torch.equal(dataset[0].waveform, dataset[0].waveform)
    assert len(calls) == 1
    assert dataset.stats.cache_hits == 1
    assert dataset.stats.cache_misses == 1


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"sampling_frequency_hz": 200.0, "end_seconds": 12.5}, "250 Hz"),
        ({"physical_units": ("V",)}, "physical unit"),
        ({"dataset_version": "other"}, "dataset identity"),
        ({"subject_id": "wrong"}, "subject"),
        ({"provenance": {"requested_channels": (1,)}}, "channel"),
    ],
)
def test_dataset_rejects_wrong_segment_contract(changed, message: str) -> None:
    dataset, _ = dataset_for(segment(**changed))
    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_dataset_rejects_nonfinite_values() -> None:
    values = np.zeros((2500, 1), dtype=np.float64)
    values[3, 0] = np.nan
    dataset, _ = dataset_for(segment(values))
    with pytest.raises(ValueError, match="non-finite"):
        dataset[0]


def test_dataset_rejects_more_than_one_channel() -> None:
    values = np.zeros((2500, 2), dtype=np.float64)
    waveform = segment(
        values,
        signal_names=("ECG0", "ECG1"),
        lead_names=("ECG0", "ECG1"),
        physical_units=("mV", "mV"),
        source_physical_units=("mV", "mV"),
    )
    dataset, _ = dataset_for(waveform)
    with pytest.raises(ValueError, match="one 2500-sample channel"):
        dataset[0]


def test_reference_rejects_wrong_stable_identity_and_interval() -> None:
    with pytest.raises(ValueError, match="stable ID"):
        reference(record_id="wrong")
    with pytest.raises(ValueError, match="2500"):
        dataset = B4WaveformDataset(
            (
                replace(
                    reference(),
                    end_sample=2000,
                    stable_id="ltstdb:s20011:0:0:2000",
                ),
            ),
            Path("/fixture"),
            _reader=lambda *args: segment(),
            _source_verifier=lambda _: {},
        )
        dataset[0]
