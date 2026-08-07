from types import SimpleNamespace

import pytest
import torch

from cardiosentinel.neural.data import B4PredictiveSample
from cardiosentinel.neural.engineering import (
    benchmark_compute_only,
    benchmark_direct_and_cached_io,
)


class TrackingDataset:
    opened_partitions: list[str] = []

    def __init__(self, first, second=None, *args, **kwargs) -> None:
        del args, kwargs
        if hasattr(first, "waveforms"):
            partition = second.partition
            self.length = second.total_count
        else:
            references = tuple(first)
            partition = references[0].partition
            self.length = len(references)
        if partition == "test":
            raise AssertionError("test partition was accessed")
        self.partition = partition
        self.opened_partitions.append(partition)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> B4PredictiveSample:
        del index
        return B4PredictiveSample(
            torch.zeros(1, 2500, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )


def _reference(partition: str, index: int):
    return SimpleNamespace(partition=partition, stable_id=f"{partition}-{index}")


def _index(partition: str, count: int):
    return SimpleNamespace(
        partition=partition,
        total_count=count,
        references=tuple(_reference(partition, index) for index in range(count)),
    )


def test_representative_io_benchmark_uses_development_and_no_metrics(
    monkeypatch,
) -> None:
    import cardiosentinel.evaluation.metrics as metrics
    import cardiosentinel.neural.engineering as engineering

    TrackingDataset.opened_partitions.clear()
    indexes = {"train": _index("train", 8), "validation": _index("validation", 8)}
    cache = SimpleNamespace(
        manifest={
            "partitions": {
                "train": {"waveform_bytes": 1, "stable_id_bytes": 1},
                "validation": {"waveform_bytes": 1, "stable_id_bytes": 1},
            }
        },
        waveforms={"train": object(), "validation": object()},
    )
    monkeypatch.setattr(engineering, "build_development_indexes", lambda _: indexes)
    monkeypatch.setattr(engineering, "validate_waveform_cache", lambda *args: cache)
    monkeypatch.setattr(engineering, "B4WaveformDataset", TrackingDataset)
    monkeypatch.setattr(engineering, "B4CachedWaveformDataset", TrackingDataset)
    monkeypatch.setattr(
        metrics,
        "select_validation_f1_threshold",
        lambda *args, **kwargs: pytest.fail("scientific metric was called"),
    )

    report = benchmark_direct_and_cached_io(
        "source", "features", "cache", train_windows=4, validation_windows=4
    )

    assert report["scientific_metrics_computed"] is False
    assert set(TrackingDataset.opened_partitions) == {"train", "validation"}
    assert report["partitions"]["train"]["selection_order"] == (
        "frozen-seed shuffled"
    )


class TinyB4(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=(1, 2)) * self.weight


def test_compute_benchmark_is_disposable_and_metric_free(monkeypatch) -> None:
    import cardiosentinel.evaluation.metrics as metrics
    import cardiosentinel.neural.engineering as engineering

    TrackingDataset.opened_partitions.clear()
    indexes = {
        "train": _index("train", 300),
        "validation": _index("validation", 300),
    }
    cache = SimpleNamespace(
        waveforms={"train": object(), "validation": object()}
    )
    monkeypatch.setattr(engineering, "build_development_indexes", lambda _: indexes)
    monkeypatch.setattr(engineering, "validate_waveform_cache", lambda *args: cache)
    monkeypatch.setattr(engineering, "B4CachedWaveformDataset", TrackingDataset)
    monkeypatch.setattr(engineering, "B4CompactCNN", TinyB4)
    monkeypatch.setattr(
        metrics,
        "select_validation_f1_threshold",
        lambda *args, **kwargs: pytest.fail("scientific metric was called"),
    )

    report = benchmark_compute_only(
        "features", "cache", batches=1, requested_device="cpu"
    )

    assert report["scientific_metrics_computed"] is False
    assert report["weights_retained"] is False
    assert report["timed_batches"] == 1
    assert report["training"]["samples_per_second"] > 0
    assert report["validation"]["samples_per_second"] > 0
    assert set(TrackingDataset.opened_partitions) == {"train", "validation"}
