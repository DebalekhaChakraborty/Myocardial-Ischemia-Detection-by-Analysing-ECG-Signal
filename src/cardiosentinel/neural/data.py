"""Lossless raw-waveform access for the frozen B4 development partitions."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cardiosentinel.neural.metadata import B4WindowReference
from cardiosentinel.neural.protocol import (
    DATASET,
    DATASET_VERSION,
    SAMPLING_FREQUENCY_HZ,
    WINDOW_SAMPLES,
    require_development_partition,
)
from cardiosentinel.neural.provenance import validate_source_verification_receipt
from cardiosentinel.signal.io import read_local_segment
from cardiosentinel.signal.models import WaveformSegment
from cardiosentinel.signal.validation import CANONICAL_PHYSICAL_UNIT


class B4PredictiveSample(NamedTuple):
    """A model-ready waveform and frozen binary target, with no identity inputs."""

    waveform: Tensor
    label: Tensor


@dataclass(frozen=True, slots=True)
class WaveformReadStats:
    source_reads: int
    cache_hits: int
    cache_misses: int
    waveform_read_seconds: float
    conversion_seconds: float


SegmentReader = Callable[
    [Path, str, str, int, int, Sequence[int] | None], WaveformSegment
]
SourceVerifier = Callable[[Path], dict]


class B4WaveformDataset(Dataset[B4PredictiveSample]):
    """Read exact canonical mV intervals; never expose metadata to the model."""

    def __init__(
        self,
        references: Sequence[B4WindowReference],
        source: Path,
        *,
        cache_windows: int = 0,
        _reader: SegmentReader = read_local_segment,
        _source_verifier: SourceVerifier = validate_source_verification_receipt,
    ) -> None:
        if cache_windows < 0:
            raise ValueError("B4 waveform cache size cannot be negative.")
        partitions = {reference.partition for reference in references}
        for partition in partitions:
            require_development_partition(partition)
        self._references = tuple(references)
        self._source = Path(source).expanduser().resolve()
        _source_verifier(self._source)
        self._reader = _reader
        self._cache_windows = cache_windows
        self._cache: OrderedDict[str, Tensor] = OrderedDict()
        self._source_reads = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._waveform_read_seconds = 0.0
        self._conversion_seconds = 0.0

    def __len__(self) -> int:
        return len(self._references)

    def __getitem__(self, index: int) -> B4PredictiveSample:
        reference = self._references[index]
        waveform = self._cache.get(reference.stable_id)
        if waveform is None:
            self._cache_misses += 1
            waveform = self._read_waveform(reference)
            if self._cache_windows:
                self._cache[reference.stable_id] = waveform
                self._cache.move_to_end(reference.stable_id)
                while len(self._cache) > self._cache_windows:
                    self._cache.popitem(last=False)
        else:
            self._cache_hits += 1
            self._cache.move_to_end(reference.stable_id)
        return B4PredictiveSample(
            waveform=waveform,
            label=torch.tensor(reference.binary_label, dtype=torch.float32),
        )

    @property
    def stats(self) -> WaveformReadStats:
        return WaveformReadStats(
            source_reads=self._source_reads,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            waveform_read_seconds=self._waveform_read_seconds,
            conversion_seconds=self._conversion_seconds,
        )

    def _read_waveform(self, reference: B4WindowReference) -> Tensor:
        if reference.end_sample - reference.start_sample != WINDOW_SAMPLES:
            raise ValueError("B4 metadata interval must contain exactly 2500 samples.")
        started = time.perf_counter()
        segment = self._reader(
            self._source,
            DATASET,
            reference.record_id,
            reference.start_sample,
            reference.end_sample,
            (reference.channel_index,),
        )
        self._waveform_read_seconds += time.perf_counter() - started
        self._source_reads += 1
        self._validate_segment(reference, segment)

        conversion_started = time.perf_counter()
        # WaveformSegment owns canonical float64 values. This is the sole cast.
        values = np.asarray(segment.values[:, 0], dtype=np.float32)
        waveform = torch.from_numpy(values.copy()).reshape(1, WINDOW_SAMPLES)
        self._conversion_seconds += time.perf_counter() - conversion_started
        return waveform

    @staticmethod
    def _validate_segment(
        reference: B4WindowReference, segment: WaveformSegment
    ) -> None:
        if segment.dataset_id != DATASET or segment.dataset_version != DATASET_VERSION:
            raise ValueError("B4 waveform source has the wrong dataset identity.")
        if segment.record_id != reference.record_id:
            raise ValueError("B4 waveform record does not match its metadata identity.")
        if segment.subject_id != reference.subject_id:
            raise ValueError(
                "B4 waveform subject does not match its metadata identity."
            )
        if (
            segment.start_sample != reference.start_sample
            or segment.end_sample != reference.end_sample
        ):
            raise ValueError(
                "B4 waveform interval does not match its metadata identity."
            )
        if segment.sampling_frequency_hz != SAMPLING_FREQUENCY_HZ:
            raise ValueError(
                "B4 requires an authoritative sampling frequency of 250 Hz."
            )
        if segment.values.dtype != np.float64:
            raise ValueError("B4 canonical waveform reader must return float64 values.")
        if segment.values.shape != (WINDOW_SAMPLES, 1):
            raise ValueError(
                "B4 waveform must contain exactly one 2500-sample channel."
            )
        if segment.physical_units != (CANONICAL_PHYSICAL_UNIT,):
            raise ValueError("B4 waveform must use the canonical physical unit mV.")
        if tuple(segment.provenance.get("requested_channels", ())) != (
            reference.channel_index,
        ):
            raise ValueError(
                "B4 waveform channel does not match its metadata identity."
            )
        if not np.isfinite(segment.values).all():
            raise ValueError("B4 waveform contains non-finite physical-mV values.")
