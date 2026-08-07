"""Bounded local and PhysioNet waveform readers using WFDB physical calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cardiosentinel.signal.catalog import dataset_spec, subject_id_for_record
from cardiosentinel.signal.models import SignalValidationError, WaveformSegment
from cardiosentinel.signal.validation import (
    CANONICAL_PHYSICAL_UNIT,
    convert_to_millivolts,
    validate_header_calibration,
    validate_waveform_segment,
)


def _validate_interval(start_sample: int, end_sample: int, sample_count: int) -> None:
    if start_sample < 0 or end_sample <= start_sample or end_sample > sample_count:
        raise SignalValidationError(
            f"Invalid bounded sample interval [{start_sample}, {end_sample}) "
            f"for record length {sample_count}."
        )


def _selected_metadata(
    header: Any, channels: Sequence[int]
) -> tuple[tuple[Any, ...], ...]:
    if not channels or len(set(channels)) != len(channels):
        raise SignalValidationError("Selected channels must be non-empty and unique.")
    if min(channels) < 0 or max(channels) >= header.n_sig:
        raise SignalValidationError("Selected channel index is outside WFDB metadata.")
    return (
        tuple(header.sig_name[index] for index in channels),
        tuple(header.units[index] for index in channels),
        tuple(header.adc_gain[index] for index in channels),
    )


def _read_segment(
    *,
    dataset_id: str,
    record_id: str,
    start_sample: int,
    end_sample: int,
    channels: Sequence[int] | None,
    root: Path | None,
) -> WaveformSegment:
    import wfdb

    spec = dataset_spec(dataset_id)
    record_path = str(root / record_id) if root is not None else record_id
    remote = root is None
    header = wfdb.rdheader(record_path, pn_dir=spec.pn_dir if remote else None)
    _validate_interval(start_sample, end_sample, int(header.sig_len))
    selected = tuple(range(header.n_sig)) if channels is None else tuple(channels)
    signal_names, source_units, gains = _selected_metadata(header, selected)
    validate_header_calibration(gains, source_units)
    record = wfdb.rdrecord(
        record_path,
        sampfrom=start_sample,
        sampto=end_sample,
        channels=list(selected),
        physical=True,
        pn_dir=spec.pn_dir if remote else None,
    )
    if record.p_signal is None:
        raise SignalValidationError("WFDB did not return calibrated physical samples.")
    values_mv = convert_to_millivolts(record.p_signal, source_units)
    segment = WaveformSegment(
        dataset_id=spec.dataset_id,
        dataset_version=spec.version,
        record_id=record_id,
        subject_id=subject_id_for_record(dataset_id, record_id),
        sampling_frequency_hz=float(header.fs),
        start_sample=start_sample,
        end_sample=end_sample,
        start_seconds=start_sample / float(header.fs),
        end_seconds=end_sample / float(header.fs),
        signal_names=signal_names,
        lead_names=signal_names,
        physical_units=(CANONICAL_PHYSICAL_UNIT,) * len(selected),
        source_physical_units=tuple(str(unit) for unit in source_units),
        values=values_mv,
        source=(
            f"physionet:{spec.pn_dir}/{record_id}"
            if remote
            else str((root / record_id).resolve())
        ),
        provenance={
            "reader": "wfdb.rdrecord",
            "physical_calibration": "wfdb",
            "requested_channels": selected,
            "source_sample_count": int(header.sig_len),
        },
    )
    validate_waveform_segment(segment)
    return segment


def read_local_segment(
    root: Path,
    dataset_id: str,
    record_id: str,
    start_sample: int,
    end_sample: int,
    channels: Sequence[int] | None = None,
) -> WaveformSegment:
    """Read only the requested local WFDB sample interval and channels."""
    return _read_segment(
        dataset_id=dataset_id,
        record_id=record_id,
        start_sample=start_sample,
        end_sample=end_sample,
        channels=channels,
        root=root,
    )


def read_remote_segment(
    dataset_id: str,
    record_id: str,
    start_sample: int,
    end_sample: int,
    channels: Sequence[int] | None = None,
) -> WaveformSegment:
    """Read a bounded official PhysioNet interval; never request a whole record."""
    return _read_segment(
        dataset_id=dataset_id,
        record_id=record_id,
        start_sample=start_sample,
        end_sample=end_sample,
        channels=channels,
        root=None,
    )


def read_remote_seconds(
    dataset_id: str,
    record_id: str,
    start_seconds: float,
    duration_seconds: float,
    channels: Sequence[int] | None = None,
) -> WaveformSegment:
    """Resolve seconds against the remote header frequency, then read that interval."""
    import wfdb

    spec = dataset_spec(dataset_id)
    header = wfdb.rdheader(record_id, pn_dir=spec.pn_dir)
    fs = float(header.fs)
    if start_seconds < 0 or duration_seconds <= 0:
        raise SignalValidationError("Start must be non-negative and duration positive.")
    start_sample_float = start_seconds * fs
    duration_sample_float = duration_seconds * fs
    start_sample = round(start_sample_float)
    duration_samples = round(duration_sample_float)
    if not np.isclose(start_sample_float, start_sample, atol=1e-9) or not np.isclose(
        duration_sample_float, duration_samples, atol=1e-9
    ):
        raise SignalValidationError(
            "Time interval must align exactly with source samples."
        )
    return read_remote_segment(
        dataset_id,
        record_id,
        start_sample,
        start_sample + duration_samples,
        channels,
    )


def waveform_summary(segment: WaveformSegment) -> dict[str, object]:
    """Return bounded physical statistics without exposing raw ECG arrays."""
    return {
        "dataset": segment.dataset_id,
        "dataset_version": segment.dataset_version,
        "record_id": segment.record_id,
        "subject_id": segment.subject_id,
        "sampling_frequency_hz": segment.sampling_frequency_hz,
        "start_sample": segment.start_sample,
        "end_sample": segment.end_sample,
        "shape": list(segment.values.shape),
        "signal_names": list(segment.signal_names),
        "lead_names": list(segment.lead_names),
        "source_physical_units": list(segment.source_physical_units),
        "canonical_physical_units": list(segment.physical_units),
        "finite": bool(np.isfinite(segment.values).all()),
        "minimum_mv": np.min(segment.values, axis=0).tolist(),
        "median_mv": np.median(segment.values, axis=0).tolist(),
        "maximum_mv": np.max(segment.values, axis=0).tolist(),
        "source": segment.source,
    }
