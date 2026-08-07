"""Bounded real-waveform engineering smoke checks across frozen partitions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cardiosentinel.baseline.cache import (
    FeatureTable,
    read_feature_table,
    require_nonversioned_path,
    write_feature_table_atomic,
    write_json_atomic,
)
from cardiosentinel.baseline.materialize import REPOSITORY_ROOT, _target_for_window
from cardiosentinel.data.provenance import git_provenance
from cardiosentinel.data.remote import parse_remote_record
from cardiosentinel.evaluation.protocol import (
    LTSTDB_V1_SPLIT_SHA256,
    PRIMARY_STRIDE_SECONDS,
    PRIMARY_WINDOW_SECONDS,
)
from cardiosentinel.evaluation.splits import (
    load_split_manifest,
    validate_split_manifest,
)
from cardiosentinel.features import (
    COMBINED_V1,
    MORPHOLOGY_V1,
    SIGNAL_V1,
    extract_morphology_features,
    extract_signal_features,
)
from cardiosentinel.signal.config import raw_profile
from cardiosentinel.signal.io import read_remote_segment
from cardiosentinel.signal.preprocessing import StreamingPreprocessor
from cardiosentinel.signal.windows import CausalWindowGenerator, seconds_to_samples


def _mechanical_records(split: dict[str, Any]) -> dict[str, str]:
    selected = {}
    for partition in ("train", "validation", "test"):
        subjects = split["partitions"][partition]["subjects"]
        records = sorted(
            record
            for subject in subjects
            for record in split["records_by_subject"][subject]
        )
        selected[partition] = records[0]
    return selected


def _smoke_record(
    output_root: Path,
    partition: str,
    record_id: str,
    duration_seconds: float,
    force: bool,
) -> dict[str, Any]:
    cache_path = output_root / partition / f"{record_id}.npz"
    expected = {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "record_id": record_id,
        "partition": partition,
        "duration_seconds": duration_seconds,
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "feature_schema_sha256": COMBINED_V1.sha256,
        "processing_profile": "raw",
    }
    if cache_path.is_file() and not force:
        _, metadata = read_feature_table(cache_path)
        if all(metadata.get(key) == value for key, value in expected.items()):
            return {**metadata["report"], "cache_resumed": True}
        raise ValueError(f"Stale smoke cache for {record_id}; pass --force.")

    record, parsed = parse_remote_record("ltstdb", record_id, "stb")
    duration_samples = seconds_to_samples(
        duration_seconds, record.sampling_frequency_hz, "smoke duration"
    )
    end_sample = min(record.sample_count, duration_samples)
    segment = read_remote_segment("ltstdb", record_id, 0, end_sample)
    preprocessor = StreamingPreprocessor(
        raw_profile(), segment.sampling_frequency_hz, segment.channel_count
    )
    processed = preprocessor.process(segment)
    if not np.array_equal(segment.values, processed.waveform.values):
        raise ValueError("Smoke raw processing was not identity.")
    generator = CausalWindowGenerator(
        segment.sampling_frequency_hz,
        PRIMARY_WINDOW_SECONDS,
        PRIMARY_STRIDE_SECONDS,
    )
    windows = generator.process(processed)
    feature_rows = []
    targets = []
    for window in windows:
        targets.append(_target_for_window(window, parsed))
        feature_rows.append(
            np.concatenate(
                (extract_signal_features(window), extract_morphology_features(window))
            )
        )
    matrix = np.vstack(feature_rows)
    table = FeatureTable(
        features=matrix,
        stable_ids=np.asarray([target.stable_id for target in targets]),
        record_ids=np.asarray([record_id] * len(targets)),
        subject_ids=np.asarray([record.subject_id] * len(targets)),
        channel_indices=np.asarray(
            [target.channel_index for target in targets], dtype=np.int64
        ),
        lead_names=np.asarray([target.lead_name or "" for target in targets]),
        window_start_samples=np.asarray(
            [target.window_start_sample for target in targets], dtype=np.int64
        ),
        window_end_samples=np.asarray(
            [target.window_end_sample for target in targets], dtype=np.int64
        ),
        partitions=np.asarray([partition] * len(targets)),
        target_families=np.asarray([target.target_family for target in targets]),
        context_flags=np.asarray(
            ["|".join(target.context_flags) for target in targets]
        ),
    )
    detected_index = len(SIGNAL_V1.names) + MORPHOLOGY_V1.names.index(
        "detected_r_peak_count"
    )
    usable_index = len(SIGNAL_V1.names) + MORPHOLOGY_V1.names.index("usable_beat_count")
    valid_index = len(SIGNAL_V1.names) + MORPHOLOGY_V1.names.index("morphology_valid")
    window_samples = round(PRIMARY_WINDOW_SECONDS * record.sampling_frequency_hz)
    stride_samples = round(PRIMARY_STRIDE_SECONDS * record.sampling_frequency_hz)
    expected_windows = (
        (end_sample - window_samples) // stride_samples + 1
    ) * record.signal_count
    if table.row_count != expected_windows:
        raise ValueError(
            "Smoke causal window count differs from deterministic geometry."
        )
    report = {
        "partition": partition,
        "record_id": record_id,
        "subject_id": record.subject_id,
        "physical_units": list(segment.physical_units),
        "source_physical_units": list(segment.source_physical_units),
        "sampling_frequency_hz": segment.sampling_frequency_hz,
        "sample_interval": [0, end_sample],
        "window_count": table.row_count,
        "feature_count": matrix.shape[1],
        "finite_feature_fraction": float(np.mean(np.isfinite(matrix))),
        "windows_with_detected_r_peak": int(np.sum(matrix[:, detected_index] >= 1)),
        "windows_with_sufficient_usable_beats": int(
            np.sum(matrix[:, usable_index] >= 2)
        ),
        "morphology_valid_windows": int(np.sum(matrix[:, valid_index] == 1)),
        "morphology_invalid_windows": int(np.sum(matrix[:, valid_index] == 0)),
        "median_usable_beats_per_window": float(np.median(matrix[:, usable_index])),
        "target_counts": dict(
            sorted(Counter(target.target_family for target in targets).items())
        ),
        "cache_readback_valid": True,
        "cache_resumed": False,
    }
    write_feature_table_atomic(cache_path, table, {**expected, "report": report})
    readback, readback_metadata = read_feature_table(cache_path)
    if readback.row_count != table.row_count or readback_metadata["report"] != report:
        raise ValueError("Smoke cache atomic readback validation failed.")
    return report


def run_remote_smoke(
    output_root: Path,
    split_path: Path,
    *,
    duration_seconds: float = 60.0,
    force: bool = False,
) -> dict[str, Any]:
    """Mechanically select and validate one bounded record per partition."""
    root = require_nonversioned_path(output_root, "Smoke output root")
    split = load_split_manifest(split_path)
    validate_split_manifest(
        split,
        expected_hash=LTSTDB_V1_SPLIT_SHA256,
        expected_subject_count=80,
        expected_record_count=86,
    )
    selected = _mechanical_records(split)
    reports = {
        partition: _smoke_record(root, partition, record_id, duration_seconds, force)
        for partition, record_id in selected.items()
    }
    summary = {
        "engineering_smoke_only": True,
        "not_benchmark_results": True,
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "feature_schema_sha256": COMBINED_V1.sha256,
        "processing_profile": "raw",
        "git": git_provenance(REPOSITORY_ROOT),
        "mechanical_record_selection": selected,
        "partitions": reports,
    }
    write_json_atomic(root / "smoke_report.json", summary)
    return json.loads((root / "smoke_report.json").read_text(encoding="utf-8"))
