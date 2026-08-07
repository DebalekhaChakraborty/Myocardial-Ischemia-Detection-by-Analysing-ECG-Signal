"""Engineering-only readiness estimates for the frozen full baseline run."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from cardiosentinel.baseline.cache import FEATURE_MANIFEST_NAME, read_json
from cardiosentinel.baseline.source import source_checksum_receipt_is_valid
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.evaluation.protocol import LTSTDB_V1_SPLIT_SHA256
from cardiosentinel.evaluation.splits import (
    load_split_manifest,
    validate_split_manifest,
)
from cardiosentinel.features import COMBINED_V1, extract_morphology_features
from cardiosentinel.signal.models import CausalWindow

PRIMARY_COUNTS = {
    "train": {"ischemic_positive": 93_613, "background_negative": 2_049_986},
    "validation": {"ischemic_positive": 21_628, "background_negative": 452_269},
    "test": {"ischemic_positive": 20_899, "background_negative": 432_905},
}
ALL_FROZEN_WINDOW_COUNTS = {
    "train": 2_202_723,
    "validation": 492_146,
    "test": 462_290,
}


def _nearest_existing(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate == candidate.parent:
            break
        candidate = candidate.parent
    return candidate


def _memory_estimate(rows: int) -> dict[str, int]:
    numeric_bytes = rows * len(COMBINED_V1.names) * np.dtype(np.float64).itemsize
    return {
        "rows": rows,
        "numeric_matrix_bytes": numeric_bytes,
        "conservative_working_bytes": int(numeric_bytes * 1.5),
    }


def _source_presence(source: Path, record_ids: set[str]) -> dict[str, Any]:
    suffix_counts = {
        suffix: sum(
            (source / f"{record_id}.{suffix}").is_file() for record_id in record_ids
        )
        for suffix in ("hea", "dat", "stb")
    }
    complete = sum(
        all(
            (source / f"{record_id}.{suffix}").is_file()
            for suffix in ("hea", "dat", "stb")
        )
        for record_id in record_ids
    )
    return {
        "source_root": str(source),
        "source_root_exists": source.is_dir(),
        "records_file_present": (source / "RECORDS").is_file(),
        "files_present_by_suffix": suffix_counts,
        "complete_record_count": complete,
    }


def _cache_presence(feature_root: Path, expected: set[str]) -> dict[str, int]:
    manifest_path = feature_root / FEATURE_MANIFEST_NAME
    if not manifest_path.is_file():
        return {"manifest_completed_records": 0, "digest_valid_resumable_records": 0}
    manifest = read_json(manifest_path)
    entries = {
        entry["record_id"]: entry
        for entry in manifest.get("records", [])
        if entry.get("status") == "complete" and entry.get("record_id") in expected
    }
    valid = 0
    for entry in entries.values():
        cache = feature_root / entry["cache_path"]
        if cache.is_file() and entry.get("cache_sha256") == sha256_file(cache):
            valid += 1
    return {
        "manifest_completed_records": len(entries),
        "digest_valid_resumable_records": valid,
    }


def _project_materialization_seconds(workers: int) -> dict[str, Any]:
    fs = 250.0
    samples = np.arange(round(10 * fs))
    values = 0.05 * np.sin(2 * np.pi * samples / fs)
    for center in range(125, values.size, 250):
        values[max(0, center - 2) : center + 3] += np.asarray([0.2, 0.7, 1.2, 0.7, 0.2])
    window = CausalWindow(
        record_id="synthetic",
        subject_id="synthetic",
        channel_index=0,
        signal_name="ecg",
        lead_name="I",
        physical_unit="mV",
        sampling_frequency_hz=fs,
        start_sample=0,
        end_sample=values.size,
        start_seconds=0.0,
        end_seconds=10.0,
        available_at_sample=values.size,
        contains_filter_warmup=False,
        values=values,
    )
    repeats = 20
    started = time.monotonic()
    for _ in range(repeats):
        extract_morphology_features(window)
    elapsed = max(time.monotonic() - started, np.finfo(float).eps)
    measured_rate = repeats / elapsed
    total_windows = sum(ALL_FROZEN_WINDOW_COUNTS.values())
    conservative_seconds = total_windows / (measured_rate * workers) * 2.0
    return {
        "basis": "20 synthetic 10-second XQRS windows; 2x safety factor",
        "measured_windows_per_second_per_worker": measured_rate,
        "projected_seconds": conservative_seconds,
        "projected_hours": conservative_seconds / 3600.0,
        "engineering_estimate_only": True,
    }


def baseline_preflight(
    source: Path, feature_root: Path, split_path: Path, *, workers: int = 1
) -> dict[str, Any]:
    """Return readiness facts without reading waveform samples."""
    if workers < 1:
        raise ValueError("Preflight workers must be at least 1.")
    source = source.expanduser().resolve()
    feature_root = feature_root.expanduser().resolve()
    split = load_split_manifest(split_path)
    validate_split_manifest(
        split,
        expected_hash=LTSTDB_V1_SPLIT_SHA256,
        expected_subject_count=80,
        expected_record_count=86,
    )
    expected = {
        record
        for subject_records in split["records_by_subject"].values()
        for record in subject_records
    }
    disk = shutil.disk_usage(_nearest_existing(feature_root))
    selected_rows = PRIMARY_COUNTS["train"]["ischemic_positive"] * 4
    source_data = _source_presence(source, expected)
    source_present = source_data["complete_record_count"] == len(expected)
    source_checksum_verified = source_checksum_receipt_is_valid(source, expected)
    blocking_reasons = []
    if not source_present:
        blocking_reasons.append("Pinned LTSTDB source data is incomplete.")
    if not source_checksum_verified:
        blocking_reasons.append(
            "Pinned LTSTDB source checksum verification is absent or invalid."
        )
    return {
        "cpu_count": os.cpu_count(),
        "requested_workers": workers,
        "available_disk_bytes": disk.free,
        "feature_root": str(feature_root),
        "expected_record_count": len(expected),
        "source_data": source_data,
        "source_present": source_present,
        "source_checksum_verified": source_checksum_verified,
        "feature_cache": _cache_presence(feature_root, expected),
        "expected_primary_benchmark_counts": PRIMARY_COUNTS,
        "conservative_memory_estimates": {
            "selected_training_matrix": _memory_estimate(selected_rows),
            "full_validation_matrix": _memory_estimate(
                ALL_FROZEN_WINDOW_COUNTS["validation"]
            ),
            "full_test_matrix": _memory_estimate(ALL_FROZEN_WINDOW_COUNTS["test"]),
        },
        "projected_full_materialization_runtime": _project_materialization_seconds(
            workers
        ),
        "blocks_execution": bool(blocking_reasons),
        "blocking_reasons": blocking_reasons,
    }
