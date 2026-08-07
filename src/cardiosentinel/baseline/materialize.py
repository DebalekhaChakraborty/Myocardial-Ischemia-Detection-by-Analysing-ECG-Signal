"""Sequential local LTSTDB waveform-to-feature materialization."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    FeatureTable,
    read_feature_table,
    require_external_path,
    write_feature_table_atomic,
    write_json_atomic,
)
from cardiosentinel.data.manifest import inspect_dataset
from cardiosentinel.data.models import DatasetRecord, ParsedAnnotations
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.evaluation.models import BenchmarkWindow
from cardiosentinel.evaluation.protocol import (
    LTSTDB_V1_SPLIT_SHA256,
    PRIMARY_ANNOTATION_DEFINITION,
    PRIMARY_STRIDE_SECONDS,
    PRIMARY_WINDOW_SECONDS,
)
from cardiosentinel.evaluation.splits import (
    load_split_manifest,
    validate_split_manifest,
)
from cardiosentinel.evaluation.targets import assign_window_target
from cardiosentinel.features import (
    COMBINED_V1,
    MORPHOLOGY_V1,
    SIGNAL_V1,
    extract_morphology_features,
    extract_signal_features,
)
from cardiosentinel.signal.config import raw_profile
from cardiosentinel.signal.io import read_local_segment
from cardiosentinel.signal.preprocessing import StreamingPreprocessor
from cardiosentinel.signal.windows import CausalWindowGenerator, seconds_to_samples

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNK_SECONDS = 300.0


def _partition_map(split: dict[str, Any]) -> dict[str, str]:
    return {
        subject: partition
        for partition, payload in split["partitions"].items()
        for subject in payload["subjects"]
    }


def _source_digest(source: Path, record_id: str) -> str:
    """Hash physical data, header, and primary annotation for cache invalidation."""
    paths = [source / f"{record_id}.{suffix}" for suffix in ("hea", "dat", "stb")]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LTSTDB record files: {missing}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _expected_cache_metadata(
    record: DatasetRecord,
    partition: str,
    source_sha256: str,
    provenance: dict[str, object],
    command: str,
) -> dict[str, Any]:
    return {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "record_id": record.record_id,
        "subject_id": record.subject_id,
        "partition": partition,
        "source_sha256": source_sha256,
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "feature_schema_version": COMBINED_V1.version,
        "feature_schema_sha256": COMBINED_V1.sha256,
        "processing_profile": "raw",
        "window_seconds": PRIMARY_WINDOW_SECONDS,
        "stride_seconds": PRIMARY_STRIDE_SECONDS,
        "annotation_definition": PRIMARY_ANNOTATION_DEFINITION,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "generation_command": command,
    }


def _target_for_window(window: Any, parsed: ParsedAnnotations) -> Any:
    geometry = BenchmarkWindow(
        dataset="ltstdb",
        record_id=window.record_id,
        subject_id=window.subject_id,
        channel_index=window.channel_index,
        lead_name=window.lead_name,
        sampling_frequency_hz=window.sampling_frequency_hz,
        start_sample=window.start_sample,
        end_sample=window.end_sample,
        available_at_sample=window.available_at_sample,
    )
    return assign_window_target(
        geometry,
        parsed.events,
        parsed.quality_intervals,
        parsed.markers,
        parsed.source_censored_intervals,
        PRIMARY_ANNOTATION_DEFINITION,
    )


def _materialize_record(
    source: Path,
    destination: Path,
    record: DatasetRecord,
    parsed: ParsedAnnotations,
    partition: str,
    metadata: dict[str, Any],
    chunk_seconds: float,
) -> dict[str, Any]:
    chunk_samples = seconds_to_samples(
        chunk_seconds, record.sampling_frequency_hz, "materialization chunk"
    )
    preprocessor = StreamingPreprocessor(
        raw_profile(), record.sampling_frequency_hz, record.signal_count
    )
    windows = CausalWindowGenerator(
        record.sampling_frequency_hz,
        PRIMARY_WINDOW_SECONDS,
        PRIMARY_STRIDE_SECONDS,
    )
    feature_rows: list[np.ndarray] = []
    stable_ids: list[str] = []
    record_ids: list[str] = []
    subject_ids: list[str] = []
    channel_indices: list[int] = []
    lead_names: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    partitions: list[str] = []
    target_families: list[str] = []
    contexts: list[str] = []
    target_counts: Counter[str] = Counter()
    started = time.monotonic()
    for start in range(0, record.sample_count, chunk_samples):
        end = min(record.sample_count, start + chunk_samples)
        segment = read_local_segment(source, "ltstdb", record.record_id, start, end)
        processed = preprocessor.process(segment)
        if not np.array_equal(segment.values, processed.waveform.values):
            raise ValueError("Raw processing profile must be exact identity.")
        for window in windows.process(processed):
            target = _target_for_window(window, parsed)
            signal_values = extract_signal_features(window)
            morphology_values = extract_morphology_features(window)
            feature_rows.append(np.concatenate((signal_values, morphology_values)))
            stable_ids.append(target.stable_id)
            record_ids.append(window.record_id)
            subject_ids.append(window.subject_id)
            channel_indices.append(window.channel_index)
            lead_names.append(window.lead_name or "")
            starts.append(window.start_sample)
            ends.append(window.end_sample)
            partitions.append(partition)
            target_families.append(target.target_family)
            contexts.append("|".join(target.context_flags))
            target_counts[target.target_family] += 1
    table = FeatureTable(
        features=np.vstack(feature_rows),
        stable_ids=np.asarray(stable_ids),
        record_ids=np.asarray(record_ids),
        subject_ids=np.asarray(subject_ids),
        channel_indices=np.asarray(channel_indices, dtype=np.int64),
        lead_names=np.asarray(lead_names),
        window_start_samples=np.asarray(starts, dtype=np.int64),
        window_end_samples=np.asarray(ends, dtype=np.int64),
        partitions=np.asarray(partitions),
        target_families=np.asarray(target_families),
        context_flags=np.asarray(contexts),
    )
    valid_index = MORPHOLOGY_V1.names.index("morphology_valid") + len(SIGNAL_V1.names)
    detected_index = MORPHOLOGY_V1.names.index("detected_r_peak_count") + len(
        SIGNAL_V1.names
    )
    usable_index = MORPHOLOGY_V1.names.index("usable_beat_count") + len(SIGNAL_V1.names)
    quality = {
        "windows_processed": table.row_count,
        "windows_with_detected_r_peak": int(
            np.sum(table.features[:, detected_index] >= 1)
        ),
        "windows_with_sufficient_usable_beats": int(
            np.sum(table.features[:, usable_index] >= 2)
        ),
        "morphology_valid_windows": int(np.sum(table.features[:, valid_index] == 1)),
        "morphology_invalid_windows": int(np.sum(table.features[:, valid_index] == 0)),
        "median_usable_beats_per_window": float(
            np.median(table.features[:, usable_index])
        ),
    }
    completed_metadata = {
        **metadata,
        "row_count": table.row_count,
        "target_counts": dict(sorted(target_counts.items())),
        "morphology_quality": quality,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_feature_table_atomic(destination, table, completed_metadata)
    return completed_metadata


def materialize_features(
    source: Path,
    feature_root: Path,
    split_path: Path,
    *,
    records: Iterable[str] | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    force: bool = False,
    command: str = "cardiosentinel baseline materialize",
) -> dict[str, Any]:
    """Materialize selected records and resume only matching complete caches."""
    source = require_external_path(source, "Waveform source")
    root = require_external_path(feature_root, "Feature root")
    split = load_split_manifest(split_path)
    validate_split_manifest(
        split,
        expected_hash=LTSTDB_V1_SPLIT_SHA256,
        expected_subject_count=80,
        expected_record_count=86,
    )
    all_records, all_parsed = inspect_dataset("ltstdb", source, "stb")
    validate_split_manifest(
        split,
        records=all_records,
        expected_hash=LTSTDB_V1_SPLIT_SHA256,
        expected_subject_count=80,
        expected_record_count=86,
    )
    requested = None if records is None else set(records)
    unknown = (
        set()
        if requested is None
        else requested - {record.record_id for record in all_records}
    )
    if unknown:
        raise ValueError(
            f"Requested records are not in frozen LTSTDB V1: {sorted(unknown)}"
        )
    partition_by_subject = _partition_map(split)
    provenance = git_provenance(REPOSITORY_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / FEATURE_MANIFEST_NAME
    prior_records: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("expected_split_sha256") != LTSTDB_V1_SPLIT_SHA256:
            raise ValueError("Existing feature root belongs to a different split.")
        prior_records = {item["record_id"]: item for item in prior.get("records", [])}
    for record, parsed in zip(all_records, all_parsed, strict=True):
        if requested is not None and record.record_id not in requested:
            continue
        partition = partition_by_subject[record.subject_id]
        cache_relative = Path(partition) / f"{record.record_id}.npz"
        cache_path = root / cache_relative
        source_sha256 = _source_digest(source, record.record_id)
        expected = _expected_cache_metadata(
            record, partition, source_sha256, provenance, command
        )
        resumed = False
        if cache_path.is_file() and not force:
            _, cached_metadata = read_feature_table(cache_path)
            comparison_keys = (
                "record_id",
                "subject_id",
                "partition",
                "source_sha256",
                "split_sha256",
                "feature_schema_sha256",
                "processing_profile",
                "window_seconds",
                "stride_seconds",
                "annotation_definition",
            )
            if all(
                cached_metadata.get(key) == expected[key] for key in comparison_keys
            ):
                completed = cached_metadata
                resumed = True
            else:
                raise ValueError(
                    f"Cache for {record.record_id} is stale; "
                    "pass --force to replace it."
                )
        if not resumed:
            completed = _materialize_record(
                source,
                cache_path,
                record,
                parsed,
                partition,
                expected,
                chunk_seconds,
            )
        prior_records[record.record_id] = {
            "record_id": record.record_id,
            "subject_id": record.subject_id,
            "partition": partition,
            "cache_path": cache_relative.as_posix(),
            "status": "complete",
            "resumed": resumed,
            "row_count": completed["row_count"],
            "target_counts": completed["target_counts"],
            "morphology_quality": completed["morphology_quality"],
            "source_sha256": source_sha256,
        }
        manifest = {
            "feature_cache_schema_version": "1",
            "dataset": "ltstdb",
            "dataset_version": "1.0.0",
            "split_sha256": LTSTDB_V1_SPLIT_SHA256,
            "expected_split_sha256": LTSTDB_V1_SPLIT_SHA256,
            "feature_schemas": {
                "signal_v1": SIGNAL_V1.as_dict(),
                "morphology_v1": MORPHOLOGY_V1.as_dict(),
                "combined_v1": COMBINED_V1.as_dict(),
            },
            "processing_profile": "raw",
            "window_seconds": PRIMARY_WINDOW_SECONDS,
            "stride_seconds": PRIMARY_STRIDE_SECONDS,
            "annotation_definition": PRIMARY_ANNOTATION_DEFINITION,
            "generation": {**provenance, "command": command},
            "records": [prior_records[key] for key in sorted(prior_records)],
        }
        write_json_atomic(manifest_path, manifest)
    if not manifest_path.is_file():
        raise ValueError("No records were selected for materialization.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
