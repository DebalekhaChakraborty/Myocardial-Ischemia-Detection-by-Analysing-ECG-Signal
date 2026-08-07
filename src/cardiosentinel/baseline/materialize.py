"""Deterministic record-parallel LTSTDB waveform-to-feature materialization."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    FeatureTable,
    finalize_feature_corpus,
    read_cache_metadata,
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


@dataclass(frozen=True)
class MaterializationJob:
    """One independent record cache owned by exactly one worker process."""

    source: Path
    destination: Path
    record: DatasetRecord
    parsed: ParsedAnnotations
    partition: str
    metadata: dict[str, Any]
    chunk_seconds: float


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


def _run_materialization_job(job: MaterializationJob) -> dict[str, Any]:
    """Execute one record while preventing nested numerical thread pools."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _materialize_record(
            job.source,
            job.destination,
            job.record,
            job.parsed,
            job.partition,
            job.metadata,
            job.chunk_seconds,
        )


def _materialization_results(
    jobs: list[MaterializationJob], workers: int
) -> Iterator[tuple[MaterializationJob, dict[str, Any]]]:
    """Yield completed jobs; callers alone decide when a record is manifested."""
    if workers < 1:
        raise ValueError("Materialization workers must be at least 1.")
    if workers == 1:
        for job in jobs:
            try:
                yield job, _run_materialization_job(job)
            except Exception as error:
                raise RuntimeError(
                    f"Materialization failed for record {job.record.record_id}."
                ) from error
        return

    pool = multiprocessing.get_context("spawn").Pool(processes=workers)
    pending = [
        (job, pool.apply_async(_run_materialization_job, (job,))) for job in jobs
    ]
    try:
        while pending:
            progressed = False
            for job, result in tuple(pending):
                if not result.ready():
                    continue
                pending.remove((job, result))
                progressed = True
                try:
                    yield job, result.get()
                except Exception as error:
                    raise RuntimeError(
                        f"Materialization failed for record {job.record.record_id}."
                    ) from error
            if not progressed:
                time.sleep(0.05)
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()


def _manifest_payload(
    prior_records: dict[str, dict[str, Any]],
    provenance: dict[str, object],
    command: str,
) -> dict[str, Any]:
    return {
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


def _completed_manifest_entry(
    job: MaterializationJob, completed: dict[str, Any], *, resumed: bool
) -> dict[str, Any]:
    return {
        "record_id": job.record.record_id,
        "subject_id": job.record.subject_id,
        "partition": job.partition,
        "cache_path": (Path(job.partition) / job.destination.name).as_posix(),
        "status": "complete",
        "resumed": resumed,
        "row_count": completed["row_count"],
        "target_counts": completed["target_counts"],
        "morphology_quality": completed["morphology_quality"],
        "source_sha256": job.metadata["source_sha256"],
        "cache_sha256": sha256_file(job.destination),
    }


def materialize_features(
    source: Path,
    feature_root: Path,
    split_path: Path,
    *,
    records: Iterable[str] | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    workers: int = 1,
    force: bool = False,
    command: str = "cardiosentinel baseline materialize",
) -> dict[str, Any]:
    """Materialize selected records and resume only matching complete caches."""
    if workers < 1:
        raise ValueError("Materialization workers must be at least 1.")
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
    jobs: list[MaterializationJob] = []
    resumed_jobs: list[tuple[MaterializationJob, dict[str, Any]]] = []
    for record, parsed in zip(all_records, all_parsed, strict=True):
        if requested is not None and record.record_id not in requested:
            continue
        partition = partition_by_subject[record.subject_id]
        cache_path = root / partition / f"{record.record_id}.npz"
        source_sha256 = _source_digest(source, record.record_id)
        expected = _expected_cache_metadata(
            record, partition, source_sha256, provenance, command
        )
        job = MaterializationJob(
            source,
            cache_path,
            record,
            parsed,
            partition,
            expected,
            chunk_seconds,
        )
        if cache_path.is_file() and not force:
            prior_entry = prior_records.get(record.record_id)
            if prior_entry is None or not prior_entry.get("cache_sha256"):
                raise ValueError(
                    f"Cache for {record.record_id} is not bound to its manifest; "
                    "pass --force to replace it."
                )
            if sha256_file(cache_path) != prior_entry["cache_sha256"]:
                raise ValueError(
                    f"Cache digest for {record.record_id} differs from its manifest."
                )
            cached_metadata = read_cache_metadata(cache_path)
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
                resumed_jobs.append((job, completed))
            else:
                raise ValueError(
                    f"Cache for {record.record_id} is stale; "
                    "pass --force to replace it."
                )
        else:
            prior_records.pop(record.record_id, None)
            jobs.append(job)
    if jobs:
        initial_manifest = _manifest_payload(prior_records, provenance, command)
        write_json_atomic(manifest_path, initial_manifest)
    for job, completed in resumed_jobs:
        prior_records[job.record.record_id] = _completed_manifest_entry(
            job, completed, resumed=True
        )
    if resumed_jobs:
        write_json_atomic(
            manifest_path, _manifest_payload(prior_records, provenance, command)
        )
    for job, completed in _materialization_results(jobs, workers):
        prior_records[job.record.record_id] = _completed_manifest_entry(
            job, completed, resumed=False
        )
        write_json_atomic(
            manifest_path, _manifest_payload(prior_records, provenance, command)
        )
    if not manifest_path.is_file():
        raise ValueError("No records were selected for materialization.")
    expected_record_ids = {
        record_id
        for subject_records in split["records_by_subject"].values()
        for record_id in subject_records
    }
    completed_ids = {
        item["record_id"]
        for item in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
        if item.get("status") == "complete" and item.get("cache_sha256")
    }
    if completed_ids == expected_record_ids:
        finalize_feature_corpus(root, expected_record_ids)
    return json.loads(manifest_path.read_text(encoding="utf-8"))
