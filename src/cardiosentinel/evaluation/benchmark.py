"""Metadata-only benchmark generation, validation, and aggregate enumeration."""

from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cardiosentinel.data import edb, ltstdb
from cardiosentinel.data.manifest import inspect_dataset
from cardiosentinel.data.models import DatasetRecord, ParsedAnnotations
from cardiosentinel.data.provenance import git_provenance
from cardiosentinel.data.remote import parse_remote_record
from cardiosentinel.data.validation import validate_dataset
from cardiosentinel.evaluation.models import EDBSecondaryCohort, EDBSecondaryCohortName
from cardiosentinel.evaluation.protocol import (
    BENCHMARK_SCHEMA_VERSION,
    CHALLENGE_EVIDENCE_POLICIES,
    DEFAULT_SEED,
    LABEL_POLICY_VERSION,
    LTSTDB_V1_GENERATOR_CODE_SHA256,
    LTSTDB_V1_SPLIT_SHA256,
    POSITIVE_CONTEXT_EVIDENCE_LEVEL,
    POSITIVE_CONTEXT_FLAGS,
    PRIMARY_ANNOTATION_DEFINITION,
    PRIMARY_STRIDE_SECONDS,
    PRIMARY_WINDOW_SECONDS,
    PROTOCOL_VERSION,
)
from cardiosentinel.evaluation.provenance import edb_secondary_cohort
from cardiosentinel.evaluation.splits import (
    PARTITIONS,
    load_split_manifest,
    validate_split_manifest,
)
from cardiosentinel.evaluation.targets import (
    assign_window_target,
    iter_causal_window_indices,
)
from cardiosentinel.signal.windows import seconds_to_samples

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_PATH = REPOSITORY_ROOT / "protocols/splits/ltstdb_v1.json"


def load_benchmark_metadata(
    dataset_id: str,
    annotation_set: str | None,
    source: Path | None = None,
) -> tuple[tuple[DatasetRecord, ...], tuple[ParsedAnnotations, ...]]:
    """Load local metadata or concurrent remote headers/annotations, never waveforms."""
    if source is not None:
        return inspect_dataset(dataset_id, source, annotation_set)
    import wfdb

    module = edb if dataset_id == "edb" else ltstdb if dataset_id == "ltstdb" else None
    if module is None:
        raise ValueError("Benchmark dataset must be edb or ltstdb.")
    selected = annotation_set or module.PRIMARY_ANNOTATION
    record_ids = tuple(wfdb.get_record_list(f"{dataset_id}/{module.DATASET_VERSION}"))

    def load_one(record_id: str) -> tuple[DatasetRecord, ParsedAnnotations]:
        return parse_remote_record(dataset_id, record_id, selected)

    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded = tuple(executor.map(load_one, record_ids))
    records = tuple(item[0] for item in loaded)
    parsed = tuple(item[1] for item in loaded)
    events = tuple(event for item in parsed for event in item.events)
    intervals = tuple(
        interval for item in parsed for interval in item.quality_intervals
    )
    markers = tuple(marker for item in parsed for marker in item.markers)
    primary = "edb.reference" if dataset_id == "edb" else "ltstdb.stb"
    report = validate_dataset(
        records,
        events,
        intervals,
        module.EXPECTED_RECORD_COUNT,
        module.EXPECTED_SUBJECT_COUNT,
        primary,
        markers,
        parsed,
    )
    report.raise_for_errors()
    return records, parsed


def _partition_mapping(split: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for partition in PARTITIONS:
        for subject in split["partitions"][partition]["subjects"]:
            mapping[subject] = partition
    return mapping


def _new_bucket() -> dict[str, Any]:
    return {
        "subjects": set(),
        "records": set(),
        "recording_seconds": 0.0,
        "ischemic_episode_count": 0,
        "ischemic_episode_seconds": 0.0,
        "rate_related_episode_count": 0,
        "rate_related_episode_seconds": 0.0,
        "window_counts": Counter(),
        "target_subjects": {},
        "positive_context_counts": Counter(),
        "positive_context_subjects": {},
    }


def _add_record_burden(
    bucket: dict[str, Any], record: DatasetRecord, parsed: ParsedAnnotations
) -> None:
    bucket["subjects"].add(record.subject_id)
    bucket["records"].add(record.record_id)
    bucket["recording_seconds"] += record.duration_seconds
    for event in parsed.events:
        duration = event.end_seconds - event.onset_seconds
        if event.event_subtype in {"ischemic", "reference_st_change"}:
            bucket["ischemic_episode_count"] += 1
            bucket["ischemic_episode_seconds"] += duration
        elif event.event_subtype == "heart_rate_related":
            bucket["rate_related_episode_count"] += 1
            bucket["rate_related_episode_seconds"] += duration


def _serialize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    positive_count = bucket["window_counts"]["ischemic_positive"]
    background_count = bucket["window_counts"]["background_negative"]
    primary_count = positive_count + background_count
    return {
        "subject_count": len(bucket["subjects"]),
        "record_count": len(bucket["records"]),
        "recording_seconds": round(bucket["recording_seconds"], 9),
        "recording_hours": round(bucket["recording_seconds"] / 3600.0, 6),
        "ischemic_episode_count": bucket["ischemic_episode_count"],
        "ischemic_episode_seconds": round(bucket["ischemic_episode_seconds"], 9),
        "rate_related_episode_count": bucket["rate_related_episode_count"],
        "rate_related_episode_seconds": round(
            bucket["rate_related_episode_seconds"], 9
        ),
        "window_counts": dict(sorted(bucket["window_counts"].items())),
        "target_subject_counts": {
            key: len(value)
            for key, value in sorted(bucket["target_subjects"].items())
        },
        "challenge_composition": {
            challenge: {
                "challenge": policy.challenge,
                "evidence_level": policy.evidence_level,
                "subject_count": len(
                    bucket["target_subjects"].get(policy.target_family, set())
                ),
                "window_count": bucket["window_counts"][policy.target_family],
                "is_headline_metric": policy.is_headline_metric,
                "supports_inferential_bootstrap": (
                    policy.supports_inferential_bootstrap
                ),
            }
            for challenge, policy in CHALLENGE_EVIDENCE_POLICIES.items()
        },
        "primary_composition": {
            "positive_windows": positive_count,
            "background_windows": background_count,
            "positive_prevalence": (
                None if primary_count == 0 else positive_count / primary_count
            ),
        },
        "positive_context_counts": dict(
            sorted(bucket["positive_context_counts"].items())
        ),
        "positive_context_subject_counts": {
            key: len(value)
            for key, value in sorted(bucket["positive_context_subjects"].items())
        },
        "positive_context_composition": {
            context_flag: {
                "context_flag": context_flag,
                "evidence_level": POSITIVE_CONTEXT_EVIDENCE_LEVEL,
                "subject_count": len(
                    bucket["positive_context_subjects"].get(context_flag, set())
                ),
                "window_count": bucket["positive_context_counts"][context_flag],
            }
            for context_flag in POSITIVE_CONTEXT_FLAGS
        },
    }


def select_edb_secondary_cohort(
    records: tuple[DatasetRecord, ...],
    parsed: tuple[ParsedAnnotations, ...],
    cohort_name: EDBSecondaryCohortName,
) -> tuple[
    tuple[DatasetRecord, ...],
    tuple[ParsedAnnotations, ...],
    EDBSecondaryCohort,
]:
    """Select a named EDB secondary cohort while retaining overlap policy metadata."""
    if len(records) != len(parsed):
        raise ValueError("EDB records and annotations are not aligned.")
    cohort = edb_secondary_cohort(cohort_name)
    selected_ids = set(cohort.record_ids)
    selected = tuple(
        (record, parsed_item)
        for record, parsed_item in zip(records, parsed, strict=True)
        if record.record_id in selected_ids
    )
    return (
        tuple(record for record, _ in selected),
        tuple(parsed_item for _, parsed_item in selected),
        cohort,
    )


def _validate_target(
    target: Any,
    record: DatasetRecord,
    events: tuple[Any, ...],
) -> None:
    if not (
        0
        <= target.window_start_sample
        < target.window_end_sample
        <= record.sample_count
    ):
        raise ValueError(f"Window exceeds record bounds in {record.record_id}.")
    overlapping_events = tuple(
        event
        for event in events
        if target.window_start_sample < event.end_sample
        and event.onset_sample < target.window_end_sample
    )
    if target.target_family == "ischemic_positive" and not any(
        event.event_subtype in {"ischemic", "reference_st_change"}
        and target.window_start_sample >= event.onset_sample
        and target.window_end_sample <= event.end_sample
        for event in overlapping_events
    ):
        raise ValueError("Primary positive is not fully within an ischemic event.")
    if target.target_family == "background_negative" and any(
        event.event_subtype
        in {"ischemic", "reference_st_change", "heart_rate_related"}
        for event in overlapping_events
    ):
        raise ValueError("Background negative overlaps an excluded ST event.")
    if target.target_family == "quality_excluded" and (
        target.eligible_for_training or target.eligible_for_primary_evaluation
    ):
        raise ValueError("Unreadable target cannot be eligible for primary use.")


def summarize_benchmark(
    records: tuple[DatasetRecord, ...],
    parsed: tuple[ParsedAnnotations, ...],
    *,
    dataset_id: str,
    annotation_definition: str,
    command: str,
    split: dict[str, Any] | None = None,
    expected_split_hash: str | None = LTSTDB_V1_SPLIT_SHA256,
    expected_subject_count: int | None = 80,
    expected_record_count: int | None = 86,
    secondary_cohort: EDBSecondaryCohort | None = None,
) -> dict[str, Any]:
    """Enumerate targets lazily and retain aggregate composition only."""
    if len(records) != len(parsed):
        raise ValueError("Benchmark records and annotations are not aligned.")
    if dataset_id == "ltstdb":
        if split is None:
            raise ValueError("LTSTDB benchmark enumeration requires the frozen split.")
        validate_split_manifest(
            split,
            records=records,
            expected_hash=expected_split_hash,
            expected_generator_code_hash=(
                LTSTDB_V1_GENERATOR_CODE_SHA256
                if expected_split_hash == LTSTDB_V1_SPLIT_SHA256
                else None
            ),
            expected_subject_count=expected_subject_count,
            expected_record_count=expected_record_count,
        )
        subject_partition = _partition_mapping(split)
    else:
        subject_partition = {record.subject_id: "secondary" for record in records}

    partition_buckets: dict[str, dict[str, Any]] = {}
    subject_buckets: dict[str, dict[str, Any]] = {}
    record_buckets: dict[str, dict[str, Any]] = {}
    for record, parsed_item in zip(records, parsed, strict=True):
        partition = subject_partition[record.subject_id]
        partition_bucket = partition_buckets.setdefault(partition, _new_bucket())
        subject_bucket = subject_buckets.setdefault(record.subject_id, _new_bucket())
        record_bucket = record_buckets.setdefault(record.record_id, _new_bucket())
        for bucket in (partition_bucket, subject_bucket, record_bucket):
            _add_record_burden(bucket, record, parsed_item)
        events_by_lead = {
            lead: tuple(event for event in parsed_item.events if event.lead == lead)
            for lead in range(record.signal_count)
        }
        quality_by_lead = {
            lead: tuple(
                interval
                for interval in parsed_item.quality_intervals
                if interval.lead is None or interval.lead == lead
            )
            for lead in range(record.signal_count)
        }
        markers_by_lead = {
            lead: tuple(
                marker
                for marker in parsed_item.markers
                if marker.lead is None or marker.lead == lead
            )
            for lead in range(record.signal_count)
        }
        censored_by_lead = {
            lead: tuple(
                interval
                for interval in parsed_item.source_censored_intervals
                if interval.lead is None or interval.lead == lead
            )
            for lead in range(record.signal_count)
        }
        marker_radius = seconds_to_samples(
            30.0, record.sampling_frequency_hz, "marker challenge vicinity"
        )
        for window in iter_causal_window_indices(
            record, PRIMARY_WINDOW_SECONDS, PRIMARY_STRIDE_SECONDS
        ):
            lead = window.channel_index
            target = assign_window_target(
                window,
                events_by_lead[lead],
                quality_by_lead[lead],
                markers_by_lead[lead],
                censored_by_lead[lead],
                annotation_definition,
                marker_vicinity_samples=marker_radius,
            )
            _validate_target(target, record, events_by_lead[lead])
            for bucket in (partition_bucket, subject_bucket, record_bucket):
                bucket["window_counts"][target.target_family] += 1
                bucket["target_subjects"].setdefault(
                    target.target_family, set()
                ).add(target.subject_id)
                if target.target_family == "ischemic_positive":
                    for context_flag in target.context_flags:
                        bucket["positive_context_counts"][context_flag] += 1
                        bucket["positive_context_subjects"].setdefault(
                            context_flag, set()
                        ).add(target.subject_id)

    if dataset_id == "ltstdb":
        for partition in PARTITIONS:
            bucket = partition_buckets.get(partition)
            if bucket is None or bucket["ischemic_episode_count"] == 0:
                raise ValueError(f"Partition {partition} has zero ischemic episodes.")
            if bucket["window_counts"]["ischemic_positive"] == 0:
                raise ValueError(f"Partition {partition} has zero positive windows.")

    provenance = git_provenance(REPOSITORY_ROOT)
    provenance.update(
        {
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_protocol_version": PROTOCOL_VERSION,
            "label_policy_version": LABEL_POLICY_VERSION,
            "dataset": dataset_id,
            "dataset_version": "1.0.0",
            "annotation_definition": annotation_definition,
            "split_sha256": split["split_sha256"] if split else None,
            "window_length_seconds": PRIMARY_WINDOW_SECONDS,
            "stride_seconds": PRIMARY_STRIDE_SECONDS,
            "seed": DEFAULT_SEED,
            "generation_command": command,
            "python_version": sys.version.split()[0],
        }
    )
    if dataset_id == "edb":
        cohort = secondary_cohort or edb_secondary_cohort("full")
        provenance["edb_secondary_cohort"] = {
            "name": cohort.name,
            "full_record_count": len(edb.EDB_RECORD_IDS),
            "cohort_record_count": len(records),
            "known_overlap_exclusion_count": len(cohort.known_overlap_record_ids),
            "known_overlap_record_ids": list(cohort.known_overlap_record_ids),
            "contains_known_source_overlap": cohort.contains_known_source_overlap,
            "fully_independent_external_validation": (
                cohort.fully_independent_external_validation
            ),
        }
    return {
        "provenance": provenance,
        "partitions": {
            key: _serialize_bucket(value)
            for key, value in sorted(partition_buckets.items())
        },
        "subjects": {
            key: _serialize_bucket(value)
            for key, value in sorted(subject_buckets.items())
        },
        "records": {
            key: _serialize_bucket(value)
            for key, value in sorted(record_buckets.items())
        },
    }


def summarize_from_sources(
    dataset_id: str,
    annotation_set: str | None,
    split_path: Path | None,
    source: Path | None,
    command: str,
    edb_cohort_name: EDBSecondaryCohortName = "full",
) -> dict[str, Any]:
    records, parsed = load_benchmark_metadata(dataset_id, annotation_set, source)
    secondary_cohort = None
    if dataset_id == "edb":
        records, parsed, secondary_cohort = select_edb_secondary_cohort(
            records, parsed, edb_cohort_name
        )
    split = load_split_manifest(split_path) if split_path else None
    definition = (
        PRIMARY_ANNOTATION_DEFINITION if dataset_id == "ltstdb" else "edb.reference"
    )
    return summarize_benchmark(
        records,
        parsed,
        dataset_id=dataset_id,
        annotation_definition=definition,
        command=command,
        split=split,
        secondary_cohort=secondary_cohort,
    )
