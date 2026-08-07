"""Deterministic subject-level balancing and frozen split-manifest validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from cardiosentinel.data.models import DatasetRecord, ParsedAnnotations
from cardiosentinel.data.validation import validate_subject_partitions
from cardiosentinel.evaluation.models import SubjectBurden
from cardiosentinel.evaluation.protocol import (
    DEFAULT_SEED,
    PRIMARY_ANNOTATION_DEFINITION,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_STRIDE_SECONDS,
    PRIMARY_WINDOW_SECONDS,
    PROTOCOL_VERSION,
)

PARTITIONS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
GENERATOR_SOURCE_PATHS = (
    "src/cardiosentinel/evaluation/models.py",
    "src/cardiosentinel/evaluation/splits.py",
)
ASSIGNMENT_ALGORITHM = (
    "deterministic greedy subject-level burden balancing over record count, "
    "recording duration, .stb ischemic episode count/duration, rate-related "
    "episode count/duration, channel count, axis-shift marker count/presence, "
    "and conduction-change marker count/presence; exact 70/15/15 capacities; "
    "deterministic improving pairwise-swap refinement; SHA-256 seed tie-breaking"
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_canonical(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def generator_code_sha256(repository_root: Path | None = None) -> str:
    """Hash stable assignment sources and configuration in deterministic order."""
    root = repository_root or Path(__file__).resolve().parents[3]
    source_payload = []
    for relative_path in GENERATOR_SOURCE_PATHS:
        content = (root / relative_path).read_text(encoding="utf-8")
        source_payload.append(
            {
                "path": relative_path,
                "content": content.replace("\r\n", "\n").replace("\r", "\n"),
            }
        )
    payload = {
        "sources": source_payload,
        "assignment_configuration": {
            "partitions": PARTITIONS,
            "split_fractions": SPLIT_FRACTIONS,
            "default_seed": DEFAULT_SEED,
            "primary_dataset": PRIMARY_DATASET,
            "primary_dataset_version": PRIMARY_DATASET_VERSION,
            "primary_annotation_definition": PRIMARY_ANNOTATION_DEFINITION,
            "axis_marker_subtype": "axis_related",
            "conduction_marker_subtype": "conduction_related",
        },
    }
    return sha256_canonical(payload)


def build_subject_burdens(
    records: Iterable[DatasetRecord], parsed: Iterable[ParsedAnnotations]
) -> tuple[SubjectBurden, ...]:
    """Aggregate only source metadata and expert annotation burden by subject."""
    records = tuple(records)
    parsed = tuple(parsed)
    if len(records) != len(parsed):
        raise ValueError("Every record must have one parsed annotation collection.")
    by_subject: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "records": [],
            "recording_seconds": 0.0,
            "ischemic_episode_count": 0,
            "ischemic_episode_seconds": 0.0,
            "rate_related_episode_count": 0,
            "rate_related_episode_seconds": 0.0,
            "signal_channel_count": 0,
            "axis_shift_marker_count": 0,
            "conduction_change_marker_count": 0,
        }
    )
    for record, parsed_item in zip(records, parsed, strict=True):
        if any(event.record_id != record.record_id for event in parsed_item.events):
            raise ValueError("Parsed events do not match their record metadata.")
        if any(marker.record_id != record.record_id for marker in parsed_item.markers):
            raise ValueError("Parsed markers do not match their record metadata.")
        bucket = by_subject[record.subject_id]
        bucket["records"].append(record.record_id)
        bucket["recording_seconds"] += record.duration_seconds
        bucket["signal_channel_count"] += record.signal_count
        for event in parsed_item.events:
            duration = event.end_seconds - event.onset_seconds
            if event.event_subtype == "ischemic":
                bucket["ischemic_episode_count"] += 1
                bucket["ischemic_episode_seconds"] += duration
            elif event.event_subtype == "heart_rate_related":
                bucket["rate_related_episode_count"] += 1
                bucket["rate_related_episode_seconds"] += duration
        for marker in parsed_item.markers:
            if marker.subtype == "axis_related":
                bucket["axis_shift_marker_count"] += 1
            elif marker.subtype == "conduction_related":
                bucket["conduction_change_marker_count"] += 1
    return tuple(
        SubjectBurden(
            subject_id=subject,
            records=tuple(sorted(bucket["records"])),
            recording_seconds=round(bucket["recording_seconds"], 9),
            ischemic_episode_count=bucket["ischemic_episode_count"],
            ischemic_episode_seconds=round(bucket["ischemic_episode_seconds"], 9),
            rate_related_episode_count=bucket["rate_related_episode_count"],
            rate_related_episode_seconds=round(
                bucket["rate_related_episode_seconds"], 9
            ),
            signal_channel_count=bucket["signal_channel_count"],
            axis_shift_marker_count=bucket["axis_shift_marker_count"],
            conduction_change_marker_count=bucket[
                "conduction_change_marker_count"
            ],
        )
        for subject, bucket in sorted(by_subject.items())
    )


def _partition_capacities(subject_count: int) -> dict[str, int]:
    raw = {
        partition: subject_count * SPLIT_FRACTIONS[partition]
        for partition in PARTITIONS
    }
    capacities = {partition: math.floor(raw[partition]) for partition in PARTITIONS}
    remaining = subject_count - sum(capacities.values())
    order = sorted(
        PARTITIONS,
        key=lambda partition: (
            -(raw[partition] - capacities[partition]),
            PARTITIONS.index(partition),
        ),
    )
    for partition in order[:remaining]:
        capacities[partition] += 1
    return capacities


def _feature_values(burden: SubjectBurden) -> tuple[float, ...]:
    return (
        float(len(burden.records)),
        burden.recording_seconds,
        float(burden.ischemic_episode_count),
        burden.ischemic_episode_seconds,
        float(burden.rate_related_episode_count),
        burden.rate_related_episode_seconds,
        float(burden.signal_channel_count),
        float(burden.axis_shift_marker_count),
        float(burden.axis_shift_marker_count > 0),
        float(burden.conduction_change_marker_count),
        float(burden.conduction_change_marker_count > 0),
    )


def assign_subjects(
    burdens: Iterable[SubjectBurden], seed: int = DEFAULT_SEED
) -> dict[str, tuple[str, ...]]:
    """Assign every subject once using pre-model annotation-burden balancing."""
    burdens = tuple(burdens)
    if not burdens or len({item.subject_id for item in burdens}) != len(burdens):
        raise ValueError("Subject burdens must be non-empty and unique.")
    capacities = _partition_capacities(len(burdens))
    totals = tuple(
        sum(_feature_values(item)[index] for item in burdens)
        for index in range(len(_feature_values(burdens[0])))
    )
    targets = {
        partition: tuple(total * SPLIT_FRACTIONS[partition] for total in totals)
        for partition in PARTITIONS
    }
    current = {
        partition: [0.0] * len(totals)
        for partition in PARTITIONS
    }
    assigned: dict[str, list[str]] = {partition: [] for partition in PARTITIONS}

    def seeded_key(subject: str, partition: str = "") -> str:
        return hashlib.sha256(f"{seed}:{subject}:{partition}".encode()).hexdigest()

    def burden_priority(item: SubjectBurden) -> tuple[float, str]:
        normalized = sum(
            value / total if total else 0.0
            for value, total in zip(_feature_values(item), totals, strict=True)
        )
        return (-normalized, seeded_key(item.subject_id))

    def objective(candidate: str, values: tuple[float, ...]) -> float:
        score = 0.0
        for partition in PARTITIONS:
            for index, target in enumerate(targets[partition]):
                actual = current[partition][index]
                if partition == candidate:
                    actual += values[index]
                score += ((actual - target) / max(abs(target), 1.0)) ** 2
            count = len(assigned[partition]) + (partition == candidate)
            score += ((count - capacities[partition]) / capacities[partition]) ** 2
        return score

    for burden in sorted(burdens, key=burden_priority):
        values = _feature_values(burden)
        candidates = [
            partition
            for partition in PARTITIONS
            if len(assigned[partition]) < capacities[partition]
        ]
        selected = min(
            candidates,
            key=lambda partition: (
                objective(partition, values),
                seeded_key(burden.subject_id, partition),
            ),
        )
        assigned[selected].append(burden.subject_id)
        current[selected] = [
            actual + value
            for actual, value in zip(current[selected], values, strict=True)
        ]

    values_by_subject = {
        burden.subject_id: _feature_values(burden) for burden in burdens
    }

    def feature_objective(state: Mapping[str, list[float]]) -> float:
        return sum(
            ((state[partition][index] - target) / max(abs(target), 1.0)) ** 2
            for partition in PARTITIONS
            for index, target in enumerate(targets[partition])
        )

    while True:
        baseline = feature_objective(current)
        best: tuple[float, str, str, str, str] | None = None
        for left_index, left_partition in enumerate(PARTITIONS):
            for right_partition in PARTITIONS[left_index + 1 :]:
                for left_subject in assigned[left_partition]:
                    left_values = values_by_subject[left_subject]
                    for right_subject in assigned[right_partition]:
                        right_values = values_by_subject[right_subject]
                        candidate_state = {
                            partition: list(values)
                            for partition, values in current.items()
                        }
                        candidate_state[left_partition] = [
                            actual - left + right
                            for actual, left, right in zip(
                                current[left_partition],
                                left_values,
                                right_values,
                                strict=True,
                            )
                        ]
                        candidate_state[right_partition] = [
                            actual - right + left
                            for actual, right, left in zip(
                                current[right_partition],
                                right_values,
                                left_values,
                                strict=True,
                            )
                        ]
                        score = feature_objective(candidate_state)
                        tie = seeded_key(
                            f"{left_subject}:{right_subject}",
                            f"{left_partition}:{right_partition}",
                        )
                        candidate = (
                            score,
                            tie,
                            left_partition,
                            left_subject,
                            right_subject,
                        )
                        if best is None or candidate < best:
                            best = candidate
        if best is None or best[0] >= baseline - 1e-12:
            break
        _, _, left_partition, left_subject, right_subject = best
        right_partition = next(
            partition
            for partition in PARTITIONS
            if partition != left_partition and right_subject in assigned[partition]
        )
        left_values = values_by_subject[left_subject]
        right_values = values_by_subject[right_subject]
        assigned[left_partition].remove(left_subject)
        assigned[left_partition].append(right_subject)
        assigned[right_partition].remove(right_subject)
        assigned[right_partition].append(left_subject)
        current[left_partition] = [
            actual - left + right
            for actual, left, right in zip(
                current[left_partition], left_values, right_values, strict=True
            )
        ]
        current[right_partition] = [
            actual - right + left
            for actual, right, left in zip(
                current[right_partition], right_values, left_values, strict=True
            )
        ]
    return {
        partition: tuple(sorted(assigned[partition])) for partition in PARTITIONS
    }


def _assignment_payload(manifest: Mapping[str, Any]) -> dict[str, object]:
    partitions = manifest["partitions"]
    return {
        "partitions": {
            partition: sorted(partitions[partition]["subjects"])
            for partition in PARTITIONS
        },
        "records_by_subject": {
            subject: sorted(records)
            for subject, records in sorted(manifest["records_by_subject"].items())
        },
    }


def split_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash only the canonical subject and record partition assignment."""
    return sha256_canonical(_assignment_payload(manifest))


def generate_split_manifest(
    records: Iterable[DatasetRecord],
    parsed: Iterable[ParsedAnnotations],
    generation_git_sha: str,
    generation_git_dirty: bool,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    records = tuple(records)
    parsed = tuple(parsed)
    burdens = build_subject_burdens(records, parsed)
    assignment = assign_subjects(burdens, seed)
    burden_by_subject = {item.subject_id: item for item in burdens}
    records_by_subject = {
        item.subject_id: list(item.records) for item in burdens
    }
    partition_summaries: dict[str, dict[str, float | int]] = {}
    for partition in PARTITIONS:
        selected = [burden_by_subject[subject] for subject in assignment[partition]]
        partition_summaries[partition] = {
            "subject_count": len(selected),
            "record_count": sum(len(item.records) for item in selected),
            "recording_seconds": round(
                sum(item.recording_seconds for item in selected), 9
            ),
            "ischemic_episode_count": sum(
                item.ischemic_episode_count for item in selected
            ),
            "ischemic_episode_seconds": round(
                sum(item.ischemic_episode_seconds for item in selected), 9
            ),
            "rate_related_episode_count": sum(
                item.rate_related_episode_count for item in selected
            ),
            "rate_related_episode_seconds": round(
                sum(item.rate_related_episode_seconds for item in selected), 9
            ),
            "axis_shift_marker_count": sum(
                item.axis_shift_marker_count for item in selected
            ),
            "axis_shift_subject_count": sum(
                item.axis_shift_marker_count > 0 for item in selected
            ),
            "conduction_change_marker_count": sum(
                item.conduction_change_marker_count for item in selected
            ),
            "conduction_change_subject_count": sum(
                item.conduction_change_marker_count > 0 for item in selected
            ),
        }
    source_payload = {
        "records": [
            {
                "record_id": record.record_id,
                "subject_id": record.subject_id,
                "sampling_frequency_hz": record.sampling_frequency_hz,
                "sample_count": record.sample_count,
                "signal_count": record.signal_count,
            }
            for record in sorted(records, key=lambda item: item.record_id)
        ],
        "burdens": [item.__dict__ for item in burdens],
    }
    manifest: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": PRIMARY_DATASET,
        "dataset_version": PRIMARY_DATASET_VERSION,
        "annotation_definition": PRIMARY_ANNOTATION_DEFINITION,
        "seed": seed,
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "sealed_test_partition": True,
        "window": {
            "length_seconds": PRIMARY_WINDOW_SECONDS,
            "stride_seconds": PRIMARY_STRIDE_SECONDS,
        },
        "partitions": {
            partition: {"subjects": list(assignment[partition])}
            for partition in PARTITIONS
        },
        "records_by_subject": records_by_subject,
        "partition_summaries": partition_summaries,
        "generation_git_sha": generation_git_sha,
        "generation_git_dirty": generation_git_dirty,
        "generator_code_sha256": generator_code_sha256(),
        "source_metadata_sha256": sha256_canonical(source_payload),
    }
    manifest["split_sha256"] = split_sha256(manifest)
    validate_split_manifest(manifest, records=records, expected_hash=None)
    return manifest


def validate_split_manifest(
    manifest: Mapping[str, Any],
    records: Iterable[DatasetRecord] | None = None,
    expected_hash: str | None = None,
    expected_subject_count: int | None = None,
    expected_record_count: int | None = None,
) -> None:
    """Fail on leakage, incomplete coverage, mapping drift, or hash changes."""
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Split protocol version is not V1.")
    if manifest.get("dataset") != PRIMARY_DATASET:
        raise ValueError("The V1 split must be for LTSTDB.")
    if manifest.get("annotation_definition") != PRIMARY_ANNOTATION_DEFINITION:
        raise ValueError("The V1 split must use the unmixed .stb definition.")
    if manifest.get("sealed_test_partition") is not True:
        raise ValueError("The V1 test partition must be sealed.")
    if not isinstance(manifest.get("generation_git_sha"), str):
        raise ValueError("Split must record generation_git_sha.")
    if not isinstance(manifest.get("generation_git_dirty"), bool):
        raise ValueError("Split must record generation_git_dirty.")
    if manifest.get("generator_code_sha256") != generator_code_sha256():
        raise ValueError("Generator code hash differs from the current implementation.")
    if not isinstance(manifest.get("source_metadata_sha256"), str):
        raise ValueError("Split must record source_metadata_sha256.")
    partitions = manifest.get("partitions")
    records_by_subject = manifest.get("records_by_subject")
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITIONS):
        raise ValueError("Split must define train, validation, and test partitions.")
    if not isinstance(records_by_subject, Mapping):
        raise ValueError("Split must define records_by_subject.")
    subject_to_partition: dict[str, str] = {}
    for partition in PARTITIONS:
        subjects = partitions[partition].get("subjects", ())
        for subject in subjects:
            if subject in subject_to_partition:
                raise ValueError(f"Subject {subject} appears in multiple partitions.")
            subject_to_partition[subject] = partition
    if set(subject_to_partition) != set(records_by_subject):
        raise ValueError("Split subjects and records_by_subject do not match.")
    if (
        expected_subject_count is not None
        and len(subject_to_partition) != expected_subject_count
    ):
        raise ValueError(
            f"Expected {expected_subject_count} split subjects, "
            f"found {len(subject_to_partition)}."
        )
    record_to_subject: dict[str, str] = {}
    for subject, subject_records in records_by_subject.items():
        if not subject_records:
            raise ValueError(f"Subject {subject} has no records.")
        for record in subject_records:
            if record in record_to_subject:
                raise ValueError(f"Record {record} appears under multiple subjects.")
            record_to_subject[record] = subject
    if (
        expected_record_count is not None
        and len(record_to_subject) != expected_record_count
    ):
        raise ValueError(
            f"Expected {expected_record_count} split records, "
            f"found {len(record_to_subject)}."
        )
    record_to_partition = {
        record: subject_to_partition[subject]
        for record, subject in record_to_subject.items()
    }
    validate_subject_partitions(
        subject_to_partition, record_to_subject, record_to_partition
    )
    if records is not None:
        actual = {record.record_id: record.subject_id for record in records}
        if actual != record_to_subject:
            raise ValueError("Record subject mapping differs from source metadata.")
    actual_hash = split_sha256(manifest)
    if manifest.get("split_sha256") != actual_hash:
        raise ValueError("Split hash does not match its canonical assignment.")
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("Test partition hash differs from committed V1.")


def load_split_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Split manifest must be a JSON object.")
    return payload


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
