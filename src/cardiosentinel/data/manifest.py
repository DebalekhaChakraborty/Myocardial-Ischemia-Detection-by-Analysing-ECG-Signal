"""Dataset discovery, acquisition manifests, and reproducible JSON serialization."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlretrieve

from cardiosentinel.data import edb, ltstdb
from cardiosentinel.data.models import DatasetRecord, ParsedAnnotations
from cardiosentinel.data.provenance import (
    git_provenance,
    selected_file_digests,
    verify_sha256_manifest,
)
from cardiosentinel.data.validation import ValidationReport, validate_dataset

PARSER_SCHEMA_VERSION = "1.0"


def dataset_module(dataset_id: str) -> Any:
    """Return the adapter only for an explicitly supported dataset identifier."""
    if dataset_id == "edb":
        return edb
    if dataset_id == "ltstdb":
        return ltstdb
    raise ValueError(f"Unsupported dataset {dataset_id}; use edb or ltstdb.")


def discover_record_ids(root: Path) -> tuple[str, ...]:
    """Read the official RECORDS file instead of guessing from arbitrary files."""
    records_file = root / "RECORDS"
    if not records_file.is_file():
        raise FileNotFoundError(f"Missing official RECORDS file in {root}.")
    records = tuple(
        line.strip()
        for line in records_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(records) != len(set(records)):
        raise ValueError("RECORDS contains duplicate record identifiers.")
    return records


def inspect_dataset(
    dataset_id: str, source: Path, annotation_set: str | None = None
) -> tuple[tuple[DatasetRecord, ...], tuple[ParsedAnnotations, ...]]:
    """Inspect local WFDB headers and one unmixed annotation definition per record."""
    module = dataset_module(dataset_id)
    record_ids = discover_record_ids(source)
    if dataset_id == "edb" and set(record_ids) != set(edb.EDB_RECORD_IDS):
        raise ValueError(
            "EDB RECORDS does not match the documented v1.0.0 record mapping."
        )
    if dataset_id == "edb" and annotation_set not in (None, edb.PRIMARY_ANNOTATION):
        raise ValueError("EDB supports only the documented atr annotation stream.")
    selected = annotation_set or module.PRIMARY_ANNOTATION
    records = tuple(module.read_record(source, record_id) for record_id in record_ids)
    if dataset_id == "edb":
        parsed = tuple(module.read_annotations(source, record) for record in records)
    else:
        parsed = tuple(
            module.read_annotations(source, record, selected) for record in records
        )
    return records, parsed


def validate_local_dataset(
    dataset_id: str, source: Path, annotation_set: str | None = None
) -> ValidationReport:
    """Run full-dataset validation; count mismatches are blocking errors."""
    module = dataset_module(dataset_id)
    records, parsed = inspect_dataset(dataset_id, source, annotation_set)
    events = tuple(event for item in parsed for event in item.events)
    intervals = tuple(
        interval for item in parsed for interval in item.quality_intervals
    )
    markers = tuple(marker for item in parsed for marker in item.markers)
    primary = (
        "edb.reference"
        if dataset_id == "edb"
        else f"ltstdb.{module.PRIMARY_ANNOTATION}"
    )
    return validate_dataset(
        records,
        events,
        intervals,
        module.EXPECTED_RECORD_COUNT,
        module.EXPECTED_SUBJECT_COUNT,
        primary,
        markers,
        parsed,
    )


def build_manifest(
    dataset_id: str,
    source: Path,
    command: str,
    annotation_set: str | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Build a deterministic manifest when `generated_at` is supplied by a caller."""
    module = dataset_module(dataset_id)
    records, parsed = inspect_dataset(dataset_id, source, annotation_set)
    events = tuple(event for item in parsed for event in item.events)
    intervals = tuple(
        interval for item in parsed for interval in item.quality_intervals
    )
    markers = tuple(marker for item in parsed for marker in item.markers)
    relevant = [source / "RECORDS", source / "SHA256SUMS.txt"]
    relevant.extend(source / f"{record.record_id}.hea" for record in records)
    selected = annotation_set or module.PRIMARY_ANNOTATION
    relevant.extend(source / f"{record.record_id}.{selected}" for record in records)
    unknown_patterns = sorted(
        {pattern for item in parsed for pattern in item.unknown_aux_patterns}
    )[:20]
    return {
        "schema_version": PARSER_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "dataset": {"id": module.DATASET_ID, "version": module.DATASET_VERSION},
        "source_location": source.name,
        "command": command,
        "provenance": git_provenance(Path(__file__).resolve().parents[3]),
        "records": [asdict(record) for record in records],
        "subject_mapping": {record.record_id: record.subject_id for record in records},
        "file_digests": selected_file_digests(source, relevant),
        "annotation_counts": {"events": len(events), "markers": len(markers)},
        "recognized_annotation_count": sum(
            item.recognized_annotation_count for item in parsed
        ),
        "ignored_known_annotation_count": sum(
            item.ignored_known_annotation_count for item in parsed
        ),
        "unknown_annotation_count": sum(
            len(item.unknown_annotations) for item in parsed
        ),
        "unknown_aux_patterns": unknown_patterns,
        "event_counts_by_type": dict(
            sorted(Counter(event.event_subtype for event in events).items())
        ),
        "event_counts_by_lead": dict(
            sorted(Counter(str(event.lead) for event in events).items())
        ),
        "signal_quality_counts": dict(
            sorted(Counter(interval.state for interval in intervals).items())
        ),
        "validation": asdict(
            validate_local_dataset(dataset_id, source, annotation_set)
        ),
    }


def write_manifest(manifest: dict[str, object], output: Path) -> None:
    """Write canonical, sorted JSON for comparison outside the repository."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def destination_is_git_ignored(destination: Path) -> bool:
    """Refuse raw downloads within a repository unless Git ignores the location."""
    repository_root = Path(__file__).resolve().parents[3]
    try:
        destination.resolve().relative_to(repository_root)
    except ValueError:
        return True
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", str(destination)],
        cwd=repository_root,
        check=False,
    )
    return result.returncode == 0


def _download_metadata_file(source_url: str, destination: Path, filename: str) -> Path:
    """Download a known release file without following a moving dataset version."""
    target = (destination / filename).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as error:
        raise ValueError(f"Invalid metadata filename {filename!r}.") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(f"{source_url}{quote(filename, safe='/')}", target)
    return target


def download_metadata(
    dataset_id: str, destination: Path, annotation_set: str | None = None
) -> Path:
    """Download headers and one annotation stream, never ECG waveform files."""
    if not destination_is_git_ignored(destination):
        raise ValueError(
            "Refusing to download physiological data into a Git-tracked location."
        )
    module = dataset_module(dataset_id)

    destination.mkdir(parents=True, exist_ok=True)
    if dataset_id == "edb" and annotation_set not in (None, edb.PRIMARY_ANNOTATION):
        raise ValueError("EDB supports only the documented atr annotation stream.")
    selected = annotation_set or module.PRIMARY_ANNOTATION
    source_url = (
        f"https://physionet.org/files/{module.DATASET_ID}/{module.DATASET_VERSION}/"
    )
    checksum_path = _download_metadata_file(source_url, destination, "SHA256SUMS.txt")
    _download_metadata_file(source_url, destination, "RECORDS")
    record_ids = discover_record_ids(destination)
    files = [f"{record_id}.hea" for record_id in record_ids]
    files.extend(f"{record_id}.{selected}" for record_id in record_ids)
    for filename in files:
        _download_metadata_file(source_url, destination, filename)
    selected_files = [destination / "RECORDS", *[destination / name for name in files]]
    digests = verify_sha256_manifest(checksum_path, destination, selected_files)
    manifest = {
        "dataset": {"id": module.DATASET_ID, "version": module.DATASET_VERSION},
        "source": source_url,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "file_list": [
            path.relative_to(destination).as_posix() for path in selected_files
        ],
        "checksums": digests,
    }
    write_manifest(manifest, destination / "cardiosentinel-acquisition.json")
    return destination
