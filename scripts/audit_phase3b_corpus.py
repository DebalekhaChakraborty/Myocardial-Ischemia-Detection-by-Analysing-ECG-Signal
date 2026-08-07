"""Read-only integrity audit for a completed Phase 3B feature corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    read_json,
    validate_feature_corpus,
)
from cardiosentinel.baseline.preflight import PRIMARY_COUNTS
from cardiosentinel.evaluation.splits import load_split_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPLIT_PATH = REPOSITORY_ROOT / "protocols/splits/ltstdb_v1.json"
PARTITIONS = ("train", "validation", "test")
PRIMARY_FAMILIES = ("ischemic_positive", "background_negative")


def _expected_record_ids(split: Mapping[str, Any]) -> set[str]:
    """Derive the required records from the committed frozen split manifest."""
    return {
        record_id
        for record_ids in split["records_by_subject"].values()
        for record_id in record_ids
    }


def _completed_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Feature manifest records must be a list.")
    return [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("status") == "complete"
    ]


def _target_counts(records: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    by_partition = {partition: Counter() for partition in PARTITIONS}
    for record in records:
        partition = record.get("partition")
        if partition not in by_partition:
            raise ValueError(f"Unexpected completed-record partition: {partition!r}")
        target_counts = record.get("target_counts")
        if not isinstance(target_counts, Mapping):
            raise ValueError("Completed record lacks target_counts metadata.")
        for family, count in target_counts.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Completed record has invalid target count metadata.")
            by_partition[partition][str(family)] += count
    return {
        partition: dict(sorted(counts.items()))
        for partition, counts in by_partition.items()
    }


def _morphology_counts(records: list[Mapping[str, Any]]) -> tuple[int, int]:
    valid = 0
    invalid = 0
    for record in records:
        quality = record.get("morphology_quality")
        if quality is None:
            continue
        if not isinstance(quality, Mapping):
            raise ValueError(
                "Completed record has invalid morphology_quality metadata."
            )
        for key, accumulator in (
            ("morphology_valid_windows", "valid"),
            ("morphology_invalid_windows", "invalid"),
        ):
            count = quality.get(key, 0)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Completed record has invalid morphology metadata.")
            if accumulator == "valid":
                valid += count
            else:
                invalid += count
    return valid, invalid


def audit_feature_corpus(
    feature_root: Path, split_path: Path = FROZEN_SPLIT_PATH
) -> dict[str, object]:
    """Validate and summarize a corpus without modifying it or its source data."""
    split = load_split_manifest(split_path)
    expected_records = _expected_record_ids(split)
    corpus_sha256 = validate_feature_corpus(feature_root, expected_records)
    manifest = read_json(feature_root / FEATURE_MANIFEST_NAME)
    completed = _completed_records(manifest)
    targets = _target_counts(completed)
    observed_primary = {
        partition: {
            family: targets[partition].get(family, 0) for family in PRIMARY_FAMILIES
        }
        for partition in PARTITIONS
    }
    if observed_primary != PRIMARY_COUNTS:
        raise ValueError(
            "Primary Benchmark V1 counts differ from the frozen protocol: "
            f"observed={observed_primary}, expected={PRIMARY_COUNTS}."
        )
    morphology_valid, morphology_invalid = _morphology_counts(completed)
    morphology_total = morphology_valid + morphology_invalid
    completed_by_partition = {
        partition: sum(record.get("partition") == partition for record in completed)
        for partition in PARTITIONS
    }
    rows_by_partition = {
        partition: sum(
            record.get("row_count", 0)
            for record in completed
            if record.get("partition") == partition
        )
        for partition in PARTITIONS
    }
    descriptive_families = {
        partition: {
            family: count
            for family, count in targets[partition].items()
            if family not in PRIMARY_FAMILIES
        }
        for partition in PARTITIONS
    }
    return {
        "feature_corpus_sha256": corpus_sha256,
        "generation": manifest.get("generation", {}),
        "expected_record_count": len(expected_records),
        "completed_records_by_partition": completed_by_partition,
        "total_rows_by_partition": rows_by_partition,
        "target_family_counts_by_partition": targets,
        "descriptive_challenge_exclusion_counts": descriptive_families,
        "total_rows": sum(rows_by_partition.values()),
        "morphology_valid_windows": morphology_valid,
        "morphology_invalid_windows": morphology_invalid,
        "morphology_validity_fraction": (
            morphology_valid / morphology_total if morphology_total else None
        ),
        "frozen_primary_counts": PRIMARY_COUNTS,
    }


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{family}={count:,}" for family, count in counts.items())


def render_audit(summary: Mapping[str, object]) -> str:
    """Render a compact audit report without adding interpretation."""
    lines = ["Phase 3B Corpus Integrity Audit", "==============================="]
    lines.append(f"Feature corpus SHA-256: {summary['feature_corpus_sha256']}")
    lines.append("Generation metadata:")
    lines.append(json.dumps(summary["generation"], indent=2, sort_keys=True))
    lines.append("")
    lines.append("Partition summaries:")
    completed = summary["completed_records_by_partition"]
    rows = summary["total_rows_by_partition"]
    targets = summary["target_family_counts_by_partition"]
    primary = summary["frozen_primary_counts"]
    descriptive = summary["descriptive_challenge_exclusion_counts"]
    for partition in PARTITIONS:
        partition_targets = targets[partition]
        lines.extend(
            (
                f"{partition}: records={completed[partition]}, "
                f"rows={rows[partition]:,}",
                f"  target families: {_format_counts(partition_targets)}",
                f"  frozen primary counts: {_format_counts(primary[partition])}",
            )
        )
        if descriptive[partition]:
            lines.append(
                "  descriptive challenge/exclusion families: "
                f"{_format_counts(descriptive[partition])}"
            )
    lines.append(f"Total rows: {summary['total_rows']:,}")
    valid = summary["morphology_valid_windows"]
    invalid = summary["morphology_invalid_windows"]
    fraction = summary["morphology_validity_fraction"]
    lines.extend(
        (
            "",
            "Algorithmic morphology-feature validity (frozen morphology_v1 criterion):",
            f"  valid windows: {valid:,}",
            f"  invalid windows: {invalid:,}",
        )
    )
    if fraction is None:
        lines.append("  validity: unavailable (no morphology windows recorded)")
    else:
        lines.append(
            f"  validity: {valid:,} / {valid + invalid:,} "
            f"({fraction:.8%})"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only integrity audit for a completed Phase 3B corpus."
    )
    parser.add_argument(
        "--feature-root",
        required=True,
        type=Path,
        help="external or approved Git-ignored feature corpus root",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = audit_feature_corpus(args.feature_root)
    except (FileNotFoundError, ValueError) as error:
        print(f"Corpus audit failed: {error}", file=sys.stderr)
        return 1
    print(render_audit(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
