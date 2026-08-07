"""Read-only progress reporting for an external Phase 3B feature corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FEATURE_ROOT = Path(
    "/home/AI_POC/cardiosentinel-features/ltstdb-baseline-v1"
)
EXPECTED_RECORD_COUNT = 86
MANIFEST_NAME = "manifest.json"
PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class CompletedRecord:
    """The small manifest subset needed for a progress display."""

    record_id: str
    partition: str
    row_count: int
    cache_sha256: str


@dataclass(frozen=True)
class MonitorReport:
    """Read-only summary derived from an existing feature root."""

    feature_root: Path
    status: str
    completed_records: int = 0
    expected_records: int = EXPECTED_RECORD_COUNT
    partitions: tuple[tuple[str, int], ...] = ()
    total_windows: int = 0
    finalized: bool = False
    feature_corpus_sha256: str | None = None
    cache_file_count: int = 0
    disk_usage_bytes: int = 0
    recent_records: tuple[CompletedRecord, ...] = ()
    morphology_valid_windows: int | None = None
    morphology_invalid_windows: int | None = None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _completed_records(records: Sequence[object]) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("status") == "complete"
    ]


def _partition_counts(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, int], ...]:
    counts = Counter(str(record.get("partition", "unknown")) for record in records)
    ordered = [(partition, counts.pop(partition, 0)) for partition in PARTITIONS]
    ordered.extend(sorted(counts.items()))
    return tuple(ordered)


def _feature_root_usage(feature_root: Path) -> tuple[int, int]:
    cache_files = 0
    disk_usage_bytes = 0
    for path in feature_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        disk_usage_bytes += stat.st_blocks * 512
        if path.suffix == ".npz":
            cache_files += 1
    return cache_files, disk_usage_bytes


def inspect_feature_root(feature_root: Path) -> MonitorReport:
    """Inspect existing manifest and cache paths without writing any artifacts."""
    root = feature_root.expanduser()
    if not root.is_dir():
        return MonitorReport(feature_root=root, status="feature root does not exist")

    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return MonitorReport(feature_root=root, status="manifest.json does not exist")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return MonitorReport(
            feature_root=root, status=f"unable to read manifest: {error}"
        )
    if not isinstance(manifest, Mapping):
        return MonitorReport(feature_root=root, status="manifest root is not an object")

    raw_records = manifest.get("records", [])
    records = raw_records if isinstance(raw_records, list) else []
    completed = _completed_records(records)
    cache_file_count, disk_usage_bytes = _feature_root_usage(root)

    morphology_entries = [
        entry["morphology_quality"]
        for entry in completed
        if isinstance(entry.get("morphology_quality"), Mapping)
    ]
    morphology_valid: int | None = None
    morphology_invalid: int | None = None
    if morphology_entries:
        morphology_valid = sum(
            _nonnegative_int(entry.get("morphology_valid_windows"))
            for entry in morphology_entries
        )
        morphology_invalid = sum(
            _nonnegative_int(entry.get("morphology_invalid_windows"))
            for entry in morphology_entries
        )

    recent = tuple(
        CompletedRecord(
            record_id=str(entry.get("record_id", "unknown")),
            partition=str(entry.get("partition", "unknown")),
            row_count=_nonnegative_int(entry.get("row_count")),
            cache_sha256=str(entry.get("cache_sha256", "")),
        )
        for entry in completed[-5:]
    )
    corpus_sha256 = manifest.get("feature_corpus_sha256")
    return MonitorReport(
        feature_root=root,
        status="ok",
        completed_records=len(completed),
        partitions=_partition_counts(completed),
        total_windows=sum(
            _nonnegative_int(entry.get("row_count")) for entry in completed
        ),
        finalized=bool(corpus_sha256),
        feature_corpus_sha256=(str(corpus_sha256) if corpus_sha256 else None),
        cache_file_count=cache_file_count,
        disk_usage_bytes=disk_usage_bytes,
        recent_records=recent,
        morphology_valid_windows=morphology_valid,
        morphology_invalid_windows=morphology_invalid,
    )


def _format_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("Byte formatter must select a unit.")


def render_monitor_report(report: MonitorReport) -> str:
    """Render a compact human-readable report without altering its input."""
    lines = ["Phase 3B Materialization Monitor", "--------------------------------"]
    lines.append(f"Feature root: {report.feature_root}")
    if report.status != "ok":
        lines.append(f"Status: {report.status}")
        return "\n".join(lines)

    percentage = 100.0 * report.completed_records / report.expected_records
    partitions = " ".join(
        f"{partition}={count}" for partition, count in report.partitions
    )
    lines.extend(
        (
            f"Completed records: {report.completed_records} / "
            f"{report.expected_records} ({percentage:.1f}%)",
            f"Partitions: {partitions}",
            f"Windows materialized: {report.total_windows:,}",
            f"Cache files: {report.cache_file_count}",
            f"Feature-root disk usage: {_format_bytes(report.disk_usage_bytes)}",
            f"Feature corpus finalized: {report.finalized}",
        )
    )
    if report.feature_corpus_sha256 is not None:
        lines.append(f"Feature corpus SHA-256: {report.feature_corpus_sha256}")

    if report.morphology_valid_windows is not None:
        assert report.morphology_invalid_windows is not None
        morphology_total = (
            report.morphology_valid_windows + report.morphology_invalid_windows
        )
        validity = (
            100.0 * report.morphology_valid_windows / morphology_total
            if morphology_total
            else 0.0
        )
        lines.extend(
            (
                "",
                "Morphology:",
                f"Valid windows: {report.morphology_valid_windows:,}",
                f"Invalid windows: {report.morphology_invalid_windows:,}",
                f"Validity: {validity:.1f}%",
            )
        )

    if report.recent_records:
        lines.extend(("", "Recently completed:"))
        for record in report.recent_records:
            digest = record.cache_sha256[:12] or "-"
            lines.append(
                f"{record.record_id}  {record.partition}  "
                f"{record.row_count:,}  {digest}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only progress monitor for a Phase 3B feature corpus."
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
        help=f"external feature root (default: {DEFAULT_FEATURE_ROOT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(render_monitor_report(inspect_feature_root(args.feature_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
