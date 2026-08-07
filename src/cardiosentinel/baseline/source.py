"""Pinned LTSTDB V1 source acquisition and checksum verification."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cardiosentinel.baseline.cache import (
    read_json,
    require_nonversioned_path,
    write_json_atomic,
)
from cardiosentinel.data.provenance import sha256_file

DATASET = "ltstdb"
DATASET_VERSION = "1.0.0"
PINNED_SOURCE_URL = "https://physionet.org/files/ltstdb/1.0.0/"
OFFICIAL_MANIFEST_NAME = "SHA256SUMS.txt"
OFFICIAL_MANIFEST_SHA256 = (
    "88b8c6d17451d758defb3355526430d607f0eeccf7fc9bb7dc992cde212c220e"
)
RECORDS_NAME = "RECORDS"
SOURCE_VERIFICATION_RECEIPT_NAME = "source_verification.json"
REQUIRED_SUFFIXES = ("hea", "dat", "stb")
_CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")


def required_source_filenames(record_ids: Iterable[str]) -> tuple[str, ...]:
    """Return the exact, minimal source file set needed by Benchmark V1."""
    records = tuple(sorted(set(record_ids)))
    if not records:
        raise ValueError("Source verification requires at least one record ID.")
    return (
        RECORDS_NAME,
        *(
            f"{record_id}.{suffix}"
            for record_id in records
            for suffix in REQUIRED_SUFFIXES
        ),
    )


def acquisition_urls(record_ids: Iterable[str]) -> tuple[str, ...]:
    """Return the pinned, minimal resumable acquisition URL set."""
    return tuple(
        f"{PINNED_SOURCE_URL}{filename}"
        for filename in (OFFICIAL_MANIFEST_NAME, *required_source_filenames(record_ids))
    )


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    """Parse a standard SHA-256 manifest, rejecting ambiguous entries."""
    if not path.is_file():
        raise FileNotFoundError(f"Official checksum manifest is absent: {path}")
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        match = _CHECKSUM_LINE.fullmatch(raw_line)
        if match is None:
            raise ValueError(
                f"Malformed checksum manifest entry on line {line_number}."
            )
        digest, filename = match.groups()
        candidate = Path(filename)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                f"Unsafe checksum manifest filename on line {line_number}."
            )
        if filename in entries:
            raise ValueError(f"Duplicate checksum manifest entry: {filename}")
        entries[filename] = digest.lower()
    if not entries:
        raise ValueError("Official checksum manifest contains no entries.")
    return entries


def _records_from_file(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required LTSTDB RECORDS file is absent: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _verify_source_against_manifest(
    source: Path,
    expected_record_ids: Iterable[str],
    *,
    official_manifest_sha256: str,
    write_receipt: bool,
) -> dict[str, Any]:
    """Verify exact required files against an explicitly pinned manifest digest.

    This lower-level function supports small offline fixtures; production callers
    use :func:`verify_pinned_source`, which always supplies the official digest.
    """
    root = require_nonversioned_path(source, "Waveform source")
    expected_records = set(expected_record_ids)
    if not expected_records:
        raise ValueError("Source verification requires expected record IDs.")
    manifest_path = root / OFFICIAL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Official checksum manifest is absent: {manifest_path}"
        )
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != official_manifest_sha256:
        raise ValueError(
            "Official SHA256SUMS.txt digest does not match the pinned release."
        )
    checksums = parse_checksum_manifest(manifest_path)
    records = _records_from_file(root / RECORDS_NAME)
    if records != expected_records:
        raise ValueError("RECORDS set differs from the frozen Benchmark V1 records.")

    required = required_source_filenames(expected_records)
    missing = [filename for filename in required if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Required LTSTDB source files are absent: {missing}")
    missing_entries = [filename for filename in required if filename not in checksums]
    if missing_entries:
        raise ValueError(
            f"Required LTSTDB files lack official checksum entries: {missing_entries}"
        )
    mismatches = [
        filename
        for filename in required
        if sha256_file(root / filename) != checksums[filename]
    ]
    if mismatches:
        raise ValueError(f"LTSTDB source checksum mismatch: {mismatches}")

    receipt = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "official_manifest": {
            "identifier": f"{PINNED_SOURCE_URL}{OFFICIAL_MANIFEST_NAME}",
            "filename": OFFICIAL_MANIFEST_NAME,
            "sha256": official_manifest_sha256,
        },
        "expected_record_count": len(expected_records),
        "verified_required_file_count": len(required),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verification_result": "passed",
        "local_source_root": str(root),
    }
    if write_receipt:
        write_json_atomic(root / SOURCE_VERIFICATION_RECEIPT_NAME, receipt)
    return receipt


def verify_pinned_source(
    source: Path, expected_record_ids: Iterable[str]
) -> dict[str, Any]:
    """Verify the minimal source set against PhysioNet's pinned V1 manifest."""
    return _verify_source_against_manifest(
        source,
        expected_record_ids,
        official_manifest_sha256=OFFICIAL_MANIFEST_SHA256,
        write_receipt=True,
    )


def source_checksum_receipt_is_valid(
    source: Path, expected_record_ids: Iterable[str]
) -> bool:
    """Return whether an external successful receipt matches this exact source plan."""
    root = source.expanduser().resolve()
    receipt_path = root / SOURCE_VERIFICATION_RECEIPT_NAME
    if not receipt_path.is_file():
        return False
    try:
        receipt = read_json(receipt_path)
    except ValueError:
        return False
    return (
        receipt.get("dataset") == DATASET
        and receipt.get("dataset_version") == DATASET_VERSION
        and receipt.get("official_manifest", {}).get("identifier")
        == f"{PINNED_SOURCE_URL}{OFFICIAL_MANIFEST_NAME}"
        and receipt.get("official_manifest", {}).get("sha256")
        == OFFICIAL_MANIFEST_SHA256
        and receipt.get("expected_record_count") == len(set(expected_record_ids))
        and receipt.get("verified_required_file_count")
        == len(required_source_filenames(expected_record_ids))
        and receipt.get("verification_result") == "passed"
        and receipt.get("local_source_root") == str(root)
    )
