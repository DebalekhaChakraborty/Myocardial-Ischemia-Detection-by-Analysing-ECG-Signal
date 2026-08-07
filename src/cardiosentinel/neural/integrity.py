"""Development-only integrity receipts for B4 metadata and waveform sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    compute_feature_corpus_sha256,
    read_cache_metadata,
    read_json,
    require_nonversioned_path,
)
from cardiosentinel.baseline.source import (
    OFFICIAL_MANIFEST_NAME,
    OFFICIAL_MANIFEST_SHA256,
    parse_checksum_manifest,
)
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.evaluation.protocol import PRIMARY_ANNOTATION_DEFINITION
from cardiosentinel.features.schema import COMBINED_V1
from cardiosentinel.neural.protocol import (
    B4_SPLIT_SHA256,
    DATASET,
    DATASET_VERSION,
    FEATURE_CORPUS_SHA256,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
)

DEVELOPMENT_PARTITIONS = frozenset(("train", "validation"))
SOURCE_SUFFIXES = ("hea", "dat", "stb")


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_cache_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Development feature cache path escapes its root.") from error
    return path


def _development_entries(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    entries = tuple(
        sorted(
            (
                entry
                for entry in manifest.get("records", ())
                if entry.get("partition") in DEVELOPMENT_PARTITIONS
            ),
            key=lambda entry: str(entry.get("record_id")),
        )
    )
    record_ids = [entry.get("record_id") for entry in entries]
    if not entries or len(set(record_ids)) != len(record_ids):
        raise ValueError("Development feature records are absent or duplicated.")
    if any(entry.get("status") != "complete" for entry in entries):
        raise ValueError("Development feature corpus contains an incomplete record.")
    return entries


def _validate_manifest_identity(
    manifest: dict[str, Any], expected_corpus_sha256: str
) -> None:
    expected = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "split_sha256": B4_SPLIT_SHA256,
        "processing_profile": "raw",
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "annotation_definition": PRIMARY_ANNOTATION_DEFINITION,
        "feature_corpus_sha256": expected_corpus_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Development feature manifest identity is not frozen.")
    schema = manifest.get("feature_schemas", {}).get("combined_v1", {})
    if schema.get("feature_schema_sha256") != COMBINED_V1.sha256:
        raise ValueError("Development feature manifest has the wrong schema.")
    if compute_feature_corpus_sha256(manifest) != expected_corpus_sha256:
        raise ValueError("Canonical feature-corpus SHA-256 does not match.")


def _expected_embedded_metadata(
    manifest: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    return {
        "dataset": manifest["dataset"],
        "dataset_version": manifest["dataset_version"],
        "record_id": entry["record_id"],
        "subject_id": entry["subject_id"],
        "partition": entry["partition"],
        "source_sha256": entry["source_sha256"],
        "split_sha256": manifest["split_sha256"],
        "feature_schema_sha256": manifest["feature_schemas"]["combined_v1"][
            "feature_schema_sha256"
        ],
        "processing_profile": manifest["processing_profile"],
        "window_seconds": manifest["window_seconds"],
        "stride_seconds": manifest["stride_seconds"],
        "annotation_definition": manifest["annotation_definition"],
        "row_count": entry["row_count"],
        "target_counts": entry["target_counts"],
    }


def _validate_development_feature_integrity(
    feature_root: Path, *, expected_corpus_sha256: str
) -> dict[str, Any]:
    root = require_nonversioned_path(feature_root, "B4 feature metadata root")
    manifest = read_json(root / FEATURE_MANIFEST_NAME)
    _validate_manifest_identity(manifest, expected_corpus_sha256)
    verified_records: list[dict[str, Any]] = []
    for entry in _development_entries(manifest):
        cache_path = _safe_cache_path(root, str(entry["cache_path"]))
        actual_cache_sha256 = sha256_file(cache_path)
        if actual_cache_sha256 != entry.get("cache_sha256"):
            raise ValueError(
                f"Development feature cache SHA-256 mismatch: {entry['record_id']}"
            )
        metadata = read_cache_metadata(cache_path)
        expected_metadata = _expected_embedded_metadata(manifest, entry)
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(
                f"Development feature metadata mismatch: {entry['record_id']}"
            )
        verified_records.append(
            {
                "record_id": entry["record_id"],
                "subject_id": entry["subject_id"],
                "partition": entry["partition"],
                "row_count": entry["row_count"],
                "cache_sha256": actual_cache_sha256,
                "source_sha256": entry["source_sha256"],
            }
        )

    scientific_payload = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": expected_corpus_sha256,
        "partitions": sorted(DEVELOPMENT_PARTITIONS),
        "records": verified_records,
    }
    return {
        **scientific_payload,
        "development_feature_integrity_sha256": canonical_sha256(
            scientific_payload
        ),
        "verified_record_count": len(verified_records),
        "local_feature_root": str(root),
        "verification_result": "passed",
    }


def validate_development_feature_integrity(feature_root: Path) -> dict[str, Any]:
    """Hash and validate train/validation NPZ files without opening test caches."""
    return _validate_development_feature_integrity(
        feature_root, expected_corpus_sha256=FEATURE_CORPUS_SHA256
    )


def source_record_sha256(record_id: str, file_digests: dict[str, str]) -> str:
    """Reproduce the Phase 3B filename-plus-file-digest record convention."""
    digest = hashlib.sha256()
    for suffix in SOURCE_SUFFIXES:
        filename = f"{record_id}.{suffix}"
        digest.update(filename.encode("ascii"))
        digest.update(file_digests[filename].encode("ascii"))
    return digest.hexdigest()


def _validate_development_source_integrity(
    source: Path,
    feature_receipt: dict[str, Any],
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = require_nonversioned_path(source, "B4 waveform source")
    manifest_path = root / OFFICIAL_MANIFEST_NAME
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Official source manifest digest is not pinned.")
    official = parse_checksum_manifest(manifest_path)
    verified_records: list[dict[str, Any]] = []
    for entry in feature_receipt.get("records", ()):
        partition = entry.get("partition")
        if partition not in DEVELOPMENT_PARTITIONS:
            raise ValueError(
                "Development source receipt contains a forbidden partition."
            )
        record_id = str(entry["record_id"])
        if Path(record_id).name != record_id:
            raise ValueError("Development source record ID is unsafe.")
        file_digests: dict[str, str] = {}
        for suffix in SOURCE_SUFFIXES:
            filename = f"{record_id}.{suffix}"
            if filename not in official:
                raise ValueError(f"Official source entry is absent: {filename}")
            actual = sha256_file(root / filename)
            if actual != official[filename]:
                raise ValueError(f"Development source SHA-256 mismatch: {filename}")
            file_digests[filename] = actual
        record_digest = source_record_sha256(record_id, file_digests)
        if record_digest != entry.get("source_sha256"):
            raise ValueError(
                f"Development source record digest mismatch: {record_id}"
            )
        verified_records.append(
            {
                "record_id": record_id,
                "partition": partition,
                "files": file_digests,
                "source_sha256": record_digest,
            }
        )

    scientific_payload = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "split_sha256": B4_SPLIT_SHA256,
        "official_manifest_sha256": expected_manifest_sha256,
        "partitions": sorted(DEVELOPMENT_PARTITIONS),
        "records": verified_records,
    }
    return {
        **scientific_payload,
        "development_source_integrity_sha256": canonical_sha256(scientific_payload),
        "verified_record_count": len(verified_records),
        "verified_file_count": len(verified_records) * len(SOURCE_SUFFIXES),
        "local_source_root": str(root),
        "verification_result": "passed",
    }


def validate_development_source_integrity(
    source: Path, feature_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Hash current train/validation source files, never sealed-test files."""
    return _validate_development_source_integrity(
        source,
        feature_receipt,
        expected_manifest_sha256=OFFICIAL_MANIFEST_SHA256,
    )
