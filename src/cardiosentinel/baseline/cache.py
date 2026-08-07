"""External, atomic, per-record feature cache structures and validation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.evaluation.protocol import (
    LTSTDB_V1_SPLIT_SHA256,
    PRIMARY_ANNOTATION_DEFINITION,
    PRIMARY_STRIDE_SECONDS,
    PRIMARY_WINDOW_SECONDS,
)
from cardiosentinel.features.schema import COMBINED_V1

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FEATURE_MANIFEST_NAME = "manifest.json"


def require_external_path(path: Path, purpose: str) -> Path:
    """Reject patient-derived data or experiment outputs inside this checkout."""
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved
    raise ValueError(f"{purpose} must be outside the Git repository: {resolved}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON via same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_cache_metadata(path: Path) -> dict[str, Any]:
    """Read only embedded cache metadata, never the numeric feature matrix."""
    with np.load(path, allow_pickle=False) as cached:
        return json.loads(str(cached["metadata_json"]))


def feature_corpus_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return only immutable scientific identities in canonical record order."""
    combined = manifest["feature_schemas"]["combined_v1"]
    records = []
    for entry in sorted(manifest["records"], key=lambda item: item["record_id"]):
        records.append(
            {
                "record_id": entry["record_id"],
                "subject_id": entry["subject_id"],
                "partition": entry["partition"],
                "source_sha256": entry["source_sha256"],
                "row_count": entry["row_count"],
                "target_counts": entry["target_counts"],
                "cache_sha256": entry["cache_sha256"],
            }
        )
    return {
        "dataset": manifest["dataset"],
        "dataset_version": manifest["dataset_version"],
        "split_sha256": manifest["split_sha256"],
        "feature_schema_sha256": combined["feature_schema_sha256"],
        "processing_profile": manifest["processing_profile"],
        "window_seconds": manifest["window_seconds"],
        "stride_seconds": manifest["stride_seconds"],
        "annotation_definition": manifest["annotation_definition"],
        "records": records,
    }


def compute_feature_corpus_sha256(manifest: dict[str, Any]) -> str:
    """Fingerprint the exact cache corpus without mutable runtime bookkeeping."""
    return _canonical_sha256(feature_corpus_payload(manifest))


def validate_feature_corpus(feature_root: Path, expected_record_ids: set[str]) -> str:
    """Validate every exact cache artifact and the persisted corpus fingerprint."""
    root = require_external_path(feature_root, "Feature root")
    manifest = read_json(root / FEATURE_MANIFEST_NAME)
    frozen_identity = {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "split_sha256": LTSTDB_V1_SPLIT_SHA256,
        "processing_profile": "raw",
        "window_seconds": PRIMARY_WINDOW_SECONDS,
        "stride_seconds": PRIMARY_STRIDE_SECONDS,
        "annotation_definition": PRIMARY_ANNOTATION_DEFINITION,
    }
    if any(manifest.get(key) != value for key, value in frozen_identity.items()):
        raise ValueError(
            "Feature corpus does not match the frozen scientific identity."
        )
    if (
        manifest.get("feature_schemas", {})
        .get("combined_v1", {})
        .get("feature_schema_sha256")
        != COMBINED_V1.sha256
    ):
        raise ValueError("Feature corpus does not use the frozen combined_v1 schema.")
    completed = [
        entry for entry in manifest["records"] if entry.get("status") == "complete"
    ]
    entries = {entry["record_id"]: entry for entry in completed}
    if len(entries) != len(completed) or set(entries) != expected_record_ids:
        raise ValueError("Feature corpus does not contain all required records.")
    for record_id in sorted(expected_record_ids):
        entry = entries[record_id]
        cache_path = (root / entry["cache_path"]).resolve()
        try:
            cache_path.relative_to(root)
        except ValueError as error:
            raise ValueError("Feature cache path escapes the feature root.") from error
        recorded_digest = entry.get("cache_sha256")
        if not cache_path.is_file() or recorded_digest != sha256_file(cache_path):
            raise ValueError(f"Feature cache digest validation failed for {record_id}.")
        metadata = read_cache_metadata(cache_path)
        expected_metadata = {
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
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(
                f"Feature cache metadata validation failed for {record_id}."
            )
    computed = compute_feature_corpus_sha256(manifest)
    if manifest.get("feature_corpus_sha256") != computed:
        raise ValueError("Feature corpus SHA-256 validation failed.")
    return computed


def finalize_feature_corpus(feature_root: Path, expected_record_ids: set[str]) -> str:
    """Persist corpus identity only after all required caches are complete."""
    root = require_external_path(feature_root, "Feature root")
    manifest_path = root / FEATURE_MANIFEST_NAME
    manifest = read_json(manifest_path)
    complete_ids = {
        entry["record_id"]
        for entry in manifest["records"]
        if entry.get("status") == "complete" and entry.get("cache_sha256")
    }
    if complete_ids != expected_record_ids:
        raise ValueError("Cannot finalize an incomplete feature corpus.")
    manifest["feature_corpus_sha256"] = compute_feature_corpus_sha256(manifest)
    write_json_atomic(manifest_path, manifest)
    return validate_feature_corpus(root, expected_record_ids)


@dataclass(frozen=True)
class FeatureTable:
    """Numeric inputs and non-predictive row metadata stored in separate arrays."""

    features: NDArray[np.float64]
    stable_ids: NDArray[np.str_]
    record_ids: NDArray[np.str_]
    subject_ids: NDArray[np.str_]
    channel_indices: NDArray[np.int64]
    lead_names: NDArray[np.str_]
    window_start_samples: NDArray[np.int64]
    window_end_samples: NDArray[np.int64]
    partitions: NDArray[np.str_]
    target_families: NDArray[np.str_]
    context_flags: NDArray[np.str_]

    def __post_init__(self) -> None:
        row_count = self.features.shape[0]
        if self.features.ndim != 2 or self.features.shape[1] != len(COMBINED_V1.names):
            raise ValueError("Feature matrix does not match combined_v1.")
        arrays = (
            self.stable_ids,
            self.record_ids,
            self.subject_ids,
            self.channel_indices,
            self.lead_names,
            self.window_start_samples,
            self.window_end_samples,
            self.partitions,
            self.target_families,
            self.context_flags,
        )
        if any(array.ndim != 1 or array.size != row_count for array in arrays):
            raise ValueError("Feature metadata arrays must align with matrix rows.")
        if len(set(self.stable_ids.tolist())) != row_count:
            raise ValueError("Feature cache contains duplicate stable IDs.")

    @property
    def row_count(self) -> int:
        return self.features.shape[0]

    def select(self, selected: NDArray[np.bool_]) -> FeatureTable:
        if selected.shape != (self.row_count,):
            raise ValueError("Feature selection mask must align with rows.")
        return FeatureTable(
            self.features[selected],
            self.stable_ids[selected],
            self.record_ids[selected],
            self.subject_ids[selected],
            self.channel_indices[selected],
            self.lead_names[selected],
            self.window_start_samples[selected],
            self.window_end_samples[selected],
            self.partitions[selected],
            self.target_families[selected],
            self.context_flags[selected],
        )


def write_feature_table_atomic(
    path: Path, table: FeatureTable, metadata: dict[str, Any]
) -> None:
    """Write one compressed record cache without exposing partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            features=table.features,
            stable_ids=table.stable_ids,
            record_ids=table.record_ids,
            subject_ids=table.subject_ids,
            channel_indices=table.channel_indices,
            lead_names=table.lead_names,
            window_start_samples=table.window_start_samples,
            window_end_samples=table.window_end_samples,
            partitions=table.partitions,
            target_families=table.target_families,
            context_flags=table.context_flags,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(temporary, path)


def read_feature_table(path: Path) -> tuple[FeatureTable, dict[str, Any]]:
    """Read a cache with object deserialization disabled."""
    with np.load(path, allow_pickle=False) as cached:
        table = FeatureTable(
            features=np.asarray(cached["features"], dtype=np.float64),
            stable_ids=np.asarray(cached["stable_ids"], dtype=np.str_),
            record_ids=np.asarray(cached["record_ids"], dtype=np.str_),
            subject_ids=np.asarray(cached["subject_ids"], dtype=np.str_),
            channel_indices=np.asarray(cached["channel_indices"], dtype=np.int64),
            lead_names=np.asarray(cached["lead_names"], dtype=np.str_),
            window_start_samples=np.asarray(
                cached["window_start_samples"], dtype=np.int64
            ),
            window_end_samples=np.asarray(cached["window_end_samples"], dtype=np.int64),
            partitions=np.asarray(cached["partitions"], dtype=np.str_),
            target_families=np.asarray(cached["target_families"], dtype=np.str_),
            context_flags=np.asarray(cached["context_flags"], dtype=np.str_),
        )
        metadata = json.loads(str(cached["metadata_json"]))
    return table, metadata


def load_partition(feature_root: Path, partition: str) -> FeatureTable:
    """Load one partition after validating every cache against its manifest."""
    root = require_external_path(feature_root, "Feature root")
    manifest = read_json(root / FEATURE_MANIFEST_NAME)
    if manifest["split_sha256"] != manifest["expected_split_sha256"]:
        raise ValueError("Feature manifest does not use the frozen split hash.")
    entries = [
        item
        for item in manifest["records"]
        if item["partition"] == partition and item["status"] == "complete"
    ]
    if not entries:
        raise ValueError(f"Feature cache contains no completed {partition} records.")
    tables = []
    for entry in entries:
        table, metadata = read_feature_table(root / entry["cache_path"])
        if metadata["split_sha256"] != manifest["split_sha256"]:
            raise ValueError("Record cache split hash differs from its manifest.")
        if metadata["feature_schema_sha256"] != COMBINED_V1.sha256:
            raise ValueError("Record cache feature schema is not combined_v1.")
        if set(table.partitions.tolist()) != {partition}:
            raise ValueError("Record cache partition metadata is inconsistent.")
        tables.append(table)
    return FeatureTable(
        features=np.vstack([table.features for table in tables]),
        stable_ids=np.concatenate([table.stable_ids for table in tables]),
        record_ids=np.concatenate([table.record_ids for table in tables]),
        subject_ids=np.concatenate([table.subject_ids for table in tables]),
        channel_indices=np.concatenate([table.channel_indices for table in tables]),
        lead_names=np.concatenate([table.lead_names for table in tables]),
        window_start_samples=np.concatenate(
            [table.window_start_samples for table in tables]
        ),
        window_end_samples=np.concatenate(
            [table.window_end_samples for table in tables]
        ),
        partitions=np.concatenate([table.partitions for table in tables]),
        target_families=np.concatenate([table.target_families for table in tables]),
        context_flags=np.concatenate([table.context_flags for table in tables]),
    )
