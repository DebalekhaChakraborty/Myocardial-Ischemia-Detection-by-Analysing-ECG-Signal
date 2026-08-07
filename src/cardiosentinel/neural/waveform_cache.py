"""Lossless, mmap-compatible external waveform cache for B4 development data."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from cardiosentinel.baseline.cache import (
    read_json,
    require_nonversioned_path,
    write_json_atomic,
)
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.neural.data import B4PredictiveSample, B4WaveformDataset
from cardiosentinel.neural.integrity import (
    DEVELOPMENT_PARTITIONS,
    canonical_sha256,
    validate_development_feature_integrity,
    validate_development_source_integrity,
)
from cardiosentinel.neural.metadata import (
    B4MetadataIndex,
    B4WindowReference,
    build_training_index,
    build_validation_index,
)
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    B4_SPLIT_SHA256,
    DATASET,
    DATASET_VERSION,
    EXPECTED_COUNTS,
    FEATURE_CORPUS_SHA256,
    REPOSITORY_ROOT,
    SAMPLING_FREQUENCY_HZ,
    STRIDE_SECONDS,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    require_development_partition,
    validate_frozen_protocol,
)

CACHE_MANIFEST_NAME = "manifest.json"
CACHE_SCHEMA_VERSION = "b4_waveform_v1"
TRAINING_SELECTION_SHA256 = (
    "318da148da5d638af44e73c06c00cc4df2815017d4ce8bb1a1b864e53eda8009"
)
DEFAULT_CACHE_ROOT = REPOSITORY_ROOT / "cardiosentinel-features" / "b4-waveform-v1"
PROGRESS_NAME = ".materialization-progress.json"
PROGRESS_INTERVAL_ROWS = 1024
FUTURE_ARTIFACT_RESERVE_BYTES = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class ValidatedB4WaveformCache:
    root: Path
    manifest: dict[str, Any]
    waveforms: Mapping[str, NDArray[np.float32]]
    stable_ids: Mapping[str, NDArray[np.str_]]


def waveform_cache_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return immutable cache identity fields covered by the corpus digest."""
    partition_payload = {}
    for partition in sorted(DEVELOPMENT_PARTITIONS):
        item = manifest["partitions"][partition]
        partition_payload[partition] = {
            "row_count": item["row_count"],
            "waveform_file": item["waveform_file"],
            "waveform_sha256": item["waveform_sha256"],
            "waveform_bytes": item["waveform_bytes"],
            "stable_id_file": item["stable_id_file"],
            "stable_id_sha256": item["stable_id_sha256"],
            "stable_id_bytes": item["stable_id_bytes"],
        }
    return {
        "cache_schema_version": manifest["cache_schema_version"],
        "dataset": manifest["dataset"],
        "dataset_version": manifest["dataset_version"],
        "protocol_sha256": manifest["protocol_sha256"],
        "split_sha256": manifest["split_sha256"],
        "feature_corpus_sha256": manifest["feature_corpus_sha256"],
        "training_selection_sha256": manifest["training_selection_sha256"],
        "development_feature_integrity_sha256": manifest[
            "development_feature_integrity_sha256"
        ],
        "development_source_integrity_sha256": manifest[
            "development_source_integrity_sha256"
        ],
        "processing_profile": manifest["processing_profile"],
        "sampling_frequency_hz": manifest["sampling_frequency_hz"],
        "window_seconds": manifest["window_seconds"],
        "stride_seconds": manifest["stride_seconds"],
        "samples_per_row": manifest["samples_per_row"],
        "dtype": manifest["dtype"],
        "physical_unit": manifest["physical_unit"],
        "partitions": partition_payload,
        "equivalence_audit": manifest["equivalence_audit"],
    }


def compute_waveform_cache_sha256(manifest: dict[str, Any]) -> str:
    return canonical_sha256(waveform_cache_payload(manifest))


def _cache_environment() -> dict[str, Any]:
    import wfdb

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "wfdb_version": wfdb.__version__,
    }


def _partition_filenames(partition: str) -> dict[str, str]:
    permitted = require_development_partition(partition)
    return {
        "waveform_file": f"{permitted}_waveforms.npy",
        "stable_id_file": f"{permitted}_stable_ids.npy",
        "waveform_partial": f".{permitted}_waveforms.npy.partial",
        "stable_id_partial": f".{permitted}_stable_ids.npy.partial",
    }


def _safe_file(root: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("B4 waveform cache filename is unsafe.")
    return root / filename


def _expected_identity(
    feature_receipt: dict[str, Any], source_receipt: dict[str, Any]
) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "protocol_sha256": B4_PROTOCOL_SHA256,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "training_selection_sha256": TRAINING_SELECTION_SHA256,
        "development_feature_integrity_sha256": feature_receipt[
            "development_feature_integrity_sha256"
        ],
        "development_source_integrity_sha256": source_receipt[
            "development_source_integrity_sha256"
        ],
        "processing_profile": "raw",
        "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "samples_per_row": WINDOW_SAMPLES,
        "dtype": "float32",
        "physical_unit": "mV",
    }


def build_development_indexes(feature_root: Path) -> dict[str, B4MetadataIndex]:
    train = build_training_index(feature_root)
    validation = build_validation_index(feature_root)
    if train.selection_sha256 != TRAINING_SELECTION_SHA256:
        raise ValueError("Frozen B4 training selection SHA-256 differs.")
    return {"train": train, "validation": validation}


def _required_free_bytes(indexes: Mapping[str, B4MetadataIndex]) -> int:
    waveform_bytes = sum(
        index.total_count * WINDOW_SAMPLES * np.dtype(np.float32).itemsize
        for index in indexes.values()
    )
    max_chars = max(
        len(reference.stable_id)
        for index in indexes.values()
        for reference in index.references
    )
    stable_id_bytes = sum(
        index.total_count * max_chars * np.dtype("U1").itemsize
        for index in indexes.values()
    )
    return (
        int((waveform_bytes + stable_id_bytes) * 1.25)
        + FUTURE_ARTIFACT_RESERVE_BYTES
    )


def cache_disk_preflight(
    cache_root: Path, indexes: Mapping[str, B4MetadataIndex]
) -> dict[str, int]:
    root = require_nonversioned_path(cache_root, "B4 waveform cache root")
    probe = root if root.exists() else root.parent
    free = shutil.disk_usage(probe).free
    required = _required_free_bytes(indexes)
    if free < required:
        raise ValueError(
            f"B4 cache requires {required} free bytes, but only {free} are available."
        )
    return {"available_bytes": free, "required_bytes": required}


def _progress_identity(
    identity: dict[str, Any], indexes: Mapping[str, B4MetadataIndex]
) -> dict[str, Any]:
    return {
        **identity,
        "partitions": {
            partition: {
                "row_count": indexes[partition].total_count,
                "stable_ids_sha256": canonical_sha256(
                    [item.stable_id for item in indexes[partition].references]
                ),
            }
            for partition in sorted(DEVELOPMENT_PARTITIONS)
        },
    }


def _new_progress(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity": identity,
        "elapsed_seconds": 0.0,
        "partitions": {
            partition: {"status": "pending", "next_row": 0}
            for partition in sorted(DEVELOPMENT_PARTITIONS)
        },
    }


def _load_or_create_progress(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    progress_path = root / PROGRESS_NAME
    if progress_path.is_file():
        progress = read_json(progress_path)
        if progress.get("identity") != identity:
            raise ValueError("Partial B4 waveform cache has different provenance.")
        return progress
    unexpected = []
    for partition in DEVELOPMENT_PARTITIONS:
        for filename in _partition_filenames(partition).values():
            if (root / filename).exists():
                unexpected.append(filename)
    if unexpected:
        raise ValueError(
            f"Unbound B4 waveform cache files exist without progress: {unexpected}"
        )
    progress = _new_progress(identity)
    write_json_atomic(progress_path, progress)
    return progress


def _open_partial_arrays(
    root: Path,
    partition: str,
    references: tuple[B4WindowReference, ...],
    next_row: int,
) -> tuple[NDArray[np.float32], NDArray[np.str_]]:
    names = _partition_filenames(partition)
    waveform_path = root / names["waveform_partial"]
    stable_id_path = root / names["stable_id_partial"]
    stable_chars = max(len(item.stable_id) for item in references)
    mode = "w+" if next_row == 0 else "r+"
    waveforms = np.lib.format.open_memmap(
        waveform_path,
        mode=mode,
        dtype=np.float32,
        shape=(len(references), WINDOW_SAMPLES),
    )
    stable_ids = np.lib.format.open_memmap(
        stable_id_path,
        mode=mode,
        dtype=f"U{stable_chars}",
        shape=(len(references),),
    )
    if waveforms.shape != (len(references), WINDOW_SAMPLES):
        raise ValueError("Partial B4 waveform array has the wrong shape.")
    if waveforms.dtype != np.float32 or stable_ids.shape != (len(references),):
        raise ValueError("Partial B4 cache arrays have the wrong dtype or shape.")
    return waveforms, stable_ids


def _checkpoint_progress(
    root: Path,
    progress: dict[str, Any],
    partition: str,
    next_row: int,
    elapsed_base: float,
    invocation_started: float,
) -> None:
    progress["partitions"][partition] = {
        "status": "writing",
        "next_row": next_row,
    }
    progress["elapsed_seconds"] = elapsed_base + (
        time.monotonic() - invocation_started
    )
    write_json_atomic(root / PROGRESS_NAME, progress)


def _materialize_partition(
    root: Path,
    partition: str,
    index: B4MetadataIndex,
    source: Path,
    progress: dict[str, Any],
    elapsed_base: float,
    invocation_started: float,
) -> None:
    state = progress["partitions"][partition]
    if state["status"] == "ready":
        return
    references = index.references
    next_row = int(state.get("next_row", 0))
    waveforms, stable_ids = _open_partial_arrays(
        root, partition, references, next_row
    )
    dataset = B4WaveformDataset(references, source)
    for row in range(next_row, len(references)):
        sample = dataset[row]
        waveforms[row] = sample.waveform.numpy()[0]
        stable_ids[row] = references[row].stable_id
        completed = row + 1
        if completed % PROGRESS_INTERVAL_ROWS == 0 or completed == len(references):
            waveforms.flush()
            stable_ids.flush()
            _checkpoint_progress(
                root,
                progress,
                partition,
                completed,
                elapsed_base,
                invocation_started,
            )
    del waveforms
    del stable_ids
    names = _partition_filenames(partition)
    waveform_path = root / names["waveform_partial"]
    stable_id_path = root / names["stable_id_partial"]
    progress["partitions"][partition] = {
        "status": "ready",
        "next_row": len(references),
        "waveform_sha256": sha256_file(waveform_path),
        "waveform_bytes": waveform_path.stat().st_size,
        "stable_id_sha256": sha256_file(stable_id_path),
        "stable_id_bytes": stable_id_path.stat().st_size,
    }
    progress["elapsed_seconds"] = elapsed_base + (
        time.monotonic() - invocation_started
    )
    write_json_atomic(root / PROGRESS_NAME, progress)


def _finalize_partition_files(
    root: Path, partition: str, state: dict[str, Any]
) -> dict[str, Any]:
    if state.get("status") != "ready":
        raise ValueError("B4 cache partition is not ready for finalization.")
    names = _partition_filenames(partition)
    for kind in ("waveform", "stable_id"):
        partial = root / names[f"{kind}_partial"]
        final = root / names[f"{kind}_file"]
        expected = state[f"{kind}_sha256"]
        if partial.is_file():
            if sha256_file(partial) != expected:
                raise ValueError("Partial B4 cache digest changed before finalization.")
            os.replace(partial, final)
        elif not final.is_file() or sha256_file(final) != expected:
            raise ValueError("Final B4 cache file is absent or corrupt.")
    return {
        "row_count": state["next_row"],
        "waveform_file": names["waveform_file"],
        "waveform_sha256": state["waveform_sha256"],
        "waveform_bytes": state["waveform_bytes"],
        "stable_id_file": names["stable_id_file"],
        "stable_id_sha256": state["stable_id_sha256"],
        "stable_id_bytes": state["stable_id_bytes"],
    }


def _audit_indices(index: B4MetadataIndex) -> tuple[int, ...]:
    groups: dict[tuple[str, int], list[int]] = {}
    for row, reference in enumerate(index.references):
        key = (reference.record_id, reference.channel_index)
        groups.setdefault(key, []).append(row)
    ordered = sorted(groups)
    selected_groups = np.linspace(0, len(ordered) - 1, min(4, len(ordered)), dtype=int)
    selected: set[int] = set()
    for group_index in selected_groups:
        rows = groups[ordered[int(group_index)]]
        selected.update((rows[0], rows[len(rows) // 2], rows[-1]))
    return tuple(sorted(selected))


def audit_waveform_cache_equivalence(
    source: Path,
    cache_root: Path,
    indexes: Mapping[str, B4MetadataIndex],
    *,
    _reader=None,
    _source_verifier=None,
) -> dict[str, Any]:
    """Require exact source-cast equality on label-independent development rows."""
    root = require_nonversioned_path(cache_root, "B4 waveform cache root")
    audited = 0
    mismatches = 0
    records: set[str] = set()
    channels: set[int] = set()
    record_channels: set[tuple[str, int]] = set()
    by_partition: dict[str, int] = {}
    for partition in sorted(DEVELOPMENT_PARTITIONS):
        index = indexes[partition]
        rows = _audit_indices(index)
        references = tuple(index.references[row] for row in rows)
        dataset_kwargs = {}
        if _reader is not None:
            dataset_kwargs["_reader"] = _reader
        if _source_verifier is not None:
            dataset_kwargs["_source_verifier"] = _source_verifier
        direct = B4WaveformDataset(references, source, **dataset_kwargs)
        names = _partition_filenames(partition)
        cached = np.load(
            root / names["waveform_file"], mmap_mode="r", allow_pickle=False
        )
        for direct_row, cache_row in enumerate(rows):
            reference = references[direct_row]
            expected = direct[direct_row].waveform.numpy()[0]
            if not np.array_equal(np.asarray(cached[cache_row]), expected):
                mismatches += 1
            audited += 1
            records.add(reference.record_id)
            channels.add(reference.channel_index)
            record_channels.add((reference.record_id, reference.channel_index))
        by_partition[partition] = len(rows)
    if mismatches:
        raise ValueError(f"B4 waveform cache equivalence mismatches: {mismatches}")
    return {
        "selection_rule": (
            "deterministic early/middle/late rows across evenly spaced "
            "record-channel groups"
        ),
        "audited_rows": audited,
        "audited_rows_by_partition": by_partition,
        "records_represented": len(records),
        "channel_indices_represented": sorted(channels),
        "record_channel_groups_represented": len(record_channels),
        "exact_mismatches": mismatches,
        "comparison": "np.array_equal",
        "result": "passed",
    }


def _manifest_candidate(
    identity: dict[str, Any],
    partitions: dict[str, Any],
    audit: dict[str, Any],
    provenance: dict[str, object],
    command: str,
    environment: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    manifest = {
        **identity,
        "cache_complete": True,
        "partitions": partitions,
        "equivalence_audit": audit,
        "creation_git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "creation_command": command,
        "environment": environment,
        "materialization_duration_seconds": duration,
    }
    manifest["waveform_cache_sha256"] = compute_waveform_cache_sha256(manifest)
    return manifest


def materialize_development_waveform_cache(
    source: Path,
    feature_root: Path,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    *,
    command: str = "cardiosentinel b4 cache-materialize",
    require_clean: bool = True,
) -> dict[str, Any]:
    """Build or resume exact train/validation arrays; never enumerate test rows."""
    invocation_started = time.monotonic()
    validate_frozen_protocol()
    root = require_nonversioned_path(cache_root, "B4 waveform cache root")
    feature_receipt = validate_development_feature_integrity(feature_root)
    source_receipt = validate_development_source_integrity(source, feature_receipt)
    indexes = build_development_indexes(feature_root)
    identity = _expected_identity(feature_receipt, source_receipt)
    provenance = git_provenance(REPOSITORY_ROOT)
    if require_clean and provenance["git_dirty"]:
        raise ValueError("B4 cache materialization requires a clean Git checkout.")

    manifest_path = root / CACHE_MANIFEST_NAME
    if manifest_path.is_file():
        validated = validate_waveform_cache(root, indexes)
        return {**validated.manifest, "reused_existing_cache": True}

    cache_disk_preflight(root, indexes)
    root.mkdir(parents=True, exist_ok=True)
    progress_identity = _progress_identity(identity, indexes)
    progress = _load_or_create_progress(root, progress_identity)
    elapsed_base = float(progress.get("elapsed_seconds", 0.0))
    for partition in ("train", "validation"):
        _materialize_partition(
            root,
            partition,
            indexes[partition],
            source,
            progress,
            elapsed_base,
            invocation_started,
        )
    partitions = {
        partition: _finalize_partition_files(
            root, partition, progress["partitions"][partition]
        )
        for partition in sorted(DEVELOPMENT_PARTITIONS)
    }
    audit = audit_waveform_cache_equivalence(source, root, indexes)
    duration = elapsed_base + (time.monotonic() - invocation_started)
    manifest = _manifest_candidate(
        identity,
        partitions,
        audit,
        provenance,
        command,
        _cache_environment(),
        duration,
    )
    write_json_atomic(manifest_path, manifest)
    (root / PROGRESS_NAME).unlink()
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "protocol_sha256": B4_PROTOCOL_SHA256,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "training_selection_sha256": TRAINING_SELECTION_SHA256,
        "processing_profile": "raw",
        "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "samples_per_row": WINDOW_SAMPLES,
        "dtype": "float32",
        "physical_unit": "mV",
        "cache_complete": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("B4 waveform cache manifest identity is not frozen.")
    if set(manifest.get("partitions", {})) != DEVELOPMENT_PARTITIONS:
        raise ValueError("B4 waveform cache must contain train and validation only.")
    audit = manifest.get("equivalence_audit", {})
    if audit.get("result") != "passed" or audit.get("exact_mismatches") != 0:
        raise ValueError("B4 waveform cache lacks a passing equivalence audit.")
    if manifest.get("waveform_cache_sha256") != compute_waveform_cache_sha256(
        manifest
    ):
        raise ValueError("B4 waveform cache corpus SHA-256 is invalid.")


def _stable_ids_equal(
    cached: NDArray[np.str_], references: tuple[B4WindowReference, ...]
) -> bool:
    if cached.shape != (len(references),):
        return False
    for start in range(0, len(references), 8192):
        end = min(len(references), start + 8192)
        expected = np.asarray(
            [item.stable_id for item in references[start:end]], dtype=np.str_
        )
        if not np.array_equal(np.asarray(cached[start:end]), expected):
            return False
    return True


def validate_waveform_cache(
    cache_root: Path,
    indexes: Mapping[str, B4MetadataIndex],
) -> ValidatedB4WaveformCache:
    """Hash and align every development cache file before predictive use."""
    root = require_nonversioned_path(cache_root, "B4 waveform cache root")
    if set(indexes) != DEVELOPMENT_PARTITIONS:
        raise ValueError("B4 cache validation requires train and validation indexes.")
    manifest_path = root / CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("B4 waveform cache is incomplete: manifest.json is absent.")
    manifest = read_json(manifest_path)
    _validate_manifest(manifest)
    waveforms: dict[str, NDArray[np.float32]] = {}
    stable_ids: dict[str, NDArray[np.str_]] = {}
    for partition in sorted(DEVELOPMENT_PARTITIONS):
        item = manifest["partitions"][partition]
        index = indexes[partition]
        expected_count = EXPECTED_COUNTS[partition]["total"]
        if (
            item.get("row_count") != expected_count
            or index.total_count != expected_count
        ):
            raise ValueError(f"B4 cache row count differs for {partition}.")
        waveform_path = _safe_file(root, str(item["waveform_file"]))
        stable_id_path = _safe_file(root, str(item["stable_id_file"]))
        for path, hash_key, bytes_key in (
            (waveform_path, "waveform_sha256", "waveform_bytes"),
            (stable_id_path, "stable_id_sha256", "stable_id_bytes"),
        ):
            if not path.is_file() or path.stat().st_size != item.get(bytes_key):
                raise ValueError(f"B4 cache file is absent or truncated: {path.name}")
            if sha256_file(path) != item.get(hash_key):
                raise ValueError(f"B4 cache file SHA-256 mismatch: {path.name}")
        waveform_array = np.load(waveform_path, mmap_mode="r", allow_pickle=False)
        stable_id_array = np.load(stable_id_path, mmap_mode="r", allow_pickle=False)
        if waveform_array.shape != (expected_count, WINDOW_SAMPLES):
            raise ValueError(f"B4 cached waveform shape differs for {partition}.")
        if waveform_array.dtype != np.float32:
            raise ValueError("B4 cached waveform dtype must be float32.")
        if stable_id_array.dtype.kind != "U" or not _stable_ids_equal(
            stable_id_array, index.references
        ):
            raise ValueError(f"B4 stable-ID alignment differs for {partition}.")
        waveforms[partition] = waveform_array
        stable_ids[partition] = stable_id_array
    return ValidatedB4WaveformCache(root, manifest, waveforms, stable_ids)


class B4CachedWaveformDataset(Dataset[B4PredictiveSample]):
    """Predictive waveform+label Dataset backed by a validated read-only mmap."""

    def __init__(
        self,
        cache: ValidatedB4WaveformCache,
        index: B4MetadataIndex,
    ) -> None:
        partition = require_development_partition(index.partition)
        if partition not in cache.waveforms:
            raise ValueError("Validated B4 cache does not contain this partition.")
        self._references = index.references
        self._waveforms = cache.waveforms[partition]
        if self._waveforms.shape[0] != len(self._references):
            raise ValueError("B4 cached Dataset rows are not aligned.")

    def __len__(self) -> int:
        return len(self._references)

    def __getitem__(self, index: int) -> B4PredictiveSample:
        waveform = torch.from_numpy(
            np.array(self._waveforms[index], dtype=np.float32, copy=True)
        ).reshape(1, WINDOW_SAMPLES)
        return B4PredictiveSample(
            waveform=waveform,
            label=torch.tensor(
                self._references[index].binary_label, dtype=torch.float32
            ),
        )
