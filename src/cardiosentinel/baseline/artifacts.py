"""Canonical experiment locks and external artifact integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cardiosentinel.baseline.cache import read_json, write_json_atomic
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.evaluation.protocol import LTSTDB_V1_SPLIT_SHA256
from cardiosentinel.features.schema import COMBINED_V1, SIGNAL_V1

LOCK_NAME = "experiment_lock.json"


def canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_experiment_lock(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Freeze a lock whose digest covers every field except the digest itself."""
    if payload.get("split_sha256") != LTSTDB_V1_SPLIT_SHA256:
        raise ValueError("Experiment lock must use the frozen LTSTDB V1 split.")
    valid_schemas = {SIGNAL_V1.sha256, COMBINED_V1.sha256}
    if payload.get("feature_schema_sha256") not in valid_schemas:
        raise ValueError("Experiment lock has an unknown feature schema.")
    locked = dict(payload)
    locked["experiment_lock_sha256"] = canonical_sha256(locked)
    write_json_atomic(run_dir / LOCK_NAME, locked)
    return locked


def validate_experiment_lock(run_dir: Path) -> dict[str, Any]:
    """Require and validate lock identity plus model and transform artifacts."""
    lock_path = run_dir / LOCK_NAME
    if not lock_path.is_file():
        raise ValueError("Test evaluation requires experiment_lock.json.")
    lock = read_json(lock_path)
    recorded_hash = lock.pop("experiment_lock_sha256", None)
    if recorded_hash is None or recorded_hash != canonical_sha256(lock):
        raise ValueError("Experiment lock hash validation failed.")
    lock["experiment_lock_sha256"] = recorded_hash
    if lock["split_sha256"] != LTSTDB_V1_SPLIT_SHA256:
        raise ValueError("Experiment lock split hash is not frozen LTSTDB V1.")
    for path_key, hash_key in (
        ("model_artifact", "trained_model_artifact_sha256"),
        ("transform_artifact", "transform_artifact_sha256"),
    ):
        path = run_dir / lock[path_key]
        if not path.is_file() or sha256_file(path) != lock[hash_key]:
            raise ValueError(f"Experiment lock artifact validation failed: {path_key}.")
    return lock


def write_predictions_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as destination:
        import numpy as np

        np.savez_compressed(destination, **arrays)
    os.replace(temporary, path)
