"""Prospective B4 provenance checks that never open sealed-test waveform files."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cardiosentinel.baseline.cache import read_json, require_nonversioned_path
from cardiosentinel.baseline.source import (
    DATASET,
    DATASET_VERSION,
    OFFICIAL_MANIFEST_NAME,
    OFFICIAL_MANIFEST_SHA256,
    PINNED_SOURCE_URL,
    SOURCE_VERIFICATION_RECEIPT_NAME,
)
from cardiosentinel.neural.protocol import validate_frozen_protocol

EXPECTED_RECORD_COUNT = 86
EXPECTED_REQUIRED_FILE_COUNT = 1 + EXPECTED_RECORD_COUNT * 3
# Scientific dependencies named for quick human verification. The full resolved
# snapshot below remains authoritative; this list never narrows what is bound.
KEY_DEPENDENCIES = ("numpy", "scikit-learn", "scipy", "torch", "wfdb")
_NAME_SEPARATORS = re.compile(r"[-_.]+")


def normalize_package_name(name: str) -> str:
    """Normalize a distribution name to its PEP 503 comparison form."""
    return _NAME_SEPARATORS.sub("-", name).strip().lower()


def installed_package_snapshot() -> tuple[dict[str, str], ...]:
    """Resolve installed distributions deterministically without any network.

    Duplicate installations of one distribution are collapsed to their highest
    sorted version so the snapshot never depends on filesystem search order.
    """
    from importlib.metadata import distributions

    versions: dict[str, set[str]] = {}
    for distribution in distributions():
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            continue
        name = normalize_package_name(str(raw_name))
        versions.setdefault(name, set()).add(str(distribution.version or ""))
    return tuple(
        {"name": name, "version": sorted(versions[name])[-1]}
        for name in sorted(versions)
    )


def dependency_environment() -> dict[str, Any]:
    """Return the resolved dependency environment bound by the experiment lock."""
    packages = installed_package_snapshot()
    resolved = {item["name"]: item["version"] for item in packages}
    canonical = json.dumps(list(packages), sort_keys=True, separators=(",", ":"))
    return {
        "installed_packages": list(packages),
        "installed_package_count": len(packages),
        "installed_packages_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "key_dependencies": {
            name: resolved.get(name) for name in KEY_DEPENDENCIES
        },
        "name_normalization": "pep503",
        "resolution_source": "importlib.metadata",
    }


def validate_source_verification_receipt(source: Path) -> dict[str, Any]:
    """Validate the prior full-source receipt without reopening waveform bytes."""
    root = require_nonversioned_path(source, "B4 waveform source")
    receipt_path = root / SOURCE_VERIFICATION_RECEIPT_NAME
    if not receipt_path.is_file():
        raise ValueError("B4 requires the pinned source-verification receipt.")
    receipt = read_json(receipt_path)
    expected = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "verified_required_file_count": EXPECTED_REQUIRED_FILE_COUNT,
        "verification_result": "passed",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("B4 source-verification receipt does not match LTSTDB V1.")
    official = receipt.get("official_manifest", {})
    if official.get("identifier") != f"{PINNED_SOURCE_URL}{OFFICIAL_MANIFEST_NAME}":
        raise ValueError("B4 source receipt has the wrong official manifest.")
    if official.get("sha256") != OFFICIAL_MANIFEST_SHA256:
        raise ValueError("B4 source receipt has the wrong official manifest digest.")
    recorded_root = receipt.get("local_source_root")
    if not isinstance(recorded_root, str) or not Path(recorded_root).is_absolute():
        raise ValueError("B4 source receipt has no absolute verification root.")
    # Dataset relocation does not rewrite frozen verification evidence. Preserve
    # both locations and do not rehash sealed-test waveform bytes in development.
    receipt["resolved_local_source_root"] = str(root)
    return receipt


def runtime_environment(device: str, worker_count: int) -> dict[str, Any]:
    """Record runtime facts without importing Torch at base-package import."""
    import torch

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": platform.processor() or _linux_cpu_model(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device,
        "gpu_model": gpu_name,
        "worker_count": worker_count,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "amp_enabled": False,
        "protocol_sha256": validate_frozen_protocol(),
        "dependencies": dependency_environment(),
    }


def _linux_cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.partition(":")[2].strip()
    return None
