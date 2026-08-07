"""Prospective B4 provenance checks that never open sealed-test waveform files."""

from __future__ import annotations

import platform
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
    }


def _linux_cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.partition(":")[2].strip()
    return None
