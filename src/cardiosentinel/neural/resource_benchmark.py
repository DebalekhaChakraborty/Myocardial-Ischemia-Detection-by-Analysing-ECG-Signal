"""Official B4 resource benchmark, frozen by `docs/B4_RESOURCE_BENCHMARK_V1.md`.

Measures locked B4-A, B4-B and B4-C checkpoints only. The procedure is
dataset-independent: it never opens a train, validation or test waveform, cache,
metadata or prediction file, and it constructs no optimizer and calls no
backward.

Peak RSS is a process high-water mark that never decreases, so each model is
measured in a fresh subprocess. Running several models in one process would
attribute the largest model's peak to every later model.
"""

from __future__ import annotations

import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from cardiosentinel.baseline.cache import read_json
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.candidates import B4CSSMCNN, B4BTransformerCNN
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.model import B4CompactCNN, trainable_parameter_count
from cardiosentinel.neural.protocol import WINDOW_SAMPLES

BENCHMARK_PROTOCOL_NAME = "B4_RESOURCE_BENCHMARK_V1"
EXPERIMENT_LOCK_NAME = "EXPERIMENT_LOCK.json"

BENCHMARK_SEED = 2026
BENCHMARK_BATCH_SIZE = 1
WARMUP_CALLS = 50
MEASURED_CALLS = 500
INTRA_OP_THREADS = 1

SUPPORTED_ARCHITECTURES: dict[str, Any] = {
    "B4CompactCNN": B4CompactCNN,
    "B4BTransformerCNN": B4BTransformerCNN,
    "B4CSSMCNN": B4CSSMCNN,
}


class LockedModelError(RuntimeError):
    """Raised when a model is not backed by a validating experiment lock."""


def _architecture_of(lock: dict[str, Any]) -> str:
    model = lock.get("model")
    if isinstance(model, dict) and model.get("architecture"):
        return str(model["architecture"])
    raise LockedModelError("The experiment lock records no model architecture.")


def validate_locked_model(run_dir: Path) -> dict[str, Any]:
    """Validate the lock digest and the bound checkpoint before any load.

    Works uniformly for B4-A and both candidates: the lock's own canonical digest
    is re-derived, `test` must be null, and the checkpoint bytes must hash to the
    value the lock recorded.
    """
    lock_path = Path(run_dir) / EXPERIMENT_LOCK_NAME
    if not lock_path.is_file():
        raise LockedModelError(f"No {EXPERIMENT_LOCK_NAME} in {run_dir}.")
    lock = read_json(lock_path)
    recorded = lock.pop("experiment_lock_sha256", None)
    if recorded is None or recorded != canonical_sha256(lock):
        raise LockedModelError("Experiment lock hash validation failed.")
    lock["experiment_lock_sha256"] = recorded
    if "test" not in lock or lock["test"] is not None:
        raise LockedModelError("Experiment lock must record test as null.")
    architecture = _architecture_of(lock)
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise LockedModelError(f"Unsupported benchmark architecture {architecture!r}.")
    if not isinstance(lock.get("trainable_parameter_count"), int):
        raise LockedModelError(
            "The experiment lock must bind a trainable parameter count."
        )
    checkpoint = Path(run_dir) / str(lock["locked_inference_model"])
    if not checkpoint.is_file():
        raise LockedModelError("The locked inference checkpoint is absent.")
    if sha256_file(checkpoint) != lock["checkpoint_sha256"]:
        raise LockedModelError("The locked checkpoint SHA-256 does not match.")
    return lock


def load_locked_model(run_dir: Path, lock: dict[str, Any]) -> torch.nn.Module:
    """Load only the locked weights; no optimizer state is read or created."""
    checkpoint = Path(run_dir) / str(lock["locked_inference_model"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "optimizer" in state:
        raise LockedModelError(
            "The locked inference artifact must not carry optimizer state."
        )
    model = SUPPORTED_ARCHITECTURES[_architecture_of(lock)]()
    model.load_state_dict(state)
    expected = lock.get("trainable_parameter_count")
    if expected is not None and trainable_parameter_count(model) != expected:
        raise LockedModelError("Locked checkpoint parameter count does not match.")
    model.eval()
    model.requires_grad_(False)
    return model


def benchmark_input() -> torch.Tensor:
    """Return the frozen deterministic synthetic window; no dataset is touched."""
    generator = torch.Generator().manual_seed(BENCHMARK_SEED)
    return torch.randn(
        BENCHMARK_BATCH_SIZE, 1, WINDOW_SAMPLES, generator=generator,
        dtype=torch.float32,
    )


def _environment() -> dict[str, Any]:
    import numpy

    from cardiosentinel.neural.provenance import dependency_environment

    cpu = platform.processor() or None
    if not cpu and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "platform": platform.platform(),
        "cpu_model": cpu,
        "device": "cpu",
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "dependencies": dependency_environment(),
    }


def measure_locked_model(run_dir: Path) -> dict[str, Any]:
    """Run the frozen procedure in **this** process; intended for a subprocess."""
    torch.set_num_threads(INTRA_OP_THREADS)
    lock = validate_locked_model(run_dir)
    model = load_locked_model(run_dir, lock)
    # Counting after `requires_grad_(False)` would report zero, so the size
    # figures come from the digest-bound lock, cross-checked on load.
    parameter_count = int(lock["trainable_parameter_count"])
    waveform = benchmark_input()

    with torch.no_grad():
        for _ in range(WARMUP_CALLS):
            model(waveform)
        samples_ns: list[int] = []
        for _ in range(MEASURED_CALLS):
            started = time.perf_counter_ns()
            model(waveform)
            samples_ns.append(time.perf_counter_ns() - started)

    ordered = sorted(samples_ns)
    checkpoint = Path(run_dir) / str(lock["locked_inference_model"])
    payload = {
        "benchmark_protocol": BENCHMARK_PROTOCOL_NAME,
        "experiment_id": lock.get("experiment_id"),
        "architecture": _architecture_of(lock),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "trainable_parameter_count": parameter_count,
        "fp32_parameter_payload_bytes": parameter_count * 4,
        "locked_checkpoint_bytes": checkpoint.stat().st_size,
        "batch_size": BENCHMARK_BATCH_SIZE,
        "warmup_calls": WARMUP_CALLS,
        "measured_calls": MEASURED_CALLS,
        "timer": "time.perf_counter_ns",
        "input_seed": BENCHMARK_SEED,
        "input_shape": [BENCHMARK_BATCH_SIZE, 1, WINDOW_SAMPLES],
        "dataset_accessed": False,
        "median_latency_ms_per_window": statistics.median(ordered) / 1e6,
        "p95_latency_ms_per_window": ordered[
            min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1)
        ]
        / 1e6,
        "tie_break_statistic": "median_latency_ms_per_window",
        "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_units": "kibibytes" if sys.platform.startswith("linux") else "bytes",
        "peak_rss_measurement_method": (
            "resource.getrusage(RUSAGE_SELF).ru_maxrss in a fresh subprocess"
        ),
        "peak_rss_available": True,
        "process_isolated": True,
        "environment": _environment(),
    }
    payload["benchmark_result_sha256"] = canonical_sha256(payload)
    return payload


def benchmark_locked_model_isolated(
    run_dir: Path, *, timeout_seconds: float = 900.0
) -> dict[str, Any]:
    """Measure one locked model in a **fresh subprocess** and return its result.

    Isolation is required because `ru_maxrss` is a non-decreasing process
    high-water mark; a shared process would attribute the largest model's peak to
    every later model.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "cardiosentinel.neural.resource_benchmark",
         str(Path(run_dir).resolve())],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise LockedModelError(
            f"Isolated benchmark failed for {run_dir}: "
            f"{completed.stderr.strip().splitlines()[-1] if completed.stderr else ''}"
        )
    return json.loads(completed.stdout)


def _main(argv: list[str]) -> int:  # pragma: no cover - subprocess entry point
    if len(argv) != 2:
        print("usage: python -m cardiosentinel.neural.resource_benchmark RUN_DIR",
              file=sys.stderr)
        return 2
    print(json.dumps(measure_locked_model(Path(argv[1])), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(_main(sys.argv))
