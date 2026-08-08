"""Official B4 resource benchmark, frozen by `docs/B4_RESOURCE_BENCHMARK_V1.md`.

Measures locked B4-A, B4-B and B4-C checkpoints only. The procedure is
dataset-independent: it never opens a train, validation or test waveform, cache,
metadata or prediction file, and it constructs no optimizer and calls no
backward.

Official evidence is produced only by `run_official_resource_suite`, which
benchmarks all three models in one exclusive invocation. Selective reruns and
per-candidate repeats are structurally impossible: there is no force, best-of,
retry-one, rerun-candidate or overwrite path.

Peak RSS is a process high-water mark that never decreases, so each model is
measured in a fresh subprocess. Running several models in one process would
attribute the largest model's peak to every later model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from cardiosentinel.baseline.cache import (
    read_json,
    require_nonversioned_path,
    write_json_atomic,
)
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.candidate_experiment import (
    ARCHITECTURE_PROTOCOL_SHA256,
    B4A_DEPENDENCY_DIGEST,
    require_exact_scientific_environment,
)
from cardiosentinel.neural.candidates import (
    B4B_EXPERIMENT_ID,
    B4B_TRAINABLE_PARAMETERS,
    B4C_EXPERIMENT_ID,
    B4C_TRAINABLE_PARAMETERS,
    B4CSSMCNN,
    B4BTransformerCNN,
)
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.model import B4CompactCNN, trainable_parameter_count
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    REPOSITORY_ROOT,
    TRAINABLE_PARAMETER_COUNT,
    WINDOW_SAMPLES,
)

BENCHMARK_PROTOCOL_NAME = "B4_RESOURCE_BENCHMARK_V1"
RESOURCE_PROTOCOL_PATH = REPOSITORY_ROOT / "docs" / "B4_RESOURCE_BENCHMARK_V1.md"
RESOURCE_PROTOCOL_SHA256 = (
    "9184f54eb2b80fd495460d0a5c8989cdc6b923ed992a87ea18253e836f4c4b98"
)
EXPERIMENT_LOCK_NAME = "EXPERIMENT_LOCK.json"

B4A_EXPERIMENT_ID = "B4_raw_compact_cnn_v1"
BENCHMARK_SEED = 2026
BENCHMARK_BATCH_SIZE = 1
WARMUP_CALLS = 50
MEASURED_CALLS = 500
INTRA_OP_THREADS = 1
P95_FRACTION = 0.95

SUITE_DIR_NAME = "B4_architecture_resource_benchmark_v1"
SUITE_ATTEMPT_NAME = "RESOURCE_BENCHMARK_ATTEMPT.json"
SUITE_RESULTS_NAME = "RESOURCE_BENCHMARK_RESULTS.json"
SUITE_STATUS_STARTED = "STARTED"
SUITE_STATUS_COMPLETE = "COMPLETE"
SUITE_STATUS_FAILED = "FAILED_OR_INTERRUPTED"

# The three official models, in the frozen suite order.
OFFICIAL_ORDER = ("B4-A", "B4-B", "B4-C")
OFFICIAL_MODELS: dict[str, dict[str, Any]] = {
    "B4-A": {
        "experiment_id": B4A_EXPERIMENT_ID,
        "architecture": "B4CompactCNN",
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "requires_architecture_protocol": False,
    },
    "B4-B": {
        "experiment_id": B4B_EXPERIMENT_ID,
        "architecture": "B4BTransformerCNN",
        "trainable_parameter_count": B4B_TRAINABLE_PARAMETERS,
        "requires_architecture_protocol": True,
    },
    "B4-C": {
        "experiment_id": B4C_EXPERIMENT_ID,
        "architecture": "B4CSSMCNN",
        "trainable_parameter_count": B4C_TRAINABLE_PARAMETERS,
        "requires_architecture_protocol": True,
    },
}
SUPPORTED_ARCHITECTURES: dict[str, Any] = {
    "B4CompactCNN": B4CompactCNN,
    "B4BTransformerCNN": B4BTransformerCNN,
    "B4CSSMCNN": B4CSSMCNN,
}
# Host identity that must be identical across all three children.
COMPARABLE_ENVIRONMENT_FIELDS = (
    "python_version",
    "torch_version",
    "numpy_version",
    "dependency_digest",
    "platform",
    "cpu_model",
    "device",
    "intra_op_threads",
    "inter_op_threads",
)


class LockedModelError(RuntimeError):
    """Raised when a model is not backed by a validating experiment lock."""


class ResourceBenchmarkError(RuntimeError):
    """Raised when the official benchmark contract forbids proceeding."""


def validate_resource_benchmark_protocol(
    path: Path = RESOURCE_PROTOCOL_PATH,
) -> str:
    """Hash the exact protocol bytes and require the frozen digest."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != RESOURCE_PROTOCOL_SHA256:
        raise ResourceBenchmarkError(
            "B4_RESOURCE_BENCHMARK_V1.md differs from its frozen SHA-256."
        )
    return digest


def _architecture_of(lock: dict[str, Any]) -> str:
    model = lock.get("model")
    if isinstance(model, dict) and model.get("architecture"):
        return str(model["architecture"])
    raise LockedModelError("The experiment lock records no model architecture.")


def validate_locked_model(
    run_dir: Path, *, official_model: str | None = None
) -> dict[str, Any]:
    """Validate the lock digest and bound checkpoint before any model is loaded.

    `official_model` selects the frozen B4-A/B4-B/B4-C mapping that official
    evidence must satisfy. Synthetic fixtures may omit it, but the official suite
    always supplies it.
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
    bound_bytes = lock.get("checkpoint_bytes")
    if bound_bytes is not None and checkpoint.stat().st_size != bound_bytes:
        raise LockedModelError("The locked checkpoint byte size does not match.")
    if official_model is not None:
        _require_official_lock(lock, official_model)
    return lock


def _require_official_lock(lock: dict[str, Any], official_model: str) -> None:
    """Apply the frozen official mapping and scientific-lock requirements."""
    specification = OFFICIAL_MODELS.get(official_model)
    if specification is None:
        raise ResourceBenchmarkError(f"Unknown official model {official_model!r}.")
    if lock.get("experiment_id") != specification["experiment_id"]:
        raise ResourceBenchmarkError(
            f"{official_model} requires experiment_id "
            f"{specification['experiment_id']!r}; observed "
            f"{lock.get('experiment_id')!r}."
        )
    if _architecture_of(lock) != specification["architecture"]:
        raise ResourceBenchmarkError(
            f"{official_model} requires architecture "
            f"{specification['architecture']!r}."
        )
    if lock["trainable_parameter_count"] != specification[
        "trainable_parameter_count"
    ]:
        raise ResourceBenchmarkError(
            f"{official_model} requires exactly "
            f"{specification['trainable_parameter_count']} trainable parameters."
        )
    if lock.get("status") != "locked_for_one_shot_test":
        raise ResourceBenchmarkError(
            f"{official_model} lock is not sealed for one-shot test."
        )
    if lock.get("git_dirty") is not False:
        raise ResourceBenchmarkError(
            f"{official_model} lock is not from a clean Git checkout."
        )
    model = lock.get("model", {})
    if model.get("verified_against_constructed_model") is not True:
        raise ResourceBenchmarkError(
            f"{official_model} lock has no verified constructed-model identity."
        )
    if lock.get("protocol_sha256") != B4_PROTOCOL_SHA256:
        raise ResourceBenchmarkError(
            f"{official_model} lock does not bind the frozen B4_PROTOCOL_V1."
        )
    if specification["requires_architecture_protocol"]:
        if lock.get("architecture_protocol_sha256") != ARCHITECTURE_PROTOCOL_SHA256:
            raise ResourceBenchmarkError(
                f"{official_model} lock does not bind the frozen architecture "
                "selection protocol."
            )
    # B4-A predates `environment_dependency_digest`; validate the strongest
    # historically available equivalent rather than fabricating the field.
    digest = lock.get("environment_dependency_digest")
    if digest is None:
        digest = (
            lock.get("environment", {})
            .get("dependencies", {})
            .get("installed_packages_sha256")
        )
    if digest != B4A_DEPENDENCY_DIGEST:
        raise ResourceBenchmarkError(
            f"{official_model} lock does not record the frozen dependency digest."
        )


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
    if trainable_parameter_count(model) != lock["trainable_parameter_count"]:
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


def nearest_rank_p95(samples: list[int]) -> int:
    """Frozen nearest-rank p95: `rank = ceil(0.95 * N)`, value at `rank - 1`."""
    if not samples:
        raise ResourceBenchmarkError("p95 requires at least one measured sample.")
    ordered = sorted(samples)
    rank = math.ceil(P95_FRACTION * len(ordered))
    return ordered[max(rank, 1) - 1]


def _cpu_model() -> str | None:
    cpu = platform.processor() or None
    if not cpu and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return cpu


def benchmark_environment() -> dict[str, Any]:
    """Collect the resolved environment, including the comparability fields."""
    import numpy

    from cardiosentinel.neural.provenance import dependency_environment

    dependencies = dependency_environment()
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "device": "cpu",
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "dependency_digest": dependencies["installed_packages_sha256"],
        "dependencies": dependencies,
    }


def _require_exact_benchmark_environment(environment: dict[str, Any]) -> str:
    """Reuse the reviewed candidate-run environment gate, no second definition."""
    return require_exact_scientific_environment(
        {
            "python_version": environment["python_version"],
            "torch_version": environment["torch_version"],
            "numpy_version": environment["numpy_version"],
            "dependencies": environment["dependencies"],
        }
    )


def measure_locked_model(
    run_dir: Path, *, official_model: str | None = None
) -> dict[str, Any]:
    """Run the frozen procedure in **this** process; intended for a subprocess.

    Both gates run before warm-up, before any timed call and before any memory
    evidence is produced: the protocol bytes must match their frozen digest and
    the process environment must equal the exact B4-A scientific environment.
    """
    protocol_sha256 = validate_resource_benchmark_protocol()
    torch.set_num_threads(INTRA_OP_THREADS)
    environment = benchmark_environment()
    dependency_digest = _require_exact_benchmark_environment(environment)

    lock = validate_locked_model(run_dir, official_model=official_model)
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

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checkpoint = Path(run_dir) / str(lock["locked_inference_model"])
    payload = {
        "benchmark_protocol": BENCHMARK_PROTOCOL_NAME,
        "resource_benchmark_protocol": BENCHMARK_PROTOCOL_NAME,
        "resource_benchmark_protocol_sha256": protocol_sha256,
        "official_model": official_model,
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
        "median_latency_ms_per_window": statistics.median(samples_ns) / 1e6,
        "p95_latency_ms_per_window": nearest_rank_p95(samples_ns) / 1e6,
        "p95_definition": "nearest_rank ceil(0.95*N)",
        "tie_break_statistic": "median_latency_ms_per_window",
        "peak_rss": peak_rss,
        "peak_rss_units": "kibibytes" if sys.platform.startswith("linux") else "bytes",
        "peak_rss_measurement_method": (
            "resource.getrusage(RUSAGE_SELF).ru_maxrss in a fresh subprocess"
        ),
        "peak_rss_available": bool(peak_rss and peak_rss > 0),
        "process_isolated": True,
        "environment_dependency_digest": dependency_digest,
        "environment": environment,
    }
    payload["benchmark_result_sha256"] = canonical_sha256(payload)
    return payload


def benchmark_locked_model_isolated(
    run_dir: Path,
    *,
    official_model: str | None = None,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Measure one locked model in a **fresh subprocess** and return its result.

    Isolation is required because `ru_maxrss` is a non-decreasing process
    high-water mark; a shared process would attribute the largest model's peak to
    every later model.
    """
    command = [
        sys.executable, "-m", "cardiosentinel.neural.resource_benchmark",
        str(Path(run_dir).resolve()),
    ]
    if official_model is not None:
        command.append(official_model)
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_seconds, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise LockedModelError(
            f"Isolated benchmark failed for {run_dir}: "
            f"{detail[-1] if detail else 'no stderr'}"
        )
    return json.loads(completed.stdout)


def _resolve_suite_dir(run_root: Path) -> Path:
    root = require_nonversioned_path(run_root, "B4 resource benchmark root")
    return root / SUITE_DIR_NAME


def _claim_suite_attempt(path: Path, payload: dict[str, Any]) -> str:
    """Create the official attempt with an atomic O_EXCL creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ResourceBenchmarkError(
            "An official B4 resource benchmark attempt already exists at "
            f"{path}. The official suite runs once; there is no force, best-of, "
            "retry-one, rerun-candidate or overwrite path, and any recovery "
            "reruns the full A/B/C suite under documented human governance."
        ) from error
    # The claim now exists on disk and is never unlinked on failure.
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_file(path)


def _require_same_host(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Refuse the suite unless every child reports an identical host identity."""
    shared: dict[str, Any] = {}
    for field in COMPARABLE_ENVIRONMENT_FIELDS:
        observed = {
            name: result["environment"].get(field) for name, result in results.items()
        }
        distinct = set(observed.values())
        if len(distinct) != 1:
            raise ResourceBenchmarkError(
                "The official suite requires one host: candidates disagree on "
                f"{field}: {observed}."
            )
        shared[field] = next(iter(distinct))
    if shared["intra_op_threads"] != INTRA_OP_THREADS:
        raise ResourceBenchmarkError(
            f"The official suite requires intra-op threads == {INTRA_OP_THREADS}."
        )
    return shared


def run_official_resource_suite(
    run_directories: dict[str, Path],
    run_root: Path,
    *,
    command: str = "cardiosentinel b4 resource-benchmark",
    timeout_seconds: float = 900.0,
    _runner=benchmark_locked_model_isolated,
) -> dict[str, Any]:
    """Benchmark B4-A, B4-B and B4-C in one exclusive official invocation.

    Exactly three locked run directories are required, measured in the frozen
    order, each in its own fresh subprocess. No model may be omitted, no fourth
    model added, and no model benchmarked officially on its own.
    """
    if set(run_directories) != set(OFFICIAL_ORDER):
        raise ResourceBenchmarkError(
            "The official suite requires exactly B4-A, B4-B and B4-C; observed "
            f"{sorted(run_directories)}."
        )
    started = time.monotonic()
    protocol_sha256 = validate_resource_benchmark_protocol()
    suite_dir = _resolve_suite_dir(run_root)
    if (suite_dir / SUITE_RESULTS_NAME).exists():
        raise ResourceBenchmarkError(
            "An official B4 resource benchmark result already exists at "
            f"{suite_dir}. The official suite runs once."
        )
    attempt_path = suite_dir / SUITE_ATTEMPT_NAME
    attempt = {
        "suite": SUITE_DIR_NAME,
        "attempt_sequence": 1,
        "attempt_status": SUITE_STATUS_STARTED,
        "repeat_attempt_permitted": False,
        "selective_candidate_retry_permitted": False,
        "resource_benchmark_protocol_sha256": protocol_sha256,
        "candidate_order": list(OFFICIAL_ORDER),
        "command": command,
        "created_at_utc_audit_only": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    attempt_sha256 = _claim_suite_attempt(attempt_path, attempt)

    try:
        results: dict[str, dict[str, Any]] = {}
        for name in OFFICIAL_ORDER:
            results[name] = _runner(
                run_directories[name],
                official_model=name,
                timeout_seconds=timeout_seconds,
            )
        shared_environment = _require_same_host(results)
        for name, result in results.items():
            if result.get("resource_benchmark_protocol_sha256") != protocol_sha256:
                raise ResourceBenchmarkError(
                    f"{name} used a different resource protocol digest."
                )

        suite = {
            "suite": SUITE_DIR_NAME,
            "resource_benchmark_protocol": BENCHMARK_PROTOCOL_NAME,
            "resource_benchmark_protocol_sha256": protocol_sha256,
            "architecture_protocol_sha256": ARCHITECTURE_PROTOCOL_SHA256,
            "b4_protocol_sha256": B4_PROTOCOL_SHA256,
            "suite_attempt_sha256": attempt_sha256,
            "attempt_sequence": 1,
            "candidate_order": list(OFFICIAL_ORDER),
            "experiment_lock_sha256": {
                name: results[name]["experiment_lock_sha256"]
                for name in OFFICIAL_ORDER
            },
            "checkpoint_sha256": {
                name: results[name]["checkpoint_sha256"] for name in OFFICIAL_ORDER
            },
            "benchmark_result_sha256": {
                name: results[name]["benchmark_result_sha256"]
                for name in OFFICIAL_ORDER
            },
            "shared_environment": shared_environment,
            "timing_procedure": {
                "batch_size": BENCHMARK_BATCH_SIZE,
                "warmup_calls": WARMUP_CALLS,
                "measured_calls": MEASURED_CALLS,
                "timer": "time.perf_counter_ns",
                "p95_definition": "nearest_rank ceil(0.95*N)",
                "tie_break_statistic": "median_latency_ms_per_window",
            },
            "memory_procedure": {
                "method": (
                    "resource.getrusage(RUSAGE_SELF).ru_maxrss in a fresh subprocess"
                ),
                "units": results[OFFICIAL_ORDER[0]]["peak_rss_units"],
                "available": {
                    name: results[name]["peak_rss_available"]
                    for name in OFFICIAL_ORDER
                },
            },
            "model_sizes": {
                name: {
                    "trainable_parameter_count": results[name][
                        "trainable_parameter_count"
                    ],
                    "fp32_parameter_payload_bytes": results[name][
                        "fp32_parameter_payload_bytes"
                    ],
                    "locked_checkpoint_bytes": results[name][
                        "locked_checkpoint_bytes"
                    ],
                }
                for name in OFFICIAL_ORDER
            },
            "median_latency_ms_per_window": {
                name: results[name]["median_latency_ms_per_window"]
                for name in OFFICIAL_ORDER
            },
            "p95_latency_ms_per_window": {
                name: results[name]["p95_latency_ms_per_window"]
                for name in OFFICIAL_ORDER
            },
            "peak_rss": {
                name: results[name]["peak_rss"] for name in OFFICIAL_ORDER
            },
            "candidate_results": {name: results[name] for name in OFFICIAL_ORDER},
            "dataset_accessed": False,
            "test_accessed": False,
            "command": command,
            "suite_duration_seconds": time.monotonic() - started,
        }
        suite["resource_benchmark_suite_sha256"] = canonical_sha256(suite)
        write_json_atomic(suite_dir / SUITE_RESULTS_NAME, suite)
        write_json_atomic(
            attempt_path,
            {
                **attempt,
                "attempt_status": SUITE_STATUS_COMPLETE,
                "resource_benchmark_suite_sha256": suite[
                    "resource_benchmark_suite_sha256"
                ],
            },
        )
        return suite
    except BaseException as error:
        write_json_atomic(
            attempt_path,
            {
                **attempt,
                "attempt_status": SUITE_STATUS_FAILED,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=20),
                "human_review_required": True,
                "selective_candidate_retry_permitted": False,
                "repeat_attempt_permitted": False,
            },
        )
        raise


def _main(argv: list[str]) -> int:  # pragma: no cover - subprocess entry point
    if len(argv) not in (2, 3):
        print(
            "usage: python -m cardiosentinel.neural.resource_benchmark "
            "RUN_DIR [OFFICIAL_MODEL]",
            file=sys.stderr,
        )
        return 2
    official = argv[2] if len(argv) == 3 else None
    print(
        json.dumps(
            measure_locked_model(Path(argv[1]), official_model=official),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(_main(sys.argv))
