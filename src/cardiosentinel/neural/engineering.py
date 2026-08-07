"""Train/validation-only B4 preflight, smoke, and I/O engineering checks."""

from __future__ import annotations

import resource
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cardiosentinel.data.provenance import git_provenance
from cardiosentinel.neural.data import B4WaveformDataset
from cardiosentinel.neural.determinism import initialize_determinism
from cardiosentinel.neural.metadata import (
    B4MetadataIndex,
    build_training_index,
    build_validation_index,
)
from cardiosentinel.neural.model import (
    B4CompactCNN,
    fp32_parameter_payload_bytes,
    local_receptive_field_samples,
    trainable_parameter_count,
)
from cardiosentinel.neural.protocol import (
    B4_SPLIT_SHA256,
    EXPECTED_COUNTS,
    FEATURE_CORPUS_SHA256,
    FP32_PARAMETER_BYTES,
    LOCAL_RECEPTIVE_FIELD_SAMPLES,
    REPOSITORY_ROOT,
    TEMPORAL_LENGTHS,
    TRAINABLE_PARAMETER_COUNT,
    require_development_partition,
    validate_frozen_protocol,
)
from cardiosentinel.neural.provenance import (
    runtime_environment,
    validate_source_verification_receipt,
)
from cardiosentinel.neural.training import (
    build_loss,
    build_optimizer,
    restore_checkpoint,
    save_checkpoint,
    validation_scores,
)

NON_SCIENTIFIC = "SMOKE / NON-SCIENTIFIC / NOT A RESULT"


def _resource_report(path: Path) -> dict[str, int | None]:
    disk = shutil.disk_usage(path if path.exists() else path.parent)
    memory_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        fields = {
            line.partition(":")[0]: line.partition(":")[2].strip()
            for line in meminfo.read_text(encoding="utf-8").splitlines()
        }
        if "MemAvailable" in fields:
            memory_bytes = int(fields["MemAvailable"].split()[0]) * 1024
    return {"available_disk_bytes": disk.free, "available_ram_bytes": memory_bytes}


def b4_preflight(
    source: Path,
    feature_root: Path,
    *,
    workers: int = 0,
    require_clean: bool = True,
    requested_device: str | None = None,
) -> dict[str, Any]:
    """Validate frozen development inputs without enumerating sealed-test rows."""
    if workers < 0:
        raise ValueError("B4 worker count cannot be negative.")
    protocol_sha256 = validate_frozen_protocol()
    source_receipt = validate_source_verification_receipt(source)
    provenance = git_provenance(REPOSITORY_ROOT)
    if require_clean and provenance["git_dirty"]:
        raise ValueError("B4 preflight requires a clean Git checkout.")
    determinism = initialize_determinism(requested_device=requested_device)
    validate_model_constants()
    train = build_training_index(feature_root)
    validation = build_validation_index(feature_root)
    # This executable assertion proves the development API has no test route.
    try:
        require_development_partition("test")
    except ValueError:
        test_firewall = "rejected_before_artifact_resolution"
    else:  # pragma: no cover - a hard invariant
        raise AssertionError("B4 sealed-test firewall is disabled.")
    return {
        "status": "ready_for_development_only",
        "protocol_sha256": protocol_sha256,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "source_provenance": source_receipt,
        "training": _index_summary(train),
        "validation": _index_summary(validation),
        "test_firewall": test_firewall,
        "git": provenance,
        "determinism": asdict(determinism),
        "environment": runtime_environment(determinism.device, workers),
        "resources": _resource_report(feature_root),
    }


def _index_summary(index: B4MetadataIndex) -> dict[str, Any]:
    return {
        "partition": index.partition,
        "positive": index.positive_count,
        "negative": index.negative_count,
        "total": index.total_count,
        "subjects": index.subject_count,
        "selection_sha256": index.selection_sha256,
    }


def _deterministic_subset(index: B4MetadataIndex, count: int):
    if count < 1:
        raise ValueError("B4 engineering subset size must be positive.")
    return tuple(sorted(index.references, key=lambda item: item.stable_id)[:count])


def b4_smoke(
    source: Path,
    feature_root: Path,
    *,
    train_windows: int = 2,
    validation_windows: int = 2,
    requested_device: str | None = None,
) -> dict[str, Any]:
    """Run one tiny optimizer step and score tiny validation data without metrics."""
    validate_frozen_protocol()
    state = initialize_determinism(requested_device=requested_device)
    train_index = build_training_index(feature_root)
    validation_index = build_validation_index(feature_root)
    train_data = B4WaveformDataset(
        _deterministic_subset(train_index, train_windows), source
    )
    validation_data = B4WaveformDataset(
        _deterministic_subset(validation_index, validation_windows), source
    )
    train_loader = DataLoader(train_data, batch_size=train_windows, shuffle=False)
    validation_loader = DataLoader(
        validation_data, batch_size=validation_windows, shuffle=False
    )
    device = torch.device(state.device)
    model = B4CompactCNN().to(device)
    optimizer = build_optimizer(model)
    loss_function = build_loss()

    waveforms, labels = next(iter(train_loader))
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(waveforms.to(device)), labels.to(device))
    loss.backward()
    optimizer.step()
    validation_labels, validation_outputs = validation_scores(
        model, validation_loader, device
    )
    with tempfile.TemporaryDirectory(prefix="cardiosentinel-b4-smoke-") as temporary:
        checkpoint = Path(temporary) / "smoke.pt"
        save_checkpoint(checkpoint, model, optimizer)
        checkpoint_bytes = checkpoint.stat().st_size
        restore_checkpoint(checkpoint, model, optimizer)

    return {
        "label": NON_SCIENTIFIC,
        "train_windows": len(train_data),
        "validation_windows": len(validation_data),
        "waveform_shape": list(waveforms.shape[1:]),
        "waveform_dtype": str(waveforms.dtype),
        "forward_backward_optimizer_step": "passed",
        "validation_score_generation": "passed",
        "validation_score_count": int(validation_outputs.size),
        "validation_label_count": int(validation_labels.size),
        "checkpoint_save_restore": "passed",
        "checkpoint_serialized_bytes": checkpoint_bytes,
        "scientific_metrics_computed": False,
        "device": state.device,
    }


def benchmark_io(
    source: Path,
    feature_root: Path,
    *,
    train_windows: int = 32,
    validation_windows: int = 32,
    batch_size: int = 256,
    cache_windows: int = 0,
) -> dict[str, Any]:
    """Measure deterministic raw reads only; never execute a model or metric."""
    if batch_size < 1:
        raise ValueError("B4 benchmark batch size must be positive.")
    validate_frozen_protocol()
    # Exclude one-time module import from direct waveform-read throughput.
    import wfdb  # noqa: F401

    train_index = build_training_index(feature_root)
    validation_index = build_validation_index(feature_root)
    results = {
        "train": _benchmark_partition(
            source,
            _deterministic_subset(train_index, train_windows),
            batch_size,
            cache_windows,
            EXPECTED_COUNTS["train"]["total"],
        ),
        "validation": _benchmark_partition(
            source,
            _deterministic_subset(validation_index, validation_windows),
            batch_size,
            cache_windows,
            EXPECTED_COUNTS["validation"]["total"],
        ),
    }
    return {
        "label": "I/O ENGINEERING / NON-SCIENTIFIC / NOT A RESULT",
        "selection_rule": "first stable IDs in ascending lexical order",
        "batch_size": batch_size,
        "scientific_metrics_computed": False,
        "partitions": results,
    }


def _benchmark_partition(
    source: Path,
    references: tuple,
    batch_size: int,
    cache_windows: int,
    full_count: int,
) -> dict[str, Any]:
    dataset = B4WaveformDataset(
        references, source, cache_windows=cache_windows
    )
    started = time.perf_counter()
    for index in range(len(dataset)):
        dataset[index]
    elapsed = time.perf_counter() - started
    stats = dataset.stats
    windows_per_second = len(dataset) / elapsed
    projected_seconds = full_count / windows_per_second
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "sample_windows": len(dataset),
        "elapsed_seconds": elapsed,
        "windows_per_second": windows_per_second,
        "batches_per_second": windows_per_second / batch_size,
        "waveform_read_seconds": stats.waveform_read_seconds,
        "conversion_seconds": stats.conversion_seconds,
        "source_reads": stats.source_reads,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "process_max_rss_kib": rss_kib,
        "projected_full_windows": full_count,
        "projected_loading_seconds": projected_seconds,
    }


def validate_model_constants() -> None:
    """Fail fast if implementation drifts from frozen architecture arithmetic."""
    model = B4CompactCNN()
    if trainable_parameter_count(model) != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("B4 model parameter count differs from the frozen protocol.")
    if fp32_parameter_payload_bytes(model) != FP32_PARAMETER_BYTES:
        raise ValueError("B4 model payload differs from the frozen protocol.")
    if local_receptive_field_samples() != LOCAL_RECEPTIVE_FIELD_SAMPLES:
        raise ValueError("B4 receptive field differs from the frozen protocol.")
    if TEMPORAL_LENGTHS != (2500, 1250, 625, 313, 157, 79):
        raise ValueError("B4 temporal lengths differ from the frozen protocol.")
