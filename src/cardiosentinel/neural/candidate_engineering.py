"""Development-only smoke and compute preflight for the B4-B/B4-C candidates.

Everything here is engineering instrumentation. Nothing in this module computes
a scientific metric, selects a checkpoint or threshold, writes a scientific
artifact, or touches the sealed-test partition: the partition guard rejects
`test` before any path, cache or source is resolved.

The scientific B4-B and B4-C runners are deliberately not built here.
"""

from __future__ import annotations

import resource
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cardiosentinel.neural.candidates import (
    B4B_EXPERIMENT_ID,
    B4C_EXPERIMENT_ID,
    B4CSSMCNN,
    B4BTransformerCNN,
    b4b_model_identity,
    b4c_model_identity,
)
from cardiosentinel.neural.determinism import initialize_determinism
from cardiosentinel.neural.protocol import (
    WINDOW_SAMPLES,
    require_development_partition,
)
from cardiosentinel.neural.training import (
    BATCH_SIZE,
    build_loss,
    build_optimizer,
)
from cardiosentinel.neural.waveform_cache import (
    B4CachedWaveformDataset,
    build_development_indexes,
    validate_waveform_cache,
)

NON_SCIENTIFIC = "SMOKE / NON-SCIENTIFIC / NOT A RESULT"
ENGINEERING = "ENGINEERING PREFLIGHT / NON-SCIENTIFIC"
DEVELOPMENT_PARTITIONS = ("train", "validation")
WARMUP_BATCHES = 3
MEASURED_BATCHES = 10

CANDIDATES = {
    B4B_EXPERIMENT_ID: (B4BTransformerCNN, b4b_model_identity),
    B4C_EXPERIMENT_ID: (B4CSSMCNN, b4c_model_identity),
}


def require_development_partitions(
    partitions: Sequence[str] = DEVELOPMENT_PARTITIONS,
) -> tuple[str, ...]:
    """Reject a sealed-test partition before any artifact path is resolved."""
    if not partitions:
        raise ValueError("Candidate engineering requires at least one partition.")
    return tuple(require_development_partition(item) for item in partitions)


def build_candidate(experiment_id: str) -> tuple[nn.Module, dict[str, Any]]:
    """Construct a candidate and verify it against its frozen identity."""
    if experiment_id not in CANDIDATES:
        raise ValueError(f"Unknown B4 candidate: {experiment_id}")
    factory, identity = CANDIDATES[experiment_id]
    model = factory()
    return model, identity(model)


def candidate_smoke(
    experiment_id: str,
    source: Path,
    feature_root: Path,
    cache_root: Path,
    *,
    train_windows: int = 2,
    validation_windows: int = 2,
    partitions: Sequence[str] = DEVELOPMENT_PARTITIONS,
    requested_device: str | None = None,
) -> dict[str, Any]:
    """Run a tiny non-scientific train/validation smoke for one candidate.

    No metric, threshold, checkpoint or scientific artifact is produced. The
    model is disposable and is discarded when this function returns.
    """
    permitted = require_development_partitions(partitions)
    if train_windows < 1 or validation_windows < 1:
        raise ValueError("Candidate smoke needs at least one window per partition.")
    state = initialize_determinism(requested_device=requested_device)
    device = torch.device(state.device)

    indexes = build_development_indexes(feature_root)
    cache = validate_waveform_cache(cache_root, indexes)
    datasets = {
        partition: B4CachedWaveformDataset(cache, indexes[partition])
        for partition in permitted
    }

    def batch(partition: str, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        samples = [datasets[partition][row] for row in range(count)]
        return (
            torch.stack([item.waveform for item in samples]),
            torch.stack([item.label for item in samples]),
        )

    train_waveforms, train_labels = batch("train", train_windows)
    validation_waveforms, _ = batch("validation", validation_windows)

    initialize_determinism(requested_device=state.device)
    model, identity = build_candidate(experiment_id)
    model.to(device)
    optimizer = build_optimizer(model)
    loss_function = build_loss()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(
        model(train_waveforms.to(device)), train_labels.to(device)
    )
    loss.backward()
    finite_gradients = all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_logits = model(validation_waveforms.to(device))

    with tempfile.TemporaryDirectory(prefix="cardiosentinel-candidate-") as folder:
        checkpoint = Path(folder) / "disposable.pt"
        torch.save(model.state_dict(), checkpoint)
        serialized_bytes = checkpoint.stat().st_size
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restore_ok = set(restored) == set(model.state_dict())

    return {
        "label": NON_SCIENTIFIC,
        "experiment_id": experiment_id,
        "model": identity,
        "partitions": list(permitted),
        "train_windows": train_windows,
        "validation_windows": validation_windows,
        "waveform_shape": list(train_waveforms.shape[1:]),
        "waveform_dtype": str(train_waveforms.dtype),
        "cached_read": "passed",
        "forward": "passed",
        "loss_is_finite": bool(torch.isfinite(loss)),
        "backward": "passed",
        "gradients_finite": bool(finite_gradients),
        "optimizer_step": "passed",
        "validation_forward": "passed",
        "validation_logit_count": int(validation_logits.numel()),
        "validation_logit_dtype": str(validation_logits.dtype),
        "state_dict_serialize_restore": "passed" if restore_ok else "failed",
        "disposable_serialized_bytes": serialized_bytes,
        "scientific_metrics_computed": False,
        "checkpoint_retained": False,
        "device": state.device,
    }


def _throughput(started: float, processed: int) -> float:
    elapsed = time.perf_counter() - started
    return processed / elapsed if elapsed > 0 else float("inf")


def candidate_compute_preflight(
    experiment_id: str,
    *,
    batch_size: int = BATCH_SIZE,
    warmup_batches: int = WARMUP_BATCHES,
    measured_batches: int = MEASURED_BATCHES,
    requested_device: str | None = None,
) -> dict[str, Any]:
    """Time disposable candidate compute on synthetic batches only.

    This is a feasibility estimate. It is explicitly **not** the official
    architecture-selection resource evidence, which is frozen and reviewed
    separately before any selection evidence is generated.
    """
    if not 1 <= measured_batches <= MEASURED_BATCHES:
        raise ValueError(f"Measured batches must be 1..{MEASURED_BATCHES}.")
    state = initialize_determinism(requested_device=requested_device)
    device = torch.device(state.device)
    model, identity = build_candidate(experiment_id)
    model.to(device)
    optimizer = build_optimizer(model)
    loss_function = build_loss()

    waveforms = torch.randn(batch_size, 1, WINDOW_SAMPLES, device=device)
    labels = torch.zeros(batch_size, device=device)

    model.train()
    for _ in range(warmup_batches):
        optimizer.zero_grad(set_to_none=True)
        loss_function(model(waveforms), labels).backward()
        optimizer.step()

    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(measured_batches):
            model(waveforms)
    forward_rate = _throughput(started, measured_batches * batch_size)

    model.train()
    started = time.perf_counter()
    for _ in range(measured_batches):
        optimizer.zero_grad(set_to_none=True)
        loss_function(model(waveforms), labels).backward()
        optimizer.step()
    training_rate = _throughput(started, measured_batches * batch_size)

    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(measured_batches):
            torch.sigmoid(model(waveforms))
    validation_rate = _throughput(started, measured_batches * batch_size)

    del optimizer, model
    return {
        "label": ENGINEERING,
        "experiment_id": experiment_id,
        "trainable_parameter_count": identity["trainable_parameter_count"],
        "fp32_parameter_payload_bytes": identity["fp32_parameter_payload_bytes"],
        "device": state.device,
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "forward_windows_per_second": forward_rate,
        "training_windows_per_second": training_rate,
        "validation_windows_per_second": validation_rate,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "official_selection_evidence": False,
        "scientific_metrics_computed": False,
        "weights_retained": False,
    }
