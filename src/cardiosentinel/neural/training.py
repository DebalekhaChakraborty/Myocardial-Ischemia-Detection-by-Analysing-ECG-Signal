"""Frozen B4 training semantics, separated from sealed-test evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from cardiosentinel.evaluation.metrics import select_validation_f1_threshold
from cardiosentinel.neural.determinism import data_order_generator, seed_worker

BATCH_SIZE = 256
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_DELTA = 1e-6


@dataclass(frozen=True, slots=True)
class CheckpointDecision:
    save_checkpoint: bool
    stop_training: bool
    best_epoch: int
    best_auprc: float
    patience: int


@dataclass(frozen=True, slots=True)
class CompletedEpoch:
    epoch: int
    mean_training_loss: float
    validation_auprc: float
    checkpoint_saved: bool
    early_stopping_patience: int


@dataclass(frozen=True, slots=True)
class FrozenTrainingResult:
    history: tuple[CompletedEpoch, ...]
    selected_checkpoint_epoch: int
    selected_validation_auprc: float
    validation_threshold: float


class CheckpointTracker:
    """Track exact maximum AUPRC while applying the separate patience delta."""

    def __init__(self) -> None:
        self.best_epoch = 0
        self.best_auprc = float("-inf")
        self.early_stopping_reference = float("-inf")
        self.patience = 0

    def update(self, epoch: int, validation_auprc: float) -> CheckpointDecision:
        if epoch < 1 or not np.isfinite(validation_auprc):
            raise ValueError("B4 checkpoint updates require a finite completed epoch.")
        save = validation_auprc > self.best_auprc
        if save:
            self.best_epoch = epoch
            self.best_auprc = validation_auprc

        if validation_auprc > self.early_stopping_reference + EARLY_STOPPING_DELTA:
            self.early_stopping_reference = validation_auprc
            self.patience = 0
        else:
            self.patience += 1
        return CheckpointDecision(
            save_checkpoint=save,
            stop_training=self.patience >= EARLY_STOPPING_PATIENCE,
            best_epoch=self.best_epoch,
            best_auprc=self.best_auprc,
            patience=self.patience,
        )


def build_loss() -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss(reduction="mean")


def build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """Construct the one-group optimizer frozen by B4_PROTOCOL_V1."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def build_training_loader(dataset: Dataset, *, workers: int) -> DataLoader:
    """Build the frozen shuffled loader with deterministic epoch-wise ordering."""
    if workers < 0:
        raise ValueError("B4 worker count cannot be negative.")
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=data_order_generator(),
    )


def build_validation_loader(dataset: Dataset, *, workers: int) -> DataLoader:
    """Build an unsampled, non-shuffled full-primary-validation loader."""
    if workers < 0:
        raise ValueError("B4 worker count cannot be negative.")
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
    )


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for waveforms, labels in batches:
        waveforms = waveforms.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(waveforms)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        count = labels.numel()
        total_loss += float(loss.detach()) * count
        total_count += count
    if total_count == 0:
        raise ValueError("B4 training requires at least one batch.")
    return total_loss / total_count


def validation_scores(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validation labels and sigmoid scores; no metric is selected here."""
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for waveforms, batch_labels in batches:
            logits = model(waveforms.to(device=device, dtype=torch.float32))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    if not labels:
        raise ValueError("B4 validation requires at least one batch.")
    return np.concatenate(labels), np.concatenate(scores)


def validation_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the frozen pooled-window checkpoint objective lazily."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def validation_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    return select_validation_f1_threshold(
        labels.tolist(), scores.tolist(), partition="validation"
    )


def run_frozen_training(
    model: nn.Module,
    training_batches: Iterable[tuple[Tensor, Tensor]],
    validation_batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    checkpoint_path: Path,
    *,
    epoch_callback: Callable[[CompletedEpoch], None] | None = None,
) -> FrozenTrainingResult:
    """Execute frozen B4 training semantics; this function has no test-data API.

    `epoch_callback` is non-scientific crash-safety instrumentation. It runs
    only after an epoch has completed and its checkpoint decision is already
    recorded, its return value is ignored, and it takes no part in checkpoint
    selection, threshold selection, or the early-stopping computation.

    If persistence raises, the exception propagates and aborts the whole
    experiment so a run can never continue with missing provenance. That aborts
    the run for integrity; it never alters the scientific early-stopping
    decision, which is computed before the callback is invoked.
    """
    optimizer = build_optimizer(model)
    loss_function = build_loss()
    tracker = CheckpointTracker()
    history: list[CompletedEpoch] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        training_loss = train_one_epoch(
            model, training_batches, optimizer, loss_function, device
        )
        labels, scores = validation_scores(model, validation_batches, device)
        auprc = validation_auprc(labels, scores)
        decision = tracker.update(epoch, auprc)
        if decision.save_checkpoint:
            save_checkpoint(checkpoint_path, model, optimizer)
        completed = CompletedEpoch(
            epoch=epoch,
            mean_training_loss=training_loss,
            validation_auprc=auprc,
            checkpoint_saved=decision.save_checkpoint,
            early_stopping_patience=decision.patience,
        )
        history.append(completed)
        if epoch_callback is not None:
            epoch_callback(completed)
        if decision.stop_training:
            break

    restore_checkpoint(checkpoint_path, model, optimizer)
    labels, scores = validation_scores(model, validation_batches, device)
    threshold = validation_f1_threshold(labels, scores)
    return FrozenTrainingResult(
        history=tuple(history),
        selected_checkpoint_epoch=tracker.best_epoch,
        selected_validation_auprc=tracker.best_auprc,
        validation_threshold=threshold,
    )


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    """Atomically serialize a checkpoint to a caller-controlled artifact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        temporary,
    )
    os.replace(temporary, path)


def restore_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
