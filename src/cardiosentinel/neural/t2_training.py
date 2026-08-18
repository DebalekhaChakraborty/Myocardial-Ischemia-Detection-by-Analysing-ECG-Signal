"""Deterministic synchronized-stream TBPTT training for the two T2 candidates.

This translates `docs/T2_CANONICAL_TRAINING_EXECUTION_SPEC_V1.md` into code and
decides no science. The parts worth understanding before editing:

**The batch is the set of active streams.** There is no mini-batch-size
hyperparameter. At TBPTT frontier `k` every still-active fitting stream
contributes its local rows `[k*256:(k+1)*256]`, and those independent chunks are
batched along the batch dimension. Streams are grouped by exact chunk length, so
no padding is ever needed and no padded position can contaminate a carried
state -- a strictly stronger guarantee than the "pad for tensor shape only"
the spec permits.

**An unavailable row never reaches the model.** Each stream chunk is compacted
to its available rows before the forward pass, so the recurrence advances over
observations only. That is exactly equivalent to carrying the state unchanged
across the gap, and it means there is no zero vector, no imputed embedding and
no mask token anywhere in this file.

**Loss is reduced over the frontier, not the group.** Weighted BCE is summed
over every PRIMARY direct-loss row in the whole frontier and divided by their
count, so an eligible PRIMARY window contributes equally regardless of which
stream or length-group it landed in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from cardiosentinel.baseline.metrics import binary_metrics
from cardiosentinel.neural.t2_models import build_t2_model, detach_state
from cardiosentinel.neural.t2_protocol import (
    T2_EARLY_STOPPING_PATIENCE_EPOCHS,
    T2_GRADIENT_CLIP_NORM,
    T2_LEARNING_RATE,
    T2_MAX_EPOCHS,
    T2_TBPTT_LENGTH,
    T2_WEIGHT_DECAY,
)

CHECKPOINT_CRITERION: Final = "internal_development_pooled_auprc"


class T2TrainingError(RuntimeError):
    """Raised when canonical T2 training cannot proceed safely."""


# ---------------------------------------------------------------------------
# Synchronized frontier planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T2ChunkSlice:
    """One stream's contribution to one synchronized frontier."""

    stream_index: int
    local_start: int
    local_stop: int

    @property
    def length(self) -> int:
        return self.local_stop - self.local_start


def synchronized_frontiers(
    stream_lengths: Sequence[int], *, tbptt: int = T2_TBPTT_LENGTH
) -> tuple[tuple[T2ChunkSlice, ...], ...]:
    """Plan the frontiers. Every stream keeps its own clock and its own state."""
    if int(tbptt) != T2_TBPTT_LENGTH:
        raise T2TrainingError(f"The frozen TBPTT length is {T2_TBPTT_LENGTH}.")
    lengths = [int(value) for value in stream_lengths]
    if any(length <= 0 for length in lengths):
        raise T2TrainingError("Every stream must carry at least one row.")
    frontier_count = max((length + tbptt - 1) // tbptt for length in lengths)
    frontiers: list[tuple[T2ChunkSlice, ...]] = []
    for index in range(frontier_count):
        start = index * tbptt
        slices = tuple(
            T2ChunkSlice(
                stream_index=position,
                local_start=start,
                local_stop=min(start + tbptt, length),
            )
            for position, length in enumerate(lengths)
            if start < length
        )
        frontiers.append(slices)
    return tuple(frontiers)


def group_by_length(
    slices: Sequence[T2ChunkSlice],
) -> dict[int, tuple[T2ChunkSlice, ...]]:
    """Group a frontier's chunks by exact length, so padding is never required."""
    grouped: dict[int, list[T2ChunkSlice]] = {}
    for item in slices:
        grouped.setdefault(item.length, []).append(item)
    return {length: tuple(items) for length, items in sorted(grouped.items())}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def positive_class_weight(*, negative_count: int, positive_count: int) -> float:
    """`N_negative / N_positive`, from the FIT partition only."""
    if positive_count <= 0 or negative_count <= 0:
        raise T2TrainingError(
            "The frozen class weight needs both classes present in the T2 fit "
            "partition."
        )
    return float(negative_count) / float(positive_count)


def direct_loss_sum(
    logits: Tensor, targets: Tensor, mask: Tensor, *, pos_weight: float
) -> tuple[Tensor, int]:
    """Weighted BCE summed over direct-loss rows, plus that row count.

    Reduction is deliberately `sum` here: the division happens once per
    optimiser step over the whole frontier, never per group.
    """
    selected = mask.bool()
    count = int(selected.sum().item())
    if count == 0:
        return logits.new_zeros(()), 0
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    total = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[selected],
        targets[selected].to(logits.dtype),
        pos_weight=weight,
        reduction="sum",
    )
    return total, count


# ---------------------------------------------------------------------------
# Chunk execution: availability compaction, then the forward pass
# ---------------------------------------------------------------------------


def compact_available(
    representation: np.ndarray, available: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unavailable rows before the model ever sees them.

    Returns the compacted representation and the original positions it came
    from, so scores can be written back without inventing a value for the gap.
    """
    positions = np.nonzero(np.asarray(available).astype(bool))[0]
    return np.asarray(representation)[positions], positions


def run_stream_group(
    model: nn.Module,
    values: Tensor,
    states: list[Any],
    indices: Sequence[int],
) -> tuple[Tensor, None]:
    """Forward one equal-length group, carrying each stream's own state."""
    incoming = _stack_states(model, [states[index] for index in indices])
    logits, outgoing = model(values, incoming)
    for position, index in enumerate(indices):
        states[index] = _slice_state(outgoing, position)
    return logits, None


def _stack_states(model: nn.Module, states: Sequence[Any]) -> Any:
    if any(state is None for state in states):
        raise T2TrainingError("Every active stream must carry an initialised state.")
    first = states[0]
    if isinstance(first, torch.Tensor):
        # GRU: [layers, batch, width]
        return torch.cat([state for state in states], dim=1)
    return tuple(
        torch.cat([state[block] for state in states], dim=0)
        for block in range(len(first))
    )


def _slice_state(state: Any, position: int) -> Any:
    if isinstance(state, torch.Tensor):
        return state[:, position : position + 1, :]
    return tuple(block[position : position + 1] for block in state)


def initial_states(model: nn.Module, count: int, *, device: Any = None) -> list[Any]:
    """One zero state per stream, at every real stream boundary."""
    return [model.initial_state(1, device=device) for _ in range(count)]


# ---------------------------------------------------------------------------
# Epoch and checkpoint bookkeeping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class T2EpochResult:
    """One completed epoch's persisted evidence."""

    epoch: int
    optimizer_steps: int
    zero_direct_loss_chunks: int
    direct_loss_rows: int
    mean_training_loss: float
    internal_dev_pooled_auprc: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "optimizer_steps": self.optimizer_steps,
            "zero_direct_loss_chunks": self.zero_direct_loss_chunks,
            "direct_loss_rows": self.direct_loss_rows,
            "mean_training_loss": self.mean_training_loss,
            "internal_dev_pooled_auprc": self.internal_dev_pooled_auprc,
        }


@dataclass(slots=True)
class T2CheckpointSelector:
    """Frozen improvement, patience and tie semantics.

    Exact equality is **not** an improvement, so the earlier epoch keeps the
    checkpoint. Patience counts consecutive completed epochs after the current
    best that fail to improve.
    """

    best_epoch: int | None = None
    best_auprc: float | None = None
    patience: int = 0
    history: list[T2EpochResult] = field(default_factory=list)

    def offer(self, result: T2EpochResult) -> bool:
        value = float(result.internal_dev_pooled_auprc)
        if not math.isfinite(value):
            raise T2TrainingError(
                f"Epoch {result.epoch} produced a non-finite internal-dev AUPRC. "
                "This is a hard failure: no retry and no substitution."
            )
        self.history.append(result)
        improved = self.best_auprc is None or value > self.best_auprc
        if improved:
            self.best_epoch = result.epoch
            self.best_auprc = value
            self.patience = 0
        else:
            self.patience += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience >= T2_EARLY_STOPPING_PATIENCE_EPOCHS

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": CHECKPOINT_CRITERION,
            "tie_break": "earlier_epoch",
            "patience_epochs": T2_EARLY_STOPPING_PATIENCE_EPOCHS,
            "max_epochs": T2_MAX_EPOCHS,
            "best_epoch": self.best_epoch,
            "best_internal_dev_pooled_auprc": self.best_auprc,
            "epochs": [item.as_dict() for item in self.history],
        }


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """The one frozen optimiser configuration."""
    return torch.optim.AdamW(
        model.parameters(), lr=T2_LEARNING_RATE, weight_decay=T2_WEIGHT_DECAY
    )


def clip_gradients(model: nn.Module) -> float:
    return float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), T2_GRADIENT_CLIP_NORM)
    )


# ---------------------------------------------------------------------------
# Internal-dev threshold: exact maximum F1, highest-threshold tie-break
# ---------------------------------------------------------------------------


def maximum_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Exact max-F1 sweep with the repository's highest-threshold tie-break.

    Identical arithmetic to `evaluation.metrics.select_validation_f1_threshold`,
    which is partition-locked to VALIDATION and therefore cannot be called for
    an internal-dev derivation. A parity test proves the two agree.
    """
    if len(labels) != len(scores) or not labels:
        raise T2TrainingError(
            "Threshold labels and scores must be aligned and non-empty."
        )
    values = [float(score) for score in scores]
    if any(label not in {0, 1} for label in labels):
        raise T2TrainingError("Threshold labels must be binary.")
    if not any(labels):
        raise T2TrainingError("Threshold derivation needs at least one positive.")
    candidates = sorted(set(values), reverse=True)
    positive_counts = dict.fromkeys(candidates, 0)
    negative_counts = dict.fromkeys(candidates, 0)
    for label, score in zip(labels, values, strict=True):
        if math.isnan(score):
            continue
        if label == 1:
            positive_counts[score] += 1
        else:
            negative_counts[score] += 1
    total_positive = sum(labels)
    true_positive = 0
    false_positive = 0
    best_f1 = -1.0
    best_threshold = candidates[0]
    for threshold in candidates:
        true_positive += positive_counts[threshold]
        false_positive += negative_counts[threshold]
        false_negative = total_positive - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        # Strictly greater keeps the highest threshold on a tie, because the
        # sweep walks candidates from high to low.
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold)


def internal_dev_threshold_evidence(
    labels: Sequence[int], scores: Sequence[float]
) -> dict[str, Any]:
    """The frozen internal-dev threshold and its TRAIN-development metrics."""
    threshold = maximum_f1_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    return {
        "evidence_class": "t2_internal_dev_threshold",
        "partition": "t2_internal_dev_8_subjects",
        "rule": "exact_maximum_f1_highest_threshold_tie_break",
        "threshold": threshold,
        "derived_before_outer_validation": True,
        "outer_validation_may_alter": False,
        "is_outer_validation_evidence": False,
        "is_train_development_evidence": True,
        **{
            key: metrics[key]
            for key in (
                "auprc",
                "auroc",
                "f1",
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "balanced_accuracy",
                "mcc",
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
            )
        },
    }


def pooled_auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Pooled AUPRC, the one checkpoint criterion."""
    metrics = binary_metrics(labels, scores, 0.5)
    value = metrics["auprc"]
    if value is None:
        raise T2TrainingError(
            "Internal-dev AUPRC is undefined (a single class was present). "
            "This is a hard failure, not a value to substitute."
        )
    return float(value)


def build_candidate(arm: str) -> nn.Module:
    """Construct one arm from a fresh seed origin, count asserted."""
    return build_t2_model(arm)


def detach_all(states: Sequence[Any]) -> list[Any]:
    """Sever gradient history at the frontier; the carried values are unchanged."""
    return [detach_state(state) for state in states]
