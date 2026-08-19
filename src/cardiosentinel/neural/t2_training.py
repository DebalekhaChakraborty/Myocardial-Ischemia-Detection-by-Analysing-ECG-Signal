"""Deterministic synchronized-stream TBPTT training for the two T2 candidates.

This translates `docs/T2_CANONICAL_TRAINING_EXECUTION_SPEC_V1.md` into code and
decides no science. The parts worth understanding before editing:

**The batch is the set of active streams.** There is no mini-batch-size
hyperparameter. At TBPTT frontier `k` every still-active fitting stream
contributes its local rows `[k*256:(k+1)*256]`, and those independent chunks are
batched along the batch dimension. Streams are grouped by the **compacted
available-observation length**, not by the raw timeline length, so no padding is
ever needed and no padded position can contaminate a carried state -- a strictly
stronger guarantee than the "pad for tensor shape only" the spec permits. Two
streams with identical raw slice lengths but different unavailable patterns
therefore land in different groups instead of being stacked incorrectly.

**An unavailable row never reaches the model.** Each stream slice is compacted
to its available rows before the forward pass, so the recurrence advances over
observations only. That is exactly equivalent to carrying the state unchanged
across the gap, and it means there is no zero vector, no imputed embedding and
no mask token anywhere in this file.

**Loss is reduced over the frontier, not the group.** Weighted BCE is summed
over every PRIMARY direct-loss row in the whole frontier and divided by their
count, so an eligible PRIMARY window contributes equally regardless of which
stream or length-group it landed in. One frontier is one optimiser step, or
none at all when the frontier holds no direct-loss row.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from cardiosentinel.baseline.metrics import binary_metrics
from cardiosentinel.neural.m1_store import OBSERVATION_STATE_FILE
from cardiosentinel.neural.t2_models import build_t2_model, detach_state
from cardiosentinel.neural.t2_protocol import (
    T2_EARLY_STOPPING_PATIENCE_EPOCHS,
    T2_GRADIENT_CLIP_NORM,
    T2_LEARNING_RATE,
    T2_MAX_EPOCHS,
    T2_OBSERVATION_AVAILABLE,
    T2_TBPTT_LENGTH,
    T2_WEIGHT_DECAY,
)
from cardiosentinel.neural.t2_timeline import (
    ROLE_CODE_PRIMARY,
    ROLE_CODE_UNAVAILABLE,
    T2Stream,
    T2Timeline,
    primary_labels_for_families,
    role_codes_for_families,
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


def _resolve_device(model: nn.Module, device: Any = None) -> Any:
    """The device the science runs on: the one asked for, or the model's own.

    Falling back to the model's parameter device rather than to a hardcoded
    CPU means a caller that has already moved the model cannot accidentally
    feed it inputs from somewhere else.
    """
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Bounded timeline reads
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T2SliceView:
    """One stream's compacted contribution to one synchronized frontier.

    `values` holds only AVAILABLE rows, in timeline order. `positions` records
    the local offsets they came from, so a score can be written back to the row
    it belongs to without inventing a value for the gap.
    """

    stream_index: int
    values: np.ndarray
    labels: np.ndarray
    direct_loss: np.ndarray
    positions: np.ndarray
    raw_length: int

    @property
    def length(self) -> int:
        return int(self.values.shape[0])


class T2TimelineReader:
    """Bounded per-frontier reads of representation, availability and role.

    Every read is a slice of the memory-mapped store. The reader holds one
    `uint8` family-code array for the partition -- 2.2 MB for TRAIN -- and never
    a per-row Python object, a second representation copy or a whole-timeline
    torch tensor.

    The role is derived here and used as a **mask**. There is deliberately no
    path from `family_codes` into `values`: the representation slice is taken
    from the store before any role is computed and is never concatenated with,
    scaled by or reordered against a label.
    """

    def __init__(self, timeline: T2Timeline, family_codes: np.ndarray) -> None:
        codes = np.asarray(family_codes, dtype=np.uint8)
        if codes.shape != (timeline.row_count,):
            raise T2TrainingError(
                f"The target authority carries {codes.shape} codes against a "
                f"timeline of {timeline.row_count} rows."
            )
        self.timeline = timeline
        self.family_codes = codes

    def slice_view(
        self, stream_index: int, stream: T2Stream, local_start: int, local_stop: int
    ) -> T2SliceView:
        begin = stream.start_index + int(local_start)
        end = stream.start_index + int(local_stop)
        if end > stream.stop_index:
            raise T2TrainingError("A frontier slice runs past the end of its stream.")
        states = self.timeline.column(OBSERVATION_STATE_FILE, begin, end)
        families = self.family_codes[begin:end]
        roles = role_codes_for_families(families, states)
        available = roles != ROLE_CODE_UNAVAILABLE
        positions = np.nonzero(available)[0]
        values = np.ascontiguousarray(
            self.timeline.representation(begin, end)[positions], dtype=np.float32
        )
        return T2SliceView(
            stream_index=int(stream_index),
            values=values,
            labels=primary_labels_for_families(families[positions]),
            direct_loss=(roles[positions] == ROLE_CODE_PRIMARY),
            positions=positions,
            raw_length=int(end - begin),
        )

    def stream_row_counts(self, streams: Sequence[T2Stream]) -> list[int]:
        return [int(stream.row_count) for stream in streams]

    def availability_census(self, streams: Sequence[T2Stream]) -> dict[str, int]:
        """Availability over a stream selection, counted without a full read."""
        available = 0
        total = 0
        for stream in streams:
            states = np.asarray(
                self.timeline.column(
                    OBSERVATION_STATE_FILE, stream.start_index, stream.stop_index
                )
            )
            available += int(np.count_nonzero(states == T2_OBSERVATION_AVAILABLE))
            total += int(states.size)
        return {
            "row_count": total,
            "available_row_count": available,
            "unavailable_row_count": total - available,
        }


def primary_class_counts(
    reader: T2TimelineReader, streams: Sequence[T2Stream]
) -> dict[str, int]:
    """Direct-loss-eligible PRIMARY class counts over a stream selection.

    The counted population is exactly the one that receives gradient: AVAILABLE
    **and** PRIMARY. A physically unavailable row is excluded because it never
    reaches the model, so counting it would put a window that produces no loss
    into the denominator of the weight that scales the loss.
    """
    positive = 0
    negative = 0
    for stream in streams:
        begin, end = stream.start_index, stream.stop_index
        states = np.asarray(reader.timeline.column(OBSERVATION_STATE_FILE, begin, end))
        families = reader.family_codes[begin:end]
        roles = role_codes_for_families(families, states)
        direct = roles == ROLE_CODE_PRIMARY
        labels = primary_labels_for_families(families)
        positive += int(np.count_nonzero(direct & (labels == 1)))
        negative += int(np.count_nonzero(direct & (labels == 0)))
    return {"positive_count": positive, "negative_count": negative}


def fit_class_weight_evidence(
    reader: T2TimelineReader,
    fit_streams: Sequence[T2Stream],
    *,
    fit_subjects: Sequence[str],
    internal_dev_subjects: Sequence[str],
) -> dict[str, Any]:
    """The frozen `pos_weight`, derived from the 48 FIT subjects and nothing else.

    The internal-dev 8 contribute no optimiser gradient at all, so they
    contribute no class count either; neither does outer VALIDATION, which this
    function has no access to.
    """
    subjects = {stream.subject_id for stream in fit_streams}
    leaked = sorted(subjects & set(internal_dev_subjects))
    if leaked:
        raise T2TrainingError(
            f"Internal-dev subjects {leaked} appear in the FIT stream selection; "
            "the class weight would then be derived from held-out rows."
        )
    if subjects != set(fit_subjects):
        raise T2TrainingError(
            "The FIT stream selection does not cover exactly the FIT subjects."
        )
    counts = primary_class_counts(reader, fit_streams)
    weight = positive_class_weight(
        negative_count=counts["negative_count"], positive_count=counts["positive_count"]
    )
    return {
        "evidence_class": "t2_fit_class_weight",
        "partition": "t2_fit_48_subjects",
        "fit_subject_count": len(subjects),
        "fit_stream_count": len(fit_streams),
        "fit_positive_count": counts["positive_count"],
        "fit_negative_count": counts["negative_count"],
        "positive_class_weight": weight,
        "rule": "n_negative_over_n_positive_on_fit_partition",
        "counted_population": "available_primary_direct_loss_rows",
        "internal_dev_rows_counted": False,
        "outer_validation_rows_counted": False,
        "all_train_subjects_counted": False,
    }


# ---------------------------------------------------------------------------
# The frontier training loop
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class T2FrontierStats:
    """One epoch's frontier-scoped optimisation record."""

    frontier_count: int = 0
    optimizer_step_count: int = 0
    zero_direct_loss_frontier_count: int = 0
    direct_loss_row_count: int = 0
    weighted_loss_sum: float = 0.0
    context_row_count: int = 0
    unavailable_row_count: int = 0
    length_group_count: int = 0

    @property
    def mean_weighted_loss_per_direct_row(self) -> float:
        if self.direct_loss_row_count == 0:
            return 0.0
        return self.weighted_loss_sum / float(self.direct_loss_row_count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frontier_count": self.frontier_count,
            "optimizer_step_count": self.optimizer_step_count,
            "zero_direct_loss_frontier_count": self.zero_direct_loss_frontier_count,
            "direct_loss_row_count": self.direct_loss_row_count,
            "weighted_loss_sum": self.weighted_loss_sum,
            "mean_weighted_loss_per_direct_row": (
                self.mean_weighted_loss_per_direct_row
            ),
            "context_row_count": self.context_row_count,
            "unavailable_row_count": self.unavailable_row_count,
            "length_group_count": self.length_group_count,
            "optimizer_steps_per_nonempty_frontier": 1,
            "state_detached_at_every_frontier": True,
            "gradient_crosses_frontier_boundary": False,
        }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    reader: T2TimelineReader,
    streams: Sequence[T2Stream],
    *,
    pos_weight: float,
    device: Any = None,
    tbptt: int = T2_TBPTT_LENGTH,
) -> T2FrontierStats:
    """One complete chronological pass over every FIT stream.

    Every stream starts this epoch at zero state -- a new epoch is another pass
    over the recordings, not a physical continuation of the patient's timeline.

    Per synchronized frontier, in this exact order:

    1. gradients are zeroed;
    2. every active stream's slice is read and compacted to available rows;
    3. slices are grouped by their **compacted** length and forwarded, each
       group carrying its own streams' states;
    4. weighted BCE is summed over the direct-loss rows of every group;
    5. the frontier's single total direct-loss count divides that one sum;
    6. one backward pass, one gradient clip, one `optimizer.step()`;
    7. every carried state is detached.

    Step 5 is the reason the sum is accumulated rather than reduced per group:
    dividing inside a group would weight a PRIMARY window by how many other
    windows happened to share its length.
    """
    if int(tbptt) != T2_TBPTT_LENGTH:
        raise T2TrainingError(f"The frozen TBPTT length is {T2_TBPTT_LENGTH}.")
    if not streams:
        raise T2TrainingError("An epoch needs at least one fitting stream.")
    model.train()
    execution_device = _resolve_device(model, device)
    stats = T2FrontierStats()
    states = initial_states(model, len(streams), device=execution_device)
    frontiers = synchronized_frontiers(reader.stream_row_counts(streams), tbptt=tbptt)

    for frontier in frontiers:
        stats.frontier_count += 1
        optimizer.zero_grad(set_to_none=True)
        views = [
            reader.slice_view(
                item.stream_index,
                streams[item.stream_index],
                item.local_start,
                item.local_stop,
            )
            for item in frontier
        ]
        for view in views:
            stats.context_row_count += view.length
            stats.unavailable_row_count += view.raw_length - view.length

        # A stream whose whole slice is unavailable contributes no observation,
        # so it is not forwarded at all and its state is left exactly as it was.
        active = [view for view in views if view.length > 0]
        grouped: dict[int, list[T2SliceView]] = {}
        for view in active:
            grouped.setdefault(view.length, []).append(view)
        stats.length_group_count += len(grouped)

        total_loss: Tensor | None = None
        total_rows = 0
        for length in sorted(grouped):
            group = grouped[length]
            indices = [view.stream_index for view in group]
            values = torch.from_numpy(
                np.stack([view.values for view in group], axis=0)
            ).to(execution_device)
            logits, _ = run_stream_group(model, values, states, indices)
            # Masks and targets follow the inputs onto the execution device;
            # a CPU mask indexing a CUDA logit tensor would silently move the
            # computation back and make the device provenance a fiction.
            mask = torch.from_numpy(
                np.stack([view.direct_loss for view in group], axis=0)
            ).to(execution_device)
            targets = torch.from_numpy(
                np.stack([view.labels for view in group], axis=0)
            ).to(execution_device)
            partial, count = direct_loss_sum(
                logits, targets, mask, pos_weight=pos_weight
            )
            if count:
                total_loss = partial if total_loss is None else total_loss + partial
                total_rows += count

        if total_rows == 0:
            stats.zero_direct_loss_frontier_count += 1
        else:
            assert total_loss is not None
            loss = total_loss / float(total_rows)
            loss.backward()
            clip_gradients(model)
            optimizer.step()
            stats.optimizer_step_count += 1
            stats.direct_loss_row_count += total_rows
            stats.weighted_loss_sum += float(total_loss.detach().item())

        # After every frontier, loss-bearing or not. A zero-loss frontier that
        # kept its graph would let the next frontier's backward pass reach
        # across the 256-window boundary.
        states = detach_all(states)
    return stats


# ---------------------------------------------------------------------------
# The internal-dev evaluator -- TRAIN-only, and never outer VALIDATION
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T2ScorePass:
    """One complete causal pass's scored rows, PRIMARY mask included."""

    scores: np.ndarray
    labels: np.ndarray
    direct_loss: np.ndarray
    subjects: np.ndarray
    positions: np.ndarray

    @property
    def primary_labels(self) -> list[int]:
        return self.labels[self.direct_loss].astype(int).tolist()

    @property
    def primary_scores(self) -> list[float]:
        return self.scores[self.direct_loss].astype(float).tolist()

    @property
    def primary_subjects(self) -> list[str]:
        return self.subjects[self.direct_loss].astype(str).tolist()

    def score_sha256(self) -> str:
        """Digest of the PRIMARY score vector, in pass order.

        This is how a retained checkpoint is proved to reproduce the exact
        predictions its best-epoch evidence was computed from, without holding
        a second copy of the score vector for the life of the run.
        """
        digest = hashlib.sha256()
        selected = np.ascontiguousarray(self.scores[self.direct_loss], dtype=np.float64)
        digest.update(repr((selected.shape, str(selected.dtype))).encode("utf-8"))
        digest.update(selected.tobytes())
        return digest.hexdigest()


def score_streams(
    model: nn.Module,
    reader: T2TimelineReader,
    streams: Sequence[T2Stream],
    *,
    device: Any = None,
    tbptt: int = T2_TBPTT_LENGTH,
) -> T2ScorePass:
    """One complete causal timeline pass. No loss, no optimiser, no gradient.

    Each stream starts at zero state and is walked in bounded slices with the
    state carried, which is the same recurrence a single unbroken pass performs.
    Challenge and other non-primary rows stay in the pass as label-blind causal
    context; only their scores are excluded from the PRIMARY mask afterwards.

    An unavailable row is skipped entirely, so it receives no score and leaves
    the state untouched -- exactly the frozen no-op semantics.
    """
    model.eval()
    execution_device = _resolve_device(model, device)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    direct: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    with torch.no_grad():
        for stream in streams:
            state = model.initial_state(1, device=execution_device)
            for local_start in range(0, stream.row_count, int(tbptt)):
                local_stop = min(local_start + int(tbptt), stream.row_count)
                view = reader.slice_view(0, stream, local_start, local_stop)
                if view.length == 0:
                    continue
                values = torch.from_numpy(view.values[None, ...]).to(execution_device)
                logits, state = model(values, state)
                scores.append(
                    torch.sigmoid(logits[0]).to(torch.float64).cpu().numpy().copy()
                )
                labels.append(view.labels)
                direct.append(view.direct_loss)
                subjects.append(np.full(view.length, stream.subject_id, dtype="<U32"))
                positions.append(view.positions + stream.start_index + local_start)
    if not scores:
        raise T2TrainingError(
            "A causal scoring pass produced no scored row; the selection holds "
            "no available observation."
        )
    return T2ScorePass(
        scores=np.concatenate(scores),
        labels=np.concatenate(labels),
        direct_loss=np.concatenate(direct),
        subjects=np.concatenate(subjects),
        positions=np.concatenate(positions),
    )


def evaluate_internal_development(
    model: nn.Module,
    reader: T2TimelineReader,
    streams: Sequence[T2Stream],
    *,
    internal_dev_subjects: Sequence[str],
    device: Any = None,
    tbptt: int = T2_TBPTT_LENGTH,
) -> tuple[float, T2ScorePass]:
    """Pooled internal-dev PRIMARY AUPRC -- the one checkpoint criterion.

    TRAIN-only. It reads the same frozen TRAIN timeline and the same target
    authority the fitting streams came from, restricted by subject identity to
    the frozen internal-dev 8. Outer VALIDATION is not reachable from here.
    """
    wanted = set(internal_dev_subjects)
    offending = sorted({stream.subject_id for stream in streams} - wanted)
    if offending:
        raise T2TrainingError(
            f"Internal-dev evaluation was offered streams from {offending}, "
            "which are not the frozen internal-dev subjects."
        )
    if not streams:
        raise T2TrainingError("Internal-dev evaluation needs at least one stream.")
    scored = score_streams(model, reader, streams, device=device, tbptt=tbptt)
    return pooled_auprc(scored.primary_labels, scored.primary_scores), scored


# ---------------------------------------------------------------------------
# Epoch and checkpoint bookkeeping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class T2EpochResult:
    """One completed epoch's persisted evidence."""

    epoch: int
    optimizer_step_count: int
    zero_direct_loss_frontier_count: int
    direct_loss_row_count: int
    weighted_loss_sum: float
    mean_weighted_loss_per_direct_row: float
    internal_dev_pooled_auprc: float
    frontier_count: int = 0
    length_group_count: int = 0

    @classmethod
    def from_stats(
        cls, epoch: int, stats: T2FrontierStats, internal_dev_pooled_auprc: float
    ) -> T2EpochResult:
        return cls(
            epoch=int(epoch),
            optimizer_step_count=stats.optimizer_step_count,
            zero_direct_loss_frontier_count=stats.zero_direct_loss_frontier_count,
            direct_loss_row_count=stats.direct_loss_row_count,
            weighted_loss_sum=stats.weighted_loss_sum,
            mean_weighted_loss_per_direct_row=(stats.mean_weighted_loss_per_direct_row),
            internal_dev_pooled_auprc=float(internal_dev_pooled_auprc),
            frontier_count=stats.frontier_count,
            length_group_count=stats.length_group_count,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "optimizer_step_count": self.optimizer_step_count,
            "zero_direct_loss_frontier_count": self.zero_direct_loss_frontier_count,
            "direct_loss_row_count": self.direct_loss_row_count,
            "weighted_loss_sum": self.weighted_loss_sum,
            "mean_weighted_loss_per_direct_row": (
                self.mean_weighted_loss_per_direct_row
            ),
            "internal_dev_pooled_auprc": self.internal_dev_pooled_auprc,
            "frontier_count": self.frontier_count,
            "length_group_count": self.length_group_count,
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


@dataclass(slots=True)
class T2RetainedCheckpoint:
    """The one state retained for an arm, and the evidence it was chosen on."""

    epoch: int
    internal_dev_pooled_auprc: float
    state_dict: dict[str, Tensor]
    internal_dev_score_sha256: str
    internal_dev_primary_row_count: int


def capture_model_state(model: nn.Module) -> dict[str, Tensor]:
    """A detached, cloned, CPU copy of the parameters at this exact epoch.

    Cloning matters: `state_dict()` returns live tensor references, so keeping
    it without a copy would silently track the next epoch's updates and the
    "retained" checkpoint would be whatever the model ended on.
    """
    return {
        key: value.detach().to("cpu").clone()
        for key, value in model.state_dict().items()
    }


def restore_model_state(
    arm: str, state_dict: dict[str, Tensor], *, device: Any = None
) -> nn.Module:
    """Rebuild the frozen architecture and load the retained state into it.

    The checkpoint itself is CPU tensors by design, so it can be promoted and
    reloaded anywhere; the rebuilt model is then moved onto the canonical
    execution device before it scores anything.
    """
    model = build_candidate(arm)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:  # pragma: no cover - strict=True already raises
        raise T2TrainingError(
            f"The retained {arm} state does not match the frozen architecture: "
            f"missing {list(missing)}, unexpected {list(unexpected)}."
        )
    if device is not None:
        model.to(torch.device(device))
    model.eval()
    return model


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


# ---------------------------------------------------------------------------
# One arm, end to end
# ---------------------------------------------------------------------------


def train_arm(
    arm: str,
    reader: T2TimelineReader,
    *,
    fit_streams: Sequence[T2Stream],
    internal_dev_streams: Sequence[T2Stream],
    internal_dev_subjects: Sequence[str],
    pos_weight: float,
    device: Any = None,
    before_model_construction: Any = None,
    on_model_constructed: Any = None,
    max_epochs: int = T2_MAX_EPOCHS,
    tbptt: int = T2_TBPTT_LENGTH,
) -> dict[str, Any]:
    """Train one frozen candidate: epochs, early stop, checkpoint, threshold.

    The choreography is the frozen one and nothing here reorders it:

    1. the runtime identity is observed **before** the model is constructed;
    2. the arm is reseeded and built, its parameter count asserted;
    3. each epoch is one full chronological FIT pass from zero state;
    4. after every completed epoch, internal-dev pooled PRIMARY AUPRC is scored
       and offered to the selector, which decides improvement, ties and
       patience;
    5. an improving epoch's state is captured immediately, so the retained
       checkpoint is that epoch's and not the last one trained;
    6. after training stops, the retained state is **reloaded** and scored once
       more, and only that pass derives the internal-dev max-F1 threshold.

    Step 6 reloads rather than reusing the live model deliberately: it proves
    the bytes that will be promoted are the bytes that produce the evidence.
    """
    if int(max_epochs) != T2_MAX_EPOCHS:
        raise T2TrainingError(f"The frozen epoch ceiling is {T2_MAX_EPOCHS}.")
    if before_model_construction is not None:
        before_model_construction(arm)
    model = build_candidate(arm)
    execution_device = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    model.to(execution_device)
    if on_model_constructed is not None:
        # The caller observes the model's REAL parameter device here, so a
        # provenance record claiming another one is refused before any
        # optimiser step rather than after promotion.
        on_model_constructed(arm, model)
    optimizer = build_optimizer(model)
    selector = T2CheckpointSelector()
    retained: T2RetainedCheckpoint | None = None
    epoch_stats: list[dict[str, Any]] = []

    for epoch in range(1, int(max_epochs) + 1):
        stats = train_one_epoch(
            model,
            optimizer,
            reader,
            fit_streams,
            pos_weight=pos_weight,
            device=execution_device,
            tbptt=tbptt,
        )
        auprc, scored = evaluate_internal_development(
            model,
            reader,
            internal_dev_streams,
            internal_dev_subjects=internal_dev_subjects,
            device=execution_device,
            tbptt=tbptt,
        )
        result = T2EpochResult.from_stats(epoch, stats, auprc)
        epoch_stats.append(result.as_dict())
        if selector.offer(result):
            retained = T2RetainedCheckpoint(
                epoch=epoch,
                internal_dev_pooled_auprc=auprc,
                state_dict=capture_model_state(model),
                internal_dev_score_sha256=scored.score_sha256(),
                internal_dev_primary_row_count=len(scored.primary_labels),
            )
        if selector.should_stop:
            break

    if retained is None:  # pragma: no cover - the first epoch always improves
        raise T2TrainingError(f"No {arm} epoch produced a retainable checkpoint.")
    if retained.epoch != selector.best_epoch:
        raise T2TrainingError(
            f"The retained {arm} checkpoint is from epoch {retained.epoch} but "
            f"the selector chose {selector.best_epoch}."
        )

    reloaded = restore_model_state(arm, retained.state_dict, device=execution_device)
    threshold_scored = score_streams(
        reloaded, reader, internal_dev_streams, device=execution_device, tbptt=tbptt
    )
    if threshold_scored.score_sha256() != retained.internal_dev_score_sha256:
        raise T2TrainingError(
            f"The retained {arm} checkpoint does not reproduce the internal-dev "
            "predictions its best-epoch evidence was computed from. This is a "
            "hard failure: the promoted bytes and the promoted evidence would "
            "not describe the same model."
        )
    threshold = internal_dev_threshold_evidence(
        threshold_scored.primary_labels, threshold_scored.primary_scores
    )
    return {
        "arm": arm,
        "state_dict": retained.state_dict,
        "best_epoch": retained.epoch,
        "best_internal_dev_pooled_auprc": retained.internal_dev_pooled_auprc,
        "internal_dev_score_sha256": retained.internal_dev_score_sha256,
        "internal_dev_primary_row_count": retained.internal_dev_primary_row_count,
        "epochs": epoch_stats,
        "epochs_completed": len(epoch_stats),
        "early_stopped": selector.should_stop,
        "checkpoint_selection": selector.as_dict(),
        "internal_dev_threshold": threshold,
        "threshold_derived_from_best_checkpoint": True,
        "threshold_derived_during_epoch_selection": False,
        "threshold_derived_from_outer_validation": False,
        "positive_class_weight": float(pos_weight),
        "execution_device": str(execution_device),
        "model_parameter_device": str(next(reloaded.parameters()).device),
        "fit_stream_count": len(fit_streams),
        "internal_dev_stream_count": len(internal_dev_streams),
    }
