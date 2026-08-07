"""Prospective in-memory B4 artifact schema; no experiment lock is written here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProspectiveB4RunRecord:
    protocol_sha256: str
    source_provenance: dict[str, Any]
    split_sha256: str
    training_selection_sha256: str
    git_sha: str
    git_dirty: bool
    environment: dict[str, Any]
    device: str
    model_config: dict[str, Any]
    trainable_parameter_count: int
    seed: int
    epoch_history: tuple[dict[str, Any], ...]
    selected_checkpoint_epoch: int | None
    validation_auprc: float | None
    threshold: float | None
    checkpoint_sha256: str | None
    checkpoint_serialized_bytes: int | None
    timings: dict[str, float]
    test: None = None
