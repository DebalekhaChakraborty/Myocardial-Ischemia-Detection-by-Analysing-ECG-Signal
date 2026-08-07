"""Strict deterministic initialization for prospective B4 execution."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from cardiosentinel.neural.protocol import SEED

CUDA_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True, slots=True)
class DeterminismState:
    seed: int
    device: str
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_workspace_config: str | None


def initialize_determinism(
    *, seed: int = SEED, requested_device: str | None = None
) -> DeterminismState:
    """Set every frozen seed and reject unsupported CUDA workspace settings."""
    if seed != SEED:
        raise ValueError(f"B4 uses the frozen seed {SEED}.")
    device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("B4 device must be cpu or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device == "cuda":
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing not in {None, CUDA_WORKSPACE_CONFIG}:
            raise ValueError("CUBLAS_WORKSPACE_CONFIG conflicts with B4 determinism.")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUDA_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return DeterminismState(
        seed=seed,
        device=device,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def seed_worker(worker_id: int) -> None:
    """Derive NumPy and Python worker seeds from PyTorch's deterministic seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_order_generator(*, seed: int = SEED) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
