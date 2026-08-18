"""The two frozen T2 causal longitudinal candidates, translated from protocol.

This module **translates** `docs/T2_LONGITUDINAL_TEMPORAL_PROTOCOL_V1.md` §13A
and §13B into PyTorch. It makes no scientific choice: every width, depth, state
dimension, activation, normalisation location, bias, dropout position,
initialiser and dtype is read from `t2_protocol`, and both constructors assert
the frozen trainable parameter count rather than discovering it.

The one thing worth understanding before reading further is **state carry**.
B4-C's `DiagonalGatedSSMBlock` created its recurrent state inside `forward` and
discarded it on return, because it modelled a fixed token sequence inside one
10-second window. T2 is an across-window stream model: both candidates here
accept an incoming state and return the outgoing state, so a stream can be
processed in TBPTT chunks that are numerically identical to one continuous pass.

A row that is physically unavailable never reaches these modules at all. There
is deliberately no "missing" embedding, no zero vector and no mask token: the
caller feeds runs of available observations and carries the state across the
gap unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import Tensor, nn

from cardiosentinel.neural.t2_protocol import (
    T2_ARM_GRU,
    T2_ARM_S4D,
    T2_ARMS,
    T2_DROPOUT,
    T2_EXPECTED_PARAMETER_COUNTS,
    T2_INPUT_DIM,
    T2_INPUT_PROJECTION_DIM,
    T2_LAYER_NORM_EPS,
    T2_OUTPUT_DIM,
    T2_S4D_SPEC,
    T2_SEED,
    T2_TEMPORAL_LAYERS,
    T2_TEMPORAL_WIDTH,
    T2ProtocolError,
    require_arm,
)

S4D_STATE_DIM: Final = int(T2_S4D_SPEC["state_dim"])
_LOG_STEP_MIN: Final = math.log(1e-3)
_LOG_STEP_MAX: Final = math.log(1e-1)


class T2ModelError(RuntimeError):
    """Raised when a constructed T2 candidate departs from the frozen spec."""


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass(frozen=True, slots=True)
class T2Scaffold:
    """The shared, identical outer scaffolding of both arms."""

    input_dim: int = T2_INPUT_DIM
    width: int = T2_TEMPORAL_WIDTH
    layers: int = T2_TEMPORAL_LAYERS
    dropout: float = T2_DROPOUT
    output_dim: int = T2_OUTPUT_DIM


def _build_scaffold(module: nn.Module) -> None:
    """Attach the frozen projection, final norm and readout. No activation."""
    module.input_projection = nn.Linear(
        T2_INPUT_DIM, T2_INPUT_PROJECTION_DIM, bias=True
    )
    module.final_norm = nn.LayerNorm(
        T2_TEMPORAL_WIDTH, eps=T2_LAYER_NORM_EPS, elementwise_affine=True
    )
    module.readout = nn.Linear(T2_TEMPORAL_WIDTH, T2_OUTPUT_DIM, bias=True)


def _validate_sequence(values: Tensor) -> tuple[int, int]:
    if values.dim() != 3:
        raise T2ModelError(
            f"A T2 candidate expects [batch, steps, {T2_INPUT_DIM}]; got "
            f"{tuple(values.shape)}."
        )
    batch, steps, dim = values.shape
    if dim != T2_INPUT_DIM:
        raise T2ModelError(
            f"T2 consumes the frozen {T2_INPUT_DIM}-dimensional representation; "
            f"got {dim}."
        )
    if values.dtype != torch.float32:
        raise T2ModelError(f"T2 input must be float32; got {values.dtype}.")
    if steps == 0:
        raise T2ModelError("A T2 forward pass needs at least one timestep.")
    return batch, steps


# ---------------------------------------------------------------------------
# T2-A: causal_gru_longitudinal_v1
# ---------------------------------------------------------------------------


class CausalGRULongitudinal(nn.Module):
    """`causal_gru_longitudinal_v1` -- the conventional recurrent comparator.

    `nn.GRU(dropout=0.10, num_layers=2)` applies dropout to the output of every
    layer *except the last*, so it acts exactly once here, between layer 1 and
    layer 2. That is the frozen semantics, not an incidental detail.
    """

    architecture: Final = T2_ARM_GRU

    def __init__(self) -> None:
        super().__init__()
        _build_scaffold(self)
        self.temporal = nn.GRU(
            input_size=T2_INPUT_PROJECTION_DIM,
            hidden_size=T2_TEMPORAL_WIDTH,
            num_layers=T2_TEMPORAL_LAYERS,
            bias=True,
            batch_first=True,
            dropout=T2_DROPOUT,
            bidirectional=False,
        )

    def initial_state(self, batch: int, *, device: Any = None) -> Tensor:
        """The frozen zero state at every real stream start."""
        return torch.zeros(T2_TEMPORAL_LAYERS, batch, T2_TEMPORAL_WIDTH, device=device)

    def forward(
        self, values: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        batch, _ = _validate_sequence(values)
        if state is None:
            state = self.initial_state(batch, device=values.device)
        if tuple(state.shape) != (T2_TEMPORAL_LAYERS, batch, T2_TEMPORAL_WIDTH):
            raise T2ModelError(
                f"GRU state must be "
                f"{(T2_TEMPORAL_LAYERS, batch, T2_TEMPORAL_WIDTH)}; got "
                f"{tuple(state.shape)}."
            )
        projected = self.input_projection(values)
        represented, next_state = self.temporal(projected, state)
        logits = self.readout(self.final_norm(represented)).squeeze(-1)
        return logits, next_state


# ---------------------------------------------------------------------------
# T2-B: causal_s4d_longitudinal_v1
# ---------------------------------------------------------------------------


class LongitudinalDiagonalSSMBlock(nn.Module):
    """One S4D-inspired diagonal gated block, carrying state across windows.

    This is **not** Mamba: the diagonal transition is time-invariant and
    input-independent, so there is no selective mechanism. The parameterisation,
    discretisation and every initialiser are B4-C's, reused verbatim; the only
    change is that the recurrent state enters and leaves the block instead of
    being created and discarded inside it.
    """

    def __init__(self) -> None:
        super().__init__()
        width = T2_TEMPORAL_WIDTH
        self.norm = nn.LayerNorm(width, eps=T2_LAYER_NORM_EPS)
        self.input_projection = nn.Linear(width, 2 * width, bias=True)
        self.output_projection = nn.Linear(width, width, bias=True)
        self.dropout = nn.Dropout(T2_DROPOUT)
        shape = (width, S4D_STATE_DIM)
        self.log_decay = nn.Parameter(torch.empty(shape))
        self.frequency = nn.Parameter(torch.empty(shape))
        self.state_input = nn.Parameter(torch.empty(shape))
        self.state_output_real = nn.Parameter(torch.empty(shape))
        self.state_output_imaginary = nn.Parameter(torch.empty(shape))
        self.skip = nn.Parameter(torch.empty(width))
        self.log_step = nn.Parameter(torch.empty(width))
        self.reset_frozen_parameters()

    def reset_frozen_parameters(self) -> None:
        """Apply the exact frozen initialisation; no substitute initialiser."""
        with torch.no_grad():
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)
            self.log_decay.fill_(math.log(0.5))
            grid = math.pi * torch.arange(
                1, S4D_STATE_DIM + 1, dtype=self.frequency.dtype
            )
            self.frequency.copy_(grid.expand(T2_TEMPORAL_WIDTH, S4D_STATE_DIM))
            self.log_step.uniform_(_LOG_STEP_MIN, _LOG_STEP_MAX)
            scale = 1.0 / math.sqrt(S4D_STATE_DIM)
            for parameter in (
                self.state_input,
                self.state_output_real,
                self.state_output_imaginary,
            ):
                parameter.normal_(mean=0.0, std=1.0).mul_(scale)
            self.skip.zero_()

    def discrete_state_space(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return `(Abar, Bbar, C)` as complex from the real float32 parameters.

        `lambda = complex(-exp(log_decay), frequency)` has a strictly negative
        real part by construction, which is the stability constraint. `zeta =
        exp(log_step) * lambda`, and `expm1` keeps relative precision for the
        small `|zeta|` that small learned steps produce.
        """
        lam = torch.complex(-torch.exp(self.log_decay), self.frequency)
        zeta = torch.exp(self.log_step).unsqueeze(-1) * lam
        transition = torch.exp(zeta)
        input_gain = torch.expm1(zeta) / lam * self.state_input.to(lam.dtype)
        output = torch.complex(self.state_output_real, self.state_output_imaginary)
        return transition, input_gain, output

    def initial_state(self, batch: int, *, device: Any = None) -> Tensor:
        transition, _, _ = self.discrete_state_space()
        return torch.zeros(
            batch,
            T2_TEMPORAL_WIDTH,
            S4D_STATE_DIM,
            dtype=transition.dtype,
            device=device,
        )

    def forward(
        self, tokens: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        batch, steps, _ = tokens.shape
        transition, input_gain, output = self.discrete_state_space()
        if state is None:
            state = torch.zeros(
                batch,
                T2_TEMPORAL_WIDTH,
                S4D_STATE_DIM,
                dtype=transition.dtype,
                device=tokens.device,
            )
        normed = self.norm(tokens)
        value, gate = self.input_projection(normed).chunk(2, dim=-1)

        transition = transition.unsqueeze(0)
        input_gain = input_gain.unsqueeze(0)
        readout = output.unsqueeze(0)
        collected: list[Tensor] = []
        for step in range(steps):
            current = value[:, step, :]
            state = transition * state + input_gain * current.unsqueeze(-1).to(
                transition.dtype
            )
            collected.append((readout * state).real.sum(-1) + self.skip * current)
        represented = torch.stack(collected, dim=1)
        branch = self.output_projection(represented * torch.nn.functional.silu(gate))
        return tokens + self.dropout(branch), state


class CausalS4DLongitudinal(nn.Module):
    """`causal_s4d_longitudinal_v1` -- compact diagonal state space across windows."""

    architecture: Final = T2_ARM_S4D

    def __init__(self) -> None:
        super().__init__()
        _build_scaffold(self)
        self.blocks = nn.ModuleList(
            LongitudinalDiagonalSSMBlock() for _ in range(T2_TEMPORAL_LAYERS)
        )
        with torch.no_grad():
            nn.init.ones_(self.final_norm.weight)
            nn.init.zeros_(self.final_norm.bias)

    def initial_state(self, batch: int, *, device: Any = None) -> tuple[Tensor, ...]:
        return tuple(block.initial_state(batch, device=device) for block in self.blocks)

    def forward(
        self, values: Tensor, state: tuple[Tensor, ...] | None = None
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        batch, _ = _validate_sequence(values)
        if state is None:
            state = self.initial_state(batch, device=values.device)
        if len(state) != len(self.blocks):
            raise T2ModelError(
                f"S4D state carries {len(state)} block states; the model has "
                f"{len(self.blocks)}."
            )
        tokens = self.input_projection(values)
        outgoing: list[Tensor] = []
        for block, block_state in zip(self.blocks, state, strict=True):
            tokens, next_state = block(tokens, block_state)
            outgoing.append(next_state)
        logits = self.readout(self.final_norm(tokens)).squeeze(-1)
        return logits, tuple(outgoing)


# ---------------------------------------------------------------------------
# Construction, with the frozen parameter count asserted, not discovered
# ---------------------------------------------------------------------------


def detach_state(state: Any) -> Any:
    """Sever gradient history while preserving the carried values exactly."""
    if isinstance(state, torch.Tensor):
        return state.detach()
    return tuple(detach_state(item) for item in state)


def seed_everything(seed: int = T2_SEED) -> dict[str, Any]:
    """Reset every relevant generator so arm construction order cannot matter.

    Each candidate is built from a fresh identical seed origin. Constructing the
    GRU, consuming RNG, then constructing the S4D would make the S4D's
    initialisation depend on the other arm existing.
    """
    import os
    import random

    import numpy as np

    if int(seed) != T2_SEED:
        raise T2ModelError(f"T2 uses the frozen seed {T2_SEED}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in this runtime
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as error:  # pragma: no cover - runtime dependent
        raise T2ModelError(
            "Deterministic algorithms could not be enabled in this runtime. "
            "This is a STOP condition: canonical T2 training does not proceed "
            "with determinism silently disabled, and nothing is installed."
        ) from error
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": int(seed),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def build_t2_model(arm: str, *, seed: int = T2_SEED) -> nn.Module:
    """Construct one frozen candidate from a fresh seed origin.

    The parameter count is asserted against the protocol, following the B4-C
    convention: a model that does not have exactly the frozen count is not the
    specified model, and construction STOPS rather than continuing.
    """
    require_arm(arm)
    seed_everything(seed)
    model = CausalGRULongitudinal() if arm == T2_ARM_GRU else CausalS4DLongitudinal()
    observed = trainable_parameter_count(model)
    expected = T2_EXPECTED_PARAMETER_COUNTS[arm]
    if observed != expected:
        raise T2ModelError(
            f"{arm} constructed {observed} trainable parameters, but the frozen "
            f"protocol specifies {expected}. The constructed model is not the "
            "specified model; nothing proceeds."
        )
    if arm == T2_ARM_GRU and model.temporal.bidirectional:
        raise T2ModelError("The T2 GRU is causal; bidirectional is forbidden.")
    return model


def model_identity(model: nn.Module) -> dict[str, Any]:
    """The identity a training artifact binds for a constructed candidate."""
    arm = getattr(model, "architecture", None)
    if arm not in T2_ARMS:
        raise T2ModelError(f"{arm!r} is not a frozen T2 candidate.")
    return {
        "architecture": arm,
        "trainable_parameters": trainable_parameter_count(model),
        "expected_trainable_parameters": T2_EXPECTED_PARAMETER_COUNTS[arm],
        "input_dim": T2_INPUT_DIM,
        "width": T2_TEMPORAL_WIDTH,
        "layers": T2_TEMPORAL_LAYERS,
        "dropout": T2_DROPOUT,
        "state_dim": S4D_STATE_DIM if arm == T2_ARM_S4D else None,
        "is_mamba": False,
        "selective_mechanism": False,
        "bidirectional": False,
        "parameter_dtype": "float32",
    }


def require_frozen_arm(arm: str) -> str:
    try:
        return require_arm(arm)
    except T2ProtocolError as error:  # pragma: no cover - re-raised for clarity
        raise T2ModelError(str(error)) from error
