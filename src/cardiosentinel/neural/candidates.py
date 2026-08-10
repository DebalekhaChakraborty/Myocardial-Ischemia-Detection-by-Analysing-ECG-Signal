"""Frozen B4-B and B4-C architecture candidates.

Both candidates are frozen by `docs/B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.md`.
They reuse the exact B4-A convolutional front end and classifier head, so the
only difference between B4-A, B4-B and B4-C is the temporal block that consumes
the 79x128 token sequence.

`B4CompactCNN` is historical frozen evidence and is not modified here. The shared
components below rebuild the same layer structure from the same low-level block
so that B4-A's module hierarchy, `state_dict` names, parameter count and locked
checkpoint remain untouched.

Neither candidate consumes subject, record, channel, context, handcrafted,
physiology, patient-memory or cross-window information. Neither carries state
between windows. No sigmoid is part of either model.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from cardiosentinel.neural.model import (
    DepthwiseSeparableBlock,
    fp32_parameter_payload_bytes,
    trainable_parameter_count,
)
from cardiosentinel.neural.protocol import WINDOW_SAMPLES

B4B_EXPERIMENT_ID = "B4B_cnn_transformer_v1"
B4C_EXPERIMENT_ID = "B4C_cnn_ssm_v1"

MODEL_DIM = 128
TOKENS = 79
ATTENTION_HEADS = 4
ATTENTION_HEAD_DIM = 32
FEED_FORWARD_DIM = 256
TRANSFORMER_BLOCKS = 2
SSM_BLOCKS = 2
SSM_STATE_DIM = 16
BRANCH_DROPOUT = 0.10
ATTENTION_DROPOUT = 0.0
LAYER_NORM_EPS = 1e-5
POSITIONAL_INIT_STD = 0.02

SHARED_FRONT_END_PARAMETERS = 26_160
SHARED_HEAD_PARAMETERS = 8_321
B4B_TRAINABLE_PARAMETERS = 309_809
B4B_FP32_PARAMETER_BYTES = 1_239_236
B4C_TRAINABLE_PARAMETERS = 155_313
B4C_FP32_PARAMETER_BYTES = 621_252

_LOG_STEP_MIN = math.log(1e-3)
_LOG_STEP_MAX = math.log(1e-1)


def _validate_waveform_input(waveforms: Tensor) -> None:
    """Apply the frozen shared input contract, identical to B4-A."""
    if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (1, WINDOW_SAMPLES):
        raise ValueError("B4 candidate input must have shape [B, 1, 2500].")
    if waveforms.dtype != torch.float32:
        raise ValueError("B4 candidate input must use torch.float32.")
    if not torch.isfinite(waveforms).all():
        raise ValueError("B4 candidate input must contain finite physical-mV values.")


class SharedLocalFrontEnd(nn.Module):
    """The exact frozen B4-A stem and four downsampling blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, 32, eps=LAYER_NORM_EPS, affine=True),
            nn.SiLU(),
        )
        self.downsampling = nn.ModuleList(
            (
                DepthwiseSeparableBlock(32, 48, 9),
                DepthwiseSeparableBlock(48, 64, 7),
                DepthwiseSeparableBlock(64, 96, 5),
                DepthwiseSeparableBlock(96, 128, 5),
            )
        )

    def forward(self, waveforms: Tensor) -> Tensor:
        values = self.stem(waveforms)
        for block in self.downsampling:
            values = block(values)
        return values


class SharedClassifierHead(nn.Module):
    """The exact frozen B4-A pooled classifier returning one raw logit."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(BRANCH_DROPOUT),
            nn.Linear(MODEL_DIM, 64),
            nn.SiLU(),
            nn.Dropout(BRANCH_DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        pooled = self.pool(values).squeeze(-1)
        return self.head(pooled).squeeze(-1)


class PreNormTransformerBlock(nn.Module):
    """Pre-norm block with exactly one dropout site per residual branch.

    A stock `TransformerEncoderLayer` is deliberately avoided: its defaults fuse
    several dropout sites, which would violate the frozen requirement that the
    only block dropout is the external residual branch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(MODEL_DIM, eps=LAYER_NORM_EPS)
        self.attention = nn.MultiheadAttention(
            embed_dim=MODEL_DIM,
            num_heads=ATTENTION_HEADS,
            dropout=ATTENTION_DROPOUT,
            bias=True,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(BRANCH_DROPOUT)
        self.norm_feed_forward = nn.LayerNorm(MODEL_DIM, eps=LAYER_NORM_EPS)
        self.feed_forward = nn.Sequential(
            nn.Linear(MODEL_DIM, FEED_FORWARD_DIM),
            nn.GELU(),
            nn.Linear(FEED_FORWARD_DIM, MODEL_DIM),
        )
        self.feed_forward_dropout = nn.Dropout(BRANCH_DROPOUT)

    def forward(self, tokens: Tensor) -> Tensor:
        normed = self.norm_attention(tokens)
        # Full bidirectional attention across the completed window: no
        # attention mask and no key padding mask are ever supplied.
        attended, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
        )
        hidden = tokens + self.attention_dropout(attended)
        projected = self.feed_forward(self.norm_feed_forward(hidden))
        return hidden + self.feed_forward_dropout(projected)


class B4BTransformerCNN(nn.Module):
    """B4-B: shared convolutional front end plus a tiny Transformer encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.front_end = SharedLocalFrontEnd()
        self.positional_embedding = nn.Parameter(torch.empty(TOKENS, MODEL_DIM))
        self.blocks = nn.ModuleList(
            PreNormTransformerBlock() for _ in range(TRANSFORMER_BLOCKS)
        )
        self.final_norm = nn.LayerNorm(MODEL_DIM, eps=LAYER_NORM_EPS)
        self.classifier = SharedClassifierHead()
        self.reset_frozen_parameters()

    def reset_frozen_parameters(self) -> None:
        """Apply the frozen initializations the protocol specifies explicitly."""
        nn.init.normal_(self.positional_embedding, mean=0.0, std=POSITIONAL_INIT_STD)
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, waveforms: Tensor) -> Tensor:
        """Return the pooled `[B, MODEL_DIM]` representation, no classification.

        This is the encoder output immediately before the classifier MLP: after
        the final LayerNorm and adaptive average pooling, and before the head's
        first dropout site. It is the representation downstream development work
        (P1 physiology fusion, later patient memory) consumes, so it is exposed
        explicitly rather than reconstructed by callers.

        `forward` is defined in terms of this method, so the two can never drift.
        """
        _validate_waveform_input(waveforms)
        values = self.front_end(waveforms)
        tokens = values.transpose(1, 2) + self.positional_embedding
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        return self.classifier.pool(tokens.transpose(1, 2)).squeeze(-1)

    def forward(self, waveforms: Tensor) -> Tensor:
        return self.classifier.head(self.encode(waveforms)).squeeze(-1)


class DiagonalGatedSSMBlock(nn.Module):
    """S4D-inspired diagonal gated state-space block.

    This is **not** Mamba: the state transition is diagonal, time-invariant and
    input-independent, so there is no selective mechanism. The linear SSM core is
    LTI; the complete block is not, because of LayerNorm, the learned
    projections, SiLU gating and the residual path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(MODEL_DIM, eps=LAYER_NORM_EPS)
        self.input_projection = nn.Linear(MODEL_DIM, 2 * MODEL_DIM)
        self.output_projection = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.dropout = nn.Dropout(BRANCH_DROPOUT)
        shape = (MODEL_DIM, SSM_STATE_DIM)
        self.log_decay = nn.Parameter(torch.empty(shape))
        self.frequency = nn.Parameter(torch.empty(shape))
        self.state_input = nn.Parameter(torch.empty(shape))
        self.state_output_real = nn.Parameter(torch.empty(shape))
        self.state_output_imaginary = nn.Parameter(torch.empty(shape))
        self.skip = nn.Parameter(torch.empty(MODEL_DIM))
        self.log_step = nn.Parameter(torch.empty(MODEL_DIM))
        self.reset_frozen_parameters()

    def reset_frozen_parameters(self) -> None:
        """Apply the exact frozen SSM initialization; no substitute initializer."""
        with torch.no_grad():
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)
            self.log_decay.fill_(math.log(0.5))
            grid = math.pi * torch.arange(
                1, SSM_STATE_DIM + 1, dtype=self.frequency.dtype
            )
            self.frequency.copy_(grid.expand(MODEL_DIM, SSM_STATE_DIM))
            self.log_step.uniform_(_LOG_STEP_MIN, _LOG_STEP_MAX)
            scale = 1.0 / math.sqrt(SSM_STATE_DIM)
            for parameter in (
                self.state_input,
                self.state_output_real,
                self.state_output_imaginary,
            ):
                parameter.normal_(mean=0.0, std=1.0).mul_(scale)
            self.skip.zero_()

    def discrete_state_space(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return `(Abar, Bbar, C)` as complex64 from the real float32 parameters."""
        lam = torch.complex(-torch.exp(self.log_decay), self.frequency)
        zeta = torch.exp(self.log_step).unsqueeze(-1) * lam
        transition = torch.exp(zeta)
        # expm1 is algebraically identical to exp(z) - 1 but retains relative
        # precision for the small |z| produced by small learned steps.
        input_gain = torch.expm1(zeta) / lam * self.state_input.to(lam.dtype)
        output = torch.complex(
            self.state_output_real, self.state_output_imaginary
        )
        return transition, input_gain, output

    def state_space(self, sequence: Tensor) -> Tensor:
        """Run the frozen discrete recurrence over the fixed token sequence.

        The state is created here and discarded on return, so nothing is carried
        between windows or between calls.
        """
        transition, input_gain, output = self.discrete_state_space()
        batch, steps, _ = sequence.shape
        state = torch.zeros(
            batch, MODEL_DIM, SSM_STATE_DIM, dtype=transition.dtype,
            device=sequence.device,
        )
        transition = transition.unsqueeze(0)
        input_gain = input_gain.unsqueeze(0)
        output = output.unsqueeze(0)
        collected: list[Tensor] = []
        for step in range(steps):
            current = sequence[:, step, :].unsqueeze(-1).to(transition.dtype)
            state = transition * state + input_gain * current
            collected.append(
                (output * state).real.sum(-1) + self.skip * sequence[:, step, :]
            )
        return torch.stack(collected, dim=1)

    def forward(self, tokens: Tensor) -> Tensor:
        normed = self.norm(tokens)
        projected, gate = self.input_projection(normed).chunk(2, dim=-1)
        represented = self.state_space(projected)
        combined = self.output_projection(
            represented * torch.nn.functional.silu(gate)
        )
        return tokens + self.dropout(combined)


class B4CSSMCNN(nn.Module):
    """B4-C: shared convolutional front end plus compact diagonal SSM blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.front_end = SharedLocalFrontEnd()
        self.blocks = nn.ModuleList(
            DiagonalGatedSSMBlock() for _ in range(SSM_BLOCKS)
        )
        self.final_norm = nn.LayerNorm(MODEL_DIM, eps=LAYER_NORM_EPS)
        self.classifier = SharedClassifierHead()
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

    def forward(self, waveforms: Tensor) -> Tensor:
        _validate_waveform_input(waveforms)
        values = self.front_end(waveforms)
        # No positional embedding: the recurrence is inherently ordered.
        tokens = values.transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        return self.classifier(tokens.transpose(1, 2))


def _identity(
    model: nn.Module,
    architecture: str,
    experiment_id: str,
    expected_parameters: int,
    expected_bytes: int,
) -> dict[str, Any]:
    parameters = trainable_parameter_count(model)
    payload = fp32_parameter_payload_bytes(model)
    if parameters != expected_parameters:
        raise ValueError(
            f"{architecture} parameter count {parameters} differs from the frozen "
            f"protocol value {expected_parameters}."
        )
    if payload != expected_bytes:
        raise ValueError(
            f"{architecture} FP32 payload {payload} differs from the frozen "
            f"protocol value {expected_bytes}."
        )
    return {
        "identity_source": "constructed_model",
        "verified_against_constructed_model": True,
        "architecture": architecture,
        "experiment_id": experiment_id,
        "trainable_parameter_count": parameters,
        "fp32_parameter_payload_bytes": payload,
        "tokens": TOKENS,
        "model_dim": MODEL_DIM,
        "output": "single_raw_logit",
    }


def b4b_model_identity(model: B4BTransformerCNN) -> dict[str, Any]:
    """Describe B4-B and fail if it drifts from the frozen protocol."""
    identity = _identity(
        model,
        "B4BTransformerCNN",
        B4B_EXPERIMENT_ID,
        B4B_TRAINABLE_PARAMETERS,
        B4B_FP32_PARAMETER_BYTES,
    )
    identity.update(
        {
            "transformer_blocks": TRANSFORMER_BLOCKS,
            "attention_heads": ATTENTION_HEADS,
            "attention_head_dim": ATTENTION_HEAD_DIM,
            "attention_dropout": ATTENTION_DROPOUT,
            "branch_dropout": BRANCH_DROPOUT,
            "feed_forward_dim": FEED_FORWARD_DIM,
            "positional_embedding_shape": [TOKENS, MODEL_DIM],
        }
    )
    return identity


def b4c_model_identity(model: B4CSSMCNN) -> dict[str, Any]:
    """Describe B4-C and fail if it drifts from the frozen protocol."""
    identity = _identity(
        model,
        "B4CSSMCNN",
        B4C_EXPERIMENT_ID,
        B4C_TRAINABLE_PARAMETERS,
        B4C_FP32_PARAMETER_BYTES,
    )
    identity.update(
        {
            "ssm_blocks": SSM_BLOCKS,
            "ssm_state_dim": SSM_STATE_DIM,
            "branch_dropout": BRANCH_DROPOUT,
            "positional_embedding": None,
            "state_dtype": "torch.complex64",
            "selective_transition": False,
        }
    )
    return identity
