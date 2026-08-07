"""Exact compact single-channel CNN frozen by B4_PROTOCOL_V1."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cardiosentinel.neural.protocol import WINDOW_SAMPLES


class DepthwiseSeparableBlock(nn.Module):
    """Frozen downsampling block: depthwise then pointwise convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=2,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(8, in_channels, eps=1e-5, affine=True),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels, eps=1e-5, affine=True),
            nn.SiLU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class ResidualContextBlock(nn.Module):
    """Frozen residual context block with no post-add activation."""

    def __init__(self, dilation: int) -> None:
        super().__init__()
        channels = 128
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                stride=1,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8, channels, eps=1e-5, affine=True),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels, eps=1e-5, affine=True),
            nn.SiLU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.layers(values)


class B4CompactCNN(nn.Module):
    """Return one raw logit per `[B, 1, 2500]` waveform batch."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, 32, eps=1e-5, affine=True),
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
        self.context = nn.ModuleList(
            tuple(ResidualContextBlock(dilation) for dilation in (2, 4, 8))
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward_features(self, waveforms: Tensor) -> Tensor:
        """Apply the frozen temporal representation without the classifier head."""
        self._validate_input(waveforms)
        values = self.stem(waveforms)
        for block in self.downsampling:
            values = block(values)
        for block in self.context:
            values = block(values)
        return values

    def forward(self, waveforms: Tensor) -> Tensor:
        values = self.forward_features(waveforms)
        values = self.pool(values).squeeze(-1)
        return self.head(values).squeeze(-1)

    @staticmethod
    def _validate_input(waveforms: Tensor) -> None:
        if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (1, WINDOW_SAMPLES):
            raise ValueError("B4 input must have shape [B, 1, 2500].")
        if waveforms.dtype != torch.float32:
            raise ValueError("B4 input must use torch.float32.")
        if not torch.isfinite(waveforms).all():
            raise ValueError("B4 input must contain only finite physical-mV values.")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def fp32_parameter_payload_bytes(model: nn.Module) -> int:
    """Return raw FP32 trainable tensor bytes, excluding serialization metadata."""
    return trainable_parameter_count(model) * torch.finfo(torch.float32).bits // 8


def local_receptive_field_samples() -> int:
    """Calculate the final local receptive field from frozen convolution stages."""
    receptive_field = 1
    sample_jump = 1
    stages = (
        (15, 2, 1),
        (9, 2, 1),
        (7, 2, 1),
        (5, 2, 1),
        (5, 2, 1),
        (5, 1, 2),
        (5, 1, 4),
        (5, 1, 8),
    )
    for kernel, stride, dilation in stages:
        receptive_field += (kernel - 1) * dilation * sample_jump
        sample_jump *= stride
    return receptive_field
