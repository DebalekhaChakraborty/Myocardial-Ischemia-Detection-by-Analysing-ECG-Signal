import pytest
import torch
from torch import nn

from cardiosentinel.neural.model import (
    B4CompactCNN,
    ResidualContextBlock,
    fp32_parameter_payload_bytes,
    local_receptive_field_samples,
    trainable_parameter_count,
)
from cardiosentinel.neural.protocol import (
    FP32_PARAMETER_BYTES,
    LOCAL_RECEPTIVE_FIELD_SAMPLES,
    TEMPORAL_LENGTHS,
    TRAINABLE_PARAMETER_COUNT,
)


def test_frozen_model_arithmetic_and_shapes() -> None:
    model = B4CompactCNN().eval()
    values = torch.zeros(2, 1, 2500, dtype=torch.float32)
    lengths = [values.shape[-1]]
    values = model.stem(values)
    lengths.append(values.shape[-1])
    for block in model.downsampling:
        values = block(values)
        lengths.append(values.shape[-1])

    assert tuple(lengths) == TEMPORAL_LENGTHS
    assert trainable_parameter_count(model) == TRAINABLE_PARAMETER_COUNT == 87_089
    assert fp32_parameter_payload_bytes(model) == FP32_PARAMETER_BYTES == 348_356
    assert local_receptive_field_samples() == LOCAL_RECEPTIVE_FIELD_SAMPLES == 1943
    assert model(torch.zeros(2, 1, 2500)).shape == (2,)


def test_model_contains_no_sigmoid_and_context_preserves_shape() -> None:
    model = B4CompactCNN()
    assert not any(isinstance(module, nn.Sigmoid) for module in model.modules())
    values = torch.randn(2, 128, 79)
    assert ResidualContextBlock(8)(values).shape == values.shape


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (torch.zeros(1, 2, 2500), "shape"),
        (torch.zeros(1, 1, 2499), "shape"),
        (torch.zeros(1, 1, 2500, dtype=torch.float64), "float32"),
    ],
)
def test_model_rejects_noncanonical_input(values: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        B4CompactCNN()(values)


def test_model_rejects_nonfinite_input() -> None:
    values = torch.zeros(1, 1, 2500)
    values[0, 0, 4] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        B4CompactCNN()(values)
