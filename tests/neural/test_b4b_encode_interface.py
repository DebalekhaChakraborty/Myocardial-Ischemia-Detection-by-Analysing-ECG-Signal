"""Regression tests for the B4-B `encode()` extraction.

B4-B is the merged, frozen global encoder. Exposing its pooled representation
must not change `forward()` by even one bit, so the reference implementation
below is a verbatim copy of the pre-refactor `forward` body and is compared
bitwise against the current one on the same weights.
"""

from __future__ import annotations

import torch

from cardiosentinel.neural.candidates import B4BTransformerCNN
from cardiosentinel.neural.protocol import WINDOW_SAMPLES

MODEL_DIM = 128


def _reference_forward(model: B4BTransformerCNN, waveforms: torch.Tensor):
    """Verbatim pre-refactor forward body, pinned here as the oracle."""
    values = model.front_end(waveforms)
    tokens = values.transpose(1, 2) + model.positional_embedding
    for block in model.blocks:
        tokens = block(tokens)
    tokens = model.final_norm(tokens)
    return model.classifier(tokens.transpose(1, 2))


def _waveforms(n: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2026)
    return torch.randn(n, 1, WINDOW_SAMPLES, generator=generator)


def test_forward_is_bitwise_unchanged_by_the_refactor() -> None:
    model = B4BTransformerCNN().eval()
    waveforms = _waveforms()
    with torch.no_grad():
        assert torch.equal(model(waveforms), _reference_forward(model, waveforms))


def test_encode_returns_the_pooled_encoder_representation() -> None:
    model = B4BTransformerCNN().eval()
    with torch.no_grad():
        embedding = model.encode(_waveforms(3))
    assert embedding.shape == (3, MODEL_DIM)
    assert embedding.dtype is torch.float32
    assert torch.isfinite(embedding).all()


def test_forward_equals_head_applied_to_encode() -> None:
    """The tap is exactly the classifier MLP's input; nothing sits between."""
    model = B4BTransformerCNN().eval()
    waveforms = _waveforms()
    with torch.no_grad():
        expected = model.classifier.head(model.encode(waveforms)).squeeze(-1)
        assert torch.equal(model(waveforms), expected)


def test_encode_is_deterministic_in_eval_mode() -> None:
    model = B4BTransformerCNN().eval()
    waveforms = _waveforms()
    with torch.no_grad():
        assert torch.equal(model.encode(waveforms), model.encode(waveforms))


def test_encode_validates_the_frozen_input_contract() -> None:
    model = B4BTransformerCNN().eval()
    for bad in (torch.zeros(2, 2, WINDOW_SAMPLES), torch.zeros(2, 1, 100)):
        try:
            model.encode(bad)
        except ValueError:
            continue
        raise AssertionError("encode accepted an out-of-contract input")


def test_checkpoint_compatibility_is_unchanged(tmp_path) -> None:
    """A checkpoint saved from the pre-refactor module still loads exactly."""
    source = B4BTransformerCNN()
    path = tmp_path / "model_selected.pt"
    torch.save(source.state_dict(), path)
    restored = B4BTransformerCNN()
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    source.eval()
    waveforms = _waveforms()
    with torch.no_grad():
        assert torch.equal(source(waveforms), restored(waveforms))
        assert torch.equal(source.encode(waveforms), restored.encode(waveforms))


def test_encode_does_not_change_model_state() -> None:
    model = B4BTransformerCNN().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.encode(_waveforms())
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
