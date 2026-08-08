"""Frozen B4-B and B4-C architecture tests, driven by synthetic fixtures.

Nothing here is a scientific result. No real training or validation performance
is computed and the sealed-test partition is never referenced.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from cardiosentinel.neural.candidate_engineering import (
    build_candidate,
    require_development_partitions,
)
from cardiosentinel.neural.candidates import (
    ATTENTION_DROPOUT,
    B4B_EXPERIMENT_ID,
    B4B_FP32_PARAMETER_BYTES,
    B4B_TRAINABLE_PARAMETERS,
    B4C_EXPERIMENT_ID,
    B4C_FP32_PARAMETER_BYTES,
    B4C_TRAINABLE_PARAMETERS,
    B4CSSMCNN,
    BRANCH_DROPOUT,
    LAYER_NORM_EPS,
    MODEL_DIM,
    SHARED_FRONT_END_PARAMETERS,
    SHARED_HEAD_PARAMETERS,
    SSM_STATE_DIM,
    TOKENS,
    B4BTransformerCNN,
    DiagonalGatedSSMBlock,
    SharedClassifierHead,
    SharedLocalFrontEnd,
    b4b_model_identity,
    b4c_model_identity,
)
from cardiosentinel.neural.determinism import initialize_determinism
from cardiosentinel.neural.model import (
    B4CompactCNN,
    fp32_parameter_payload_bytes,
    trainable_parameter_count,
)
from cardiosentinel.neural.protocol import TRAINABLE_PARAMETER_COUNT

WINDOW = 2500


@pytest.fixture(autouse=True)
def deterministic():
    initialize_determinism(requested_device="cpu")


def _waveforms(batch: int = 3) -> torch.Tensor:
    return torch.randn(batch, 1, WINDOW, dtype=torch.float32)


# --------------------------------------------------------------------------
# Shared components and B4-A compatibility
# --------------------------------------------------------------------------


def test_shared_front_end_matches_the_frozen_geometry() -> None:
    front_end = SharedLocalFrontEnd()
    output = front_end(_waveforms(4))

    assert trainable_parameter_count(front_end) == SHARED_FRONT_END_PARAMETERS
    assert tuple(output.shape) == (4, MODEL_DIM, TOKENS)


def test_shared_head_matches_the_frozen_count_and_shape() -> None:
    head = SharedClassifierHead()
    logits = head(torch.randn(4, MODEL_DIM, TOKENS))

    assert trainable_parameter_count(head) == SHARED_HEAD_PARAMETERS
    assert tuple(logits.shape) == (4,)


def test_b4a_is_untouched_by_the_candidate_modules() -> None:
    model = B4CompactCNN()

    assert trainable_parameter_count(model) == TRAINABLE_PARAMETER_COUNT
    assert trainable_parameter_count(model) == 87_089
    assert fp32_parameter_payload_bytes(model) == 348_356


def test_b4a_state_dict_names_are_unchanged() -> None:
    keys = set(B4CompactCNN().state_dict())

    # A representative sample of the frozen B4-A checkpoint key namespace.
    for expected in (
        "stem.0.weight",
        "stem.1.weight",
        "downsampling.0.layers.0.weight",
        "context.0.layers.0.weight",
        "head.1.weight",
        "head.4.weight",
    ):
        assert expected in keys, expected
    # The candidates must not have leaked their own module names into B4-A.
    assert not any(key.startswith("front_end.") for key in keys)
    assert not any(key.startswith("classifier.") for key in keys)
    assert not any("positional" in key for key in keys)


def test_b4a_locked_checkpoint_still_loads_into_b4a() -> None:
    donor = B4CompactCNN()
    state = donor.state_dict()
    receiver = B4CompactCNN()
    receiver.load_state_dict(state)

    assert set(receiver.state_dict()) == set(state)


# --------------------------------------------------------------------------
# B4-B identity
# --------------------------------------------------------------------------


def test_b4b_exact_parameter_count_and_payload() -> None:
    model = B4BTransformerCNN()
    identity = b4b_model_identity(model)

    assert trainable_parameter_count(model) == B4B_TRAINABLE_PARAMETERS == 309_809
    assert fp32_parameter_payload_bytes(model) == B4B_FP32_PARAMETER_BYTES == 1_239_236
    assert identity["experiment_id"] == "B4B_cnn_transformer_v1"
    assert identity["verified_against_constructed_model"] is True


def test_b4b_identity_rejects_a_drifted_model(monkeypatch) -> None:
    import cardiosentinel.neural.candidates as candidates

    monkeypatch.setattr(candidates, "B4B_TRAINABLE_PARAMETERS", 1)
    with pytest.raises(ValueError, match="parameter count"):
        candidates.b4b_model_identity(B4BTransformerCNN())


def test_b4b_structure_matches_the_frozen_protocol() -> None:
    model = B4BTransformerCNN()

    assert len(model.blocks) == 2
    assert tuple(model.positional_embedding.shape) == (TOKENS, MODEL_DIM)
    for block in model.blocks:
        assert block.attention.num_heads == 4
        assert block.attention.embed_dim == MODEL_DIM
        assert block.attention.embed_dim // block.attention.num_heads == 32
        assert block.attention.dropout == ATTENTION_DROPOUT == 0.0
        assert block.attention_dropout.p == BRANCH_DROPOUT == 0.10
        assert block.feed_forward_dropout.p == BRANCH_DROPOUT
        assert isinstance(block.feed_forward[0], nn.Linear)
        assert block.feed_forward[0].in_features == 128
        assert block.feed_forward[0].out_features == 256
        assert isinstance(block.feed_forward[1], nn.GELU)
        assert block.feed_forward[2].in_features == 256
        assert block.feed_forward[2].out_features == 128
        assert block.norm_attention.eps == LAYER_NORM_EPS == 1e-5
        assert block.norm_feed_forward.eps == LAYER_NORM_EPS
    assert model.final_norm.eps == LAYER_NORM_EPS


def test_b4b_uses_no_stock_transformer_encoder_layer() -> None:
    model = B4BTransformerCNN()

    assert not any(
        isinstance(module, (nn.TransformerEncoder, nn.TransformerEncoderLayer))
        for module in model.modules()
    )


def test_b4b_layernorm_initialization_is_frozen() -> None:
    model = B4BTransformerCNN()

    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))


def test_b4b_positional_initialization_follows_the_frozen_rule() -> None:
    initialize_determinism(requested_device="cpu")
    embedding = B4BTransformerCNN().positional_embedding.detach()

    assert embedding.numel() == TOKENS * MODEL_DIM
    # Normal(0, 0.02): the sample statistics must sit near the frozen values.
    assert abs(float(embedding.mean())) < 0.005
    assert 0.015 < float(embedding.std()) < 0.025
    assert float(embedding.abs().max()) < 0.02 * 6


def test_b4b_forward_shape_and_absence_of_sigmoid() -> None:
    model = B4BTransformerCNN().eval()
    with torch.no_grad():
        logits = model(_waveforms(5))

    assert tuple(logits.shape) == (5,)
    assert logits.dtype == torch.float32
    assert not any(isinstance(m, nn.Sigmoid) for m in model.modules())


def test_b4b_attention_uses_no_masks() -> None:
    """The forward path must never supply an attention or padding mask."""
    model = B4BTransformerCNN().eval()
    captured: list[tuple] = []
    block = model.blocks[0]
    original = block.attention.forward

    def spy(*args, **kwargs):
        captured.append((kwargs.get("attn_mask"), kwargs.get("key_padding_mask")))
        return original(*args, **kwargs)

    block.attention.forward = spy  # type: ignore[method-assign]
    with torch.no_grad():
        model(_waveforms(2))

    assert captured and all(mask is None for pair in captured for mask in pair)


def test_b4b_construction_is_deterministic_after_reseeding() -> None:
    initialize_determinism(requested_device="cpu")
    first = B4BTransformerCNN().state_dict()
    initialize_determinism(requested_device="cpu")
    second = B4BTransformerCNN().state_dict()

    assert set(first) == set(second)
    for key in first:
        assert torch.equal(first[key], second[key]), key


# --------------------------------------------------------------------------
# B4-C identity
# --------------------------------------------------------------------------


def test_b4c_exact_parameter_count_and_payload() -> None:
    model = B4CSSMCNN()
    identity = b4c_model_identity(model)

    assert trainable_parameter_count(model) == B4C_TRAINABLE_PARAMETERS == 155_313
    assert fp32_parameter_payload_bytes(model) == B4C_FP32_PARAMETER_BYTES == 621_252
    assert identity["experiment_id"] == "B4C_cnn_ssm_v1"
    assert identity["selective_transition"] is False
    assert identity["positional_embedding"] is None


def test_b4c_structure_matches_the_frozen_protocol() -> None:
    model = B4CSSMCNN()

    assert len(model.blocks) == 2
    for block in model.blocks:
        assert tuple(block.log_decay.shape) == (MODEL_DIM, SSM_STATE_DIM)
        assert tuple(block.frequency.shape) == (MODEL_DIM, SSM_STATE_DIM)
        assert tuple(block.state_input.shape) == (MODEL_DIM, SSM_STATE_DIM)
        assert tuple(block.state_output_real.shape) == (MODEL_DIM, SSM_STATE_DIM)
        assert tuple(block.state_output_imaginary.shape) == (MODEL_DIM, SSM_STATE_DIM)
        assert tuple(block.skip.shape) == (MODEL_DIM,)
        assert tuple(block.log_step.shape) == (MODEL_DIM,)
        assert block.input_projection.out_features == 2 * MODEL_DIM
        assert block.output_projection.out_features == MODEL_DIM
        assert block.dropout.p == BRANCH_DROPOUT
    assert MODEL_DIM == 128
    assert SSM_STATE_DIM == 16


def test_b4c_has_no_positional_embedding() -> None:
    names = dict(B4CSSMCNN().named_parameters())

    assert not any("positional" in name for name in names)


def test_b4c_real_parameters_are_float32() -> None:
    for name, parameter in B4CSSMCNN().named_parameters():
        assert parameter.dtype == torch.float32, name
        assert not parameter.is_complex(), name


def test_b4c_recurrence_tensors_are_complex64() -> None:
    block = DiagonalGatedSSMBlock()
    transition, input_gain, output = block.discrete_state_space()

    assert transition.dtype == torch.complex64
    assert input_gain.dtype == torch.complex64
    assert output.dtype == torch.complex64


def test_b4c_ssm_output_is_float32() -> None:
    block = DiagonalGatedSSMBlock()
    represented = block.state_space(torch.randn(2, TOKENS, MODEL_DIM))

    assert represented.dtype == torch.float32
    assert tuple(represented.shape) == (2, TOKENS, MODEL_DIM)


def test_b4c_initialization_follows_the_frozen_rule() -> None:
    block = DiagonalGatedSSMBlock()

    assert torch.allclose(
        block.log_decay, torch.full_like(block.log_decay, math.log(0.5))
    )
    expected = math.pi * torch.arange(1, SSM_STATE_DIM + 1, dtype=torch.float32)
    assert torch.allclose(block.frequency[0], expected)
    assert torch.allclose(block.frequency[-1], expected)
    assert torch.equal(block.skip, torch.zeros_like(block.skip))
    steps = torch.exp(block.log_step.detach())
    assert float(steps.min()) >= 1e-3 - 1e-9
    assert float(steps.max()) <= 1e-1 + 1e-9


def test_b4c_transition_is_unconditionally_stable() -> None:
    block = DiagonalGatedSSMBlock()
    transition, _, _ = block.discrete_state_space()

    assert bool((transition.abs() < 1.0).all())
    # Stability must survive adversarial finite parameters, not only the init.
    with torch.no_grad():
        block.log_decay.fill_(-8.0)
        block.log_step.fill_(math.log(1e-1))
        block.frequency.fill_(50.0)
    stressed, _, _ = block.discrete_state_space()
    assert bool((stressed.abs() < 1.0).all())


def test_b4c_performs_exactly_one_state_update_per_token() -> None:
    """Each loop iteration performs one state update and emits one row.

    The emitted row count is therefore exactly the number of state updates, so a
    79-token window performs exactly 79 updates.
    """
    block = DiagonalGatedSSMBlock()

    with torch.no_grad():
        full = block.state_space(torch.randn(2, TOKENS, MODEL_DIM))
        short = block.state_space(torch.randn(2, 5, MODEL_DIM))

    assert full.shape[1] == TOKENS == 79
    assert short.shape[1] == 5


def test_b4c_full_model_feeds_exactly_seventy_nine_tokens() -> None:
    model = B4CSSMCNN().eval()
    observed: list[int] = []
    block = model.blocks[0]
    original = block.state_space

    def spy(sequence):
        observed.append(sequence.shape[1])
        return original(sequence)

    block.state_space = spy  # type: ignore[method-assign]
    with torch.no_grad():
        model(_waveforms(2))

    assert observed == [TOKENS]


def test_b4c_forward_shape_and_absence_of_sigmoid() -> None:
    model = B4CSSMCNN().eval()
    with torch.no_grad():
        logits = model(_waveforms(5))

    assert tuple(logits.shape) == (5,)
    assert logits.dtype == torch.float32
    assert not any(isinstance(m, nn.Sigmoid) for m in model.modules())


def test_b4c_backward_gradients_are_finite() -> None:
    model = B4CSSMCNN()
    logits = model(_waveforms(2))
    nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0, 0.0])).backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name


# --------------------------------------------------------------------------
# SSM numerical reference and statelessness
# --------------------------------------------------------------------------


def test_ssm_core_matches_an_independent_reference_recurrence() -> None:
    """Recompute the frozen recurrence with plain tensor ops and compare."""
    torch.manual_seed(7)
    block = DiagonalGatedSSMBlock()
    sequence = torch.randn(3, TOKENS, MODEL_DIM)

    with torch.no_grad():
        produced = block.state_space(sequence)

        lam = torch.complex(-torch.exp(block.log_decay), block.frequency)
        zeta = torch.exp(block.log_step).unsqueeze(-1) * lam
        transition = torch.exp(zeta)
        gain = torch.expm1(zeta) / lam * block.state_input.to(torch.complex64)
        output = torch.complex(
            block.state_output_real, block.state_output_imaginary
        )
        state = torch.zeros(
            3, MODEL_DIM, SSM_STATE_DIM, dtype=torch.complex64
        )
        rows = []
        for step in range(TOKENS):
            drive = sequence[:, step, :].unsqueeze(-1).to(torch.complex64)
            state = transition.unsqueeze(0) * state + gain.unsqueeze(0) * drive
            rows.append(
                (output.unsqueeze(0) * state).real.sum(-1)
                + block.skip * sequence[:, step, :]
            )
        reference = torch.stack(rows, dim=1)

    assert produced.shape == reference.shape
    assert torch.allclose(produced, reference, rtol=1e-5, atol=1e-5)


def test_ssm_carries_no_state_between_calls() -> None:
    """B4-C is intra-window only; it is not the later T2 longitudinal SSM."""
    block = DiagonalGatedSSMBlock().eval()
    sequence = torch.randn(2, TOKENS, MODEL_DIM)

    with torch.no_grad():
        first = block.state_space(sequence)
        other = block.state_space(torch.randn(2, TOKENS, MODEL_DIM))
        repeated = block.state_space(sequence)

    assert torch.equal(first, repeated)
    assert not torch.allclose(first, other)


def test_b4c_repeated_evaluation_is_identical() -> None:
    model = B4CSSMCNN().eval()
    waveforms = _waveforms(3)

    with torch.no_grad():
        first = model(waveforms)
        model(_waveforms(3))
        repeated = model(waveforms)

    assert torch.equal(first, repeated)


def test_no_module_retains_a_state_buffer() -> None:
    for model in (B4BTransformerCNN(), B4CSSMCNN()):
        for name, buffer in model.named_buffers():
            assert "state" not in name, name
            assert buffer is not None


# --------------------------------------------------------------------------
# Prohibited inputs and development-only firewall
# --------------------------------------------------------------------------


def test_candidates_accept_only_a_waveform_tensor() -> None:
    import inspect

    for factory in (B4BTransformerCNN, B4CSSMCNN):
        parameters = inspect.signature(factory.forward).parameters
        assert list(parameters) == ["self", "waveforms"]


def test_candidates_reject_malformed_input() -> None:
    for factory in (B4BTransformerCNN, B4CSSMCNN):
        model = factory().eval()
        with pytest.raises(ValueError, match=r"\[B, 1, 2500\]"):
            model(torch.randn(2, 1, 100))
        with pytest.raises(ValueError, match="float32"):
            model(torch.randn(2, 1, WINDOW, dtype=torch.float64))
        with pytest.raises(ValueError, match="finite"):
            broken = torch.randn(2, 1, WINDOW)
            broken[0, 0, 0] = float("nan")
            model(broken)


def test_candidate_module_never_references_forbidden_inputs() -> None:
    import ast
    from pathlib import Path

    import cardiosentinel.neural.candidates as candidates

    tree = ast.parse(Path(candidates.__file__).read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in (
        "subject_id",
        "record_id",
        "channel_index",
        "context_flags",
        "target_family",
    ):
        assert forbidden not in names, forbidden
        assert not any(forbidden in text for text in literals), forbidden


def test_engineering_firewall_rejects_the_test_partition() -> None:
    assert require_development_partitions(("train", "validation")) == (
        "train",
        "validation",
    )
    for attempt in (("test",), ("train", "test"), ("validation", "test")):
        with pytest.raises(ValueError, match="train and validation only"):
            require_development_partitions(attempt)


def test_candidate_smoke_rejects_test_before_resolving_any_path(monkeypatch) -> None:
    """The partition guard must fire before any cache or source is resolved."""
    import cardiosentinel.neural.candidate_engineering as engineering

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("A path was resolved before the partition guard.")

    monkeypatch.setattr(engineering, "build_development_indexes", forbidden)
    monkeypatch.setattr(engineering, "validate_waveform_cache", forbidden)
    monkeypatch.setattr(engineering, "initialize_determinism", forbidden)

    with pytest.raises(ValueError, match="train and validation only"):
        engineering.candidate_smoke(
            B4B_EXPERIMENT_ID,
            source=None,
            feature_root=None,
            cache_root=None,
            partitions=("test",),
        )


def test_build_candidate_rejects_an_unknown_identifier() -> None:
    with pytest.raises(ValueError, match="Unknown B4 candidate"):
        build_candidate("B4D_hybrid_v1")


def test_build_candidate_returns_verified_identities() -> None:
    for experiment_id, expected in (
        (B4B_EXPERIMENT_ID, B4B_TRAINABLE_PARAMETERS),
        (B4C_EXPERIMENT_ID, B4C_TRAINABLE_PARAMETERS),
    ):
        model, identity = build_candidate(experiment_id)
        assert identity["trainable_parameter_count"] == expected
        assert identity["verified_against_constructed_model"] is True
        assert trainable_parameter_count(model) == expected
