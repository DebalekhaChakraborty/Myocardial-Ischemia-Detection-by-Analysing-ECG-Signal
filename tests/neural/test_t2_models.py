"""The two frozen T2 candidates, proven synthetically (§37 A-C).

Nothing here touches the real corpus. Every tensor is synthetic, every forward
pass is on random data, and no optimiser step is taken against a real row.

State-carry equivalence is asserted with `allclose` rather than bitwise
equality: chunking changes the sequence length a BLAS kernel sees, so float32
reassociation moves the last ulp or two. That is numerical equivalence, which is
what §10 requires; claiming bit-identity would be false.
"""

from __future__ import annotations

import math

import pytest
import torch

from cardiosentinel.neural.t2_models import (
    S4D_STATE_DIM,
    CausalGRULongitudinal,
    CausalS4DLongitudinal,
    LongitudinalDiagonalSSMBlock,
    T2ModelError,
    build_t2_model,
    detach_state,
    model_identity,
    seed_everything,
    trainable_parameter_count,
)
from cardiosentinel.neural.t2_protocol import (
    T2_ARM_GRU,
    T2_ARM_S4D,
    T2_EXPECTED_PARAMETER_COUNTS,
    T2_INPUT_DIM,
    T2_TEMPORAL_LAYERS,
    T2_TEMPORAL_WIDTH,
)

CARRY_TOLERANCE = 1e-5


def _values(batch: int = 3, steps: int = 12) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(batch, steps, T2_INPUT_DIM, generator=generator)


# --- A. model construction ------------------------------------------------


@pytest.mark.parametrize(
    ("arm", "expected"),
    [(T2_ARM_GRU, 59_521), (T2_ARM_S4D, 45_313)],
)
def test_parameter_count_is_exactly_the_frozen_value(arm, expected):
    model = build_t2_model(arm)
    assert trainable_parameter_count(model) == expected
    assert T2_EXPECTED_PARAMETER_COUNTS[arm] == expected


def test_construction_refuses_an_unknown_arm():
    with pytest.raises(Exception):
        build_t2_model("causal_lstm_longitudinal_v1")


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_one_logit_per_timestep(arm):
    model = build_t2_model(arm)
    model.eval()
    values = _values(batch=2, steps=9)
    with torch.no_grad():
        logits, _ = model(values)
    assert logits.shape == (2, 9)
    assert logits.dtype == torch.float32


def test_gru_state_shape_and_dtype():
    model = build_t2_model(T2_ARM_GRU)
    state = model.initial_state(4)
    assert state.shape == (T2_TEMPORAL_LAYERS, 4, T2_TEMPORAL_WIDTH)
    assert state.dtype == torch.float32
    assert bool(torch.all(state == 0))


def test_s4d_state_shape_and_dtype():
    model = build_t2_model(T2_ARM_S4D)
    state = model.initial_state(4)
    assert len(state) == T2_TEMPORAL_LAYERS
    for block_state in state:
        assert block_state.shape == (4, T2_TEMPORAL_WIDTH, S4D_STATE_DIM)
        assert block_state.dtype == torch.complex64
        assert bool(torch.all(block_state == 0))


def test_gru_is_not_bidirectional():
    model = build_t2_model(T2_ARM_GRU)
    assert model.temporal.bidirectional is False
    assert model.temporal.num_layers == T2_TEMPORAL_LAYERS
    assert model.temporal.hidden_size == T2_TEMPORAL_WIDTH
    assert model.temporal.input_size == 64
    assert model.temporal.batch_first is True
    assert model.temporal.dropout == pytest.approx(0.10)


def test_models_reject_the_wrong_representation():
    model = build_t2_model(T2_ARM_S4D)
    with pytest.raises(T2ModelError, match="frozen 146"):
        model(torch.randn(1, 4, 128))
    with pytest.raises(T2ModelError, match="float32"):
        model(torch.randn(1, 4, T2_INPUT_DIM, dtype=torch.float64))


def test_no_selective_or_mamba_mechanism():
    """The transition is time-invariant and input-independent by construction."""
    identity = model_identity(build_t2_model(T2_ARM_S4D))
    assert identity["is_mamba"] is False
    assert identity["selective_mechanism"] is False
    block = LongitudinalDiagonalSSMBlock()
    transition_a, _, _ = block.discrete_state_space()
    transition_b, _, _ = block.discrete_state_space()
    assert torch.equal(transition_a, transition_b)
    # and it does not depend on any input at all: it takes no argument
    import inspect

    assert list(inspect.signature(block.discrete_state_space).parameters) == []


def test_model_identity_reports_the_frozen_shape():
    for arm in (T2_ARM_GRU, T2_ARM_S4D):
        identity = model_identity(build_t2_model(arm))
        assert identity["input_dim"] == 146
        assert identity["width"] == 64
        assert identity["layers"] == 2
        assert identity["dropout"] == 0.10
        assert identity["bidirectional"] is False
        assert identity["parameter_dtype"] == "float32"


# --- B. S4D mathematics ---------------------------------------------------


def test_lambda_real_part_is_always_negative():
    block = LongitudinalDiagonalSSMBlock()
    with torch.no_grad():
        block.log_decay.uniform_(-5.0, 5.0)
    lam = torch.complex(-torch.exp(block.log_decay), block.frequency)
    assert bool(torch.all(lam.real < 0)), "the stability constraint must hold"


def test_zoh_discretization_matches_the_frozen_formula():
    block = LongitudinalDiagonalSSMBlock()
    transition, input_gain, output = block.discrete_state_space()
    lam = torch.complex(-torch.exp(block.log_decay), block.frequency)
    zeta = torch.exp(block.log_step).unsqueeze(-1) * lam
    assert torch.allclose(transition, torch.exp(zeta))
    expected_gain = torch.expm1(zeta) / lam * block.state_input.to(lam.dtype)
    assert torch.allclose(input_gain, expected_gain)
    assert torch.allclose(
        output,
        torch.complex(block.state_output_real, block.state_output_imaginary),
    )


def test_input_gain_uses_expm1_not_exp_minus_one():
    """`expm1` keeps relative precision for the small |zeta| small steps give."""
    import ast
    from pathlib import Path

    import cardiosentinel.neural.t2_models as module

    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    names = {
        getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "expm1" in names


def test_skip_term_is_initialized_exactly_zero():
    block = LongitudinalDiagonalSSMBlock()
    assert bool(torch.all(block.skip == 0))
    assert block.skip.shape == (T2_TEMPORAL_WIDTH,)


def test_frozen_initializers():
    block = LongitudinalDiagonalSSMBlock()
    assert torch.allclose(
        block.log_decay, torch.full_like(block.log_decay, math.log(0.5))
    )
    expected = math.pi * torch.arange(1, S4D_STATE_DIM + 1, dtype=torch.float32)
    assert torch.allclose(block.frequency[0], expected)
    assert bool(torch.all(block.log_step >= math.log(1e-3)))
    assert bool(torch.all(block.log_step <= math.log(1e-1)))


def test_step_recurrence_agrees_with_a_reference_calculation():
    """The loop must be exactly `state = Abar*state + Bbar*value`."""
    block = LongitudinalDiagonalSSMBlock()
    block.eval()
    tokens = torch.randn(2, 5, T2_TEMPORAL_WIDTH)
    with torch.no_grad():
        _, final_state = block(tokens)
        transition, input_gain, _ = block.discrete_state_space()
        normed = block.norm(tokens)
        value, _ = block.input_projection(normed).chunk(2, dim=-1)
        reference = torch.zeros_like(final_state)
        for step in range(tokens.shape[1]):
            reference = transition.unsqueeze(0) * reference + input_gain.unsqueeze(
                0
            ) * value[:, step, :].unsqueeze(-1).to(transition.dtype)
    assert torch.allclose(final_state, reference, atol=1e-6)


# --- B/C. state carry equivalence for both arms ---------------------------


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_one_sequence_equals_two_carried_chunks(arm):
    model = build_t2_model(arm)
    model.eval()
    values = _values(batch=3, steps=14)
    with torch.no_grad():
        whole, whole_state = model(values)
        first, first_state = model(values[:, :6])
        second, second_state = model(values[:, 6:], first_state)
    joined = torch.cat([first, second], dim=1)
    assert torch.allclose(whole, joined, atol=CARRY_TOLERANCE)
    if isinstance(whole_state, torch.Tensor):
        assert torch.allclose(whole_state, second_state, atol=CARRY_TOLERANCE)
    else:
        for left, right in zip(whole_state, second_state, strict=True):
            assert torch.allclose(left, right, atol=CARRY_TOLERANCE)


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_detaching_state_preserves_forward_values(arm):
    model = build_t2_model(arm)
    model.eval()
    values = _values(batch=2, steps=10)
    with torch.no_grad():
        _, state = model(values[:, :5])
        attached, _ = model(values[:, 5:], state)
        detached, _ = model(values[:, 5:], detach_state(state))
    assert torch.equal(attached, detached)


def _state_tensors(state):
    return [state] if isinstance(state, torch.Tensor) else list(state)


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_detaching_state_severs_gradient_history(arm):
    """Detach is a gradient boundary, not a value boundary.

    Proven by contrast: the *attached* state carries a `grad_fn` back into the
    first chunk's graph, and the detached one does not. Asserting on the graph
    is the real claim -- checking `.grad` on a non-leaf tensor would pass
    vacuously.
    """
    model = build_t2_model(arm)
    model.train()
    values = _values(batch=2, steps=8)
    _, state = model(values[:, :4])

    attached = _state_tensors(state)
    assert all(tensor.grad_fn is not None for tensor in attached), (
        "the un-detached state must still be connected to the first chunk"
    )

    carried = detach_state(state)
    detached = _state_tensors(carried)
    assert all(tensor.grad_fn is None for tensor in detached)
    assert all(tensor.requires_grad is False for tensor in detached)
    for before, after in zip(attached, detached, strict=True):
        assert torch.equal(before.detach(), after), "values must be unchanged"

    # A backward through the detached chunk still reaches the parameters, so the
    # boundary stops history, not learning.
    model.zero_grad(set_to_none=True)
    second_logits, _ = model(values[:, 4:], carried)
    second_logits.sum().backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.parameters()
    )


def test_state_carry_is_available_on_both_arms_by_the_same_contract():
    for arm in (T2_ARM_GRU, T2_ARM_S4D):
        model = build_t2_model(arm)
        state = model.initial_state(1)
        logits, outgoing = model(_values(batch=1, steps=3), state)
        assert logits.shape == (1, 3)
        assert type(outgoing) is type(state)


# --- I. deterministic construction ----------------------------------------


def test_each_arm_is_reseeded_independently_of_construction_order():
    """Building the GRU first must not change the S4D's initialisation."""
    s4d_alone = build_t2_model(T2_ARM_S4D)
    build_t2_model(T2_ARM_GRU)
    s4d_after_gru = build_t2_model(T2_ARM_S4D)
    for left, right in zip(
        s4d_alone.state_dict().values(),
        s4d_after_gru.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(left, right)


def test_repeated_construction_is_bit_identical():
    for arm in (T2_ARM_GRU, T2_ARM_S4D):
        first = build_t2_model(arm).state_dict()
        second = build_t2_model(arm).state_dict()
        assert first.keys() == second.keys()
        for key in first:
            assert torch.equal(first[key], second[key]), key


def test_seed_everything_refuses_a_substitute_seed():
    with pytest.raises(T2ModelError, match="frozen seed 2026"):
        seed_everything(1234)
    state = seed_everything()
    assert state["seed"] == 2026


def test_direct_class_construction_matches_the_builder():
    seed_everything()
    direct = CausalGRULongitudinal()
    assert trainable_parameter_count(direct) == 59_521
    seed_everything()
    direct_s4d = CausalS4DLongitudinal()
    assert trainable_parameter_count(direct_s4d) == 45_313
