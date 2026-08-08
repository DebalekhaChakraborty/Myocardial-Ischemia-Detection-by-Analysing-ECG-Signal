"""Protocol arithmetic checks for B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.

These are governance tests, not science. They recompute the frozen parameter
tables from the documented formulas and require the protocol document to state
the same totals, so the document's numbers cannot drift from its own arithmetic.

Nothing here instantiates or trains B4-B or B4-C: the candidate architectures
are deliberately unimplemented at this phase.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from cardiosentinel.neural.protocol import (
    FP32_PARAMETER_BYTES,
    TEMPORAL_LENGTHS,
    TRAINABLE_PARAMETER_COUNT,
)

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.md"
)
B4_PROTOCOL_V1_SHA256 = (
    "f6f5e9ed728c86a9b2bd75b2327b9199f0e097b91387525a192c212e6771b28b"
)

TOKENS = 79
MODEL_DIM = 128
HEADS = 4
STATE_DIM = 16
BLOCKS = 2
PARAMETER_CEILING = 1_000_000

SHARED_FRONT_END = 26_160
SHARED_HEAD = 8_321
B4B_TOTAL = 309_809
B4C_TOTAL = 155_313


def _group_norm(channels: int) -> int:
    return 2 * channels


def _linear(inputs: int, outputs: int) -> int:
    return inputs * outputs + outputs


def _layer_norm(dim: int) -> int:
    return 2 * dim


def _downsampling_block(cin: int, cout: int, kernel: int) -> int:
    """Depthwise then pointwise convolution, both bias-free, with GroupNorm."""
    return cin * kernel + _group_norm(cin) + cout * cin + _group_norm(cout)


def shared_front_end_parameters() -> int:
    stem = 1 * 32 * 15 + _group_norm(32)
    return stem + sum(
        _downsampling_block(cin, cout, kernel)
        for cin, cout, kernel in (
            (32, 48, 9),
            (48, 64, 7),
            (64, 96, 5),
            (96, 128, 5),
        )
    )


def shared_head_parameters() -> int:
    return _linear(128, 64) + _linear(64, 1)


def b4a_context_block_parameters() -> int:
    """B4-A's dilated residual convolution block, for decomposition checking."""
    return (
        128 * 5 + _group_norm(128) + 128 * 128 + _group_norm(128)
    )


def b4b_encoder_block_parameters() -> int:
    attention = 3 * (MODEL_DIM * MODEL_DIM) + 3 * MODEL_DIM
    attention += MODEL_DIM * MODEL_DIM + MODEL_DIM
    feed_forward = _linear(MODEL_DIM, 2 * MODEL_DIM) + _linear(2 * MODEL_DIM, MODEL_DIM)
    return _layer_norm(MODEL_DIM) + attention + _layer_norm(MODEL_DIM) + feed_forward


def b4b_total_parameters() -> int:
    positional = TOKENS * MODEL_DIM
    return (
        shared_front_end_parameters()
        + positional
        + BLOCKS * b4b_encoder_block_parameters()
        + _layer_norm(MODEL_DIM)
        + shared_head_parameters()
    )


def b4c_ssm_core_parameters() -> int:
    """State decay, frequency, B, complex C, skip D and log step."""
    per_state = MODEL_DIM * STATE_DIM
    return 5 * per_state + 2 * MODEL_DIM


def b4c_block_parameters() -> int:
    return (
        _layer_norm(MODEL_DIM)
        + _linear(MODEL_DIM, 2 * MODEL_DIM)
        + b4c_ssm_core_parameters()
        + _linear(MODEL_DIM, MODEL_DIM)
    )


def b4c_total_parameters() -> int:
    return (
        shared_front_end_parameters()
        + BLOCKS * b4c_block_parameters()
        + _layer_norm(MODEL_DIM)
        + shared_head_parameters()
    )


def _document() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


def _normalized() -> str:
    """Collapse whitespace so phrase checks survive markdown line wrapping."""
    return " ".join(_document().split())


# --------------------------------------------------------------------------
# The shared front end must decompose the frozen B4-A model exactly
# --------------------------------------------------------------------------


def test_shared_front_end_and_head_decompose_frozen_b4a() -> None:
    front_end = shared_front_end_parameters()
    head = shared_head_parameters()

    assert front_end == SHARED_FRONT_END
    assert head == SHARED_HEAD
    # Front end + three dilated context blocks + head is exactly frozen B4-A.
    assert (
        front_end + 3 * b4a_context_block_parameters() + head
        == TRAINABLE_PARAMETER_COUNT
    )
    assert TRAINABLE_PARAMETER_COUNT * 4 == FP32_PARAMETER_BYTES


def test_token_geometry_matches_the_frozen_temporal_lengths() -> None:
    assert TEMPORAL_LENGTHS[-1] == TOKENS
    assert TEMPORAL_LENGTHS[0] == 2500
    assert MODEL_DIM == 128


# --------------------------------------------------------------------------
# B4-B arithmetic
# --------------------------------------------------------------------------


def test_b4b_parameter_arithmetic_matches_the_protocol() -> None:
    assert b4b_encoder_block_parameters() == 132_480
    assert b4b_total_parameters() == B4B_TOTAL
    assert b4b_total_parameters() < PARAMETER_CEILING


def test_b4b_component_breakdown_is_exact() -> None:
    assert TOKENS * MODEL_DIM == 10_112
    assert 3 * (MODEL_DIM**2) + 3 * MODEL_DIM + MODEL_DIM**2 + MODEL_DIM == 66_048
    assert (
        _linear(MODEL_DIM, 2 * MODEL_DIM) + _linear(2 * MODEL_DIM, MODEL_DIM)
        == 65_920
    )
    assert MODEL_DIM % HEADS == 0
    assert MODEL_DIM // HEADS == 32


# --------------------------------------------------------------------------
# B4-C arithmetic
# --------------------------------------------------------------------------


def test_b4c_parameter_arithmetic_matches_the_protocol() -> None:
    assert b4c_ssm_core_parameters() == 10_496
    assert b4c_block_parameters() == 60_288
    assert b4c_total_parameters() == B4C_TOTAL
    assert b4c_total_parameters() < PARAMETER_CEILING


def test_b4c_is_the_more_compact_temporal_candidate() -> None:
    # Recorded so the Pareto review starts from a real resource difference.
    assert b4c_total_parameters() < b4b_total_parameters()
    assert TRAINABLE_PARAMETER_COUNT < b4c_total_parameters()


def test_candidate_payload_bytes_are_stated_correctly() -> None:
    assert b4b_total_parameters() * 4 == 1_239_236
    assert b4c_total_parameters() * 4 == 621_252


# --------------------------------------------------------------------------
# The document must state the same numbers it derives
# --------------------------------------------------------------------------


def test_protocol_states_the_recomputed_totals() -> None:
    text = _document()

    for value in (
        f"{shared_front_end_parameters():,}",
        f"{shared_head_parameters():,}",
        f"{b4b_total_parameters():,}",
        f"{b4c_total_parameters():,}",
        f"{b4b_encoder_block_parameters():,}",
        f"{b4c_block_parameters():,}",
        f"{b4c_ssm_core_parameters():,}",
        f"{TRAINABLE_PARAMETER_COUNT:,}",
    ):
        assert value in text, value


def test_protocol_freezes_the_shared_benchmark_identity() -> None:
    text = _document()

    for digest in (
        "66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7",
        "f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5",
        "318da148da5d638af44e73c06c00cc4df2815017d4ce8bb1a1b864e53eda8009",
        "dcac70260c92a8a4934dfcaa120e22fee939a976b63bf880f75ae176993d3ed2",
    ):
        assert digest in text
    for count in ("374,452", "93,613", "280,839", "473,897", "21,628", "452,269"):
        assert count in text


def test_protocol_declares_the_governance_boundaries() -> None:
    text = _normalized()

    # Exactly one configuration; no sweeps anywhere.
    assert "No grid search" in text or "no grid search" in text
    for forbidden in ("sweep is authorized", "hyperparameter search"):
        assert forbidden in text
    # B4-D is not authorized in this phase.
    assert "not authorized" in text
    # The SSM is named honestly.
    assert "DiagonalGatedSSMBlock" in text
    assert "must not be called Mamba" in text
    # Selection uses validation only and never test.
    assert "development validation evidence only" in text
    assert "uncalibrated sigmoid model scores" in text


def test_protocol_separates_b4c_from_the_t2_longitudinal_experiment() -> None:
    text = _normalized()

    assert "### B4-C does not satisfy T2" in text
    assert "carries no information between windows" in text
    assert "T1 and T2 remain required" in text


def test_protocol_records_the_experiment_identifier_alias() -> None:
    text = _normalized()

    for identifier in (
        "B4A_cnn_v1",
        "B4_raw_compact_cnn_v1",
        "B4B_cnn_transformer_v1",
        "B4C_cnn_ssm_v1",
        "B4_architecture_selection_v1",
    ):
        assert identifier in text
    assert "never renamed" in text


def test_b4_protocol_v1_is_unchanged_by_this_phase() -> None:
    path = PROTOCOL_PATH.parent / "B4_PROTOCOL_V1.md"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest == B4_PROTOCOL_V1_SHA256


# --------------------------------------------------------------------------
# B4-B frozen implementation semantics
# --------------------------------------------------------------------------


def test_b4b_freezes_attention_semantics() -> None:
    text = _normalized()

    for clause in (
        "embed_dim = 128",
        "num_heads = 4",
        "head_dim = 32",
        "batch_first = true",
        "attn_mask = none",
        "key_padding_mask = none",
    ):
        assert clause in text, clause


def test_b4b_internal_attention_dropout_is_zero() -> None:
    text = _normalized()

    assert "attention dropout = 0.0" in text
    assert "Internal attention-weight dropout is **0.0**" in text
    # The only block dropout is the external residual branch at 0.10.
    assert "Dropout_0.10( MHSA(LayerNorm(x)) )" in text
    assert "Dropout_0.10( FFN(LayerNorm(h)) )" in text
    assert "must not be applied together" in text
    assert "dropout=0.0" in text


def test_b4b_freezes_positional_and_layernorm_initialization() -> None:
    text = _normalized()

    assert "P ~ Normal(mean=0.0, std=0.02)" in text
    assert "`weight = 1`, `bias = 0`, `eps = 1e-5`" in text
    assert "initialize_determinism(seed=2026)" in text


def test_b4b_states_the_default_initialization_policy() -> None:
    text = _normalized()

    policy = "PyTorch module default initialization of the resolved PyTorch version"
    assert policy in text
    assert "records the exact resolved PyTorch version" in text
    assert "reference environment is PyTorch 2.13" in text


# --------------------------------------------------------------------------
# B4-C frozen implementation semantics
# --------------------------------------------------------------------------


def test_b4c_log_step_initialization_is_unambiguous() -> None:
    text = _normalized()

    assert "d_c ~ Uniform( log(1e-3), log(1e-1) )" in text
    assert "Delta_c = exp( d_c )" in text
    assert "log-uniform on `[1e-3, 1e-1]`" in text
    # The previously ambiguous phrasing is explicitly superseded.
    assert "was ambiguous about which quantity was log-uniform" in text


def test_b4c_state_index_and_frequency_grid_are_explicit() -> None:
    text = _normalized()

    assert "n = 1, 2, ..., 16" in text
    assert "w_{c,n} = pi * n" in text
    assert "one-based" in text
    # No exact-reproduction claim about a published initializer.
    assert "in the spirit of S4D" in text
    assert "does not reproduce" in text or "No claim is made that it reproduces" in text


def test_b4c_freezes_the_numerical_dtype_contract() -> None:
    text = _normalized()

    assert "torch.complex64" in text
    assert "torch.float32" in text
    for quantity in ("`lambda`", "`Abar`", "`Bbar`", "Recurrent state `x`"):
        assert quantity in text
    assert "No `float64` or `complex128` scientific forward path is authorized" in text
    assert "every trainable parameter is stored and counted as real `float32`" in text


def test_b4c_freezes_the_stable_expm1_zoh_computation() -> None:
    text = _normalized()

    assert "Bbar_{c,n} = expm1( z ) / lambda_{c,n} * B_{c,n}" in text
    assert "algebraically identical" in text
    assert "cancels the leading digits" in text
    assert "nonzero by parameterization" in text


def test_b4c_separates_lti_core_from_the_nonlinear_block() -> None:
    text = _normalized()

    block_clause = (
        "complete `DiagonalGatedSSMBlock` is not LTI and is not "
        "convolution-equivalent"
    )
    assert block_clause in text
    core_clause = (
        "**SSM core alone** therefore has an exact causal-convolution "
        "representation"
    )
    assert core_clause in text
    assert "temporal memory operator" in text
    assert "not Mamba" in text
    assert "**no selective or input-dependent state transition**" in text


def test_b4c_freezes_one_canonical_recurrence_implementation() -> None:
    text = _normalized()

    assert "x[0] = 0" in text
    assert "for k = 1..79:" in text
    assert "only the temporal dimension is iterated" in text
    for excluded in (
        "FFT",
        "parallel scan",
        "custom CUDA kernels",
        "external state-space library",
    ):
        assert excluded in text
    immutability = (
        "must not be changed after any validation or latency result is observed"
    )
    assert immutability in text


# --------------------------------------------------------------------------
# Selection rule formalization
# --------------------------------------------------------------------------


def test_selection_defines_the_resource_vector() -> None:
    text = _normalized()

    for component in (
        "trainable_parameter_count",
        "serialized_FP32_bytes",
        "measured_CPU_inference_latency",
        "measured_peak_inference_memory",
    ):
        assert component in text
    assert "Lower is better in every component" in text


def test_selection_defines_formal_pareto_dominance() -> None:
    text = _normalized()

    assert "Pareto-dominates" in text
    assert "`AUPRC(X) >= AUPRC(Y)`" in text
    assert "no worse than `Y` in **every available** predeclared resource" in text
    assert "strictly better than `Y` in at least one of" in text
    assert "neither candidate is Pareto-dominant on resources" in text
    assert "A Pareto-dominated candidate must not be selected" in text


def test_support_dimensions_are_excluded_from_dominance() -> None:
    text = _normalized()

    assert "deliberately **not** folded into the dominance test" in text
    assert "must not be converted into a post-hoc scalar score" in text


def test_selection_freezes_the_lexicographic_resource_tie_break() -> None:
    text = _normalized()

    assert "lexicographic" in text
    assert "median CPU inference latency" in text
    assert "consulted only if every higher-priority item is tied" in text
    assert "never imputed, estimated or substituted" in text
    assert "same recorded CPU environment" in text


# --------------------------------------------------------------------------
# Historical provenance honesty
# --------------------------------------------------------------------------


def test_b4a_provenance_does_not_overclaim_blindness() -> None:
    text = _normalized()

    assert "No claim of blindness to the B4-A validation result is made" in text
    assert "that blindness did not exist" in text
    assert "architecture **families** were predeclared" in text
    ordering = (
        "frozen in this document *after* the B4-A development validation result"
    )
    assert ordering in text
    assert "*before* either candidate was implemented or trained" in text
    assert "not** an optimization objective" in text
    # The historical values remain recorded as context.
    assert "0.3156014611186772" in text
    assert "0.8675598293803359" in text


def test_hardening_did_not_change_the_frozen_totals() -> None:
    # The precision pass must not move any accepted architecture number.
    assert b4b_total_parameters() == 309_809
    assert b4c_total_parameters() == 155_313
    assert b4b_total_parameters() * 4 == 1_239_236
    assert b4c_total_parameters() * 4 == 621_252
    assert shared_front_end_parameters() == 26_160
    assert shared_head_parameters() == 8_321
