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
