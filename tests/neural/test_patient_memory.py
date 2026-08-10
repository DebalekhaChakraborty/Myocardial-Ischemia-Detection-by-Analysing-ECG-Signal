"""Causal integrity tests for the M1 dual-timescale patient memory.

These are the invariants that make M1 a *causal* experiment rather than a
plausible-looking one. Several of them encode defects that were real risks in
this codebase's history: a memory that peeks at its own window, a stream key
that merges simultaneous channels, and an admission rule that quietly reads
a label.

Everything here is synthetic. No real corpus, no real waveform, no real model
and no test partition is touched.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m1_experiment, patient_memory
from cardiosentinel.neural.metadata import B4WindowReference
from cardiosentinel.neural.patient_memory import (
    ALPHA_LONG,
    ALPHA_SHORT,
    COLD_START_BINS,
    M1D_EXPERIMENT_ID,
    M1L_EXPERIMENT_ID,
    M1S_EXPERIMENT_ID,
    REPRESENTATION_DIM,
    DualTimescaleMemory,
    M1MemoryError,
    build_causal_streams,
    build_m1_head,
    cold_start_bin,
    fit_distance_standardizer,
    generate_stream_memory,
    m1_alpha_identity,
    m1_arm_features,
    m1_boundary_statement,
    m1_head_identity,
    select_rows,
    stream_key,
)

WINDOW_SAMPLES = 2_500
STRIDE_SAMPLES = 1_250  # the frozen 5 s stride at 250 Hz


def reference(
    record: str,
    channel: int,
    index: int,
    *,
    subject: str = "ltstdb:s0001",
    family: str = "background_negative",
    partition: str = "train",
) -> B4WindowReference:
    """Build one synthetic window whose stable ID matches the frozen scheme."""
    start = index * STRIDE_SAMPLES
    end = start + WINDOW_SAMPLES
    return B4WindowReference(
        stable_id=f"ltstdb:{record}:{channel}:{start}:{end}",
        record_id=record,
        subject_id=subject,
        channel_index=channel,
        start_sample=start,
        end_sample=end,
        partition=partition,
        target_family=family,
        context_flags=(),
    )


def vector(seed: int, scale: float = 1.0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return (generator.normal(size=REPRESENTATION_DIM) * scale).astype(np.float32)


@pytest.fixture
def standardizer():
    rows = np.stack([vector(seed) for seed in range(200)]).astype(np.float64)
    return fit_distance_standardizer(rows, partition="train")


def source_of(function) -> ast.AST:
    """Parse a function body with docstrings stripped.

    Asserting against source text has bitten this suite before: a docstring
    that merely *describes* a forbidden behaviour matched a naive substring
    check. Only executable code is inspected here.
    """
    tree = ast.parse(inspect.getsource(function).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return tree


# --------------------------------------------------------------------------
# A. Past-only invariance
# --------------------------------------------------------------------------


def test_a_future_windows_cannot_change_earlier_memory_features(standardizer):
    rows = [reference("r1", 0, i) for i in range(12)]
    streams = build_causal_streams(rows)
    base = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    first = generate_stream_memory(
        streams, partition="train", representations=base, standardizer=standardizer
    )

    # Replace every window strictly after position 5 with a wildly different
    # representation. Nothing at or before position 5 may move.
    perturbed = dict(base)
    for row in rows[6:]:
        perturbed[row.stable_id] = vector(999, scale=50.0)
    second = generate_stream_memory(
        streams, partition="train", representations=perturbed, standardizer=standardizer
    )

    assert first.stable_ids == second.stable_ids
    np.testing.assert_array_equal(first.d_short[:6], second.d_short[:6])
    np.testing.assert_array_equal(first.d_long[:6], second.d_long[:6])
    assert not np.array_equal(first.d_short[6:], second.d_short[6:])


# --------------------------------------------------------------------------
# B. Score before update
# --------------------------------------------------------------------------


def test_b_prototype_at_t_contains_only_strictly_earlier_windows(standardizer):
    prior = standardizer.prior_vector()
    memory = DualTimescaleMemory(prior)
    x1 = standardizer.standardize(vector(1))[0]
    x2 = standardizer.standardize(vector(2))[0]

    first = memory.observe(x1)
    # The very first window can only be compared against the cold-start prior.
    np.testing.assert_allclose(
        first.d_short, float(np.sqrt(np.mean((x1 - prior) ** 2)))
    )
    assert first.past_observed_count == 0
    assert first.past_update_count == 0

    expected_short = (1.0 - ALPHA_SHORT) * prior + ALPHA_SHORT * x1
    second = memory.observe(x2)
    np.testing.assert_allclose(
        second.d_short, float(np.sqrt(np.mean((x2 - expected_short) ** 2)))
    )
    assert second.past_observed_count == 1
    assert second.past_update_count == 1


def test_b_observe_scores_before_it_updates(standardizer):
    memory = DualTimescaleMemory(standardizer.prior_vector())
    x = standardizer.standardize(vector(7))[0]
    before = memory.mu_short.copy()
    memory.observe(x)
    assert not np.array_equal(before, memory.mu_short)


# --------------------------------------------------------------------------
# C. No label path
# --------------------------------------------------------------------------


def test_c_changing_every_label_leaves_memory_bit_identical(standardizer):
    rows = [reference("r1", 0, i) for i in range(10)]
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    original = generate_stream_memory(
        build_causal_streams(rows),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )

    relabelled = [
        reference(
            "r1",
            0,
            index,
            family="ischemic_positive" if index % 2 else "rate_related",
        )
        for index in range(10)
    ]
    changed = generate_stream_memory(
        build_causal_streams(relabelled),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )

    np.testing.assert_array_equal(original.d_short, changed.d_short)
    np.testing.assert_array_equal(original.d_long, changed.d_long)
    assert original.chronology_sha256 == changed.chronology_sha256


def test_c_memory_update_accepts_no_label_or_score_argument():
    parameters = list(
        inspect.signature(DualTimescaleMemory.update).parameters
    )
    assert parameters == ["self", "standardized"]
    forbidden = {
        "label",
        "target_family",
        "score",
        "threshold",
        "uncertainty",
        "event_state",
        "morphology_valid",
    }
    assert not forbidden.intersection(
        inspect.signature(DualTimescaleMemory.observe).parameters
    )


# --------------------------------------------------------------------------
# D. Stream key correctness
# --------------------------------------------------------------------------


def test_d_simultaneous_channels_keep_independent_histories(standardizer):
    rows = [reference("r1", channel, i) for channel in (0, 1) for i in range(8)]
    streams = build_causal_streams(rows)
    assert set(streams) == {("r1", 0), ("r1", 1)}

    # Channel 0 sees a constant signal; channel 1 sees a violently varying one.
    values: dict[str, np.ndarray] = {}
    for row in rows:
        if row.channel_index == 0:
            values[row.stable_id] = vector(0)
        else:
            values[row.stable_id] = vector(int(row.start_sample), scale=40.0)

    memory = generate_stream_memory(
        streams, partition="train", representations=values, standardizer=standardizer
    )
    index = memory.index()
    channel0 = [
        memory.d_short[index[row.stable_id]] for row in rows if row.channel_index == 0
    ]
    # A constant stream collapses toward its own prototype. If channel 1 leaked
    # into channel 0's history this would not hold.
    assert channel0 == sorted(channel0, reverse=True)
    assert channel0[-1] < channel0[0]


def test_d_stream_key_is_record_and_channel():
    assert stream_key(reference("r1", 1, 0)) == ("r1", 1)
    assert stream_key(reference("r1", 0, 0)) != stream_key(reference("r1", 1, 0))


def test_d_channel_history_is_unaffected_by_the_other_channel(standardizer):
    alone = [reference("r1", 0, i) for i in range(6)]
    paired = alone + [reference("r1", 1, i) for i in range(6)]
    values = {row.stable_id: vector(hash(row.stable_id) % 1000) for row in paired}

    solo = generate_stream_memory(
        build_causal_streams(alone),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    both = generate_stream_memory(
        build_causal_streams(paired),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    shared = select_rows(both, solo.stable_ids)
    np.testing.assert_array_equal(solo.d_short, both.d_short[shared])
    np.testing.assert_array_equal(solo.d_long, both.d_long[shared])


# --------------------------------------------------------------------------
# E. Record reset
# --------------------------------------------------------------------------


def test_e_two_records_of_one_subject_start_from_the_global_prior(standardizer):
    subject = "ltstdb:s2027"
    first = [reference("rA", 0, i, subject=subject) for i in range(6)]
    second = [reference("rB", 0, i, subject=subject) for i in range(6)]
    values = {row.stable_id: vector(3) for row in first + second}

    memory = generate_stream_memory(
        build_causal_streams(first + second),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    index = memory.index()
    opening_a = memory.d_short[index[first[0].stable_id]]
    opening_b = memory.d_short[index[second[0].stable_id]]

    # Identical content in both recordings: if state carried across the record
    # boundary merely because the subject matched, rB would open much closer to
    # its prototype than rA did.
    assert opening_a == pytest.approx(opening_b)
    assert memory.past_observed_count[index[second[0].stable_id]] == 0
    assert memory.past_update_count[index[second[0].stable_id]] == 0


# --------------------------------------------------------------------------
# F. Subject-ID non-leakage
# --------------------------------------------------------------------------


def test_f_head_input_width_leaves_no_room_for_identity_features():
    for experiment_id, width in (
        (M1S_EXPERIMENT_ID, REPRESENTATION_DIM + 1),
        (M1L_EXPERIMENT_ID, REPRESENTATION_DIM + 1),
        (M1D_EXPERIMENT_ID, REPRESENTATION_DIM + 2),
    ):
        head = build_m1_head(experiment_id)
        identity = m1_head_identity(experiment_id, head)
        assert head.input_dim == width
        assert identity["patient_identifier_features"] == []
        assert identity["learned_patient_embedding"] is False


def test_f_arm_features_refuse_any_extra_identity_column():
    base = np.zeros((4, REPRESENTATION_DIM), dtype=np.float32)
    memory = np.zeros((4, 1), dtype=np.float32)
    assert m1_arm_features(M1S_EXPERIMENT_ID, base, memory).shape == (
        4,
        REPRESENTATION_DIM + 1,
    )
    with pytest.raises(M1MemoryError):
        m1_arm_features(M1S_EXPERIMENT_ID, base, np.zeros((4, 2), dtype=np.float32))


def test_f_feature_assembly_never_reads_an_identifier():
    tree = source_of(patient_memory.m1_arm_features)
    names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not {"subject_id", "record_id", "patient_id"}.intersection(names)


# --------------------------------------------------------------------------
# G. Frozen dual-timescale constants
# --------------------------------------------------------------------------


def test_g_alpha_constants_are_exact_and_ordered():
    assert ALPHA_SHORT == 1.0 - 2.0 ** (-1.0 / 60)
    assert ALPHA_LONG == 1.0 - 2.0 ** (-1.0 / 720)
    assert ALPHA_SHORT == pytest.approx(0.01148597964710385, abs=0.0)
    assert ALPHA_LONG == pytest.approx(0.0009622411662165709, abs=0.0)
    assert ALPHA_SHORT > ALPHA_LONG > 0.0

    identity = m1_alpha_identity()
    assert identity["short_half_life_updates"] == 60
    assert identity["long_half_life_updates"] == 720
    assert identity["stride_seconds"] == 5.0
    assert identity["swept"] is False and identity["tuned"] is False


def test_g_short_prototype_moves_faster_than_long(standardizer):
    memory = DualTimescaleMemory(standardizer.prior_vector())
    x = standardizer.standardize(vector(11, scale=8.0))[0]
    features = memory.observe(x)
    assert features.d_short == pytest.approx(features.d_long)
    after = memory.deviations(x)
    # Having admitted x, the fast prototype must now sit nearer to it.
    assert after.d_short < after.d_long


# --------------------------------------------------------------------------
# H. Cold start
# --------------------------------------------------------------------------


def test_h_every_stream_opens_against_the_exact_persisted_prior(standardizer):
    rows = [reference(record, 0, i) for record in ("rA", "rB") for i in range(4)]
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    memory = generate_stream_memory(
        build_causal_streams(rows),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    prior = standardizer.prior_vector()
    index = memory.index()
    for record in ("rA", "rB"):
        opening = next(row for row in rows if row.record_id == record)
        x = standardizer.standardize(values[opening.stable_id])[0]
        position = index[opening.stable_id]
        assert memory.d_short[position] == pytest.approx(
            float(np.sqrt(np.mean((x - prior) ** 2)))
        )
        assert memory.past_observed_count[position] == 0


def test_h_prior_is_persisted_rather_than_assumed_zero(standardizer):
    payload = standardizer.as_dict()
    assert len(payload["prior"]) == REPRESENTATION_DIM
    assert "never assumed zero" in payload["prior_semantics"]
    restored = type(standardizer).from_dict(payload)
    np.testing.assert_array_equal(restored.prior_vector(), standardizer.prior_vector())


def test_h_cold_start_bins_are_frozen():
    assert [name for name, _, _ in COLD_START_BINS] == [
        "0_5_minutes",
        "5_60_minutes",
        "over_60_minutes",
    ]
    assert cold_start_bin(0.0) == "0_5_minutes"
    assert cold_start_bin(299.9) == "0_5_minutes"
    assert cold_start_bin(300.0) == "5_60_minutes"
    assert cold_start_bin(3599.9) == "5_60_minutes"
    assert cold_start_bin(3600.0) == "over_60_minutes"
    with pytest.raises(M1MemoryError):
        cold_start_bin(-1.0)


# --------------------------------------------------------------------------
# I. Full-stream, label-independent history
# --------------------------------------------------------------------------


def test_i_challenge_observations_remain_in_history_without_their_labels(
    standardizer,
):
    interleaved = [
        reference(
            "r1",
            0,
            index,
            partition="validation",
            family="rate_related" if index % 3 == 1 else "background_negative",
        )
        for index in range(12)
    ]
    values = {row.stable_id: vector(index) for index, row in enumerate(interleaved)}
    labelled = generate_stream_memory(
        build_causal_streams(interleaved),
        partition="validation",
        representations=values,
        standardizer=standardizer,
    )

    # Strip the challenge *labels* while keeping the same observations at the
    # same causal positions. Memory must be unchanged.
    unlabelled = [
        reference("r1", 0, index, partition="validation")
        for index in range(12)
    ]
    stripped = generate_stream_memory(
        build_causal_streams(unlabelled),
        partition="validation",
        representations=values,
        standardizer=standardizer,
    )
    np.testing.assert_array_equal(labelled.d_short, stripped.d_short)
    assert len(stripped.stable_ids) == 12


def test_i_removing_the_observations_does_change_later_memory(standardizer):
    full = [reference("r1", 0, index, partition="validation") for index in range(12)]
    values = {row.stable_id: vector(index) for index, row in enumerate(full)}
    complete = generate_stream_memory(
        build_causal_streams(full),
        partition="validation",
        representations=values,
        standardizer=standardizer,
    )
    thinned = [row for index, row in enumerate(full) if index % 3 != 1]
    partial = generate_stream_memory(
        build_causal_streams(thinned),
        partition="validation",
        representations=values,
        standardizer=standardizer,
    )
    shared = select_rows(complete, partial.stable_ids)
    assert not np.array_equal(partial.d_short[1:], complete.d_short[shared][1:])


# --------------------------------------------------------------------------
# J. Always-update: M1 is deliberately not contamination-safe
# --------------------------------------------------------------------------


def test_j_an_abnormal_like_observation_still_enters_memory(standardizer):
    memory = DualTimescaleMemory(standardizer.prior_vector())
    normal = standardizer.standardize(vector(4))[0]
    abnormal = standardizer.standardize(vector(4) + 60.0)[0]

    memory.observe(normal)
    before = memory.mu_short.copy()
    scored = memory.observe(abnormal)
    # It is flagged as deviant when scored...
    assert scored.d_short > 10.0
    # ...and then admitted anyway. This is the documented M1-v1 limitation.
    assert not np.array_equal(before, memory.mu_short)
    assert memory.past_update_count == 2

    follow_up = memory.deviations(abnormal)
    assert follow_up.d_short < scored.d_short


def test_j_boundary_statement_declares_the_limitation():
    boundary = m1_boundary_statement()
    assert boundary["contamination_safe"] is False
    assert boundary["update_policy"] == "finite_observation_always_update"
    assert boundary["m2_required_before_safe_adaptation_claim"] is True
    assert boundary["label_gated_update"] is False
    assert boundary["score_gated_update"] is False


# --------------------------------------------------------------------------
# K. Non-finite refusal
# --------------------------------------------------------------------------


def test_k_non_finite_observation_fails_rather_than_being_skipped(standardizer):
    memory = DualTimescaleMemory(standardizer.prior_vector())
    broken = standardizer.standardize(vector(5))[0].copy()
    broken[3] = np.nan
    with pytest.raises(M1MemoryError, match="Non-finite"):
        memory.observe(broken)
    assert memory.past_update_count == 0


def test_k_non_finite_representation_is_refused_by_the_standardizer(standardizer):
    values = vector(6).astype(np.float64)
    values[0] = np.inf
    with pytest.raises(M1MemoryError, match="non-finite"):
        standardizer.standardize(values)


def test_k_stream_generation_propagates_the_refusal(standardizer):
    rows = [reference("r1", 0, i) for i in range(3)]
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    values[rows[1].stable_id] = values[rows[1].stable_id].copy()
    values[rows[1].stable_id][0] = np.nan
    with pytest.raises(M1MemoryError):
        generate_stream_memory(
            build_causal_streams(rows),
            partition="train",
            representations=values,
            standardizer=standardizer,
        )


# --------------------------------------------------------------------------
# L. Physiology freeze
# --------------------------------------------------------------------------


def test_l_m1_never_fits_a_physiology_transform():
    module = ast.parse(Path(m1_experiment.__file__).read_text())
    called = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "fit_physiology_transform" not in called
    imported = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "fit_physiology_transform" not in imported


def test_l_representation_dim_is_the_retained_fused_width():
    assert REPRESENTATION_DIM == 146
    assert m1_experiment.REPRESENTATION_DIM == 146


# --------------------------------------------------------------------------
# M. Primary supervision separation
# --------------------------------------------------------------------------


def test_m_history_holds_every_window_while_training_selects_only_primary(
    standardizer,
):
    rows = [
        reference(
            "r1",
            0,
            index,
            family="rate_related" if index % 4 == 3 else "background_negative",
        )
        for index in range(16)
    ]
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    memory = generate_stream_memory(
        build_causal_streams(rows),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    assert len(memory.stable_ids) == 16

    primary = [row.stable_id for row in rows if row.target_family != "rate_related"]
    selected = select_rows(memory, primary)
    assert selected.shape[0] == 12
    matrix = memory.memory_matrix(M1D_EXPERIMENT_ID)[selected]
    assert matrix.shape == (12, 2)


def test_m_select_rows_refuses_an_unknown_identifier(standardizer):
    rows = [reference("r1", 0, i) for i in range(3)]
    values = {row.stable_id: vector(i) for i, row in enumerate(rows)}
    memory = generate_stream_memory(
        build_causal_streams(rows),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    with pytest.raises(M1MemoryError):
        select_rows(memory, ["ltstdb:zz:0:0:2500"])


# --------------------------------------------------------------------------
# N. Challenge causal position
# --------------------------------------------------------------------------


def test_n_challenge_window_sees_only_earlier_stream_observations(standardizer):
    rows = [reference("r1", 0, index, partition="validation") for index in range(9)]
    challenge_at = 5
    rows[challenge_at] = reference(
        "r1", 0, challenge_at, partition="validation", family="axis_shift"
    )
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    memory = generate_stream_memory(
        build_causal_streams(rows),
        partition="validation",
        representations=values,
        standardizer=standardizer,
    )

    replay = DualTimescaleMemory(standardizer.prior_vector())
    for row in rows[:challenge_at]:
        replay.observe(standardizer.standardize(values[row.stable_id])[0])
    expected = replay.deviations(
        standardizer.standardize(values[rows[challenge_at].stable_id])[0]
    )
    position = memory.index()[rows[challenge_at].stable_id]
    assert memory.d_short[position] == pytest.approx(expected.d_short)
    assert memory.past_observed_count[position] == challenge_at


# --------------------------------------------------------------------------
# O. Deterministic replay
# --------------------------------------------------------------------------


def test_o_identical_streams_produce_identical_memory(standardizer):
    rows = [
        reference(record, channel, index)
        for record in ("rA", "rB")
        for channel in (0, 1)
        for index in range(7)
    ]
    values = {row.stable_id: vector(hash(row.stable_id) % 5000) for row in rows}
    first = generate_stream_memory(
        build_causal_streams(rows),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    shuffled = list(reversed(rows))
    second = generate_stream_memory(
        build_causal_streams(shuffled),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    assert first.stable_ids == second.stable_ids
    assert first.chronology_sha256 == second.chronology_sha256
    np.testing.assert_array_equal(first.d_short, second.d_short)
    np.testing.assert_array_equal(first.d_long, second.d_long)


def test_o_duplicate_start_samples_in_one_stream_are_refused():
    duplicated = [reference("r1", 0, 0), reference("r1", 0, 0)]
    with pytest.raises(M1MemoryError, match="strictly increasing"):
        build_causal_streams(duplicated)


# --------------------------------------------------------------------------
# P. Test firewall
# --------------------------------------------------------------------------


def test_p_no_m1_entry_point_accepts_the_test_partition(standardizer):
    with pytest.raises(Exception):
        fit_distance_standardizer(
            np.zeros((4, REPRESENTATION_DIM)), partition="test"
        )
    with pytest.raises(Exception):
        generate_stream_memory(
            {}, partition="test", representations={}, standardizer=standardizer
        )


def test_p_m1_modules_never_import_a_sealed_test_evaluator():
    for module in (patient_memory, m1_experiment):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "sealed_test" not in (node.module or "")
                for alias in node.names:
                    assert "sealed_test" not in alias.name
                    assert "evaluate_locked_test" not in alias.name
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "sealed_test" not in alias.name


def test_p_stage1_result_and_preflight_record_the_firewall():
    tree = ast.parse(Path(m1_experiment.__file__).read_text())
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "test_accessed" in literals
    assert "unopened" in literals


# --------------------------------------------------------------------------
# Q. M1 / M2 boundary
# --------------------------------------------------------------------------


def test_q_m1_declares_no_m2_mechanism():
    boundary = m1_boundary_statement()
    for field in (
        "rollback",
        "uncertainty_admission",
        "event_state_admission",
        "conformal_admission",
        "label_gated_update",
        "score_gated_update",
        "patient_identity_is_a_feature",
        "cross_recording_state_carryover",
    ):
        assert boundary[field] is False
    assert boundary["memory_resets_at_recording_channel_boundary"] is True


def test_q_memory_state_exposes_no_rollback_path():
    forbidden = {"rollback", "revert", "undo", "restore", "reset"}
    assert not forbidden.intersection(dir(DualTimescaleMemory))


def test_q_no_force_or_retry_flag_exists_in_the_m1_modules():
    for module in (patient_memory, m1_experiment):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"unlink", "rmtree", "rmdir"}
            if isinstance(node, ast.keyword):
                assert node.arg not in {"force", "overwrite", "retry", "fresh_seed"}


# --------------------------------------------------------------------------
# D2. Three-channel records
#
# The frozen development corpus holds both 2-channel and 3-channel LTSTDB
# records (observed indices {0, 1, 2}). An earlier chronology audit wrongly
# reported two channels everywhere, and that claim reached the protocol text.
# The stream key was already generic, and these tests pin that: nothing may
# start assuming exactly two channels.
# --------------------------------------------------------------------------


def _three_channel_record(windows: int = 6):
    return [
        reference("rA", channel, index, partition="train")
        for channel in (0, 1, 2)
        for index in range(windows)
    ]


def test_d2_a_three_channel_record_yields_three_independent_streams():
    streams = build_causal_streams(_three_channel_record())
    assert set(streams) == {("rA", 0), ("rA", 1), ("rA", 2)}
    assert len(streams) == 3
    for key, rows in streams.items():
        assert len(rows) == 6
        assert all(row.channel_index == key[1] for row in rows)


def test_d2_channel_two_is_accepted_and_keyed_independently():
    assert stream_key(reference("rA", 2, 0)) == ("rA", 2)
    assert stream_key(reference("rA", 2, 0)) != stream_key(reference("rA", 1, 0))
    assert stream_key(reference("rA", 2, 0)) != stream_key(reference("rA", 0, 0))


@pytest.mark.parametrize("perturbed", [0, 1, 2])
def test_d2_no_channel_contaminates_any_other(standardizer, perturbed):
    """Perturb one lead entirely; the other two must be bit-identical."""
    rows = _three_channel_record()
    baseline = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    altered = dict(baseline)
    for row in rows:
        if row.channel_index == perturbed:
            altered[row.stable_id] = vector(4242, scale=75.0)

    streams = build_causal_streams(rows)
    first = generate_stream_memory(
        streams, partition="train", representations=baseline, standardizer=standardizer
    )
    second = generate_stream_memory(
        streams, partition="train", representations=altered, standardizer=standardizer
    )

    index = first.index()
    untouched = [row for row in rows if row.channel_index != perturbed]
    assert {row.channel_index for row in untouched} == {0, 1, 2} - {perturbed}
    for row in untouched:
        position = index[row.stable_id]
        assert first.d_short[position] == second.d_short[position]
        assert first.d_long[position] == second.d_long[position]

    touched = [row for row in rows if row.channel_index == perturbed]
    assert any(
        first.d_short[index[row.stable_id]] != second.d_short[index[row.stable_id]]
        for row in touched
    )


def test_d2_every_channel_cold_starts_from_the_global_prior(standardizer):
    rows = _three_channel_record()
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    memory = generate_stream_memory(
        build_causal_streams(rows),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    prior = standardizer.prior_vector()
    index = memory.index()
    for channel in (0, 1, 2):
        opening = next(row for row in rows if row.channel_index == channel)
        position = index[opening.stable_id]
        expected = standardizer.standardize(values[opening.stable_id])[0]
        assert memory.d_short[position] == pytest.approx(
            float(np.sqrt(np.mean((expected - prior) ** 2)))
        )
        assert memory.past_observed_count[position] == 0
        assert memory.recording_age_seconds[position] == 0.0


def test_d2_ordering_within_each_channel_follows_start_samples(standardizer):
    rows = _three_channel_record()
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    memory = generate_stream_memory(
        build_causal_streams(list(reversed(rows))),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    seen: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for position, channel in enumerate(memory.channel_indices.tolist()):
        seen[channel].append(int(memory.start_samples[position]))
    for channel, starts in seen.items():
        assert starts == sorted(starts), channel
        assert len(starts) == 6


def test_d2_mixed_two_and_three_channel_records_coexist(standardizer):
    """The corpus holds both shapes; stream counts must follow the data."""
    rows = _three_channel_record(4) + [
        reference("rB", channel, index, partition="train")
        for channel in (0, 1)
        for index in range(4)
    ]
    streams = build_causal_streams(rows)
    assert len(streams) == 5  # 3 + 2, exactly as the real corpus mixes them
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    memory = generate_stream_memory(
        streams, partition="train", representations=values, standardizer=standardizer
    )
    assert sorted({int(v) for v in memory.channel_indices}) == [0, 1, 2]
    assert len(memory.streams) == 5


def test_d2_no_module_assumes_a_two_channel_record():
    """Nothing may hard-code the channel set."""
    for module in (patient_memory, m1_experiment):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Set | ast.Tuple | ast.List):
                        values = [
                            element.value
                            for element in comparator.elts
                            if isinstance(element, ast.Constant)
                        ]
                        assert values != [0, 1], (
                            "a hard-coded {0, 1} channel set would exclude the "
                            "third lead present in the frozen corpus"
                        )
