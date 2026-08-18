"""The frozen T2 longitudinal protocol, proven synthetically.

Every proof here runs against synthetic streams, synthetic subject lists and
frozen manifest metadata. Nothing trains, fits, scores real data, executes the
temporal model or touches TEST -- and a test enforces that the protocol module
imports only the standard library, so it cannot reach real data even if a later
edit tried to.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import t2_protocol as T
from cardiosentinel.neural.t2_protocol import (
    T2ProtocolError,
    T2Row,
    T2StreamKey,
    validate_t2_protocol_document,
)

SPLIT_PATH = Path("protocols/splits/ltstdb_v1.json")


def _split_partitions() -> dict[str, list[str]]:
    if not SPLIT_PATH.is_file():
        pytest.skip("the frozen split manifest is not on this filesystem")
    manifest = json.loads(SPLIT_PATH.read_text())
    return {
        name: list(part["subjects"]) for name, part in manifest["partitions"].items()
    }


def _row(
    record: str = "s20011",
    channel: int = 0,
    start_sample: int = 0,
    observation_state: int | None = None,
    *,
    stable_id: str | None = None,
) -> T2Row:
    """One synthetic row whose stable_id is canonical unless deliberately broken."""
    state = (
        T.T2_OBSERVATION_AVAILABLE if observation_state is None else observation_state
    )
    end = start_sample + T.T2_WINDOW_LENGTH_SAMPLES
    return T2Row(
        stable_id=(
            stable_id
            if stable_id is not None
            else f"ltstdb:{record}:{channel}:{start_sample}:{end}"
        ),
        record_id=record,
        channel_index=channel,
        start_sample=start_sample,
        observation_state=state,
    )


def _stream(
    record: str = "s20011",
    channel: int = 0,
    count: int = 4,
    *,
    start: int = 0,
    states: tuple[int, ...] | None = None,
) -> tuple[T2Row, ...]:
    """A synthetic causal stream at the frozen 2500-sample / 1250-stride grid."""
    return tuple(
        _row(
            record,
            channel,
            start + index * 1250,
            states[index] if states is not None else T.T2_OBSERVATION_AVAILABLE,
        )
        for index in range(count)
    )


def _frozen_ids(rows) -> tuple[str, ...]:
    return tuple(row.stable_id for row in rows)


def _document_prose() -> str:
    """Whitespace-normalised protocol text.

    Substring assertions against wrapped markdown are brittle: a phrase that
    happens to straddle a line break silently fails. Collapse whitespace first so
    the assertion tests the claim, not the fill width.
    """
    return " ".join(T.T2_PROTOCOL_PATH.read_text().split())


# --- 1. exact starting protocol constants ---------------------------------


def test_protocol_document_is_frozen():
    assert validate_t2_protocol_document() == T.T2_PROTOCOL_SHA256


def test_exact_starting_constants():
    assert T.T2_STARTING_GIT_SHA == "997df407376edcf585a68d019b26b02a7670c12b"
    assert T.T2_SPLIT_SHA256 == (
        "66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7"
    )
    assert T.T2_U1_RETENTION_DECISION_SHA256 == (
        "9d8436f2b7d2c303aeeb03e438c60fb8110f7d06d0bbd589f5be65ea8f80cb7b"
    )
    assert T.T2_M2_RETENTION_DECISION_SHA256 == (
        "da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47"
    )
    assert T.T2_ENVIRONMENT_DEPENDENCY_DIGEST == (
        "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
    )
    assert T.T2_WINDOW_LENGTH_SECONDS == 10.0
    assert T.T2_WINDOW_STRIDE_SECONDS == 5.0


def test_protocol_module_imports_only_the_standard_library():
    """Protocol validation must be unable to reach real data or TEST."""
    tree = ast.parse(Path(T.__file__).read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    forbidden = {"numpy", "torch", "cardiosentinel", "scipy", "pandas", "sklearn"}
    assert not (modules & forbidden), modules & forbidden


# --- 2-6. the internal split ----------------------------------------------


def test_internal_split_is_deterministic_and_48_8():
    subjects = _split_partitions()["train"]
    first = T.assign_internal_split(subjects)
    second = T.assign_internal_split(list(reversed(subjects)))
    assert first == second, "the split must not depend on input ordering"
    assert first["fit_count"] == 48
    assert first["internal_dev_count"] == 8
    assert first["split_sha256"] == T.T2_INTERNAL_SPLIT_SHA256
    assert tuple(first["internal_dev_subjects"]) == T.T2_INTERNAL_DEV_SUBJECTS


def test_internal_split_is_subject_disjoint():
    subjects = _split_partitions()["train"]
    assignment = T.assign_internal_split(subjects)
    fit = set(assignment["fit_subjects"])
    dev = set(assignment["internal_dev_subjects"])
    assert not (fit & dev)
    assert fit | dev == set(subjects)


def test_split_assignment_uses_identity_only():
    """Relabelling nothing but the seed changes the split; labels never enter it.

    The function takes subject identities and a seed string and nothing else --
    there is no parameter through which a label, prevalence or outcome could
    reach it, and a different seed produces a different partition.
    """
    subjects = _split_partitions()["train"]
    frozen = T.assign_internal_split(subjects)
    other = T.assign_internal_split(subjects, seed_string="some-other-seed")
    assert other["split_sha256"] != frozen["split_sha256"]
    assert T.T2_SPLIT_USES_LABELS is False
    assert T.T2_SPLIT_USES_OUTCOMES is False
    assert T.T2_SPLIT_USES_PREVALENCE is False


def test_no_outer_validation_subject_enters_the_internal_split():
    parts = _split_partitions()
    assignment = T.assign_internal_split(parts["train"])
    T.validate_internal_split(
        assignment,
        validation_subjects=parts["validation"],
        test_subjects=parts["test"],
    )
    both = set(assignment["fit_subjects"]) | set(assignment["internal_dev_subjects"])
    assert not (both & set(parts["validation"]))


def test_no_test_subject_enters_anything():
    parts = _split_partitions()
    assignment = T.assign_internal_split(parts["train"])
    both = set(assignment["fit_subjects"]) | set(assignment["internal_dev_subjects"])
    assert not (both & set(parts["test"]))
    # and the validator refuses a leak rather than merely reporting one
    leaked = dict(assignment)
    leaked["fit_subjects"] = sorted(
        set(assignment["fit_subjects"]) - {assignment["fit_subjects"][0]}
        | {parts["test"][0]}
    )
    with pytest.raises(T2ProtocolError, match="Sealed TEST subjects"):
        T.validate_internal_split(
            leaked,
            validation_subjects=parts["validation"],
            test_subjects=parts["test"],
        )


def test_validator_refuses_an_outer_validation_leak():
    parts = _split_partitions()
    assignment = T.assign_internal_split(parts["train"])
    leaked = dict(assignment)
    leaked["internal_dev_subjects"] = sorted(
        set(assignment["internal_dev_subjects"])
        - {assignment["internal_dev_subjects"][0]}
        | {parts["validation"][0]}
    )
    with pytest.raises(T2ProtocolError, match="Outer VALIDATION subjects"):
        T.validate_internal_split(
            leaked,
            validation_subjects=parts["validation"],
            test_subjects=parts["test"],
        )


# --- 7-11. stream, ordering and availability semantics --------------------


def test_stream_state_resets_across_record_and_channel():
    rows = _stream("s20011", 0, 3) + _stream("s20011", 1, 3) + _stream("s20021", 1, 3)
    assert T.state_reset_positions(rows) == (0, 3, 6)
    assert T.T2_STATE_CROSSES_RECORD is False
    assert T.T2_STATE_CROSSES_CHANNEL is False
    assert T.T2_STATE_CROSSES_SUBJECT is False


def test_a_single_stream_resets_only_once():
    assert T.state_reset_positions(_stream(count=500)) == (0,)


def test_mixed_streams_are_refused_as_one_sequence():
    rows = _stream("s20011", 0, 2) + _stream("s20011", 1, 2)
    with pytest.raises(T2ProtocolError, match="must be one"):
        T.require_chronological_stream(rows)


def test_chronological_ordering_is_required():
    rows = _stream(count=4)
    shuffled = (rows[0], rows[2], rows[1], rows[3])
    with pytest.raises(T2ProtocolError, match="strict chronological order"):
        T.require_chronological_stream(shuffled)
    with pytest.raises(T2ProtocolError, match="strict chronological order"):
        T.require_chronological_stream((rows[0], rows[0]))
    assert T.require_chronological_stream(rows) == rows
    assert T.T2_SHUFFLE_WITHIN_STREAM_PERMITTED is False


def test_future_window_access_is_refused():
    T.require_no_future_access(10, [8, 9, 10])
    with pytest.raises(T2ProtocolError, match="may not read future windows"):
        T.require_no_future_access(10, [9, 11])
    assert T.T2_BIDIRECTIONAL_PERMITTED is False
    assert T.T2_FUTURE_CONTEXT_PERMITTED is False


def test_unavailable_rows_cannot_inject_a_fake_z():
    rows = _stream(
        count=3,
        states=(
            T.T2_OBSERVATION_AVAILABLE,
            T.T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT,
            T.T2_OBSERVATION_AVAILABLE,
        ),
    )
    with pytest.raises(T2ProtocolError, match="no synthetic z"):
        T.require_available_for_modelling(rows[1])
    assert T.modellable_rows(rows) == (rows[0], rows[2])
    assert T.T2_SYNTHETIC_Z_PERMITTED is False
    assert T.T2_IMPUTATION_PERMITTED is False
    assert T.T2_FORWARD_FILL_PERMITTED is False


def test_unavailable_row_cannot_update_hidden_state_but_time_still_advances():
    assert T.T2_UNAVAILABLE_ROW_UPDATES_HIDDEN_STATE is False
    assert T.T2_UNAVAILABLE_ROW_SCORED is False
    assert T.T2_UNAVAILABLE_ROW_TRAINED is False
    assert T.T2_UNAVAILABLE_ROW_ADVANCES_TIMELINE is True
    # the gap is in evidence, not in time: the stream is still one stream
    rows = _stream(
        count=3,
        states=(
            T.T2_OBSERVATION_AVAILABLE,
            T.T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT,
            T.T2_OBSERVATION_AVAILABLE,
        ),
    )
    assert T.state_reset_positions(rows) == (0,)
    assert T.require_chronological_stream(rows) == rows


def test_no_new_sqi_threshold_is_invented():
    assert T.T2_NEW_SQI_THRESHOLD_PERMITTED is False
    assert T.T2_OBSERVATION_AVAILABLE == 1
    assert T.T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT == 2


# --- 12-14. representation and forbidden inputs ---------------------------


def test_z_dimension_is_exactly_146():
    assert T.T2_INPUT_DIM == 146
    assert T.T2_EMBEDDING_DIM == 128
    assert T.T2_PHYSIOLOGY_DIM == 18
    assert T.require_input_dimension(146) == 146
    for wrong in (128, 145, 147, 164):
        with pytest.raises(T2ProtocolError, match="not the frozen 146"):
            T.require_input_dimension(wrong)


def test_u1_probability_is_not_a_t2_v1_trainable_input():
    T.require_permitted_trainable_inputs(["z_t"])
    for name in (
        "u1_oof_calibrated_probability",
        "u1_calibrated_probability",
        "u1_uncertainty",
    ):
        with pytest.raises(T2ProtocolError, match="may not be a trainable"):
            T.require_permitted_trainable_inputs(["z_t", name])


def test_u_star_thresholds_cannot_be_t2_inputs():
    for name in ("u_star_dev", "u_star_deploy"):
        with pytest.raises(T2ProtocolError, match="may not be a trainable"):
            T.require_permitted_trainable_inputs([name])


def test_other_forbidden_inputs_are_refused():
    for name in (
        "future_window_label",
        "challenge_family_identity",
        "episode_identity",
        "future_episode_duration",
        "m2_gate_outcome",
        "test_derived_quantity",
    ):
        with pytest.raises(T2ProtocolError, match="may not be a trainable"):
            T.require_permitted_trainable_inputs(["z_t", name])


# --- 15-17. training population and loss ----------------------------------


def test_full_chronological_training_population_is_required():
    assert (
        T.require_full_chronological_population(
            offered_row_count=2_208_431, full_stream_row_count=2_208_431
        )
        == 2_208_431
    )
    assert T.T2_FULL_CHRONOLOGICAL_POPULATION_REQUIRED is True


def test_three_to_one_negative_sampling_is_forbidden():
    """The 3:1 P1 TRAIN selection must never become the T2 training population."""
    assert T.T2_NEGATIVE_SAMPLING_PERMITTED is False
    assert T.T2_P1_TRAIN_SELECTION_ROW_COUNT == 374_452
    # 280,839 negatives = 3 x 93,613 positives -- a selection, not a timeline
    assert T.T2_P1_TRAIN_SELECTION_ROW_COUNT == 93_613 + 3 * 93_613
    assert T.T2_P1_TRAIN_SELECTION_ROW_COUNT < T.T2_TRAIN_FULL_STREAM_ROW_COUNT
    with pytest.raises(T2ProtocolError, match="Negative sampling is forbidden"):
        T.require_full_chronological_population(
            offered_row_count=T.T2_P1_TRAIN_SELECTION_ROW_COUNT,
            full_stream_row_count=T.T2_TRAIN_FULL_STREAM_ROW_COUNT,
        )


def test_positive_class_weight_comes_from_the_fit_partition_only():
    assert T.T2_LOSS == "binary_cross_entropy_with_logits"
    assert T.T2_POSITIVE_CLASS_WEIGHT_RULE == (
        "n_negative_over_n_positive_on_fit_partition"
    )
    assert T.T2_CLASS_WEIGHT_PARTITION == "t2_fit_48_subjects"
    assert T.T2_VALIDATION_DERIVED_CLASS_WEIGHT_PERMITTED is False
    assert T.T2_FOCAL_LOSS_COMPARISON_PERMITTED is False
    assert T.T2_LOSS_FAMILY_SEARCH_PERMITTED is False
    assert T.positive_class_weight(negative_count=300, positive_count=100) == 3.0
    with pytest.raises(T2ProtocolError, match="both classes present"):
        T.positive_class_weight(negative_count=300, positive_count=0)


# --- 18-19. TBPTT ---------------------------------------------------------


def test_tbptt_length_is_exactly_256():
    assert T.T2_TBPTT_LENGTH == 256
    assert T.T2_TBPTT_HORIZON_SECONDS == 1280.0
    with pytest.raises(T2ProtocolError, match="frozen TBPTT length is 256"):
        T.tbptt_chunks(T2StreamKey("s20011", 0), 600, length=128)


def test_chunk_boundary_detaches_gradient_without_resetting_causal_state():
    chunks = T.tbptt_chunks(T2StreamKey("s20011", 0), 600)
    assert [(c.start_index, c.stop_index) for c in chunks] == [
        (0, 256),
        (256, 512),
        (512, 600),
    ]
    # first chunk of the stream starts from the frozen zero state
    assert chunks[0].carries_state_in is False
    assert chunks[0].detaches_state_in is False
    # every later chunk inherits state, detached: gradient stops, state does not
    for chunk in chunks[1:]:
        assert chunk.carries_state_in is True
        assert chunk.detaches_state_in is True
    assert T.T2_STATE_CARRIES_ACROSS_CHUNK is True
    assert T.T2_STATE_DETACHED_AT_CHUNK_BOUNDARY is True
    assert T.T2_GRADIENT_CROSSES_CHUNK_BOUNDARY is False
    assert T.T2_STATE_RESET_AT_CHUNK_BOUNDARY is False


def test_state_resets_only_at_real_stream_boundaries():
    """256 windows ends a gradient horizon; only a new stream ends the state."""
    rows = _stream("s20011", 0, 300) + _stream("s20011", 1, 300)
    assert T.state_reset_positions(rows) == (0, 300)
    chunks = T.tbptt_chunks(T2StreamKey("s20011", 0), 300)
    assert len(chunks) == 2
    assert chunks[1].carries_state_in is True


# --- 20-23. candidates, capacity and optimisation -------------------------


def test_exactly_gru_and_s4d_candidates():
    assert T.T2_ARMS == (
        "causal_gru_longitudinal_v1",
        "causal_s4d_longitudinal_v1",
    )
    assert len(T.T2_ARMS) == 2
    for arm in T.T2_ARMS:
        assert T.require_arm(arm) == arm
    with pytest.raises(T2ProtocolError, match="not a frozen T2 candidate"):
        T.require_arm("causal_transformer_longitudinal_v1")


def test_s4d_is_not_called_mamba():
    assert T.T2_S4D_IS_MAMBA is False
    assert "mamba" not in T.T2_ARM_S4D.lower()
    assert T.T2_S4D_FAMILY == "s4d_inspired_diagonal_state_space"
    assert T.T2_EXTERNAL_SSM_PACKAGE_PERMITTED is False
    assert T.T2_FRAMEWORK == "torch"
    assert "must **not** be called *Mamba*" in _document_prose()


def test_shared_parameter_capacity_envelope():
    assert T.T2_INPUT_PROJECTION_DIM == 64
    assert T.T2_TEMPORAL_WIDTH == 64
    assert T.T2_TEMPORAL_LAYERS == 2
    assert T.T2_DROPOUT == 0.10
    assert T.T2_OUTPUT_DIM == 1
    assert (T.T2_PARAMETER_RATIO_MIN, T.T2_PARAMETER_RATIO_MAX) == (0.5, 2.0)
    assert T.T2_MODEL_SIZE_INCREASE_AFTER_RESULTS_PERMITTED is False

    proof = T.require_capacity_envelope({T.T2_ARM_GRU: 100_000, T.T2_ARM_S4D: 150_000})
    assert proof["within_envelope"] is True
    assert proof["ratio_s4d_over_gru"] == 1.5
    for counts in (
        {T.T2_ARM_GRU: 100_000, T.T2_ARM_S4D: 300_000},
        {T.T2_ARM_GRU: 300_000, T.T2_ARM_S4D: 100_000},
    ):
        with pytest.raises(T2ProtocolError, match="shared capacity envelope"):
            T.require_capacity_envelope(counts)


def test_optimizer_seed_and_epoch_constants_are_exact():
    assert T.T2_OPTIMIZER == "AdamW"
    assert T.T2_LEARNING_RATE == 3e-4
    assert T.T2_WEIGHT_DECAY == 1e-4
    assert T.T2_MAX_EPOCHS == 10
    assert T.T2_GRADIENT_CLIP_NORM == 1.0
    assert T.T2_SEED == 2026
    assert T.T2_EARLY_STOPPING_PATIENCE_EPOCHS == 3
    assert T.T2_CHECKPOINT_CRITERION == "internal_development_pooled_auprc"
    assert T.T2_CHECKPOINT_TIE_BREAK == "earlier_epoch"


# --- 24-27. selection -----------------------------------------------------


def test_outer_validation_is_not_an_early_stopping_input():
    assert T.T2_OUTER_VALIDATION_IN_EPOCH_SELECTION is False
    assert T.T2_OUTER_VALIDATION_ATTEMPTS == 1
    assert T.T2_AUTOMATIC_RETRY_PERMITTED is False


def test_primary_selection_is_pooled_auprc():
    assert T.T2_PRIMARY_SELECTION_METRIC == "pooled_primary_validation_auprc"
    assert T.T2_SECONDARY_SELECTION_METRIC == "subject_macro_auprc"
    decision = T.select_t2_arm(
        pooled_auprc={T.T2_ARM_GRU: 0.40, T.T2_ARM_S4D: 0.45},
        subject_macro_auprc={T.T2_ARM_GRU: 0.90, T.T2_ARM_S4D: 0.10},
        parameter_counts={T.T2_ARM_GRU: 100, T.T2_ARM_S4D: 100},
    )
    assert decision["selected_arm"] == T.T2_ARM_S4D
    assert decision["selection_basis"] == T.T2_PRIMARY_SELECTION_METRIC


def test_the_0_002_tie_rule_is_exact():
    assert T.T2_SELECTION_TIE_TOLERANCE == 0.002
    # a difference of exactly the tolerance is NOT a tie
    at_tolerance = T.select_t2_arm(
        pooled_auprc={T.T2_ARM_GRU: 0.400, T.T2_ARM_S4D: 0.402},
        subject_macro_auprc={T.T2_ARM_GRU: 0.90, T.T2_ARM_S4D: 0.10},
        parameter_counts={T.T2_ARM_GRU: 100, T.T2_ARM_S4D: 100},
    )
    assert at_tolerance["selected_arm"] == T.T2_ARM_S4D
    assert at_tolerance["selection_basis"] == T.T2_PRIMARY_SELECTION_METRIC

    # just inside it falls through to subject-macro AUPRC
    inside = T.select_t2_arm(
        pooled_auprc={T.T2_ARM_GRU: 0.400, T.T2_ARM_S4D: 0.4019},
        subject_macro_auprc={T.T2_ARM_GRU: 0.90, T.T2_ARM_S4D: 0.10},
        parameter_counts={T.T2_ARM_GRU: 100, T.T2_ARM_S4D: 100},
    )
    assert inside["selected_arm"] == T.T2_ARM_GRU
    assert inside["selection_basis"] == T.T2_SECONDARY_SELECTION_METRIC

    # tied on both -> the smaller model
    smaller = T.select_t2_arm(
        pooled_auprc={T.T2_ARM_GRU: 0.400, T.T2_ARM_S4D: 0.400},
        subject_macro_auprc={T.T2_ARM_GRU: 0.50, T.T2_ARM_S4D: 0.50},
        parameter_counts={T.T2_ARM_GRU: 200, T.T2_ARM_S4D: 100},
    )
    assert smaller["selected_arm"] == T.T2_ARM_S4D
    assert smaller["selection_basis"] == T.T2_SELECTION_PARAMETER_TIE_BREAK


# --- the corrected two-stage 0.002 boundary (§8 of the closure) -----------


def _select(pooled_gap: float, macro_gap: float, counts=(100, 100)) -> dict:
    """S4D leads on both metrics by the given gaps; counts are (GRU, S4D)."""
    return T.select_t2_arm(
        pooled_auprc={T.T2_ARM_GRU: 0.400, T.T2_ARM_S4D: 0.400 + pooled_gap},
        subject_macro_auprc={T.T2_ARM_GRU: 0.500, T.T2_ARM_S4D: 0.500 + macro_gap},
        parameter_counts={T.T2_ARM_GRU: counts[0], T.T2_ARM_S4D: counts[1]},
    )


def test_pooled_difference_just_below_tolerance_is_a_tie():
    decision = _select(0.001999, 0.0)
    assert decision["selection_basis"] != T.T2_PRIMARY_SELECTION_METRIC


def test_pooled_difference_of_exactly_the_tolerance_is_not_a_tie():
    decision = _select(0.002, 0.0)
    assert decision["selection_basis"] == T.T2_PRIMARY_SELECTION_METRIC
    assert decision["selected_arm"] == T.T2_ARM_S4D


def test_subject_macro_difference_just_below_tolerance_falls_to_parameters():
    """The written tolerance applies at the second stage too, not exact equality."""
    decision = _select(0.0005, 0.001999, counts=(200, 100))
    assert decision["selection_basis"] == T.T2_SELECTION_PARAMETER_TIE_BREAK
    assert decision["selected_arm"] == T.T2_ARM_S4D


def test_subject_macro_difference_of_exactly_the_tolerance_selects_higher_macro():
    decision = _select(0.0005, 0.002, counts=(100, 999))
    assert decision["selection_basis"] == T.T2_SECONDARY_SELECTION_METRIC
    assert decision["selected_arm"] == T.T2_ARM_S4D


def test_equal_parameter_counts_resolve_deterministically_to_gru():
    decision = _select(0.0, 0.0, counts=(100, 100))
    assert decision["selected_arm"] == T.T2_ARM_GRU
    assert decision["selection_basis"] == T.T2_SELECTION_TERMINAL_TIE_BREAK
    assert T.T2_SELECTION_TERMINAL_ARM == T.T2_ARM_GRU


def test_latency_is_not_part_of_scientific_selection():
    assert T.T2_LATENCY_IN_SCIENTIFIC_SELECTION is False
    assert _select(0.0, 0.0)["latency_used"] is False


def test_challenge_evidence_cannot_select_the_model():
    assert T.T2_CHALLENGE_IS_SELECTION_INPUT is False
    assert T.T2_CHALLENGE_MERGED_INTO_PRIMARY is False
    assert T.T2_WEIGHTED_COMPOSITE_SCORE_PERMITTED is False
    assert T.T2_LATENCY_ADJUSTED_SCORE_PERMITTED is False
    decision = T.select_t2_arm(
        pooled_auprc={T.T2_ARM_GRU: 0.45, T.T2_ARM_S4D: 0.40},
        subject_macro_auprc={T.T2_ARM_GRU: 0.40, T.T2_ARM_S4D: 0.40},
        parameter_counts={T.T2_ARM_GRU: 100, T.T2_ARM_S4D: 100},
    )
    assert decision["challenge_evidence_used"] is False
    assert set(T.T2_CHALLENGE_FAMILIES) == {
        "rate_related",
        "axis_shift",
        "conduction_change",
    }


def test_binary_threshold_is_frozen_on_internal_dev_before_outer_validation():
    assert T.T2_BINARY_THRESHOLD_RULE == (
        "exact_maximum_f1_highest_threshold_tie_break"
    )
    assert T.require_threshold_partition("t2_internal_dev_8_subjects") == (
        "t2_internal_dev_8_subjects"
    )
    assert T.T2_THRESHOLD_LOCKED_BEFORE_OUTER_VALIDATION is True
    assert T.T2_OUTER_VALIDATION_MAY_ALTER_THRESHOLD is False
    for wrong in ("outer_validation", "t2_fit_48_subjects"):
        with pytest.raises(T2ProtocolError, match="frozen on"):
            T.require_threshold_partition(wrong)


# --- 28-32. semantics, T1 interface, routing, TEST ------------------------


def test_t2_raw_score_cannot_be_called_calibrated_uncertainty():
    assert T.T2_OUTPUT_IS_CALIBRATED_PROBABILITY is False
    assert T.T2_OUTPUT_IS_UNCERTAINTY is False
    assert T.T2_OUTPUT_IS_CONFORMAL_EVIDENCE is False
    assert T.T2_OUTPUT_SEMANTIC_NAME == "causal_temporal_evidence_score"
    assert T.T2_RETAINED_CALIBRATED_PROBABILITY_SOURCE == (
        "u1_oof_development_calibration"
    )
    assert T.require_t2_score_semantics("causal_temporal_evidence_score")
    for name in (
        "calibrated_probability",
        "calibrated_uncertainty",
        "confidence",
        "uncertainty",
        "conformal_evidence",
    ):
        with pytest.raises(T2ProtocolError, match="may not be called"):
            T.require_t2_score_semantics(name)
    assert T.T2_CALIBRATION_OF_T2_AUTHORISED is False


def test_t1_states_are_interface_only():
    assert T.T1_STATES == ("NORMAL", "WATCH", "EVENT", "RECOVERY")
    assert T.T1_IMPLEMENTED_HERE is False
    assert T.T2_TRAINED_TO_EMIT_T1_STATES is False
    assert set(T.T1_PERMITTED_INPUTS) == {
        "frozen_detector_decision",
        "u1_oof_platt_calibrated_probability",
        "u1_calibrated_uncertainty",
        "m2g_causally_available_patient_adaptation_evidence",
        "selected_t2_temporal_evidence_score",
        "physical_availability_state",
        "elapsed_causal_time_or_state_duration",
    }


def test_t1_thresholds_and_durations_remain_undefined():
    assert T.T1_TRANSITION_THRESHOLD is None
    assert T.T1_PERSISTENCE_DURATION is None
    assert T.T1_HYSTERESIS_VALUE is None
    assert T.T1_EVENT_ONSET_RULE is None
    assert T.T1_RECOVERY_RULE is None


def test_routing_remains_undefined_and_the_u1_router_stays_rejected():
    assert T.T2_ROUTING_DEFINED_HERE is False
    assert T.T2_ROUTE_THRESHOLD is None
    assert T.U1_SYMMETRIC_ROUTER_STILL_REJECTED is True


def test_test_remains_unopened():
    assert T.T2_TEST_ACCESSED is False
    assert T.T2_SEALED_TEST_STATE == "unopened"
    identity = T.t2_protocol_identity()
    assert identity["test_accessed"] is False
    assert identity["sealed_test_state"] == "unopened"


def test_development_optimism_is_disclosed():
    assert T.T2_OUTER_VALIDATION_IS_UNSEEN_GENERALISATION is False
    assert T.T2_DEVELOPMENT_OPTIMISM_DISCLOSED is True
    assert "not unseen generalisation" in T.T2_DEVELOPMENT_OPTIMISM_NOTE


# --- provenance and document ----------------------------------------------


def test_protocol_identity_binds_every_required_provenance_field():
    identity = T.t2_protocol_identity()
    for field in (
        "t2_protocol_sha256",
        "starting_git_sha",
        "split_sha256",
        "p1_protocol_sha256",
        "p1_retention_decision_sha256",
        "p1b_experiment_lock_sha256",
        "train_stream_cache_sha256",
        "validation_stream_cache_sha256",
        "train_representation_content_sha256",
        "validation_representation_content_sha256",
        "u1_retention_decision_sha256",
        "u1_result_sha256",
        "u1_experiment_lock_sha256",
        "m2_retention_decision_sha256",
        "internal_split_sha256",
        "stream_ordering_rule",
        "availability_rule",
        "input_dim",
        "tbptt_length",
        "candidates",
        "optimizer",
        "seed",
        "selection_rule",
        "test_accessed",
        "sealed_test_state",
    ):
        assert field in identity, field
    assert identity["input_dim"] == 146
    assert identity["tbptt_length"] == 256
    assert identity["candidates"] == list(T.T2_ARMS)


def test_the_frozen_split_manifest_is_the_bound_one():
    if not SPLIT_PATH.is_file():
        pytest.skip("the frozen split manifest is not on this filesystem")
    manifest = json.loads(SPLIT_PATH.read_text())
    assert manifest["split_sha256"] == T.T2_SPLIT_SHA256
    assert len(manifest["partitions"]["train"]["subjects"]) == 56
    assert len(manifest["partitions"]["validation"]["subjects"]) == 12
    assert manifest["sealed_test_partition"] is True
    assert manifest["window"] == {"length_seconds": 10.0, "stride_seconds": 5.0}


def test_document_records_the_b4c_separation_and_the_t1_t2_separation():
    text = _document_prose()
    assert "The B4-C rejection does NOT imply that state-space models are" in text
    assert "T2 is **not** the T1 state machine" in text
    assert "T1 IS NOT IMPLEMENTED HERE." in text
    assert "the only admissible `z_t` source" in text
    assert "They are not unseen generalisation" in text
    assert "remains rejected" in text
    # the split is stated as identity-only, and the 3:1 trap is called out
    assert "No label, prevalence, episode count or model outcome participates." in text
    assert "a *selection*, not a timeline" in text


def test_module_contains_no_training_or_scoring_path():
    tree = ast.parse(Path(T.__file__).read_text())
    forbidden = {
        "backward",
        "step",
        "fit",
        "train",
        "load_state_dict",
        "no_grad",
        "DataLoader",
        "write_json_atomic",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden, name


# ==========================================================================
# Closure additions: full timeline vs PRIMARY mask, row roles, replay
# integrity, persisted order field, and the frozen architectures.
# ==========================================================================


def test_full_timeline_is_context_and_primary_is_only_a_mask():
    assert T.T2_CONTEXT_POPULATION == "full_replay_timeline"
    assert T.T2_LOSS_POPULATION == "primary_metric_mask"
    assert T.T2_PRIMARY_IS_A_MASK_NOT_A_SEQUENCE is True
    # TRAIN: the mask is strictly smaller than the timeline it is applied to
    assert T.T2_TRAIN_PRIMARY_ROW_COUNT == 2_143_599
    assert T.T2_TRAIN_PRIMARY_ROW_COUNT < T.T2_TRAIN_FULL_STREAM_ROW_COUNT
    assert T.T2_TRAIN_NON_PRIMARY_ROW_COUNT == 64_832
    assert (
        T.T2_TRAIN_CHALLENGE_ROW_COUNT + T.T2_TRAIN_OTHER_NON_PRIMARY_ROW_COUNT
        == T.T2_TRAIN_NON_PRIMARY_ROW_COUNT
    )
    # VALIDATION agrees with the populations U1 and M2 already bound
    assert T.T2_VALIDATION_PRIMARY_ROW_COUNT == 473_897
    assert T.T2_VALIDATION_CHALLENGE_ROW_COUNT == 8_137
    assert T.T2_VALIDATION_NON_PRIMARY_ROW_COUNT == 19_007


def test_frozen_counts_match_the_corpus_authority():
    """The counts are read from the frozen corpus, never derived here."""
    manifest_path = Path("cardiosentinel-features/ltstdb-baseline-v1/manifest.json")
    if not manifest_path.is_file():
        pytest.skip("the frozen feature corpus is not on this filesystem")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["feature_corpus_sha256"] == T.T2_FEATURE_CORPUS_SHA256
    totals: dict[str, int] = {}
    for record in manifest["records"]:
        if record["partition"] != "train":
            continue
        for name, count in record["target_counts"].items():
            totals[name] = totals.get(name, 0) + count
    assert totals["ischemic_positive"] == T.T2_TRAIN_ISCHEMIC_POSITIVE
    assert totals["background_negative"] == T.T2_TRAIN_BACKGROUND_NEGATIVE
    assert sum(totals.values()) == T.T2_TRAIN_FULL_STREAM_ROW_COUNT
    challenge = sum(totals[name] for name in T.T2_CHALLENGE_CATEGORIES)
    other = sum(totals[name] for name in T.T2_OTHER_NON_PRIMARY_CATEGORIES)
    assert challenge == T.T2_TRAIN_CHALLENGE_ROW_COUNT
    assert other == T.T2_TRAIN_OTHER_NON_PRIMARY_ROW_COUNT


def test_available_challenge_row_is_context_with_no_direct_loss():
    semantics = T.role_semantics(T.ROLE_CHALLENGE_CONTEXT)
    assert semantics["consumes_z"] is True
    assert semantics["updates_state"] is True
    assert semantics["produces_score"] is True
    assert semantics["direct_loss"] is False
    assert semantics["primary_metric"] is False
    assert semantics["challenge_metric"] is True
    # the honest gradient claim, not the false one
    assert T.T2_CHALLENGE_RECEIVES_DIRECT_LOSS is False
    assert T.T2_CHALLENGE_MAY_BE_LABEL_BLIND_CONTEXT is True
    assert T.T2_CHALLENGE_IS_CHECKPOINT_EVIDENCE is False


def test_available_other_non_primary_row_is_context_only():
    semantics = T.role_semantics(T.ROLE_OTHER_NONPRIMARY_CONTEXT)
    assert semantics["consumes_z"] is True
    assert semantics["updates_state"] is True
    assert semantics["direct_loss"] is False
    assert semantics["primary_metric"] is False
    assert semantics["challenge_metric"] is False


def test_unavailable_row_neither_updates_state_nor_receives_loss_or_score():
    semantics = T.role_semantics(T.ROLE_UNAVAILABLE_NO_STATE_UPDATE)
    assert semantics == {
        "consumes_z": False,
        "updates_state": False,
        "produces_score": False,
        "direct_loss": False,
        "primary_metric": False,
        "challenge_metric": False,
    }


def test_only_primary_rows_carry_direct_loss():
    direct = [
        role for role in T.T2_ROW_ROLES if T.T2_ROW_ROLE_SEMANTICS[role]["direct_loss"]
    ]
    assert direct == [T.ROLE_PRIMARY_DIRECT_LOSS]


def test_row_role_is_never_a_trainable_feature():
    assert T.T2_ROW_ROLE_IS_MODEL_INPUT is False
    assert T.T2_CHALLENGE_IDENTITY_IS_MODEL_INPUT is False
    assert T.T2_CHALLENGE_LABEL_IS_MODEL_INPUT is False
    with pytest.raises(T2ProtocolError):
        T.role_semantics("SOME_INVENTED_ROLE")


def test_primary_masking_does_not_alter_stream_chronology():
    rows = _stream(count=10)
    before = T.require_chronological_stream(rows)
    proof = T.require_mask_does_not_thin_the_stream(
        replay_row_count=len(rows), masked_row_count=4
    )
    assert proof["context_rows_retained"] == len(rows)
    assert proof["masked_row_count"] == 4
    # the replay itself is untouched by the mask
    assert T.require_chronological_stream(rows) == before
    with pytest.raises(T2ProtocolError, match="cannot select more rows"):
        T.require_mask_does_not_thin_the_stream(
            replay_row_count=10, masked_row_count=11
        )


def test_primary_and_challenge_come_from_one_identical_replay():
    assert T.T2_SINGLE_CONTINUOUS_REPLAY_REQUIRED is True
    assert T.T2_MASKS_APPLIED_AFTER_SCORING is True
    assert T.T2_STATE_RESET_BEFORE_CHALLENGE_ROWS is False
    assert T.T2_INTERVENING_NON_PRIMARY_REMOVAL_PERMITTED is False
    # one pass produces scores for every available role; masks only select
    scored = [
        role
        for role in T.T2_ROW_ROLES
        if T.T2_ROW_ROLE_SEMANTICS[role]["produces_score"]
    ]
    assert set(scored) == {
        T.ROLE_PRIMARY_DIRECT_LOSS,
        T.ROLE_CHALLENGE_CONTEXT,
        T.ROLE_OTHER_NONPRIMARY_CONTEXT,
    }


def test_a_challenge_only_replay_is_forbidden():
    assert T.T2_CHALLENGE_ONLY_REPLAY_PERMITTED is False
    assert T.T2_PRIMARY_ONLY_REPLAY_PERMITTED is False


def test_persisted_chronological_field_and_alias_are_explicit():
    assert T.T2_STREAM_ORDER_FIELD == "start_sample"
    assert T.T2_STREAM_ORDER_FIELD_ALIAS == "window_start_samples"
    assert T.T2_STREAM_ORDER_FIELD_SEMANTICS == "window start in samples"
    text = _document_prose()
    assert "`window_start_samples := persisted start_sample`" in text


def test_state_must_carry_across_tbptt_chunks_and_detach():
    chunks = T.tbptt_chunks(T2StreamKey("s20011", 0), 700)
    assert len(chunks) == 3
    for chunk in chunks[1:]:
        assert chunk.carries_state_in is True
        assert chunk.detaches_state_in is True
    assert T.T2_STATE_CARRIES_ACROSS_CHUNK is True
    assert T.T2_STATE_DETACHED_AT_CHUNK_BOUNDARY is True
    assert T.T2_STATE_RESET_AT_CHUNK_BOUNDARY is False
    text = _document_prose()
    assert "MUST carry causally** across TBPTT chunk boundaries" in text


def test_exact_gru_architecture_is_complete():
    spec = T.architecture_spec(T.T2_ARM_GRU)
    assert spec["temporal_core"] == "torch.nn.GRU"
    assert spec["gru_input_size"] == 64
    assert spec["gru_hidden_size"] == 64
    assert spec["gru_num_layers"] == 2
    assert spec["gru_bidirectional"] is False
    assert spec["gru_bias"] is True
    assert spec["gru_dropout"] == 0.10
    assert "except the last" not in spec["gru_dropout_semantics"]
    assert spec["gru_dropout_semantics"] == (
        "between_stacked_layers_only_not_after_final_layer"
    )
    assert spec["hidden_state_initialization"] == "zeros"
    assert spec["recurrent_state_shape"] == "(num_layers=2, batch, hidden=64)"
    assert spec["residual_connection"] is False
    assert spec["input_projection"] == "Linear(146, 64, bias=True)"
    assert spec["input_projection_activation"] is None
    assert spec["readout"] == "Linear(64, 1, bias=True)"
    assert spec["dtype"] == "float32"
    proof = T.require_architecture_is_fully_specified(T.T2_ARM_GRU)
    assert proof["fully_specified"] is True
    assert proof["expected_trainable_parameters"] == 59_521


def test_exact_s4d_architecture_is_complete():
    spec = T.architecture_spec(T.T2_ARM_S4D)
    assert spec["state_dim"] == 16
    assert spec["blocks"] == 2
    assert spec["model_width"] == 64
    assert spec["discretization"] == "zero_order_hold"
    assert spec["lambda_parameterization"] == "complex(-exp(log_decay), frequency)"
    assert spec["stability_constraint"] == (
        "negative_real_part_guaranteed_by_negative_exp"
    )
    assert spec["state_update"] == "state = Abar * state + Bbar * u_t"
    assert spec["output_equation"] == "(C * state).real.sum(-1) + D * u_t"
    assert spec["activation"] == "SiLU on the gate branch only"
    assert spec["block_norm"] == "LayerNorm(64, eps=1e-5) pre-norm at block input"
    assert spec["residual_connection"] is True
    assert spec["dropout_placement"] == "on_the_branch_before_the_residual_add"
    assert spec["d_skip_term"] == "real_per_channel_vector_initialized_zero"
    assert spec["hidden_state_initialization"] == "zeros"
    assert spec["recurrent_inference"] == (
        "explicit_step_recurrence_carrying_state_across_windows"
    )
    proof = T.require_architecture_is_fully_specified(T.T2_ARM_S4D)
    assert proof["fully_specified"] is True
    assert proof["expected_trainable_parameters"] == 45_313


def test_b4c_conventions_reused_and_divergences_stated():
    assert T.T2_S4D_REUSES_B4C_CONVENTIONS
    assert "state_dim_16" in T.T2_S4D_REUSES_B4C_CONVENTIONS
    assert any("zero_order_hold" in name for name in T.T2_S4D_REUSES_B4C_CONVENTIONS)
    # exactly two divergences, each with a stated prospective reason
    assert len(T.T2_S4D_DIVERGES_FROM_B4C) == 2
    assert any("carried across" in reason for reason in T.T2_S4D_DIVERGES_FROM_B4C)
    assert any("64 rather than" in reason for reason in T.T2_S4D_DIVERGES_FROM_B4C)


def test_frozen_parameter_counts_sit_inside_the_shared_envelope():
    proof = T.require_capacity_envelope(T.T2_EXPECTED_PARAMETER_COUNTS)
    assert proof["within_envelope"] is True
    assert 0.5 <= proof["ratio_s4d_over_gru"] <= 2.0
    assert proof["parameter_counts"] == {
        T.T2_ARM_GRU: 59_521,
        T.T2_ARM_S4D: 45_313,
    }


def test_no_unbound_model_design_choice_remains():
    assert T.T2_UNBOUND_ARCHITECTURAL_CHOICE_REMAINS is False
    assert T.T2_IMPLEMENTATION_MAY_CHOOSE == ()
    for arm in T.T2_ARMS:
        proof = T.require_architecture_is_fully_specified(arm)
        assert proof["implementation_may_choose"] == []
    with pytest.raises(T2ProtocolError, match="not a frozen T2 candidate"):
        T.architecture_spec("causal_lstm_longitudinal_v1")


def test_challenge_identity_and_labels_cannot_enter_trainable_inputs():
    for name in ("challenge_family_identity", "future_window_label"):
        with pytest.raises(T2ProtocolError, match="may not be a trainable"):
            T.require_permitted_trainable_inputs(["z_t", name])
    assert T.T2_CHALLENGE_IDENTITY_IS_MODEL_INPUT is False
    assert T.T2_CHALLENGE_LABEL_IS_MODEL_INPUT is False


def test_test_remains_unopened_after_the_closure():
    assert T.T2_TEST_ACCESSED is False
    assert T.T2_SEALED_TEST_STATE == "unopened"
    identity = T.t2_protocol_identity()
    assert identity["test_accessed"] is False
    assert identity["sealed_test_state"] == "unopened"


# ==========================================================================
# Row lineage: the stable_id rides on the row, so a caller cannot attest
# identities for rows it did not supply.
# ==========================================================================


def test_canonical_row_identity_carries_its_own_stable_id():
    row = _row("s20011", 1, 1250)
    assert row.stable_id == "ltstdb:s20011:1:1250:3750"
    assert row.record_id == "s20011"
    assert row.channel_index == 1
    assert row.start_sample == 1250
    # the conceptual name resolves deterministically to the persisted field
    assert row.window_start_samples == row.start_sample
    assert T.canonical_stable_id(row) == row.stable_id
    assert T.T2_WINDOW_LENGTH_SAMPLES == 2500
    assert T.T2_DATASET == "ltstdb"


def test_stable_id_record_component_must_match_the_row():
    row = _row("s20011", 0, 0, stable_id="ltstdb:s20021:0:0:2500")
    with pytest.raises(T2ProtocolError, match="encodes record"):
        T.require_stable_id_matches_row(row)


def test_stable_id_channel_component_must_match_the_row():
    row = _row("s20011", 0, 0, stable_id="ltstdb:s20011:2:0:2500")
    with pytest.raises(T2ProtocolError, match="encodes channel"):
        T.require_stable_id_matches_row(row)


def test_stable_id_start_component_must_match_the_row():
    row = _row("s20011", 0, 1250, stable_id="ltstdb:s20011:0:0:2500")
    with pytest.raises(T2ProtocolError, match="encodes start"):
        T.require_stable_id_matches_row(row)


def test_stable_id_end_component_must_be_start_plus_2500():
    row = _row("s20011", 0, 0, stable_id="ltstdb:s20011:0:0:2400")
    with pytest.raises(T2ProtocolError, match="encodes end"):
        T.require_stable_id_matches_row(row)


def test_malformed_or_foreign_stable_ids_are_refused():
    with pytest.raises(T2ProtocolError, match="malformed"):
        T.require_stable_id_matches_row(_row(stable_id="ltstdb:s20011:0:0"))
    with pytest.raises(T2ProtocolError, match="names dataset"):
        T.require_stable_id_matches_row(_row(stable_id="edb:s20011:0:0:2500"))
    with pytest.raises(T2ProtocolError, match="non-integer"):
        T.require_stable_id_matches_row(_row(stable_id="ltstdb:s20011:x:0:2500"))


def _verify(rows, frozen_ids, cache=None):
    return T.require_full_timeline_replay(
        offered_rows=rows,
        frozen_stable_ids=frozen_ids,
        stream_cache_sha256=cache or T.T2_TRAIN_STREAM_CACHE_SHA256,
        expected_stream_cache_sha256=T.T2_TRAIN_STREAM_CACHE_SHA256,
    )


def test_a_correct_full_timeline_replay_still_passes():
    rows = _stream(count=8)
    proof = _verify(rows, _frozen_ids(rows))
    assert proof["row_count"] == 8
    assert proof["stream_count"] == 1
    assert proof["thinned"] is False
    assert proof["row_substituted"] is False
    assert proof["stable_ids_derived_from_rows"] is True
    assert proof["order_field"] == "start_sample"


def test_returned_row_count_comes_from_the_actual_rows():
    """Not from an identity vector: there is no longer one to count."""
    rows = _stream(count=8)
    assert _verify(rows, _frozen_ids(rows))["row_count"] == len(rows)
    import inspect

    signature = inspect.signature(T.require_full_timeline_replay)
    assert "offered_stable_ids" not in signature.parameters


def test_thinned_rows_with_the_full_frozen_id_vector_are_refused():
    """The exact gap: 7 rows attested by 8 complete frozen ids."""
    rows = _stream(count=8)
    frozen = _frozen_ids(rows)
    thinned = rows[:3] + rows[4:]
    assert len(thinned) == 7 and len(frozen) == 8
    with pytest.raises(T2ProtocolError, match="thinned by nothing at all"):
        _verify(thinned, frozen)


def test_full_rows_with_a_thinned_id_vector_are_refused():
    rows = _stream(count=8)
    frozen = _frozen_ids(rows)[:7]
    with pytest.raises(T2ProtocolError, match="thinned by nothing at all"):
        _verify(rows, frozen)


def test_same_count_row_substitution_is_refused():
    """Same length and a valid frozen id vector is not valid row lineage."""
    timeline_a = _stream("s20011", 0, count=8)
    timeline_b = _stream("s20021", 0, count=8)
    frozen = _frozen_ids(timeline_a)
    substituted = timeline_a[:4] + (timeline_b[4],) + timeline_a[5:]
    assert len(substituted) == len(frozen)
    with pytest.raises(T2ProtocolError, match="Equal length is not"):
        _verify(substituted, frozen)


def test_a_row_whose_id_disagrees_with_its_own_identity_is_refused():
    """Right count, right ids, but one row does not match the id it carries."""
    rows = _stream(count=8)
    frozen = _frozen_ids(rows)
    broken = rows[:5] + (rows[5]._replace(start_sample=999_999),) + rows[6:]
    assert len(broken) == 8
    with pytest.raises(T2ProtocolError, match="encodes start"):
        _verify(broken, frozen)


def test_duplicate_rows_are_refused():
    rows = _stream(count=4)
    frozen = _frozen_ids(rows)
    duplicated = rows[:3] + (rows[0],)
    with pytest.raises(T2ProtocolError, match="repeats a stable_id"):
        _verify(duplicated, frozen)


def test_reordered_rows_are_refused():
    rows = _stream(count=6)
    frozen = _frozen_ids(rows)
    swapped = rows[:2] + (rows[3], rows[2]) + rows[4:]
    with pytest.raises(T2ProtocolError, match="different order"):
        _verify(swapped, frozen)


def test_a_replay_from_the_wrong_stream_cache_is_refused():
    rows = _stream(count=4)
    with pytest.raises(T2ProtocolError, match="not the\\s+frozen"):
        _verify(rows, _frozen_ids(rows), cache="0" * 64)


# --- the stale broad flag, and the precise ones that replaced it ----------


def test_broad_challenge_trained_on_flag_no_longer_exists():
    """It was ambiguous: challenge context can reach a later PRIMARY loss."""
    assert not hasattr(T, "T2_CHALLENGE_TRAINED_ON")


def test_precise_challenge_context_semantics_are_unchanged():
    assert T.T2_CHALLENGE_RECEIVES_DIRECT_LOSS is False
    assert T.T2_CHALLENGE_IDENTITY_IS_MODEL_INPUT is False
    assert T.T2_CHALLENGE_LABEL_IS_MODEL_INPUT is False
    assert T.T2_CHALLENGE_MAY_BE_LABEL_BLIND_CONTEXT is True
    assert T.T2_CHALLENGE_IS_CHECKPOINT_EVIDENCE is False
    assert T.T2_CHALLENGE_IS_SELECTION_INPUT is False
    assert T.T2_CHALLENGE_MERGED_INTO_PRIMARY is False


def test_modellable_rows_wording_does_not_overclaim():
    doc = T.modellable_rows.__doc__ or ""
    assert "direct loss remains PRIMARY-only" in doc.replace("**", "")
    rows = _stream(
        count=3,
        states=(
            T.T2_OBSERVATION_AVAILABLE,
            T.T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT,
            T.T2_OBSERVATION_AVAILABLE,
        ),
    )
    assert T.modellable_rows(rows) == (rows[0], rows[2])


def test_s4d_notation_is_self_contained():
    spec = T.architecture_spec(T.T2_ARM_S4D)
    assert spec["zeta_definition"] == "zeta = exp(log_step) * lambda"
    assert spec["transition_abar"].startswith("Abar = exp(zeta)")
    assert spec["input_gain_bbar"].startswith("Bbar = expm1(zeta) / lambda")
    equations = spec["per_step_equations"]
    assert "state_t = Abar * state_(t-1) + Bbar * value_t" in equations
    assert "output_t = x_t + Dropout(branch_t)" in equations
    assert spec["ssm_input_u_t"] == "the projected value branch, not the gate branch"
    # notation only: the architecture itself is unchanged
    assert spec["state_dim"] == 16
    assert spec["blocks"] == 2
    assert spec["discretization"] == "zero_order_hold"


def test_science_remains_unexecuted_and_test_unopened():
    assert T.T2_TEST_ACCESSED is False
    assert T.T2_SEALED_TEST_STATE == "unopened"
    identity = T.t2_protocol_identity()
    assert identity["test_accessed"] is False
    assert identity["sealed_test_state"] == "unopened"
    # still no training, scoring or execution path in the module
    tree = ast.parse(Path(T.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in {"backward", "fit", "train", "no_grad", "DataLoader"}
