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
        T2Row(
            record_id=record,
            channel_index=channel,
            window_start_samples=start + index * 1250,
            observation_state=(
                states[index] if states is not None else T.T2_OBSERVATION_AVAILABLE
            ),
        )
        for index in range(count)
    )


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
    assert smaller["selection_basis"] == T.T2_SELECTION_FINAL_TIE_BREAK


def test_challenge_evidence_cannot_select_the_model():
    assert T.T2_CHALLENGE_IS_SELECTION_INPUT is False
    assert T.T2_CHALLENGE_TRAINED_ON is False
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
