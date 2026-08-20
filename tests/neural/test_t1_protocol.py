"""The frozen T1-v1 causal episode-state protocol, proven structurally.

Nothing here executes science. No T1 development run, no candidate search, no
score quantile over real data, no state trace over real rows, no episode result,
no metric, no bootstrap, no final configuration selection and no TEST access.
Every transition case is driven with hand-built rows whose values are chosen to
exercise a rule, never read from a real distribution.

The point of this file is that the protocol is *prospective*: each frozen
constant, formula and tie-break is pinned here so it cannot drift once real T1
numbers exist and start to look inconvenient.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_protocol as P
from cardiosentinel.neural.t1_protocol import (
    T1_PROFILE_BALANCED,
    T1_PROFILE_CONSERVATIVE,
    T1_PROFILE_FAST,
    T1ProtocolError,
    T1Row,
    T1Streaks,
    T1Thresholds,
    next_state,
)

THRESHOLDS = T1Thresholds(p_watch=0.30, s_watch=0.40, p_event=0.80, s_event=0.70)
MATURE = 10_000.0
COLD = 10.0


def _row(
    *,
    d=False,
    p=0.0,
    s=0.0,
    age=MATURE,
    present=True,
    stable_id="ltstdb:s20041:0:0:2500",
):
    return T1Row(
        stable_id=stable_id,
        score_present=present,
        detector_decision=d,
        calibrated_probability=p,
        decision_error_uncertainty=(1.0 - p) if d else p,
        temporal_evidence=s,
        elapsed_stream_seconds=age,
    )


def _event_row(age=MATURE):
    return _row(d=True, p=0.95, s=0.95, age=age)


def _normal_row(age=MATURE):
    return _row(d=False, p=0.01, s=0.01, age=age)


def _ambiguous_row(age=MATURE):
    """WATCH-level: not EVENT evidence, not NORMAL evidence."""
    return _row(d=False, p=0.50, s=0.10, age=age)


# --- 1-7. state space, streams, causality ----------------------------------


def test_there_are_exactly_four_states():
    assert P.T1_STATES == ("NORMAL", "WATCH", "EVENT", "RECOVERY")
    assert len(P.T1_STATES) == 4


def test_there_is_no_fifth_state():
    for invented in ("ALERT", "CRITICAL", "UNKNOWN", "ESCALATE", ""):
        assert invented not in P.T1_STATES
        with pytest.raises(T1ProtocolError):
            P.require_state(invented)


def test_the_initial_state_is_normal():
    assert P.T1_INITIAL_STATE == "NORMAL"


def test_the_stream_key_is_record_and_channel():
    assert P.T1_STREAM_KEY == ("record_id", "channel_index")
    assert P.T1_TIMELINE_ORDER == "start_sample"


def test_state_never_crosses_a_stream():
    assert P.T1_STATE_CROSSES_STREAM is False
    assert P.T1_PER_STREAM is True
    assert P.T1_PATIENT_LEVEL_CHANNEL_FUSION_DEFINED is False


def test_the_transition_function_is_causal_only():
    """It receives one row. There is no sequence argument to look ahead in."""
    parameters = list(inspect.signature(next_state).parameters)
    assert parameters == ["state", "streaks", "row", "thresholds", "profile"]
    # The row structure is exactly the causal present: current evidence and
    # elapsed time. No sequence, no index, no neighbour, no future field.
    assert T1Row._fields == (
        "stable_id",
        "score_present",
        "detector_decision",
        "calibrated_probability",
        "decision_error_uncertainty",
        "temporal_evidence",
        "elapsed_stream_seconds",
    )
    assert not [
        field for field in T1Row._fields if "next" in field or "future" in field
    ]


def test_future_context_is_forbidden():
    for forbidden in ("future_row", "future_score"):
        assert forbidden in P.T1_FORBIDDEN_TRANSITION_INPUTS
        with pytest.raises(T1ProtocolError, match="transition function"):
            P.require_transition_input_permitted(forbidden)


# --- 8-9. the T2 inheritance -----------------------------------------------


def test_the_retained_t2_arm_is_s4d():
    assert P.T1_T2_RETAINED_ARM == "causal_s4d_longitudinal_v1"
    assert P.T1_T2_COMPARATOR_ARM == "causal_gru_longitudinal_v1"


def test_the_t2_binary_threshold_cannot_be_t1_policy():
    assert P.T1_T2_THRESHOLD_IS_T1_POLICY is False
    reporting = P.T1_T2_REPORTING_THRESHOLD_NOT_T1_POLICY
    assert reporting == 0.8972153067588806
    # It is not any generated threshold level, and not a quantile level.
    assert reporting not in P.Q_WATCH
    assert reporting not in P.Q_EVENT
    assert "t2_frozen_reporting_threshold" in P.T1_FORBIDDEN_TRANSITION_INPUTS
    with pytest.raises(T1ProtocolError):
        P.require_transition_input_permitted("t2_frozen_reporting_threshold")


# --- 10-11. the calibration source -----------------------------------------


def test_the_u1_oof_source_is_required_for_development():
    assert P.T1_U1_OOF_CALIBRATOR_REQUIRED_FOR_DEVELOPMENT is True
    assert P.require_development_calibrator("u1_oof_development_calibration") == (
        "u1_oof_development_calibration"
    )
    assert P.T1_U1_OOF_EVIDENCE_STORE_SHA256 == (
        "b95f484c9a7b08447f5a5d4330528136e040cf05acb9e2f7e54305e20bdffcba"
    )


def test_the_u1_deployment_calibrator_is_forbidden_for_development():
    assert P.T1_U1_DEPLOYMENT_CALIBRATOR_PERMITTED_FOR_DEVELOPMENT is False
    for source in ("u1_deployment_calibrator", "all_validation_calibrator"):
        with pytest.raises(T1ProtocolError, match="subject-disjoint"):
            P.require_development_calibrator(source)


# --- 12-15. leakage firewalls ----------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "target_family",
        "label",
        "challenge_family_identity",
        "m2_update_admitted",
        "m2_gate_outcome",
        "subject_outcome",
        "episode_identity",
        "gru_score",
        "s4d_binary_decision",
        "u_star_dev",
        "u_star_deploy",
        "test_derived_quantity",
    ],
)
def test_forbidden_transition_inputs_are_refused(forbidden):
    assert forbidden in P.T1_FORBIDDEN_TRANSITION_INPUTS
    assert forbidden not in P.T1_ALLOWED_ROW_INPUTS
    with pytest.raises(T1ProtocolError):
        P.require_transition_input_permitted(forbidden)


def test_the_row_structure_carries_no_label_or_family():
    """The leakage firewall is structural, not merely documented."""
    assert not {"label", "target_family", "subject_id", "episode"} & set(T1Row._fields)


# --- 16-19. the full-timeline calibration contract -------------------------


def test_the_full_timeline_calibration_contract_exists():
    assert P.T1_FULL_TIMELINE_CALIBRATION_CONTRACT == (
        "apply_frozen_held_out_subject_platt_calibrator_to_every_scored_m2g_row"
    )
    assert P.T1_U1_CALIBRATOR_FAMILY == "platt_logistic_on_recovered_logit"
    assert P.T1_U1_CLAMP_DELTA == 1e-7
    assert P.T1_DETECTOR_THRESHOLD == 0.7554003000259399


def test_no_upstream_refit_or_replay_is_permitted():
    assert P.T1_U1_REFIT_PERMITTED is False
    assert P.T1_M2_REPLAY_PERMITTED is False
    assert P.T1_T2_REPLAY_PERMITTED is False
    assert P.T1_TIMELINE_REGENERATED_FROM_WAVEFORM is False


# --- 20-22. prospective threshold generation -------------------------------


def test_the_watch_quantiles_are_exact():
    assert P.Q_WATCH == (0.90, 0.95)


def test_the_event_quantiles_are_exact():
    assert P.Q_EVENT == (0.99, 0.995)


def test_the_empirical_order_statistic_is_exact():
    """k = ceil(q * N), 1-based, no interpolation."""
    values = [float(v) for v in range(1, 11)]  # 1..10, N = 10
    ids = [f"id{v:02d}" for v in range(1, 11)]
    assert P.empirical_order_statistic(values, ids, 0.90) == 9.0
    assert P.empirical_order_statistic(values, ids, 0.95) == 10.0
    assert P.empirical_order_statistic(values, ids, 0.99) == 10.0
    assert P.empirical_order_statistic(values, ids, 0.10) == 1.0
    # A library quantile would interpolate to 9.1 at q=0.90 over 1..10.
    assert P.empirical_order_statistic(values, ids, 0.90) != 9.1
    # Order independence: ties break on stable_id, so input order cannot matter.
    shuffled = list(reversed(values))
    shuffled_ids = list(reversed(ids))
    assert P.empirical_order_statistic(shuffled, shuffled_ids, 0.90) == 9.0
    assert P.T1_THRESHOLD_TIE_ORDER == ("value", "stable_id")
    assert P.T1_THRESHOLD_INTERPOLATION_PERMITTED is False
    assert P.T1_THRESHOLD_SOURCE_POPULATION == (
        "fit_subject_primary_background_negative"
    )
    assert P.T1_THRESHOLD_USES_CHALLENGE_ROWS is False
    assert P.T1_THRESHOLD_USES_LABEL_WEIGHTING is False
    with pytest.raises(T1ProtocolError):
        P.empirical_order_statistic([], [], 0.9)


# --- 23-27. the persistence grid -------------------------------------------


def test_there_are_exactly_three_persistence_profiles():
    assert len(P.T1_PERSISTENCE_PROFILES) == 3
    assert {p.name for p in P.T1_PERSISTENCE_PROFILES} == {
        "FAST",
        "BALANCED",
        "CONSERVATIVE",
    }


def test_there_are_exactly_twelve_candidate_policies():
    assert P.T1_CANDIDATE_POLICY_COUNT == 12
    policies = P.candidate_policies()
    assert len(policies) == 12
    assert len({p.name for p in policies}) == 12


def test_the_fast_profile_is_exact():
    assert T1_PROFILE_FAST == P.T1PersistenceProfile(
        name="FAST",
        watch_clear_windows=2,
        event_confirm_windows=2,
        event_release_windows=2,
        re_event_confirm_windows=1,
        recovery_clear_windows=3,
        cold_event_confirm_windows=4,
    )


def test_the_balanced_profile_is_exact():
    assert T1_PROFILE_BALANCED == P.T1PersistenceProfile(
        name="BALANCED",
        watch_clear_windows=3,
        event_confirm_windows=3,
        event_release_windows=3,
        re_event_confirm_windows=2,
        recovery_clear_windows=6,
        cold_event_confirm_windows=6,
    )


def test_the_conservative_profile_is_exact():
    assert T1_PROFILE_CONSERVATIVE == P.T1PersistenceProfile(
        name="CONSERVATIVE",
        watch_clear_windows=6,
        event_confirm_windows=6,
        event_release_windows=6,
        re_event_confirm_windows=3,
        recovery_clear_windows=12,
        cold_event_confirm_windows=12,
    )


# --- 28-31. evidence formulas ----------------------------------------------


def test_the_watch_evidence_formula_is_frozen():
    """Any one of the three signals suffices."""
    assert P.is_watch_evidence(_row(d=True, p=0.0, s=0.0), THRESHOLDS)
    assert P.is_watch_evidence(_row(d=False, p=0.30, s=0.0), THRESHOLDS)
    assert P.is_watch_evidence(_row(d=False, p=0.0, s=0.40), THRESHOLDS)
    assert not P.is_watch_evidence(_row(d=False, p=0.29, s=0.39), THRESHOLDS)


def test_the_mature_event_evidence_formula_is_frozen():
    """All three are required once the stream is mature."""
    assert P.is_event_evidence(_row(d=True, p=0.80, s=0.70), THRESHOLDS)
    assert not P.is_event_evidence(_row(d=False, p=0.99, s=0.99), THRESHOLDS)
    assert not P.is_event_evidence(_row(d=True, p=0.79, s=0.99), THRESHOLDS)
    assert not P.is_event_evidence(_row(d=True, p=0.99, s=0.69), THRESHOLDS)


def test_the_cold_start_event_formula_is_frozen():
    """Below 300 s the S4D term is not required; above it, it is."""
    assert P.T1_COLD_START_SECONDS == 300.0
    assert P.T1_COLD_START_REQUIRES_S4D is False
    assert P.T1_MATURE_REQUIRES_S4D is True
    cold = _row(d=True, p=0.95, s=0.0, age=299.0)
    assert P.is_cold_start(cold)
    assert P.is_event_evidence(cold, THRESHOLDS)
    mature = _row(d=True, p=0.95, s=0.0, age=300.0)
    assert not P.is_cold_start(mature)
    assert not P.is_event_evidence(mature, THRESHOLDS)
    # The detector and calibrated probability are still both required when cold.
    assert not P.is_event_evidence(_row(d=False, p=0.95, s=0.95, age=10.0), THRESHOLDS)
    assert not P.is_event_evidence(_row(d=True, p=0.79, s=0.95, age=10.0), THRESHOLDS)
    assert P.T1_COLD_START_MODIFIES_T2 is False


def test_the_normal_evidence_formula_is_frozen():
    assert P.is_normal_evidence(_row(d=False, p=0.29, s=0.39), THRESHOLDS)
    assert not P.is_normal_evidence(_row(d=True, p=0.0, s=0.0), THRESHOLDS)
    assert not P.is_normal_evidence(_row(d=False, p=0.30, s=0.0), THRESHOLDS)
    assert not P.is_normal_evidence(_row(d=False, p=0.0, s=0.40), THRESHOLDS)


def test_the_uncertainty_definition_is_frozen():
    assert P.decision_error_uncertainty(True, 0.9) == pytest.approx(0.1)
    assert P.decision_error_uncertainty(False, 0.9) == pytest.approx(0.9)
    # No independent T1 uncertainty threshold exists to tune.
    source = Path(P.__file__).read_text()
    assert "u_watch" not in source
    assert "u_event" not in source


# --- 32-33. unavailable rows -----------------------------------------------


def test_an_unavailable_row_holds_state_and_invents_nothing():
    assert P.T1_UNAVAILABLE_HOLDS_STATE is True
    assert P.T1_UNAVAILABLE_PERMITS_TRANSITION is False
    assert P.T1_IMPUTATION_PERMITTED is False
    assert P.T1_FORWARD_FILL_PERMITTED is False
    assert P.T1_SYNTHETIC_ZERO_PERMITTED is False
    gap = _row(present=False)
    for state in P.T1_STATES:
        held, streaks = next_state(
            state, T1Streaks(event_confirm=5), gap, THRESHOLDS, T1_PROFILE_FAST
        )
        assert held == state
        assert streaks == P.T1_ZERO_STREAKS
    # Evidence cannot be computed for it at all.
    with pytest.raises(T1ProtocolError, match="no evidence"):
        P.is_watch_evidence(gap, THRESHOLDS)


def test_an_unavailable_row_resets_confirmation_streaks():
    """A gap must not be able to confirm an escalation across itself."""
    assert P.T1_UNAVAILABLE_RESETS_STREAKS is True
    profile = T1_PROFILE_FAST  # event_confirm_windows == 2
    state, streaks = next_state(
        "WATCH", P.T1_ZERO_STREAKS, _event_row(), THRESHOLDS, profile
    )
    assert state == "WATCH" and streaks.event_confirm == 1
    state, streaks = next_state(
        state, streaks, _row(present=False), THRESHOLDS, profile
    )
    assert streaks == P.T1_ZERO_STREAKS
    # The next EVENT row starts from one again, so no escalation yet.
    state, streaks = next_state(state, streaks, _event_row(), THRESHOLDS, profile)
    assert state == "WATCH" and streaks.event_confirm == 1


# --- 34-38. transition semantics -------------------------------------------


def test_normal_transition_semantics():
    profile = T1_PROFILE_FAST
    # No watch evidence: hold.
    state, _ = next_state(
        "NORMAL", P.T1_ZERO_STREAKS, _normal_row(), THRESHOLDS, profile
    )
    assert state == "NORMAL"
    # Watch evidence: immediate WATCH, one row.
    state, _ = next_state(
        "NORMAL", P.T1_ZERO_STREAKS, _ambiguous_row(), THRESHOLDS, profile
    )
    assert state == "WATCH"
    # Event evidence escalates through WATCH and reaches EVENT on the budget.
    state, streaks = next_state(
        "NORMAL", P.T1_ZERO_STREAKS, _event_row(), THRESHOLDS, profile
    )
    assert state == "WATCH" and streaks.event_confirm == 1
    state, _ = next_state(state, streaks, _event_row(), THRESHOLDS, profile)
    assert state == "EVENT"


def test_watch_transition_semantics():
    profile = T1_PROFILE_BALANCED  # confirm 3, watch_clear 3
    state, streaks = "WATCH", P.T1_ZERO_STREAKS
    for _ in range(2):
        state, streaks = next_state(state, streaks, _event_row(), THRESHOLDS, profile)
        assert state == "WATCH"
    state, _ = next_state(state, streaks, _event_row(), THRESHOLDS, profile)
    assert state == "EVENT"

    state, streaks = "WATCH", P.T1_ZERO_STREAKS
    for _ in range(2):
        state, streaks = next_state(state, streaks, _normal_row(), THRESHOLDS, profile)
        assert state == "WATCH"
    state, _ = next_state(state, streaks, _normal_row(), THRESHOLDS, profile)
    assert state == "NORMAL"

    # Ambiguous rows hold WATCH and clear both streaks.
    state, streaks = next_state(
        "WATCH", T1Streaks(watch_clear=2), _ambiguous_row(), THRESHOLDS, profile
    )
    assert state == "WATCH" and streaks == P.T1_ZERO_STREAKS


def test_event_transition_semantics():
    profile = T1_PROFILE_FAST  # event_release_windows == 2
    # EVENT evidence holds EVENT and resets the release streak.
    state, streaks = next_state(
        "EVENT", T1Streaks(event_release=1), _event_row(), THRESHOLDS, profile
    )
    assert state == "EVENT" and streaks.event_release == 0
    # NORMAL evidence releases to RECOVERY on the budget.
    state, streaks = next_state(
        "EVENT", P.T1_ZERO_STREAKS, _normal_row(), THRESHOLDS, profile
    )
    assert state == "EVENT" and streaks.event_release == 1
    state, _ = next_state(state, streaks, _normal_row(), THRESHOLDS, profile)
    assert state == "RECOVERY"
    # An ambiguous row does not contribute to release and does not lose it.
    state, streaks = next_state(
        "EVENT", T1Streaks(event_release=1), _ambiguous_row(), THRESHOLDS, profile
    )
    assert state == "EVENT" and streaks.event_release == 1


def test_recovery_transition_semantics():
    profile = T1_PROFILE_BALANCED  # re_event 2, recovery_clear 6
    state, streaks = next_state(
        "RECOVERY", P.T1_ZERO_STREAKS, _event_row(), THRESHOLDS, profile
    )
    assert state == "RECOVERY" and streaks.re_event_confirm == 1
    state, _ = next_state(state, streaks, _event_row(), THRESHOLDS, profile)
    assert state == "EVENT"

    state, streaks = "RECOVERY", P.T1_ZERO_STREAKS
    for _ in range(5):
        state, streaks = next_state(state, streaks, _normal_row(), THRESHOLDS, profile)
        assert state == "RECOVERY"
    state, _ = next_state(state, streaks, _normal_row(), THRESHOLDS, profile)
    assert state == "NORMAL"


def test_recovery_never_automatically_becomes_watch():
    """Ambiguous evidence in RECOVERY holds RECOVERY; it does not fall to WATCH."""
    for profile in P.T1_PERSISTENCE_PROFILES:
        state, _ = next_state(
            "RECOVERY", P.T1_ZERO_STREAKS, _ambiguous_row(), THRESHOLDS, profile
        )
        assert state == "RECOVERY"
    assert ("RECOVERY", "otherwise", "RECOVERY", "never automatically WATCH") in (
        P.T1_TRANSITION_TABLE
    )
    # No row of the frozen table sends RECOVERY to WATCH.
    assert not [
        row
        for row in P.T1_TRANSITION_TABLE
        if row[0] == "RECOVERY" and row[2] == "WATCH"
    ]


# --- 39-40. the development split ------------------------------------------


def test_the_twelve_fold_loso_split_is_exact():
    assert P.T1_FOLD_COUNT == 12
    assert P.T1_FOLD_DESIGN == "leave_one_subject_out"
    assert P.T1_VALIDATION_SUBJECTS == (
        "ltstdb:s2004",
        "ltstdb:s2005",
        "ltstdb:s2019",
        "ltstdb:s2020",
        "ltstdb:s2023",
        "ltstdb:s2031",
        "ltstdb:s2057",
        "ltstdb:s2058",
        "ltstdb:s2059",
        "ltstdb:s3068",
        "ltstdb:s3072",
        "ltstdb:s3073",
    )
    folds = P.t1_folds()
    assert len(folds) == 12
    assert [f.held_out_subject for f in folds] == sorted(P.T1_VALIDATION_SUBJECTS)
    for fold in folds:
        assert len(fold.fit_subjects) == 11
        assert fold.held_out_subject not in fold.fit_subjects
        assert set(fold.fit_subjects) | {fold.held_out_subject} == set(
            P.T1_VALIDATION_SUBJECTS
        )


def test_held_out_labels_are_unavailable_during_policy_selection():
    assert P.T1_HELD_OUT_LABELS_AVAILABLE_DURING_SELECTION is False
    assert P.T1_FOLD_RETRY_PERMITTED is False
    assert P.T1_FOLD_MANUAL_OVERRIDE_PERMITTED is False


# --- 41-43. episodes and matching ------------------------------------------


def test_episode_grouping_requires_the_exact_cadence():
    starts = [0, 1250, 2500, 3750]
    positives = [True, True, True, True]
    assert P.group_reference_episodes(starts, positives) == ((0, 4),)
    assert P.T1_EPISODE_CADENCE_SAMPLES == 1250


def test_episode_grouping_does_not_bridge_a_gap():
    """A cadence break splits an episode; it is never stitched."""
    assert P.T1_EPISODE_GAP_BRIDGING_PERMITTED is False
    assert P.T1_EPISODE_MINIMUM_DURATION_FILTER is None
    assert P.T1_EPISODE_ANNOTATION_REREAD is False
    # A non-positive row breaks the run.
    assert P.group_reference_episodes(
        [0, 1250, 2500, 3750], [True, True, False, True]
    ) == ((0, 2), (3, 4))
    # So does a jump larger than one stride, even with positives throughout.
    assert P.group_reference_episodes(
        [0, 1250, 5000, 6250], [True, True, True, True]
    ) == ((0, 2), (2, 4))


def test_event_matching_is_one_to_one_and_penalizes_overmerging():
    episodes = [(0, 2), (6, 8)]
    # One predicted run spanning both episodes matches only the first.
    overmerged = [(0, 10)]
    matched = P.match_runs_to_episodes(episodes, overmerged)
    assert matched == {0: 0}
    assert len(matched) == 1
    # Two separate runs match one episode each.
    separate = [(0, 2), (6, 8)]
    assert P.match_runs_to_episodes(episodes, separate) == {0: 0, 1: 1}
    # A run overlapping nothing matches nothing.
    assert P.match_runs_to_episodes(episodes, [(3, 5)]) == {}
    assert P.T1_EPISODE_MATCHING == ("one_to_one_earliest_unmatched_overlapping_run")


# --- 44-49. policy selection -----------------------------------------------


def test_pooled_episode_f1_is_the_primary_selection_metric():
    assert P.T1_PRIMARY_SELECTION_METRIC == "pooled_episode_f1"


def test_pooled_window_mcc_is_the_second_metric():
    assert P.T1_SECONDARY_SELECTION_METRIC == "pooled_primary_window_mcc"
    assert P.T1_THIRD_SELECTION_METRIC == "false_event_onsets_per_physical_hour"
    assert P.T1_FOURTH_SELECTION_METRIC == "event_exposure_fraction"


def test_challenge_and_latency_are_not_selection_inputs():
    assert P.T1_SELECTION_USES_CHALLENGE_EVIDENCE is False
    assert P.T1_SELECTION_USES_LATENCY is False
    assert P.T1_CHALLENGE_IS_SELECTION_INPUT is False
    assert P.T1_CHALLENGE_IS_TRANSITION_INPUT is False


def test_no_weighted_composite_is_used():
    assert P.T1_SELECTION_USES_WEIGHTED_COMPOSITE is False
    source = Path(P.__file__).read_text()
    assert "weighted_score" not in source
    assert "composite_score" not in source


def test_the_deterministic_safety_tie_break_is_exact():
    """When every metric ties, the frozen preference decides -- not luck."""
    assert P.T1_SELECTION_ORDER == (
        "pooled_episode_f1_desc",
        "pooled_primary_window_mcc_desc",
        "false_event_onsets_per_hour_asc",
        "event_exposure_fraction_asc",
        "q_event_desc",
        "q_watch_desc",
        "persistence_profile_conservative_first",
    )
    tied = dict(
        episode_f1=0.5,
        window_mcc=0.5,
        false_onsets_per_hour=1.0,
        event_exposure_fraction=0.1,
    )
    winner = min(P.candidate_policies(), key=lambda p: P.policy_sort_key(p, **tied))
    assert winner.q_event == max(P.Q_EVENT)
    assert winner.q_watch == max(P.Q_WATCH)
    assert winner.profile.name == "CONSERVATIVE"
    # CONSERVATIVE sorts before BALANCED before FAST.
    assert [p.name for p in P.T1_PERSISTENCE_PROFILES] == [
        "CONSERVATIVE",
        "BALANCED",
        "FAST",
    ]


# --- 50-52. evidence status and inference unit -----------------------------


def test_t1_evidence_remains_development_evidence():
    assert P.T1_EVIDENCE_CLASS == (
        "cross_fitted_t1_development_evidence_conditional_on_frozen_upstream_components"
    )
    assert P.T1_IS_UNSEEN_GENERALIZATION is False
    assert P.T1_IS_EXTERNAL_VALIDATION is False
    assert P.T1_IS_INDEPENDENT_VALIDATION is False
    assert P.T1_IS_CLINICAL_VALIDATION is False


def test_the_subject_is_the_inferential_unit():
    assert P.T1_BOOTSTRAP_UNIT == "subject"
    assert P.T1_BOOTSTRAP_RESELECTS_POLICY is False
    assert P.T1_BOOTSTRAP_CLAIM_SCOPE == (
        "between_subject_variation_conditional_on_the_cross_fitted_t1_procedure"
    )


def test_the_bootstrap_is_frozen_at_one_thousand_seed_2026():
    assert P.T1_BOOTSTRAP_REPLICATES == 1000
    assert P.T1_BOOTSTRAP_SEED == 2026


# --- 53-55. firewalls ------------------------------------------------------


def test_test_remains_unopened():
    assert P.T1_TEST_ACCESSED is False
    assert P.T1_SEALED_TEST_STATE == "unopened"
    source = Path(P.__file__).read_text().lower()
    for forbidden in ("sealed_test_reader", "load_test", "test_partition"):
        assert forbidden not in source


def test_routing_remains_undefined():
    assert P.T1_ROUTING_DEFINED is False
    assert P.T1_U1_SYMMETRIC_ROUTER_REJECTED is True
    source = Path(P.__file__).read_text()
    for forbidden in (
        "bandwidth",
        "cloud_escalation",
        "edge_capacity",
        "latency_limit",
    ):
        assert forbidden not in source


def test_no_llm_participates_in_state_determination():
    assert P.T1_LLM_PARTICIPATES_IN_STATE is False
    source = Path(P.__file__).read_text().lower()
    for forbidden in ("openai", "anthropic", "prompt", "completion", "chat"):
        assert forbidden not in source


# --- 56-57. the module cannot compute --------------------------------------


def test_the_protocol_module_cannot_read_artifacts():
    """A protocol binder must be incapable of touching scientific state."""
    tree = ast.parse(Path(P.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    # Standard library only: no scientific stack, no repository run machinery.
    assert imported <= {
        "__future__",
        "hashlib",
        "math",
        "dataclasses",
        "pathlib",
        "typing",
    }
    for forbidden in ("numpy", "torch", "scipy", "sklearn", "pandas", "cardiosentinel"):
        assert forbidden not in imported, forbidden

    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for forbidden in (
        "np_load",
        "load",
        "read_text",
        "write_text",
        "write_bytes",
        "savez",
        "unlink",
        "rmtree",
        "mkdir",
    ):
        assert forbidden not in called, forbidden
    # The single `open` is the protocol document's own digest check, read-only.
    source = Path(P.__file__).read_text()
    assert source.count("open(") == 1
    assert 'open(path, "rb")' in source


def test_the_protocol_module_cannot_perform_model_inference():
    tree = ast.parse(Path(P.__file__).read_text())
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for forbidden in (
        "build_t2_model",
        "forward",
        "load_state_dict",
        "eval",
        "no_grad",
        "predict",
        "fit",
        "sigmoid",
    ):
        assert forbidden not in called, forbidden


def test_the_protocol_document_digest_is_frozen():
    assert P.validate_t1_protocol_document() == P.T1_PROTOCOL_SHA256
    assert P.T1_PROTOCOL_PATH.is_file()


def test_a_mutated_protocol_document_is_refused(tmp_path):
    forged = tmp_path / "forged.md"
    forged.write_text(P.T1_PROTOCOL_PATH.read_text() + "\nappended\n")
    with pytest.raises(T1ProtocolError, match="immutable"):
        P.validate_t1_protocol_document(forged)


def test_the_frozen_upstream_identities_are_bound():
    assert P.T1_STARTING_GIT_SHA == "b3004da9dcd8e7462d69eac81eb82ca9da86b8cb"
    assert P.T1_M2_RETAINED_ARM == "M2-G"
    assert P.T1_TIMELINE_ROW_COUNT == 492_904
    assert P.T1_EXPECTED_SCORE_PRESENT_ROWS == 492_898
    assert P.T1_EXPECTED_UNAVAILABLE_ROWS == 6
    assert (
        P.T1_EXPECTED_SCORE_PRESENT_ROWS + P.T1_EXPECTED_UNAVAILABLE_ROWS
        == P.T1_TIMELINE_ROW_COUNT
    )
    assert P.T1_TIMELINE_STREAM_COUNT == 30
    assert P.T1_TIMELINE_SUBJECT_COUNT == 12


def test_the_bound_identities_match_the_merged_upstream_modules():
    """The protocol must bind what the repository actually froze."""
    from cardiosentinel.neural import m2_selection as M2S
    from cardiosentinel.neural import t2_protocol as T2P
    from cardiosentinel.neural import t2_selection as T2S
    from cardiosentinel.neural import u1_protocol as U1P
    from cardiosentinel.neural import u1_selection as U1S

    assert P.T1_M2_RETENTION_DECISION_SHA256 == M2S.M2_RETENTION_DECISION_SHA256
    assert P.T1_M2_RETAINED_ARM == M2S.M2_RETAINED_ARM
    assert P.T1_U1_PROTOCOL_SHA256 == U1P.U1_PROTOCOL_SHA256
    assert P.T1_U1_OOF_EVIDENCE_STORE_SHA256 == U1S.U1_OOF_EVIDENCE_STORE_SHA256
    assert P.T1_U1_CLAMP_DELTA == U1P.U1_CLAMP_DELTA
    assert P.T1_DETECTOR_THRESHOLD == U1P.U1_CLASSIFICATION_THRESHOLD
    assert P.T1_T2_PROTOCOL_SHA256 == T2P.T2_PROTOCOL_SHA256
    assert P.T1_T2_RETENTION_DECISION_SHA256 == T2S.T2_RETENTION_DECISION_SHA256
    assert P.T1_T2_RETAINED_ARM == T2S.T2_RETAINED_ARM
    assert P.T1_T2_COMPARATOR_ARM == T2S.T2_COMPARATOR_ARM
    assert P.T1_T2_OUTER_RESULT_SHA256 == T2S.T2_OUTER_RESULT_SHA256
    assert P.T1_T2_ROW_EVIDENCE_CONTENT_SHA256 == T2S.T2_ROW_EVIDENCE_CONTENT_SHA256
    assert P.T1_SPLIT_SHA256 == T2P.T2_SPLIT_SHA256
    assert P.T1_FEATURE_CORPUS_SHA256 == T2P.T2_FEATURE_CORPUS_SHA256
    assert P.T1_T2_REPORTING_THRESHOLD_NOT_T1_POLICY == (
        T2S.T2_RETAINED_INTERNAL_DEV_THRESHOLD
    )
