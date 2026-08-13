"""Synthetic causal verification of the frozen M2-v1 update policy.

Every test here is synthetic engineering verification. No canonical M1 run is
repeated, no real DEVELOPMENT corpus is read, no VALIDATION or TEST partition
is touched, and no scientific result is produced.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m2_evidence as E
from cardiosentinel.neural import m2_gate as G
from cardiosentinel.neural import m2_policy as P
from cardiosentinel.neural.metadata import B4WindowReference
from cardiosentinel.neural.patient_memory import (
    ALPHA_LONG,
    OBSERVATION_AVAILABLE,
    OBSERVATION_UNAVAILABLE_EXACT_FLAT,
    REPRESENTATION_DIM,
    DualTimescaleMemory,
    M1DistanceStandardizer,
    build_causal_streams,
    generate_stream_memory,
)
from cardiosentinel.neural.protocol import WINDOW_SAMPLES

STRIDE_SAMPLES = 1250  # the frozen 5 s stride at 250 Hz
THRESHOLD = G.NORMAL_EVIDENCE_THRESHOLD
CLEAN_SQI = {column: 0.0 for column in G.G3_SQI_COLUMNS}


def identity_standardizer() -> M1DistanceStandardizer:
    """A synthetic pass-through standardizer with a zero cold-start prior."""
    return M1DistanceStandardizer(
        means=tuple([0.0] * REPRESENTATION_DIM),
        scales=tuple([1.0] * REPRESENTATION_DIM),
        prior=tuple([0.0] * REPRESENTATION_DIM),
        zero_variance_dimensions=(),
        fitted_rows=1,
        fitted_population="synthetic_test_only",
        input_identities={},
    )


def representation(value: float) -> np.ndarray:
    return np.full(REPRESENTATION_DIM, float(value), dtype=np.float64)


def row(
    index: int,
    *,
    record_id: str = "s00001",
    channel_index: int = 0,
    value: float = 1.0,
    state: int = OBSERVATION_AVAILABLE,
    sqi: dict[str, float] | None = None,
    finite_sample_fraction: float | None = 1.0,
    morphology_valid: float | None = 1.0,
    start_sample: int | None = None,
) -> P.M2TimelineRow:
    return P.M2TimelineRow(
        record_id=record_id,
        channel_index=channel_index,
        start_sample=index * STRIDE_SAMPLES if start_sample is None else start_sample,
        observation_state=state,
        representation=None
        if state != OBSERVATION_AVAILABLE
        else representation(value),
        finite_sample_fraction=finite_sample_fraction,
        sqi=dict(CLEAN_SQI if sqi is None else sqi),
        morphology_valid=morphology_valid,
    )


def code_identifiers(source: str) -> set[str]:
    """Every identifier and non-docstring literal a module's CODE references.

    Docstrings are excluded on purpose: a module must stay free to document,
    in prose, exactly which annotation-derived quantities it refuses to
    consult. What matters is that no executable path names one.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            found.add(node.arg)
        elif isinstance(node, ast.alias):
            found.add(node.name)
            if node.asname:
                found.add(node.asname)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.add(node.value)
    return found


def constant_scorer(score: float) -> P.M2Scorer:
    return lambda _representation, _d_long: score


def admitting_scorer() -> P.M2Scorer:
    """Always at the normal-evidence margin, so G4 passes and never re-arms."""
    return constant_scorer(THRESHOLD)


def replay(rows, *, arm=P.M2_ARM_GATED, scorer=None):
    return P.replay_stream(
        rows,
        arm=arm,
        standardizer=identity_standardizer(),
        scorer=scorer or admitting_scorer(),
    )


# --------------------------------------------------------------------------
# 11. M2-0 parity with the frozen naive M1L control
# --------------------------------------------------------------------------


def test_m2_zero_matches_frozen_dual_timescale_memory_exactly():
    """M2-0 reproduces `DualTimescaleMemory.observe` trajectory-for-trajectory."""
    values = [0.5, 1.5, -2.0, 3.25, 0.125, 4.0, -1.0, 2.5]
    rows = [row(i, value=v) for i, v in enumerate(values)]
    evidence = replay(rows, arm=P.M2_ARM_NAIVE, scorer=constant_scorer(0.9))

    standardizer = identity_standardizer()
    reference = DualTimescaleMemory(standardizer.prior_vector())
    expected_d_long = []
    for value in values:
        features = reference.observe(standardizer.standardize(representation(value))[0])
        expected_d_long.append(features.d_long)

    assert [item.d_long for item in evidence] == expected_d_long
    assert all(item.update_admitted for item in evidence)
    assert evidence[-1].past_update_count_after == len(values)

    # The prototype trajectory itself, not merely the distances it produced.
    trajectory: list[np.ndarray] = []
    P.replay_stream(
        rows,
        arm=P.M2_ARM_NAIVE,
        standardizer=standardizer,
        scorer=constant_scorer(0.9),
        prototype_observer=lambda _index, _time, mu: trajectory.append(mu),
    )
    expected = DualTimescaleMemory(standardizer.prior_vector())
    for step, value in enumerate(values):
        expected.update(standardizer.standardize(representation(value))[0])
        assert np.array_equal(trajectory[step], expected.mu_long)


def test_m2_zero_matches_generate_stream_memory():
    """Parity against M1's own stream-level naive replay entry point."""
    values = [0.5, 1.5, -2.0, 3.25]
    rows = [row(i, value=v) for i, v in enumerate(values)]
    standardizer = identity_standardizer()

    references = tuple(
        B4WindowReference(
            stable_id=(
                f"ltstdb:s00001:0:{i * STRIDE_SAMPLES}:"
                f"{i * STRIDE_SAMPLES + WINDOW_SAMPLES}"
            ),
            record_id="s00001",
            subject_id="synthetic",
            channel_index=0,
            start_sample=i * STRIDE_SAMPLES,
            end_sample=i * STRIDE_SAMPLES + WINDOW_SAMPLES,
            partition="train",
            target_family="background_negative",
            context_flags=(),
        )
        for i in range(len(values))
    )
    streams = build_causal_streams(references)
    representations = {
        reference.stable_id: representation(value)
        for reference, value in zip(references, values, strict=True)
    }
    m1 = generate_stream_memory(
        streams,
        partition="train",
        representations=representations,
        standardizer=standardizer,
    )
    evidence = replay(rows, arm=P.M2_ARM_NAIVE, scorer=constant_scorer(0.9))
    assert [item.d_long for item in evidence] == list(m1.d_long)


def test_m2_zero_does_not_change_alpha_semantics():
    """One admitted update moves the prototype by exactly the frozen alpha."""
    evidence = replay([row(0, value=1.0)], arm=P.M2_ARM_NAIVE)
    standardizer = identity_standardizer()
    memory = DualTimescaleMemory(standardizer.prior_vector())
    memory.update(standardizer.standardize(representation(1.0))[0])
    assert evidence[0].update_admitted
    assert np.allclose(memory.mu_long, ALPHA_LONG * representation(1.0))


# --------------------------------------------------------------------------
# 12.1-12.5 causal ordering, stream isolation, boundaries
# --------------------------------------------------------------------------


def test_current_observation_never_affects_its_own_score():
    """`d_long` is measured against the pre-update prototype."""
    seen: list[float] = []

    def recording_scorer(_representation, d_long):
        seen.append(d_long)
        return THRESHOLD

    evidence = replay([row(0, value=3.0)], scorer=recording_scorer)
    # Cold-start prior is zero, so the first row's d_long is |value|, which is
    # only true if its own update had not yet been applied.
    assert seen == [3.0]
    assert evidence[0].d_long == 3.0


def test_future_observations_never_affect_earlier_decisions():
    rows = [row(0, value=1.0), row(1, value=1.0)]
    first_only = replay(rows[:1])
    both = replay(rows)
    assert both[0].d_long == first_only[0].d_long
    assert both[0].update_admitted == first_only[0].update_admitted
    assert both[0].refractory_until_after == first_only[0].refractory_until_after


def test_stream_ordering_is_causal_and_rejects_ambiguity():
    shuffled = [row(2, value=1.0), row(0, value=1.0), row(1, value=1.0)]
    evidence = replay(shuffled)
    assert [item.start_sample for item in evidence] == [
        0,
        STRIDE_SAMPLES,
        2 * STRIDE_SAMPLES,
    ]
    with pytest.raises(P.M2PolicyError, match="strictly increasing"):
        replay([row(0, value=1.0), row(0, value=2.0)])


def test_channel_states_are_independent():
    rows = [
        row(0, channel_index=0, value=5.0),
        row(0, channel_index=1, value=1.0),
        row(1, channel_index=0, value=5.0),
        row(1, channel_index=1, value=1.0),
    ]
    streams = P.replay_streams(
        rows,
        arm=P.M2_ARM_GATED,
        standardizer=identity_standardizer(),
        scorer=admitting_scorer(),
    )
    assert set(streams) == {("s00001", 0), ("s00001", 1)}
    # Each channel cold-starts from the shared prior, so the first row of each
    # sees its own value as its deviation -- no cross-channel carryover.
    assert streams[("s00001", 0)][0].d_long == 5.0
    assert streams[("s00001", 1)][0].d_long == 1.0


def test_recording_boundaries_reset_state():
    rows = [
        row(0, record_id="s00001", value=5.0),
        row(1, record_id="s00001", value=5.0),
        row(0, record_id="s00002", value=5.0),
    ]
    streams = P.replay_streams(
        rows,
        arm=P.M2_ARM_GATED,
        standardizer=identity_standardizer(),
        scorer=admitting_scorer(),
    )
    assert streams[("s00001", 0)][0].d_long == streams[("s00002", 0)][0].d_long
    assert streams[("s00002", 0)][0].past_update_count_before == 0


# --------------------------------------------------------------------------
# 12.6-12.7 physical availability
# --------------------------------------------------------------------------


def test_unavailable_row_produces_no_score_and_no_update():
    evidence = replay(
        [row(0, state=OBSERVATION_UNAVAILABLE_EXACT_FLAT), row(1, value=1.0)]
    )
    first = evidence[0]
    assert first.decision.score is None
    assert first.d_long is None
    assert not first.update_admitted
    assert not first.decision.g1_available
    assert not first.refractory_rearmed_after_decision
    assert first.past_update_count_after == first.past_update_count_before
    assert first.past_observed_count_before == 0
    # The next AVAILABLE row still sees a pristine cold-start prototype.
    assert evidence[1].d_long == 1.0


def test_unavailable_time_still_lets_refractory_expire_naturally():
    """An unavailable gap advances real time, so a freeze can lapse across it."""
    rows = [
        row(0, value=1.0),  # suspicious -> arms refractory until t0 + 60
        *[
            row(i, state=OBSERVATION_UNAVAILABLE_EXACT_FLAT)
            for i in range(1, 14)  # 13 windows x 5 s = 65 s of unavailable time
        ],
        row(14, value=1.0),
    ]
    scores = iter([0.9, THRESHOLD])
    evidence = P.replay_stream(
        rows,
        arm=P.M2_ARM_GATED,
        standardizer=identity_standardizer(),
        scorer=lambda _r, _d: next(scores),
    )
    # Arm on the first row, then confirm the last row is past the freeze.
    first, last = evidence[0], evidence[-1]
    assert first.refractory_rearmed_after_decision
    assert last.available_time >= first.refractory_until_after
    assert last.decision.g5_not_in_refractory


# --------------------------------------------------------------------------
# 12.8-12.10 G3 boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize("column", G.G3_SQI_COLUMNS)
def test_exact_g3_boundary_value_passes(column):
    sqi = dict(CLEAN_SQI)
    sqi[column] = G.G3_UPPER_BOUNDS[column]
    evidence = replay([row(0, sqi=sqi)])
    assert evidence[0].decision.g3_sqi_admissible
    assert evidence[0].update_admitted


@pytest.mark.parametrize("column", G.G3_SQI_COLUMNS)
def test_epsilon_above_each_g3_boundary_fails(column):
    sqi = dict(CLEAN_SQI)
    sqi[column] = np.nextafter(G.G3_UPPER_BOUNDS[column], np.inf)
    evidence = replay([row(0, sqi=sqi)])
    assert not evidence[0].decision.g3_sqi_admissible
    assert not evidence[0].decision.g3_feature_results[column]
    assert not evidence[0].update_admitted


def test_g3_finite_sample_precondition_is_hard():
    evidence = replay([row(0, finite_sample_fraction=0.999999)])
    assert not evidence[0].decision.g3_finite_sample_precondition
    assert not evidence[0].decision.g3_sqi_admissible


def test_g3_failure_alone_does_not_arm_refractory():
    sqi = dict(CLEAN_SQI)
    sqi["flatline_fraction"] = 1.0
    evidence = replay([row(0, sqi=sqi), row(1)])
    assert not evidence[0].decision.g3_sqi_admissible
    assert not evidence[0].update_admitted
    assert not evidence[0].refractory_rearmed_after_decision
    assert evidence[0].refractory_until_after == -np.inf
    # The following row is therefore not frozen.
    assert evidence[1].decision.g5_not_in_refractory


# --------------------------------------------------------------------------
# 12.11-12.14 G4 and the refractory
# --------------------------------------------------------------------------


def test_exact_g4_threshold_passes():
    evidence = replay([row(0)], scorer=constant_scorer(THRESHOLD))
    assert evidence[0].decision.g4_normal_evidence
    assert evidence[0].update_admitted
    assert not evidence[0].refractory_rearmed_after_decision


def test_value_above_g4_threshold_fails():
    evidence = replay([row(0)], scorer=constant_scorer(np.nextafter(THRESHOLD, np.inf)))
    assert not evidence[0].decision.g4_normal_evidence
    assert not evidence[0].update_admitted


def test_g4_failure_arms_refractory_only_after_current_decision():
    """The arming row itself is judged against the PRIOR refractory state."""
    evidence = replay([row(0)], scorer=constant_scorer(0.9))
    decision = evidence[0].decision
    assert decision.refractory_until_before == -np.inf
    assert decision.g5_not_in_refractory  # its own arming did not freeze it
    assert not decision.g4_normal_evidence
    assert evidence[0].refractory_rearmed_after_decision
    assert evidence[0].refractory_until_after == (
        evidence[0].available_time + G.REFRACTORY_DURATION_SECONDS
    )


def test_suspicious_row_inside_refractory_re_arms_future_refractory():
    evidence = replay([row(0), row(1), row(2)], scorer=constant_scorer(0.9))
    first, second, third = evidence
    assert second.available_time < first.refractory_until_after
    assert not second.decision.g5_not_in_refractory
    # Still scored while frozen, and its suspicious score extends the freeze.
    assert second.decision.score == 0.9
    assert second.refractory_rearmed_after_decision
    assert second.refractory_until_after > first.refractory_until_after
    assert third.refractory_until_after > second.refractory_until_after


def test_refractory_blocks_an_otherwise_admissible_row():
    scores = iter([0.9, THRESHOLD])
    evidence = replay([row(0), row(1)], scorer=lambda _r, _d: next(scores))
    assert not evidence[0].update_admitted
    second = evidence[1]
    assert second.decision.g4_normal_evidence
    assert second.decision.g3_sqi_admissible
    assert second.decision.g6_morphology_computable
    assert not second.decision.g5_not_in_refractory
    assert not second.update_admitted
    assert second.decision.refusal_reasons() == ("G5",)


# --------------------------------------------------------------------------
# 12.15-12.18 G6 and combined admission
# --------------------------------------------------------------------------


def test_g6_zero_refuses_update_but_still_permits_score():
    evidence = replay([row(0, morphology_valid=0.0)])
    assert evidence[0].decision.score == THRESHOLD
    assert not evidence[0].decision.g6_morphology_computable
    assert not evidence[0].update_admitted


def test_g6_zero_alone_does_not_arm_refractory():
    evidence = replay([row(0, morphology_valid=0.0), row(1)])
    assert not evidence[0].refractory_rearmed_after_decision
    assert evidence[0].refractory_until_after == -np.inf
    assert evidence[1].decision.g5_not_in_refractory


def test_sqi_fail_plus_g4_fail_still_rearms_via_g4():
    sqi = dict(CLEAN_SQI)
    sqi["flatline_fraction"] = 1.0
    evidence = replay([row(0, sqi=sqi)], scorer=constant_scorer(0.9))
    assert not evidence[0].decision.g3_sqi_admissible
    assert not evidence[0].decision.g4_normal_evidence
    assert evidence[0].refractory_rearmed_after_decision


def test_update_occurs_only_when_all_six_conditions_pass():
    evidence = replay([row(0)])
    decision = evidence[0].decision
    assert all(
        (
            decision.g1_available,
            decision.g2_finite_representation,
            decision.g3_sqi_admissible,
            decision.g4_normal_evidence,
            decision.g5_not_in_refractory,
            decision.g6_morphology_computable,
        )
    )
    assert decision.admitted and evidence[0].update_admitted
    assert decision.refusal_reasons() == ()


def test_m2_gated_refuses_where_m2_zero_updates():
    """The two arms differ only in admission, on identical input."""
    rows = [row(0, morphology_valid=0.0)]
    gated = replay(rows, arm=P.M2_ARM_GATED)
    naive = replay(rows, arm=P.M2_ARM_NAIVE)
    assert not gated[0].update_admitted
    assert naive[0].update_admitted
    assert gated[0].d_long == naive[0].d_long  # same score-before-update evidence


def test_m2_zero_ignores_every_gate_refusal_condition():
    """M2-0 updates whenever inherited M1 observation semantics permit it."""
    bad_sqi = dict(CLEAN_SQI)
    bad_sqi["flatline_fraction"] = 1.0
    cases = {
        "g3_sqi": (row(0, sqi=bad_sqi), THRESHOLD),
        "g4_normal_evidence": (row(0), 0.9),
        "g6_morphology": (row(0, morphology_valid=0.0), THRESHOLD),
        "g3_precondition": (row(0, finite_sample_fraction=0.5), THRESHOLD),
    }
    for name, (candidate, score) in cases.items():
        gated = replay([candidate], arm=P.M2_ARM_GATED, scorer=constant_scorer(score))
        naive = replay([candidate], arm=P.M2_ARM_NAIVE, scorer=constant_scorer(score))
        assert not gated[0].update_admitted, name
        assert naive[0].update_admitted, name

    # An UNAVAILABLE row is refused by BOTH arms: that is inherited M1
    # physical-observation semantics, not an M2 gate condition.
    unavailable = [row(0, state=OBSERVATION_UNAVAILABLE_EXACT_FLAT)]
    assert not replay(unavailable, arm=P.M2_ARM_NAIVE)[0].update_admitted
    assert not replay(unavailable, arm=P.M2_ARM_GATED)[0].update_admitted


# --------------------------------------------------------------------------
# 12.20-12.21 label and identity firewall
# --------------------------------------------------------------------------


def test_labels_cannot_alter_m2_gate_decisions():
    """No annotation field exists on the row or in the gate signature."""
    fields = set(P.M2TimelineRow.__dataclass_fields__)
    forbidden = {
        "target_family",
        "label",
        "binary_label",
        "ischemic",
        "challenge",
        "annotation",
        "quality_label",
        "subject_id",
    }
    assert not (fields & forbidden)
    parameters = set(inspect.signature(P.evaluate_gate).parameters)
    assert not (parameters & forbidden)
    replay_parameters = set(inspect.signature(P.replay_stream).parameters)
    assert not (replay_parameters & forbidden)


def test_patient_identifier_cannot_alter_m2_gate_decisions():
    """`record_id` selects a namespace; it is not visible to the gate."""
    assert "record_id" not in inspect.signature(P.evaluate_gate).parameters
    assert "subject_id" not in P.M2TimelineRow.__dataclass_fields__
    baseline = replay([row(0, record_id="s00001")])
    renamed = replay([row(0, record_id="s99999")])
    assert baseline[0].decision.admitted == renamed[0].decision.admitted
    assert baseline[0].d_long == renamed[0].d_long


def test_gate_module_has_no_annotation_identifier_anywhere():
    """No *code* in the policy module names an annotation-derived quantity.

    Checked over code identifiers and non-docstring literals rather than raw
    text, so the module stays free to document in prose exactly which labels
    it refuses to look at.
    """
    identifiers = code_identifiers(Path(P.__file__).read_text())
    for banned in ("target_family", "binary_label", "ischemic_positive", "labels"):
        assert banned not in identifiers, banned


# --------------------------------------------------------------------------
# 12.22-12.24 physical time, gaps, counters
# --------------------------------------------------------------------------


def test_refractory_uses_real_elapsed_seconds_not_row_count():
    """A dense stream needs many rows to clear 60 s; a sparse one needs few."""
    dense = replay(
        [row(i, start_sample=i * 250) for i in range(3)],  # 1 s apart
        scorer=constant_scorer(0.9),
    )
    assert not dense[1].decision.g5_not_in_refractory
    assert not dense[2].decision.g5_not_in_refractory

    sparse = [
        P.M2TimelineRow(
            record_id="s00001",
            channel_index=0,
            start_sample=start,
            observation_state=OBSERVATION_AVAILABLE,
            representation=representation(1.0),
            finite_sample_fraction=1.0,
            sqi=dict(CLEAN_SQI),
            morphology_valid=1.0,
        )
        for start in (0, 250 * 100)  # 100 s apart
    ]
    scores = iter([0.9, THRESHOLD])
    spaced = P.replay_stream(
        sparse,
        arm=P.M2_ARM_GATED,
        standardizer=identity_standardizer(),
        scorer=lambda _r, _d: next(scores),
    )
    assert spaced[1].decision.g5_not_in_refractory
    assert spaced[1].update_admitted


def test_available_time_matches_the_frozen_rule():
    assert P.available_time_seconds(0) == (0 + 2500) / 250.0
    assert P.available_time_seconds(1250) == (1250 + 2500) / 250.0


def test_streams_with_long_unavailable_gaps_resume_by_elapsed_time():
    rows = [
        row(0, value=1.0),
        *[row(i, state=OBSERVATION_UNAVAILABLE_EXACT_FLAT) for i in range(1, 20)],
        row(20, value=1.0),
    ]
    scores = iter([0.9] + [THRESHOLD])
    evidence = P.replay_stream(
        rows,
        arm=P.M2_ARM_GATED,
        standardizer=identity_standardizer(),
        scorer=lambda _r, _d: next(scores),
    )
    last = evidence[-1]
    assert last.decision.g5_not_in_refractory
    assert last.update_admitted
    # The gap contributed no observations and no updates.
    assert last.past_observed_count_before == 1
    assert last.past_update_count_before == 0


def test_counters_obey_inherited_m1_semantics():
    rows = [row(0), row(1, morphology_valid=0.0), row(2)]
    evidence = replay(rows)
    assert evidence[0].past_observed_count_before == 0
    assert evidence[1].past_observed_count_before == 1  # scored even though refused
    assert evidence[2].past_observed_count_before == 2
    assert evidence[0].past_update_count_after == 1
    assert evidence[1].past_update_count_after == 1  # refused row did not update
    assert evidence[2].past_update_count_after == 2


def test_time_since_last_admitted_update_is_undefined_before_any_update():
    evidence = replay([row(0, morphology_valid=0.0), row(1)])
    assert evidence[0].time_since_last_admitted_update is None
    assert evidence[1].time_since_last_admitted_update == 0.0


# --------------------------------------------------------------------------
# 12.25 / 9. no rollback anywhere
# --------------------------------------------------------------------------


def test_no_rollback_path_exists():
    """No executable rollback machinery in either M2 module."""
    for module in (P, E):
        identifiers = {
            name.lower() for name in code_identifiers(Path(module.__file__).read_text())
        }
        for banned in (
            "snapshot",
            "restore",
            "rollback_prototype",
            "m2-gr",
            "m2_gr",
            "oracle",
        ):
            assert banned not in identifiers, (module.__name__, banned)
    assert G.M2_ROLLBACK_IN_CORE is False
    assert P.m2_policy_identity(P.M2_ARM_GATED)["rollback"] is False
    with pytest.raises(P.M2PolicyError):
        P.require_m2_arm("M2-RB-ORACLE")


def test_policy_module_has_no_training_or_mutation_path():
    tree = ast.parse(Path(P.__file__).read_text())
    forbidden = {
        "train_m1_arm",
        "execute_m1_stage1",
        "materialize_stream_store",
        "write_json_atomic",
        "save",
        "unlink",
        "rmtree",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden, name
    imported = {
        alias.name
        for statement in ast.walk(tree)
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
    }
    assert not (imported & forbidden)
    assert "sealed_test" not in Path(P.__file__).read_text()


def test_only_two_claim_bearing_arms_exist():
    assert P.M2_ARMS == ("M2-0", "M2-G") == G.M2_CORE_ARMS
    with pytest.raises(P.M2PolicyError, match="exactly two"):
        P.require_m2_arm("M2-GR")


# --------------------------------------------------------------------------
# Frozen constants must never be restated incorrectly
# --------------------------------------------------------------------------


def test_policy_binds_the_frozen_constants_not_copies():
    identity = P.m2_policy_identity(P.M2_ARM_GATED)
    assert identity["g4_normal_evidence_threshold"] == 0.0002997174742631614
    assert identity["g5_refractory_seconds"] == 60.0
    assert identity["classification_threshold_used_for_admission"] is False
    assert G.M1L_CLASSIFICATION_THRESHOLD != G.NORMAL_EVIDENCE_THRESHOLD
    assert identity["retained_experiment_id"] == "M1L_long_memory_v2"


def test_classification_threshold_is_never_the_admission_threshold():
    """A row scoring below the classification threshold is still refused."""
    between = (G.NORMAL_EVIDENCE_THRESHOLD + G.M1L_CLASSIFICATION_THRESHOLD) / 2
    evidence = replay([row(0)], scorer=constant_scorer(between))
    assert between < G.M1L_CLASSIFICATION_THRESHOLD
    assert not evidence[0].decision.g4_normal_evidence
    assert not evidence[0].update_admitted


# --------------------------------------------------------------------------
# Evidence, schema and drift
# --------------------------------------------------------------------------


def test_admission_summary_reports_overlapping_refusals_honestly():
    sqi = dict(CLEAN_SQI)
    sqi["flatline_fraction"] = 1.0
    evidence = replay([row(0, sqi=sqi), row(1)], scorer=constant_scorer(0.9))
    summary = E.summarize_admission(evidence)
    assert summary["arm"] == P.M2_ARM_GATED
    assert summary["rows"] == 2
    assert summary["update_admitted_count"] == 0
    assert summary["update_admission_fraction"] == 0.0
    assert summary["update_admission_denominator"] == "all_timeline_rows"
    assert summary["freeze_fraction"] == 1.0
    assert "overlap" in summary["refusal_semantics"]
    # Both rows scored suspiciously, so both are applicable G4 failures; only
    # the first also fails G3. The two causes overlap on row 0 and are NOT
    # forced into exclusive attribution.
    assert summary["refusals"]["normal_evidence"]["failed_count"] == 2
    assert summary["refusals"]["sqi"]["failed_count"] == 1


def test_row_evidence_exposes_every_required_audit_field():
    evidence = replay([row(0)])
    payload = evidence[0].as_dict()
    for key in (
        "observation_state",
        "g1_available",
        "g2_finite_representation",
        "g3_sqi_admissible",
        "g3_feature_results",
        "score",
        "g4_normal_evidence",
        "refractory_until_before",
        "g5_not_in_refractory",
        "morphology_valid",
        "g6_morphology_computable",
        "update_admitted",
        "refractory_rearmed_after_current_decision",
        "time_since_last_admitted_update",
        "past_update_count_before",
        "past_update_count_after",
    ):
        assert key in payload, key


def test_prototype_drift_is_root_mean_square():
    reference = np.zeros(REPRESENTATION_DIM)
    current = np.full(REPRESENTATION_DIM, 2.0)
    assert E.prototype_drift(current, reference) == 2.0


def test_interval_drift_excludes_missing_pre_stress_prototype_with_reason():
    trajectory = E.PrototypeTrajectory(
        times=np.asarray([10.0, 20.0]),
        prototypes=np.zeros((2, REPRESENTATION_DIM)),
    )
    result = E.interval_drift_evidence(
        trajectory, stress_start_time=0.0, stress_end_time=30.0
    )
    assert result["mu_ref_available"] is False
    assert result["excluded_reason"] == E.EXCLUDED_NO_PRE_STRESS_PROTOTYPE
    assert result["peak_drift_during_stress"] is None


def test_interval_drift_never_fabricates_follow_up():
    prototypes = np.zeros((3, REPRESENTATION_DIM))
    prototypes[2] = 1.0
    trajectory = E.PrototypeTrajectory(
        times=np.asarray([0.0, 10.0, 20.0]), prototypes=prototypes
    )
    result = E.interval_drift_evidence(
        trajectory, stress_start_time=5.0, stress_end_time=25.0
    )
    assert result["mu_ref_available"] is True
    assert result["peak_drift_during_stress"] == 1.0
    for label, value in result["residual_drift"].items():
        assert value is None, label
        assert (
            result["residual_drift_excluded_reasons"][label]
            == E.EXCLUDED_NO_ELIGIBLE_FOLLOW_UP
        )


def test_result_schema_is_unpopulated_and_claims_nothing():
    for arm in P.M2_ARMS:
        schema = E.validate_unpopulated(E.empty_m2_result_schema(arm))
        assert schema["populated"] is False
        assert schema["scientific_execution_performed"] is False
        assert schema["validation_accessed"] is False
        assert schema["test_accessed"] is False
        assert all(value is None for value in schema["metrics"].values())
        assert all(value is None for value in schema["policy_evidence"].values())


def test_populated_result_is_refused():
    schema = E.empty_m2_result_schema(P.M2_ARM_GATED)
    schema["metrics"]["pooled_auprc"] = 0.5
    with pytest.raises(P.M2PolicyError):
        E.validate_unpopulated(schema)


# --------------------------------------------------------------------------
# Human review correction: M2-0 must remain a TRUE naive control, and gate
# evidence must distinguish an applicable failure from a non-applicable one.
# --------------------------------------------------------------------------


def test_m2_zero_never_acquires_an_operative_refractory_state():
    """§9.1 -- the naive control has no memory-update safety refractory."""
    evidence = replay(
        [row(0), row(1), row(2)], arm=P.M2_ARM_NAIVE, scorer=constant_scorer(0.9)
    )
    for item in evidence:
        assert item.refractory_until_after is None
        assert item.refractory_rearmed_after_decision is False
        assert item.decision.refractory_until_before is None
        assert item.decision.g5_not_in_refractory is None
        assert item.update_admitted  # every AVAILABLE finite row still updates


def test_high_m2_zero_score_does_not_create_an_actual_refractory_refusal():
    """§9.2 -- a suspicious score cannot freeze the naive control."""
    evidence = replay(
        [row(i) for i in range(5)], arm=P.M2_ARM_NAIVE, scorer=constant_scorer(0.99)
    )
    assert all(item.update_admitted for item in evidence)
    assert all("G5" not in item.decision.refusal_reasons() for item in evidence)
    assert evidence[-1].past_update_count_after == 5


def test_m2_zero_summary_reports_no_actual_normal_evidence_refusal():
    """§9.3 -- M2-0 updated the row, so it cannot also have refused it."""
    evidence = replay(
        [row(i) for i in range(3)], arm=P.M2_ARM_NAIVE, scorer=constant_scorer(0.99)
    )
    summary = E.summarize_admission(evidence)
    assert summary["arm"] == P.M2_ARM_NAIVE
    assert summary["update_admission_fraction"] == 1.0
    normal = summary["refusals"]["normal_evidence"]
    assert normal["applicable"] is False
    assert normal["failed_count"] == 0
    assert normal["evaluated_count"] == 0
    assert normal["fraction"] is None


def test_m2_zero_summary_reports_no_counterfactual_refractory_refusal():
    """§9.4 -- no counterfactual G5 state may reach a claim-bearing field."""
    evidence = replay(
        [row(i) for i in range(5)], arm=P.M2_ARM_NAIVE, scorer=constant_scorer(0.99)
    )
    summary = E.summarize_admission(evidence)
    for key in ("sqi", "normal_evidence", "refractory", "morphology"):
        entry = summary["refusals"][key]
        assert entry["applicable"] is False, key
        assert entry["failed_count"] == 0, key
        assert entry["fraction"] is None, key
    assert "counterfactual" not in summary


def test_unavailable_row_is_not_a_g4_normal_evidence_refusal():
    """§9.5 -- absence of a score is not a normal-evidence failure."""
    evidence = replay([row(0, state=OBSERVATION_UNAVAILABLE_EXACT_FLAT)])
    decision = evidence[0].decision
    assert decision.g1_available is False
    assert decision.g4_normal_evidence is None
    assert decision.refusal_reasons() == ("G1",)
    assert set(decision.not_applicable_conditions()) == {"G2", "G3", "G4", "G5", "G6"}
    summary = E.summarize_admission(evidence)
    assert summary["refusals"]["normal_evidence"]["evaluated_count"] == 0
    assert summary["refusals"]["normal_evidence"]["failed_count"] == 0


def test_unscored_nonfinite_representation_row_is_not_a_g4_refusal():
    """§9.6 -- G2 fails, no score exists, so G4 is not applicable."""
    broken = P.M2TimelineRow(
        record_id="s00001",
        channel_index=0,
        start_sample=0,
        observation_state=OBSERVATION_AVAILABLE,
        representation=np.full(REPRESENTATION_DIM, np.nan),
        finite_sample_fraction=1.0,
        sqi=dict(CLEAN_SQI),
        morphology_valid=1.0,
    )
    evidence = replay([broken])
    decision = evidence[0].decision
    assert decision.g1_available is True
    assert decision.g2_finite_representation is False
    assert decision.score is None
    assert decision.g4_normal_evidence is None
    assert "G4" not in decision.refusal_reasons()
    assert decision.refusal_reasons() == ("G2",)
    assert not evidence[0].update_admitted  # fail-closed
    # G3/G5/G6 still had their own inputs and remain evaluated.
    assert decision.g3_sqi_admissible is True
    assert decision.g5_not_in_refractory is True
    assert decision.g6_morphology_computable is True


def test_g4_denominator_is_exactly_the_scored_population():
    """§9.7 -- unscored rows enter neither numerator nor denominator."""
    rows = [
        row(0),  # scored, passes G4
        row(1, state=OBSERVATION_UNAVAILABLE_EXACT_FLAT),  # unscored
        row(2),  # scored, fails G4
    ]
    scores = iter([THRESHOLD, 0.99])
    evidence = P.replay_stream(
        rows,
        arm=P.M2_ARM_GATED,
        standardizer=identity_standardizer(),
        scorer=lambda _r, _d: next(scores),
    )
    summary = E.summarize_admission(evidence)
    normal = summary["refusals"]["normal_evidence"]
    assert summary["rows"] == 3
    assert summary["scored_rows"] == 2
    assert normal["evaluated_count"] == 2  # NOT 3
    assert normal["failed_count"] == 1
    assert normal["fraction"] == 0.5
    assert normal["not_applicable_count"] == 1
    assert normal["denominator"] == "rows_where_a_score_exists_and_g4_was_evaluated"


def test_scored_row_above_threshold_is_an_actual_g4_refusal_for_m2_gated():
    """§9.8 -- a real normal-evidence failure is still counted."""
    evidence = replay([row(0)], scorer=constant_scorer(np.nextafter(THRESHOLD, np.inf)))
    assert evidence[0].decision.g4_normal_evidence is False
    assert evidence[0].decision.refusal_reasons() == ("G4",)
    summary = E.summarize_admission(evidence)
    assert summary["refusals"]["normal_evidence"]["failed_count"] == 1
    assert summary["refusals"]["normal_evidence"]["evaluated_count"] == 1
    assert summary["refusals"]["normal_evidence"]["fraction"] == 1.0


def test_refusal_counts_and_fractions_match_explicit_integer_counts():
    """§9.9 -- every fraction is numerator/denominator over integers."""
    bad_sqi = dict(CLEAN_SQI)
    bad_sqi["flatline_fraction"] = 1.0
    rows = [
        row(0),  # clean, admitted
        row(1, sqi=bad_sqi),  # G3 failure
        row(2, morphology_valid=0.0),  # G6 failure
        row(3),  # clean
    ]
    evidence = replay(rows, scorer=constant_scorer(THRESHOLD))
    summary = E.summarize_admission(evidence)
    assert summary["rows"] == 4
    assert summary["refusals"]["sqi"]["failed_count"] == 1
    assert summary["refusals"]["sqi"]["evaluated_count"] == 4
    assert summary["refusals"]["sqi"]["fraction"] == 1 / 4
    assert summary["refusals"]["morphology"]["failed_count"] == 1
    assert summary["refusals"]["morphology"]["fraction"] == 1 / 4
    assert summary["refusals"]["normal_evidence"]["failed_count"] == 0
    assert summary["update_admitted_count"] == 2
    assert summary["update_admission_fraction"] == 2 / 4


def test_overlapping_actual_refusals_are_not_forced_into_exclusive_attribution():
    """§9.10 -- one row may genuinely fail several applicable conditions."""
    bad_sqi = dict(CLEAN_SQI)
    bad_sqi["flatline_fraction"] = 1.0
    evidence = replay(
        [row(0, sqi=bad_sqi, morphology_valid=0.0)], scorer=constant_scorer(0.99)
    )
    reasons = evidence[0].decision.refusal_reasons()
    assert set(reasons) == {"G3", "G4", "G6"}
    summary = E.summarize_admission(evidence)
    assert summary["refusals"]["sqi"]["failed_count"] == 1
    assert summary["refusals"]["normal_evidence"]["failed_count"] == 1
    assert summary["refusals"]["morphology"]["failed_count"] == 1
    # Overlapping causes do not sum to the freeze fraction, and the summary
    # says so rather than implying a partition.
    assert summary["freeze_fraction"] == 1.0


def test_m2_zero_parity_remains_exact_after_the_correction():
    """§9.11 -- removing the refractory changed no M2-0 trajectory."""
    values = [0.5, 1.5, -2.0, 3.25, 0.125]
    rows = [row(i, value=v) for i, v in enumerate(values)]
    evidence = replay(rows, arm=P.M2_ARM_NAIVE, scorer=constant_scorer(0.99))

    standardizer = identity_standardizer()
    reference = DualTimescaleMemory(standardizer.prior_vector())
    expected = [
        reference.observe(standardizer.standardize(representation(v))[0]).d_long
        for v in values
    ]
    assert [item.d_long for item in evidence] == expected
    assert all(item.update_admitted for item in evidence)
    assert evidence[-1].past_update_count_after == len(values)


def test_scorer_receives_raw_representation_and_pre_update_d_long():
    """§6 -- exact [raw 146-d z_t, pre-update d_long] scorer-input parity."""
    seen: list[tuple[np.ndarray, float]] = []

    def recording_scorer(vector, d_long):
        seen.append((np.array(vector, copy=True), d_long))
        return THRESHOLD

    values = [2.0, 4.0]
    rows = [row(i, value=v) for i, v in enumerate(values)]
    standardizer = identity_standardizer()
    evidence = P.replay_stream(
        rows,
        arm=P.M2_ARM_NAIVE,
        standardizer=standardizer,
        scorer=recording_scorer,
    )

    # The RAW frozen representation reaches the scorer, not a standardized or
    # post-update derivative of it.
    for (vector, _d_long), value in zip(seen, values, strict=True):
        assert np.array_equal(vector, representation(value))

    # The PRE-update d_long reaches the scorer: recomputing against a
    # reference memory that has not yet seen the current row reproduces it,
    # and it equals what the evidence recorded.
    reference = DualTimescaleMemory(standardizer.prior_vector())
    for step, value in enumerate(values):
        standardized = standardizer.standardize(representation(value))[0]
        expected_pre_update = reference.deviations(standardized).d_long
        assert seen[step][1] == expected_pre_update
        assert evidence[step].d_long == expected_pre_update
        reference.update(standardized)  # only now does the prototype move

    # No post-update value entered the scorer: the second row's d_long is
    # measured against a prototype that already moved for the first row only.
    assert seen[1][1] != seen[0][1]


def test_correction_introduces_no_label_or_identity_leakage():
    """§9.14 -- the firewall still holds after the semantics correction."""
    identifiers = code_identifiers(Path(P.__file__).read_text())
    for banned in ("target_family", "binary_label", "ischemic_positive", "labels"):
        assert banned not in identifiers, banned
    evidence_identifiers = code_identifiers(Path(E.__file__).read_text())
    for banned in ("target_family", "binary_label", "ischemic_positive"):
        assert banned not in evidence_identifiers, banned
    parameters = set(inspect.signature(P.evaluate_gate).parameters)
    assert "arm" in parameters  # the only new parameter
    assert not (parameters & {"label", "target_family", "subject_id"})


def test_correction_introduces_no_rollback_or_execution_route():
    """§9.15 -- still no rollback path and still no way to execute science."""
    for module in (P, E):
        source = Path(module.__file__).read_text()
        identifiers = {name.lower() for name in code_identifiers(source)}
        for banned in ("snapshot", "restore", "oracle", "m2_gr"):
            assert banned not in identifiers, (module.__name__, banned)
        assert "torch" not in identifiers
        assert "load_stream_store" not in identifiers
