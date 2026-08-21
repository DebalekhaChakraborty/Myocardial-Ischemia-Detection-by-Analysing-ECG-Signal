"""Tests for the canonical T1 fold evaluator.

The evaluator is the scientific body the graph was missing. These tests run it
-- on a synthetic timeline, outside the canonical namespace, with a target
source built in this file -- and prove the governance boundaries hold while it
runs. Nothing here authorizes execution, claims the canonical attempt, creates
the canonical run directory or reaches TEST, and a test at the bottom asserts
each of those directly.

The synthetic corpus is deliberately not LTSTDB. It carries the twelve frozen
subject identities because the authority refuses anything else, but its scores
and episodes are generated here, so no assertion below is a scientific claim
about ischemia detection.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import t1_canonical_driver as D
from cardiosentinel.neural import t1_capability_gate as G
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_development_run as R
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_fold_evaluation as N
from cardiosentinel.neural import t1_fold_evaluator as V
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.t1_fold_authority import (
    T1FoldAuthorityError,
    T1SubjectTargets,
    fit_evaluation_authority,
    held_out_evaluation_authority,
    require_active_scoped_request,
    require_validation_partition,
)
from cardiosentinel.neural.t1_protocol import (
    T1_STATE_EVENT,
    T1_STATES,
    T1_VALIDATION_SUBJECTS,
    T1Thresholds,
    candidate_policies,
    t1_folds,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CADENCE = 1250
ROWS_PER_SUBJECT = 120
EPISODE = range(40, 62)


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


# ---------------------------------------------------------------------------
# A synthetic timeline and a source that honours the scoped-request contract
# ---------------------------------------------------------------------------


def _corpus(seed: int = 11):
    rng = np.random.default_rng(seed)
    names = (
        "stable_id",
        "record_id",
        "channel_index",
        "start_sample",
        "subject_id",
        "score_present",
        "m2g_detector_score",
        "detector_decision_d_t",
        "oof_calibrated_probability_p_t",
        "decision_error_uncertainty_u_t",
        "s4d_temporal_evidence_s_t",
        "elapsed_stream_seconds",
    )
    columns: dict[str, list] = {name: [] for name in names}
    targets: dict[str, T1SubjectTargets] = {}
    for number, subject in enumerate(T1_VALIDATION_SUBJECTS):
        record = f"synthetic{number:02d}"
        identifiers, positives, masks = [], [], []
        for position in range(ROWS_PER_SUBJECT):
            stable_id = f"{subject}:{position}"
            present = bool(rng.random() > 0.05)
            positive = position in EPISODE
            probability = float(
                rng.uniform(0.80, 0.99) if positive else rng.uniform(0.0, 0.35)
            )
            temporal = float(
                rng.uniform(0.75, 0.99) if positive else rng.uniform(0.0, 0.45)
            )
            decision = bool(probability >= SPEC.T1_DETECTOR_THRESHOLD)
            columns["stable_id"].append(stable_id)
            columns["record_id"].append(record)
            columns["channel_index"].append(0)
            columns["start_sample"].append(position * CADENCE)
            columns["subject_id"].append(subject)
            columns["score_present"].append(present)
            columns["m2g_detector_score"].append(probability)
            columns["detector_decision_d_t"].append(decision if present else False)
            columns["oof_calibrated_probability_p_t"].append(
                probability if present else -1.0
            )
            columns["decision_error_uncertainty_u_t"].append(
                (1.0 - probability if decision else probability) if present else -1.0
            )
            columns["s4d_temporal_evidence_s_t"].append(temporal if present else -1.0)
            columns["elapsed_stream_seconds"].append(position * CADENCE / 250.0)
            identifiers.append(stable_id)
            positives.append(positive)
            masks.append(True)
        targets[subject] = T1SubjectTargets(
            subject_id=subject,
            stable_id=tuple(identifiers),
            primary_positive=tuple(positives),
            primary_mask=tuple(masks),
        )
    return {name: np.asarray(values) for name, values in columns.items()}, targets


class _Source:
    """A target source that proves the authorization exactly as the real one does.

    Records every subject it was asked for, so a test can assert the evaluator
    never reached past the authority's scope.
    """

    def __init__(self, targets):
        self._targets = targets
        self.asked: list[str] = []

    def read_subject_targets(self, subject_id, *, partition):
        partition = require_validation_partition(partition)
        require_active_scoped_request(subject_id, partition)
        self.asked.append(subject_id)
        return self._targets[subject_id]


class _UnauthorizedSource(_Source):
    """A source that answers with someone else's rows."""

    def read_subject_targets(self, subject_id, *, partition):
        super().read_subject_targets(subject_id, partition=partition)
        other = next(s for s in T1_VALIDATION_SUBJECTS if s != subject_id)
        return self._targets[other]


@pytest.fixture(scope="module")
def corpus():
    return _corpus()


def _promoted_state() -> dict:
    return {
        "selection_promoted": True,
        "selection_digest_verified": True,
        SPEC.T1_HELD_OUT_ACCESS_FLAG: True,
        "selection_sha256": "0" * 64,
    }


def _run_fold(columns, targets, fold_index: int = 0):
    fold = t1_folds()[fold_index]
    source = _Source(targets)
    evaluator = V.T1CanonicalFoldEvaluator()
    selection = evaluator(fold, fit_evaluation_authority(fold, source=source), columns)
    evaluated = evaluator.evaluate_held_out(
        fold,
        held_out_evaluation_authority(fold, _promoted_state(), source=source),
        columns,
        selection["artifact"],
    )
    return fold, source, selection["artifact"], evaluated


# ---------------------------------------------------------------------------
# 1. It runs, and it produces what the assembly layer consumes
# ---------------------------------------------------------------------------


def test_a_fold_selects_one_of_the_twelve_frozen_candidates(corpus):
    columns, targets = corpus
    _, _, artifact, _ = _run_fold(columns, targets)
    names = {policy.name for policy in candidate_policies()}
    assert artifact["selected_policy_id"] in names
    assert artifact["candidate_count"] == SPEC.T1_CANDIDATE_POLICIES_PER_FOLD
    assert set(artifact["candidate_metrics"]) == names


def test_the_selection_carries_four_generated_thresholds(corpus):
    columns, targets = corpus
    _, _, artifact, _ = _run_fold(columns, targets)
    for name in ("p_watch", "s_watch", "p_event", "s_event"):
        assert isinstance(artifact[name], float)
    assert artifact["p_event"] >= artifact["p_watch"]
    assert artifact["s_event"] >= artifact["s_watch"]
    assert artifact["threshold_population"] == V.T1_THRESHOLD_POPULATION


def test_the_held_out_evaluation_produces_every_trace_column(corpus):
    columns, targets = corpus
    fold, _, _, evaluated = _run_fold(columns, targets)
    trace = evaluated["trace_columns"]
    assert set(trace) == set(V.T1_HELD_OUT_TRACE_COLUMNS)
    width = {len(values) for values in trace.values()}
    assert width == {len(evaluated["row_positions"])}
    assert set(trace["emitted_state"]) <= set(T1_STATES)
    assert all(index == fold.fold_index for index in trace["fold_index"])


def test_the_twelve_traces_tile_the_timeline_exactly(corpus):
    columns, targets = corpus
    source = _Source(targets)
    evaluator = V.T1CanonicalFoldEvaluator()
    traces = {}
    for fold in t1_folds():
        selection = evaluator(
            fold, fit_evaluation_authority(fold, source=source), columns
        )
        traces[fold.fold_index] = evaluator.evaluate_held_out(
            fold,
            held_out_evaluation_authority(fold, _promoted_state(), source=source),
            columns,
            selection["artifact"],
        )
    widened = V.widen_with_held_out_traces(columns, traces)
    assert set(widened) == set(SPEC.T1_OOF_STATE_EVIDENCE_COLUMNS)
    assert len(widened["emitted_state"]) == len(columns["stable_id"])
    covered = sorted({int(v) for v in widened["fold_index"].tolist()})
    assert covered == list(range(SPEC.T1_FOLD_COUNT))


def test_a_row_evaluated_twice_is_refused(corpus):
    columns, targets = corpus
    _, _, _, evaluated = _run_fold(columns, targets)
    with pytest.raises(V.T1FoldEvaluatorError, match="do not tile the timeline"):
        V.widen_with_held_out_traces(columns, {0: evaluated, 1: evaluated})


def test_the_state_machine_actually_moves(corpus):
    """A trace stuck in NORMAL would satisfy the schema and prove nothing."""
    columns, targets = corpus
    _, _, _, evaluated = _run_fold(columns, targets)
    emitted = set(evaluated["trace_columns"]["emitted_state"])
    assert T1_STATE_EVENT in emitted
    assert len(emitted) > 1
    assert any(evaluated["trace_columns"]["transition_occurred"])


# ---------------------------------------------------------------------------
# 2. Deterministic
# ---------------------------------------------------------------------------


def test_the_same_input_produces_the_same_output(corpus):
    columns, targets = corpus
    first = _run_fold(columns, targets)[3]
    second = _run_fold(columns, targets)[3]
    assert first["trace_columns"] == second["trace_columns"]
    assert first["episode_evidence"] == second["episode_evidence"]
    assert first["primary_confusion"] == second["primary_confusion"]
    assert first["onset_latency_seconds"] == second["onset_latency_seconds"]


def test_selection_does_not_depend_on_the_order_targets_arrive_in(corpus):
    """Causal order is physical order, never the order a source answered in."""
    columns, targets = corpus
    reversed_targets = {
        subject: T1SubjectTargets(
            subject_id=t.subject_id,
            stable_id=tuple(reversed(t.stable_id)),
            primary_positive=tuple(reversed(t.primary_positive)),
            primary_mask=tuple(reversed(t.primary_mask)),
        )
        for subject, t in targets.items()
    }
    straight = _run_fold(columns, targets)[2]
    flipped = _run_fold(columns, reversed_targets)[2]
    assert straight["selected_policy_id"] == flipped["selected_policy_id"]
    assert straight["p_event"] == flipped["p_event"]


def test_the_trace_stepper_agrees_with_the_harness_runner(corpus):
    """The richer trace is a refinement, not a second opinion."""
    columns, targets = corpus
    fold = t1_folds()[0]
    source = _Source(targets)
    view = V._build_fold_view(
        columns, fit_evaluation_authority(fold, source=source), fold.fit_subjects
    )
    thresholds = T1Thresholds(p_watch=0.3, s_watch=0.3, p_event=0.7, s_event=0.7)
    policy = candidate_policies()[0]
    rows = {key: view.streams[key].rows for key in view.streams}
    harness = R.run_policy_over_streams(rows, thresholds, policy)
    for key in view.streams:
        traced = V.trace_stream(view.streams[key].rows, thresholds, policy.profile)
        assert list(traced.emitted) == harness[key]


def test_the_evaluator_module_reads_no_clock_and_no_random_source():
    code = ast.unparse(ast.parse(Path(V.__file__).read_text(encoding="utf-8")))
    for forbidden in ("random", "time.", "datetime", "uuid", "default_rng", "shuffle"):
        assert forbidden not in code, f"the evaluator uses {forbidden}"


# ---------------------------------------------------------------------------
# 3. It sees only what the authority approved
# ---------------------------------------------------------------------------


def test_selection_never_asks_for_the_held_out_subject(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    source = _Source(targets)
    V.T1CanonicalFoldEvaluator()(
        fold, fit_evaluation_authority(fold, source=source), columns
    )
    assert fold.held_out_subject not in source.asked
    assert sorted(source.asked) == sorted(fold.fit_subjects)


def test_the_held_out_phase_asks_for_exactly_one_subject(corpus):
    columns, targets = corpus
    fold, source, _, _ = _run_fold(columns, targets)
    held_out_only = [s for s in source.asked if s == fold.held_out_subject]
    assert len(held_out_only) == 1


def test_a_fit_authority_cannot_be_used_for_the_held_out_phase(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    source = _Source(targets)
    evaluator = V.T1CanonicalFoldEvaluator()
    selection = evaluator(fold, fit_evaluation_authority(fold, source=source), columns)
    with pytest.raises(V.T1FoldEvaluatorError, match="needs a 'held_out_subject_only'"):
        evaluator.evaluate_held_out(
            fold,
            fit_evaluation_authority(fold, source=source),
            columns,
            selection["artifact"],
        )


def test_a_held_out_authority_cannot_be_used_for_selection(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    source = _Source(targets)
    authority = held_out_evaluation_authority(fold, _promoted_state(), source=source)
    with pytest.raises(V.T1FoldEvaluatorError, match="needs a 'fit_subjects_only'"):
        V.T1CanonicalFoldEvaluator()(fold, authority, columns)


def test_the_held_out_phase_is_unreachable_before_the_barrier(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    source = _Source(targets)
    unpromoted = {"selection_promoted": False, "selection_digest_verified": False}
    with pytest.raises(Exception):
        held_out_evaluation_authority(fold, unpromoted, source=source)


def test_a_source_answering_for_another_subject_is_refused(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    source = _UnauthorizedSource(targets)
    with pytest.raises(T1FoldAuthorityError, match="refuses rows it did not ask for"):
        V.T1CanonicalFoldEvaluator()(
            fold, fit_evaluation_authority(fold, source=source), columns
        )


def test_the_evaluator_cannot_be_handed_a_bare_authority(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    with pytest.raises(V.T1FoldEvaluatorError, match="looser type"):
        V.T1CanonicalFoldEvaluator()(fold, R.fit_authority(fold.fit_subjects), columns)


def test_the_evaluator_holds_nothing_it_could_reach_data_with():
    evaluator = V.T1CanonicalFoldEvaluator()
    assert N.require_no_independent_access(evaluator) is evaluator
    assert getattr(evaluator, "__slots__", ()) == ()
    assert V.evaluator_capability()["holds_a_target_source"] is False


def test_membership_is_never_inferred_from_the_columns():
    """Row sets come from the authority's stable ids, not a subject predicate."""
    code = ast.unparse(ast.parse(Path(V.__file__).read_text(encoding="utf-8")))
    assert 'columns["subject_id"]' not in code
    assert "subject_id ==" not in code
    assert "targets_for_subject" in code


def test_a_labelled_column_in_the_timeline_is_refused(corpus):
    columns, targets = corpus
    poisoned = dict(columns)
    poisoned["label"] = np.zeros(len(columns["stable_id"]), dtype=bool)
    fold = t1_folds()[0]
    source = _Source(targets)
    with pytest.raises(V.T1FoldEvaluatorError, match="carries 'label'"):
        V.T1CanonicalFoldEvaluator()(
            fold, fit_evaluation_authority(fold, source=source), columns=poisoned
        )


# ---------------------------------------------------------------------------
# 4. Fails closed on a missing frozen dependency
# ---------------------------------------------------------------------------


def test_a_missing_input_column_fails_closed(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    source = _Source(targets)
    incomplete = {
        name: values
        for name, values in columns.items()
        if name != "s4d_temporal_evidence_s_t"
    }
    with pytest.raises(V.T1FoldEvaluatorError, match="missing"):
        V.T1CanonicalFoldEvaluator()(
            fold, fit_evaluation_authority(fold, source=source), incomplete
        )


def test_a_row_the_authority_names_but_the_timeline_lacks_fails_closed(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    broken = dict(targets)
    subject = fold.fit_subjects[0]
    original = targets[subject]
    broken[subject] = T1SubjectTargets(
        subject_id=subject,
        stable_id=original.stable_id + ("nowhere:0",),
        primary_positive=original.primary_positive + (False,),
        primary_mask=original.primary_mask + (True,),
    )
    source = _Source(broken)
    with pytest.raises(V.T1FoldEvaluatorError, match="does not contain"):
        V.T1CanonicalFoldEvaluator()(
            fold, fit_evaluation_authority(fold, source=source), columns
        )


def test_an_empty_background_population_fails_closed(corpus):
    columns, targets = corpus
    fold = t1_folds()[0]
    all_positive = {
        subject: T1SubjectTargets(
            subject_id=t.subject_id,
            stable_id=t.stable_id,
            primary_positive=tuple(True for _ in t.primary_positive),
            primary_mask=t.primary_mask,
        )
        for subject, t in targets.items()
    }
    source = _Source(all_positive)
    with pytest.raises(V.T1FoldEvaluatorError, match="no FIT-subject PRIMARY"):
        V.T1CanonicalFoldEvaluator()(
            fold, fit_evaluation_authority(fold, source=source), columns
        )


def test_a_selection_naming_an_unknown_policy_fails_closed(corpus):
    columns, targets = corpus
    fold, source, artifact, _ = _run_fold(columns, targets)
    forged = {**artifact, "selected_policy_id": "qw0.5_qe0.5_INVENTED"}
    with pytest.raises(V.T1FoldEvaluatorError, match="not one of the twelve"):
        V.T1CanonicalFoldEvaluator().evaluate_held_out(
            fold,
            held_out_evaluation_authority(fold, _promoted_state(), source=source),
            columns,
            forged,
        )


def test_a_selection_missing_its_thresholds_fails_closed(corpus):
    columns, targets = corpus
    fold, source, artifact, _ = _run_fold(columns, targets)
    stripped = {k: v for k, v in artifact.items() if k != "p_event"}
    with pytest.raises(V.T1FoldEvaluatorError, match="missing"):
        V.T1CanonicalFoldEvaluator().evaluate_held_out(
            fold,
            held_out_evaluation_authority(fold, _promoted_state(), source=source),
            columns,
            stripped,
        )


def test_a_fold_mismatch_fails_closed(corpus):
    columns, targets = corpus
    source = _Source(targets)
    with pytest.raises(V.T1FoldEvaluatorError, match="One evaluation is one fold"):
        V.T1CanonicalFoldEvaluator()(
            t1_folds()[1],
            fit_evaluation_authority(t1_folds()[0], source=source),
            columns,
        )


# ---------------------------------------------------------------------------
# 5. The capability gate accepts the real evaluator and only the real one
# ---------------------------------------------------------------------------


def test_the_gate_accepts_the_real_evaluator():
    attestation = G.require_completable("evaluate_fold", V.T1CanonicalFoldEvaluator())
    assert attestation.executes is True
    assert attestation.provider == "T1CanonicalFoldEvaluator"


def test_the_gate_still_rejects_the_non_executing_evaluator():
    with pytest.raises(G.T1CapabilityError, match="cannot complete"):
        G.require_completable("evaluate_fold", N.T1NonExecutingFoldEvaluator())


def test_the_gate_rejects_an_evaluator_with_no_second_phase():
    class HalfAnEvaluator:
        def __call__(self, fold, authority, columns):
            return {"artifact": {}}

        def t1_execution_capability(self):
            return G.attest(
                "evaluate_fold",
                provider="HalfAnEvaluator",
                executes=True,
                reason="Selects and cannot evaluate.",
            )

    with pytest.raises(G.T1CapabilityError, match="evaluate_held_out"):
        G.require_completable("evaluate_fold", HalfAnEvaluator())


def test_the_evaluator_matches_the_harness_call_shape():
    harness = Path(R.__file__).read_text(encoding="utf-8")
    assert "evaluate_fold(\n" in harness
    assert "evaluate_fold.evaluate_held_out(" in harness
    assert G.CAPABILITY_CALL_CONTRACT["evaluate_fold"] == (3, ())
    assert G.SECOND_PHASE_POSITIONALS == 4


# ---------------------------------------------------------------------------
# 6. Nothing was authorized, claimed, created or opened
# ---------------------------------------------------------------------------


def test_authorization_is_untouched():
    """Agreement between the two views is the invariant that survives the flip.

    A divergent copy of the constant is how a gate opens on one code path and
    not another, so the two are compared to each other rather than to a value.
    """
    assert D.T1_EXECUTION_SPECIFICATION_AUTHORIZED is (
        CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED
    )


def test_the_canonical_attempt_is_untouched(corpus):
    columns, targets = corpus
    assert not _canonical_root().exists()
    _run_fold(columns, targets)
    assert not _canonical_root().exists()


def test_the_evaluator_creates_nothing_and_opens_no_file():
    code = ast.unparse(ast.parse(Path(V.__file__).read_text(encoding="utf-8")))
    for forbidden in (
        "mkdir",
        "makedirs",
        "write_text",
        "write_bytes",
        "np.load",
        "np.save",
        "open(",
        "shutil",
    ):
        assert forbidden not in code, f"the evaluator calls {forbidden}"


def test_the_evaluator_reaches_no_test_partition():
    tree = ast.parse(Path(V.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "cardiosentinel.neural.t1_persistence" not in imported
    assert V.evaluator_capability()["test_accessed"] is False
    with pytest.raises(SPEC.T1ExecutionSpecError, match="TEST is sealed"):
        SPEC.require_no_test_access("test")


def test_the_evaluator_alters_no_frozen_science():
    """It composes frozen components; it defines no threshold and no ordering."""
    code = ast.unparse(ast.parse(Path(V.__file__).read_text(encoding="utf-8")))
    for forbidden in ("policy_sort_key", "Q_WATCH =", "Q_EVENT =", "T1_PROFILE_"):
        assert forbidden not in code, f"the evaluator redefines {forbidden}"
    assert "select_policy" in code
    assert "generate_thresholds" in code
    assert "next_state" in code


def test_the_upstream_contracts_are_unchanged():
    assert SPEC.T1_REQUIRED_M2_RETAINED_ARM == "M2-G"
    assert SPEC.T1_REQUIRED_U1_FAMILY == "platt_logistic_on_recovered_logit"
    assert SPEC.T1_REQUIRED_T2_RETAINED_ARM == "causal_s4d_longitudinal_v1"
    assert SPEC.T1_U1_REFIT_PERMITTED is False
    assert SPEC.T1_M2_REPLAY_PERMITTED is False
    assert SPEC.T1_T2_REPLAY_PERMITTED is False
