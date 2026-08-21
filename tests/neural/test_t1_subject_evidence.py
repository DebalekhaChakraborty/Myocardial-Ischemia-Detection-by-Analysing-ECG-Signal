"""Tests for subject-level evidence assembled from the held-out evaluations.

Subject-level metrics are label-dependent and the OOF state evidence store is
label-free by design, so these do not come from `oof_columns`. They come from
the held-out evaluations, which is the same truth seen at the right stage: each
fold holds out exactly one subject, so per-fold held-out evidence *is*
per-subject evidence.

Nothing here opens a label, re-runs a fold, re-derives a policy, authorizes
execution, claims the canonical attempt or reaches TEST.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_assembly as A
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.t1_protocol import T1_VALIDATION_SUBJECTS, t1_folds

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _trace(fold_index: int, **overrides) -> dict:
    fold = t1_folds()[fold_index]
    trace = {
        "fold_index": fold_index,
        "held_out_subject": fold.held_out_subject,
        "selected_policy_id": "qw0.9_qe0.99_FAST",
        "policy_runs": SPEC.T1_HELD_OUT_POLICY_RUNS_PER_FOLD,
        "episode_evidence": {
            "reference_episodes": 4,
            "predicted_event_runs": 5,
            "matched_episodes": 3,
            "unmatched_predicted_runs": 2,
        },
        "primary_confusion": {"tp": 30, "fp": 10, "tn": 900, "fn": 20},
        "onset_latency_seconds": (10.0, 20.0, 30.0),
    }
    trace.update(overrides)
    return trace


def _traces(**per_fold) -> dict[int, dict]:
    traces = {index: _trace(index) for index in range(SPEC.T1_FOLD_COUNT)}
    for index, overrides in per_fold.items():
        traces[index] = _trace(index, **overrides)
    return traces


# ---------------------------------------------------------------------------
# 1. The fold <-> subject bijection
# ---------------------------------------------------------------------------


def test_each_subject_maps_to_exactly_one_held_out_fold():
    by_subject = A.require_held_out_bijection(_traces())
    assert sorted(by_subject) == sorted(T1_VALIDATION_SUBJECTS)
    assert len(by_subject) == SPEC.T1_FOLD_COUNT
    folds = sorted(int(trace["fold_index"]) for trace in by_subject.values())
    assert folds == list(range(SPEC.T1_FOLD_COUNT))


def test_a_repeated_subject_is_refused():
    traces = _traces()
    traces[1] = _trace(1, held_out_subject=traces[0]["held_out_subject"])
    with pytest.raises(A.T1AssemblyError, match="more than one fold"):
        A.require_held_out_bijection(traces)


def test_a_missing_fold_is_refused():
    traces = _traces()
    del traces[7]
    with pytest.raises(A.T1AssemblyError, match="covers folds"):
        A.require_held_out_bijection(traces)


def test_a_fold_that_misnames_itself_is_refused():
    traces = _traces()
    traces[3] = {**_trace(3), "fold_index": 9}
    with pytest.raises(A.T1AssemblyError, match="A fold names itself"):
        A.require_held_out_bijection(traces)


def test_a_subject_outside_the_frozen_roster_is_refused():
    traces = _traces()
    traces[0] = _trace(0, held_out_subject="ltstdb:s9999")
    with pytest.raises(A.T1AssemblyError, match="No fold held out"):
        A.require_held_out_bijection(traces)


# ---------------------------------------------------------------------------
# 2. Only after the held-out barrier
# ---------------------------------------------------------------------------


def test_more_than_one_held_out_policy_run_is_refused():
    traces = _traces()
    traces[2] = _trace(2, policy_runs=2)
    with pytest.raises(A.T1AssemblyError, match="held-out policy"):
        A.require_held_out_bijection(traces)
    assert SPEC.T1_HELD_OUT_POLICY_RUNS_PER_FOLD == 1


def test_evidence_without_a_promoted_policy_is_refused():
    traces = _traces()
    traces[5] = {
        key: value for key, value in _trace(5).items() if key != "selected_policy_id"
    }
    with pytest.raises(A.T1AssemblyError, match="selected_policy_id"):
        A.require_held_out_bijection(traces)


def test_a_partial_set_of_folds_cannot_produce_subject_evidence():
    """A barrier that opened for some folds is not a cross-fitted result."""
    partial = {index: _trace(index) for index in range(4)}
    with pytest.raises(A.T1AssemblyError):
        A.derive_subject_evidence(held_out_traces=partial)


# ---------------------------------------------------------------------------
# 3. Metrics match the evaluator's own counts
# ---------------------------------------------------------------------------


def test_metrics_are_computed_from_the_evaluators_counts():
    evidence = A.derive_subject_evidence(held_out_traces=_traces())
    subject = T1_VALIDATION_SUBJECTS[0]
    entry = evidence[subject]
    assert entry["reference_episodes"] == 4
    assert entry["predicted_event_runs"] == 5
    assert entry["matched_episodes"] == 3
    assert entry["primary_true_positive"] == 30
    assert entry["primary_false_negative"] == 20
    # 2*3 / (2*3 + (5-3) + (4-3))
    assert entry["episode_f1"] == pytest.approx(6 / 9)


def test_the_frozen_helpers_compute_the_metrics():
    from cardiosentinel.neural.t1_development_run import episode_f1, window_mcc

    evidence = A.derive_subject_evidence(held_out_traces=_traces())
    entry = evidence[T1_VALIDATION_SUBJECTS[0]]
    assert entry["episode_f1"] == episode_f1(3, 5, 4)
    predicted, actual = A._confusion_arrays({"tp": 30, "fp": 10, "tn": 900, "fn": 20})
    assert entry["primary_window_mcc"] == window_mcc(predicted, actual)
    assert int(predicted.sum()) == 40
    assert int(actual.sum()) == 50


def test_an_undefined_metric_stays_undefined_and_never_becomes_zero():
    traces = _traces()
    traces[0] = _trace(
        0,
        episode_evidence={
            "reference_episodes": 0,
            "predicted_event_runs": 0,
            "matched_episodes": 0,
            "unmatched_predicted_runs": 0,
        },
        onset_latency_seconds=(),
    )
    entry = A.derive_subject_evidence(held_out_traces=traces)[T1_VALIDATION_SUBJECTS[0]]
    assert entry["episode_f1"] is None
    assert entry["median_onset_latency_seconds"] is None
    assert entry["detected_episode_count"] == 0


def test_an_undefined_statistic_is_carried_as_nan_not_zero():
    traces = _traces()
    traces[0] = _trace(
        0,
        episode_evidence={
            "reference_episodes": 0,
            "predicted_event_runs": 0,
            "matched_episodes": 0,
            "unmatched_predicted_runs": 0,
        },
    )
    statistic = A.derive_subject_statistic(held_out_traces=traces)
    assert math.isnan(statistic[T1_VALIDATION_SUBJECTS[0]])
    assert not any(statistic[s] == 0.0 for s in T1_VALIDATION_SUBJECTS), (
        "zero would be indistinguishable from a real measurement"
    )


def test_the_bootstrap_statistic_is_named_and_matches_the_evidence():
    assert A.BOOTSTRAP_SUBJECT_STATISTIC == "episode_f1"
    evidence = A.derive_subject_evidence(held_out_traces=_traces())
    statistic = A.derive_subject_statistic(held_out_traces=_traces())
    for subject in T1_VALIDATION_SUBJECTS:
        assert statistic[subject] == pytest.approx(evidence[subject]["episode_f1"])


def test_the_median_is_the_median():
    assert A._median((30.0, 10.0, 20.0)) == 20.0
    assert A._median((10.0, 20.0, 30.0, 40.0)) == 25.0
    assert A._median(()) is None


# ---------------------------------------------------------------------------
# 4. Deterministic and completely shaped
# ---------------------------------------------------------------------------


def test_assembly_is_deterministic_and_ordered():
    first = A.derive_subject_evidence(held_out_traces=_traces())
    second = A.derive_subject_evidence(held_out_traces=_traces())
    assert first == second
    assert list(first) == sorted(first)
    assert list(A.derive_subject_statistic(held_out_traces=_traces())) == sorted(
        T1_VALIDATION_SUBJECTS
    )


def test_every_named_field_is_present_for_every_subject():
    evidence = A.derive_subject_evidence(held_out_traces=_traces())
    for subject in T1_VALIDATION_SUBJECTS:
        assert set(evidence[subject]) == set(A.SUBJECT_EVIDENCE_FIELDS)


def test_the_collaborators_produce_the_driver_shaped_artifacts():
    from cardiosentinel.neural import t1_capability_gate as G

    subject = A.assemble_subject_evidence(held_out_traces=_traces())
    bootstrap = A.assemble_bootstrap(held_out_traces=_traces())
    assert G.require_completable("assemble_subject_evidence", subject).executes
    assert G.require_completable("assemble_bootstrap", bootstrap).executes


# ---------------------------------------------------------------------------
# 5. No independent label access, no archive, no authority bypass
# ---------------------------------------------------------------------------


def test_the_assembly_layer_still_opens_nothing():
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))

    class _Strip(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and " " in node.value:
                return ast.Constant(value="")
            return node

    code = ast.unparse(_Strip().visit(tree))
    for forbidden in (
        "np.load",
        "read_t2_identity_members",
        "read_m2g_row_evidence",
        "read_subject_targets",
        "open(",
        "read_bytes",
        "Path(",
    ):
        assert forbidden not in code, f"the assembly layer calls {forbidden}"


def test_no_independent_label_reader_exists():
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "cardiosentinel.neural.t1_evidence_store" not in imported
    assert "cardiosentinel.neural.t1_fold_authority" not in imported


def test_no_fold_is_rerun_and_no_policy_is_rederived():
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    code = ast.unparse(tree)
    for forbidden in (
        "select_policy",
        "generate_thresholds",
        "run_policy_over_streams",
        "next_state",
        "candidate_policies",
    ):
        assert forbidden not in code, f"the assembly layer calls {forbidden}"


# ---------------------------------------------------------------------------
# 6. Nothing authorized, claimed or opened
# ---------------------------------------------------------------------------


def test_authorization_remains_false():
    A.derive_subject_evidence(held_out_traces=_traces())
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is False


def test_the_canonical_attempt_is_untouched():
    assert not _canonical_root().exists()
    A.derive_subject_statistic(held_out_traces=_traces())
    assert not _canonical_root().exists()


def test_final_configuration_is_not_implemented_here():
    """Out of scope for this PR, and deliberately still injected."""
    import inspect

    parameters = inspect.signature(A.assemble_final_configuration).parameters
    assert "configuration" in parameters
