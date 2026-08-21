"""Tests for the T1 label-bearing assembly collaborators.

These exercise the composition layer on synthetic structures. Nothing here
runs a fold, evaluates a policy, generates a prediction, opens a VALIDATION
label, touches TEST, creates a canonical directory or changes authorization.

Every input is built in the test. That is the point: these collaborators take
values, they do not fetch them, and a test that had to fetch anything would be
evidence the module reads something it should not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import t1_assembly as A
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.t1_protocol import (
    T1_STATE_EVENT,
    T1_STATE_NORMAL,
    T1_STATE_WATCH,
    T1_VALIDATION_SUBJECTS,
    t1_folds,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROWS = 24


class _Source:
    """Minimal target source: the protocol shape and nothing wider."""

    def read_subject_targets(self, subject_id, *, partition):  # pragma: no cover
        raise AssertionError("no test in this module opens targets")


IDENTITY_PATH = Path("/nonexistent/t2_outer_row_identity.npz")


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _selections(count: int = 12) -> list[dict]:
    return [{"fold_index": i, "selection_sha256": f"{i:064d}"} for i in range(count)]


def _columns(rows: int = ROWS) -> dict[str, np.ndarray]:
    """A synthetic OOF state trace with every frozen column present."""
    subjects = np.asarray(
        [T1_VALIDATION_SUBJECTS[i % len(T1_VALIDATION_SUBJECTS)] for i in range(rows)]
    )
    states = np.asarray(
        [
            (
                T1_STATE_EVENT
                if i % 6 == 0
                else T1_STATE_WATCH
                if i % 3 == 0
                else T1_STATE_NORMAL
            )
            for i in range(rows)
        ]
    )
    base = {
        "stable_id": np.asarray([f"row-{i}" for i in range(rows)]),
        "record_id": np.asarray([f"s{2004 + (i % 12):04d}1" for i in range(rows)]),
        "channel_index": np.zeros(rows, dtype=np.int32),
        "start_sample": np.arange(rows, dtype=np.int64) * 1250,
        "subject_id": subjects,
        "score_present": np.ones(rows, dtype=bool),
        "m2g_detector_score": np.linspace(0.1, 0.9, rows),
        "detector_decision_d_t": np.zeros(rows, dtype=bool),
        "oof_calibrated_probability_p_t": np.linspace(0.1, 0.9, rows),
        "decision_error_uncertainty_u_t": np.linspace(0.1, 0.5, rows),
        "s4d_temporal_evidence_s_t": np.linspace(0.2, 0.8, rows),
        "elapsed_stream_seconds": np.arange(rows, dtype=np.float64) * 5.0,
        "fold_index": np.asarray([i % 12 for i in range(rows)], dtype=np.int32),
        "selected_policy_id": np.asarray(["P1"] * rows),
        "p_watch": np.full(rows, 0.6),
        "s_watch": np.full(rows, 0.6),
        "p_event": np.full(rows, 0.8),
        "s_event": np.full(rows, 0.8),
        "emitted_state": states,
        "state_elapsed_seconds": np.zeros(rows, dtype=np.float64),
        "transition_from": np.asarray([T1_STATE_NORMAL] * rows),
        "transition_to": states,
        "transition_occurred": np.zeros(rows, dtype=bool),
    }
    return base


def _code_only() -> str:
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]

    class _DropProse(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            if isinstance(node.value, str) and " " in node.value:
                return ast.Constant(value="")
            return node

    return ast.unparse(_DropProse().visit(tree))


# ---------------------------------------------------------------------------
# 1. Each collaborator exists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", A.assembly_capability()["collaborators"])
def test_each_collaborator_exists(name):
    assert hasattr(A, name), name
    assert callable(getattr(A, name))


def _identity_file(tmp_path):
    """A synthetic row identity whose stable ids match `_columns()`."""
    import numpy as _np

    stable = _np.asarray(_columns()["stable_id"])
    families = _np.asarray(
        [
            "rate_related_confounder" if index % 4 == 0 else "background_negative"
            for index in range(len(stable))
        ]
    )
    path = tmp_path / "t2_outer_row_identity.npz"
    _np.savez(path, stable_id=stable, target_family=families)
    return path


def _held_out_traces(undefined_subjects=()):
    """Twelve held-out evaluations, one per subject, matching the bijection.

    `episode_f1` is 0.5 for every subject by construction (matched 1, predicted
    2, reference 2 gives 2/(2+1+1)), so a test that needs a known statistic
    gets one without injecting it.
    """
    traces = {}
    for fold in t1_folds():
        undefined = fold.held_out_subject in undefined_subjects
        traces[fold.fold_index] = {
            "fold_index": fold.fold_index,
            "held_out_subject": fold.held_out_subject,
            "selected_policy_id": "qw0.9_qe0.99_FAST",
            "policy_runs": 1,
            "episode_evidence": {
                "reference_episodes": 0 if undefined else 2,
                "predicted_event_runs": 0 if undefined else 2,
                "matched_episodes": 0 if undefined else 1,
                "unmatched_predicted_runs": 0 if undefined else 1,
            },
            "primary_confusion": {"tp": 5, "fp": 5, "tn": 90, "fn": 5},
            "onset_latency_seconds": () if undefined else (12.0,),
        }
    return traces


def test_the_collaborators_are_exactly_the_drivers_missing_ones():
    from cardiosentinel.neural.t1_canonical_driver import (
        REQUIRED_COLLABORATOR_CALLABLES,
    )

    provided = set(A.assembly_capability()["collaborators"])
    required = set(REQUIRED_COLLABORATOR_CALLABLES)
    assert provided == required - {"evaluate_fold"}, (
        "the assembly layer must supply every required callable except the fold "
        "evaluator, which is a separate scientific capability"
    )


# ---------------------------------------------------------------------------
# §7 subject identity
# ---------------------------------------------------------------------------


def test_subject_of_record_resolves_through_the_frozen_authority():
    authority = A.subject_of_record()
    assert authority("s20041") == "ltstdb:s2004"
    assert authority.subject_of_record("s20041") == "ltstdb:s2004"


def test_an_off_roster_subject_is_refused():
    with pytest.raises(A.T1AssemblyError, match="not in the frozen"):
        A.subject_of_record()("s20011")


def test_a_persisted_identity_disagreement_is_a_hard_stop():
    authority = A.subject_of_record()
    assert authority.require_agreement("s20041", "ltstdb:s2004") == "ltstdb:s2004"
    with pytest.raises(A.T1AssemblyError, match="Exact agreement"):
        authority.require_agreement("s20041", "ltstdb:s2005")


def test_identity_is_never_a_transition_feature_or_label_derived():
    described = A.subject_of_record().as_dict()
    assert described["is_transition_feature"] is False
    assert described["derived_from_label"] is False
    assert described["selects"] == ["state_namespace", "u1_loso_calibrator"]
    assert SPEC.T1_SUBJECT_IDENTITY_IS_TRANSITION_FEATURE is False
    assert SPEC.T1_SUBJECT_IDENTITY_DERIVED_FROM_LABEL is False


def test_the_authority_returns_only_a_namespace_string():
    """It cannot leak an outcome because it returns nothing that could carry one."""
    assert isinstance(A.subject_of_record()("s20041"), str)


# ---------------------------------------------------------------------------
# 2. Unauthorized / dishonest input is rejected
# ---------------------------------------------------------------------------


def test_oof_state_columns_refuse_a_partial_input():
    columns = _columns()
    del columns["emitted_state"]
    with pytest.raises(A.T1AssemblyError, match="missing columns"):
        A.assemble_oof_state_columns(columns=columns, selections=_selections())


def test_oof_state_columns_refuse_an_incomplete_fold_set():
    with pytest.raises(A.T1AssemblyError, match="frozen design is 12"):
        A.assemble_oof_state_columns(columns=_columns(), selections=_selections(11))


def test_oof_state_columns_refuse_a_trace_missing_a_fold():
    columns = _columns()
    columns["fold_index"] = np.zeros(ROWS, dtype=np.int32)
    with pytest.raises(A.T1AssemblyError, match="every row is held out|not the frozen"):
        A.assemble_oof_state_columns(columns=columns, selections=_selections())


def test_oof_state_columns_refuse_a_non_state():
    columns = _columns()
    columns["emitted_state"] = np.asarray(["ELEVATED"] * ROWS)
    with pytest.raises(A.T1AssemblyError, match="non-states"):
        A.assemble_oof_state_columns(columns=columns, selections=_selections())


def test_oof_state_columns_return_exactly_the_frozen_schema():
    assembled = A.assemble_oof_state_columns(
        columns=_columns(), selections=_selections()
    )
    assert tuple(assembled) == tuple(SPEC.T1_OOF_STATE_EVIDENCE_COLUMNS)
    for forbidden in SPEC.T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS:
        assert forbidden not in assembled


def test_oof_result_refuses_impossible_episode_counts():
    with pytest.raises(A.T1AssemblyError, match="cannot exceed"):
        A.assemble_oof_result(
            episode_evidence={
                "reference_episodes": 2,
                "predicted_event_runs": 2,
                "matched_episodes": 5,
            },
            onset_latency_seconds=[],
            primary_confusion={
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 1,
                "true_negative": 1,
            },
        )(oof_columns=_columns(), selections=_selections())


def test_final_configuration_refuses_before_the_oof_result_is_promoted():
    with pytest.raises(A.T1AssemblyError, match="only after"):
        A.assemble_final_configuration(
            configuration=dict.fromkeys(A.FINAL_CONFIGURATION_FIELDS, 0.5),
            oof_result_promoted=False,
        )(oof_columns=_columns(), selections=_selections())


def test_final_configuration_refuses_a_missing_field():
    configuration = dict.fromkeys(A.FINAL_CONFIGURATION_FIELDS, 0.5)
    del configuration["p_event"]
    with pytest.raises(A.T1AssemblyError, match="missing"):
        A.assemble_final_configuration(
            configuration=configuration, oof_result_promoted=True
        )(oof_columns=_columns(), selections=_selections())


def test_challenge_families_cannot_be_unknown_by_construction(tmp_path):
    """The guarantee moved: membership is derived, so it cannot be forged.

    The builder used to refuse an injected family it did not recognise. It no
    longer takes one, so an unknown family is not refused -- it is unreachable,
    which is the stronger property.
    """
    import inspect

    assert "challenge_rows" not in inspect.signature(A.assemble_challenge).parameters
    artifact = A.assemble_challenge(t2_identity=_identity_file(tmp_path))(
        oof_columns=_columns()
    )
    assert set(artifact["families"]) == set(A.CHALLENGE_FAMILIES)


def test_challenge_rows_cannot_fall_outside_the_trace(tmp_path):
    """Rows are positions found in the trace, so out-of-range cannot arise."""
    artifact = A.assemble_challenge(t2_identity=_identity_file(tmp_path))(
        oof_columns=_columns()
    )
    width = len(_columns()["stable_id"])
    for family in A.CHALLENGE_FAMILIES:
        assert artifact["families"][family]["row_count"] <= width


def test_subject_evidence_refuses_an_incomplete_fold_set():
    with pytest.raises(A.T1AssemblyError, match="covers folds"):
        A.assemble_subject_evidence(held_out_traces={})(oof_columns=_columns())


def test_bootstrap_refuses_an_incomplete_fold_set():
    with pytest.raises(A.T1AssemblyError, match="covers folds"):
        A.assemble_bootstrap(held_out_traces={})(oof_columns=_columns())


# ---------------------------------------------------------------------------
# Spec conformance of the assembled artifacts
# ---------------------------------------------------------------------------


def _oof_result():
    return A.assemble_oof_result(
        episode_evidence={
            "reference_episodes": 6,
            "predicted_event_runs": 4,
            "matched_episodes": 3,
        },
        onset_latency_seconds=[5.0, 10.0, 15.0],
        primary_confusion={
            "true_positive": 3,
            "false_positive": 2,
            "false_negative": 3,
            "true_negative": 16,
        },
    )(oof_columns=_columns(), selections=_selections())


def test_the_oof_result_reports_the_sections_the_spec_names():
    result = _oof_result()
    assert set(result["episode"]) >= {
        "reference_episodes",
        "predicted_event_runs",
        "matched_episodes",
        "episode_precision",
        "episode_sensitivity",
        "episode_f1",
    }
    assert set(result["onset_latency_seconds"]) >= {"median", "iqr", "p90"}
    assert set(result["primary_window"]) >= {
        "f1",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "balanced_accuracy",
        "mcc",
    }
    assert set(result["state_burden"]) == set(SPEC.T1_STAGE_ORDER[:0]) | {
        "NORMAL",
        "WATCH",
        "EVENT",
        "RECOVERY",
    }
    assert len(result["state_flows"]) == 6


def test_no_categorical_state_auprc_is_reported():
    assert _oof_result()["categorical_state_auprc_reported"] is False
    assert SPEC.T1_CATEGORICAL_STATE_AUPRC_REPORTED is False


def test_exposure_includes_unavailable_positions():
    result = _oof_result()
    assert result["physical_exposure_hours"] == (ROWS * 5.0) / 3600.0
    assert SPEC.T1_EXPOSURE_INCLUDES_UNAVAILABLE_POSITIONS is True
    assert SPEC.T1_EXPOSURE_IS_PRIMARY_ONLY is False


def test_the_bootstrap_is_the_frozen_design():
    bootstrap = A.assemble_bootstrap(held_out_traces=_held_out_traces())(
        oof_columns=_columns()
    )
    assert bootstrap["replicates"] == 1000
    assert bootstrap["seed"] == 2026
    assert bootstrap["unit"] == "subject"
    assert bootstrap["policy_reselected_inside_bootstrap"] is False
    assert bootstrap["resampled_with_multiplicity"] is True


def test_the_bootstrap_is_deterministic():
    build = A.assemble_bootstrap(held_out_traces=_held_out_traces())
    assert build(oof_columns=_columns()) == build(oof_columns=_columns())


def test_undefined_replicates_are_preserved_not_zeroed():
    bootstrap = A.assemble_bootstrap(
        held_out_traces=_held_out_traces(
            undefined_subjects=(T1_VALIDATION_SUBJECTS[0],)
        )
    )(oof_columns=_columns())
    assert bootstrap["undefined_replicates"] > 0
    assert bootstrap["defined_replicates"] + bootstrap["undefined_replicates"] == 1000


def test_subject_order_is_the_frozen_roster_not_the_observed_order():
    evidence = A.assemble_subject_evidence(held_out_traces=_held_out_traces())(
        oof_columns=_columns()
    )
    assert evidence["subject_order"] == [
        s for s in T1_VALIDATION_SUBJECTS if s in evidence["subject_order"]
    ]
    assert evidence["inferential_unit"] == "subject"


def test_challenge_is_annotation_never_an_input(tmp_path):
    challenge = A.assemble_challenge(t2_identity=_identity_file(tmp_path))(
        oof_columns=_columns()
    )
    assert challenge["is_selection_input"] is False
    assert challenge["is_transition_input"] is False
    assert challenge["is_threshold_generation_input"] is False
    assert challenge["joined_after_state_trace"] is True
    assert challenge["families"]["CONDUCTION"]["evidence_level"] == (
        "exploratory_descriptive"
    )
    assert challenge["families"]["RATE"]["evidence_level"] == "quantitative_secondary"


def test_the_final_configuration_is_never_development_evidence():
    configuration = A.assemble_final_configuration(
        configuration=dict.fromkeys(A.FINAL_CONFIGURATION_FIELDS, 0.5),
        oof_result_promoted=True,
    )(oof_columns=_columns(), selections=_selections())
    assert configuration["is_development_evidence"] is False
    assert configuration["replaces_oof_result"] is False
    assert configuration["in_sample_on_all_twelve_subjects"] is True


def test_assembly_is_deterministic():
    first = A.assemble_oof_state_columns(columns=_columns(), selections=_selections())
    second = A.assemble_oof_state_columns(columns=_columns(), selections=_selections())
    assert tuple(first) == tuple(second)
    for name in first:
        assert np.array_equal(first[name], second[name])


# ---------------------------------------------------------------------------
# 3-6. Firewall: no labels, no TEST, no artifacts, no authorization change
# ---------------------------------------------------------------------------


def test_no_collaborator_opens_labels_or_reads_a_dataset():
    code = _code_only()
    for reader in (
        "np.load",
        "read_store",
        "read_m2g_row_evidence",
        "read_t2_identity_members",
        "read_t2_selected_scores",
        "read_subject_targets",
        "open(",
        "read_text",
        "read_bytes",
    ):
        assert reader not in code, f"the assembly layer calls {reader}"


def test_no_collaborator_names_a_path():
    code = _code_only()
    assert "Path(" not in code
    assert "pathlib" not in code


def test_no_collaborator_bypasses_the_fold_authority():
    code = _code_only()
    assert "T1CorpusTargetSource" not in code
    assert "FoldScopedEvaluationAuthority" not in code


def test_construction_opens_nothing(tmp_path):
    before = sorted(tmp_path.iterdir())
    A.subject_of_record()
    A.assembly_capability()
    assert sorted(tmp_path.iterdir()) == before


def test_no_collaborator_accesses_test():
    assert SPEC.T1_TEST_ACCESSED is False
    assert SPEC.T1_SEALED_TEST_STATE == "unopened"
    assert A.assembly_capability()["test_accessed"] is False
    with pytest.raises(SPEC.T1ExecutionSpecError):
        A.subject_of_record()("test")
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()


def test_no_collaborator_creates_a_scientific_artifact(tmp_path):
    """Whole-word: `oof_result_promoted` is a flag this layer reads, not a call."""
    import re

    code = _code_only()
    for writer in ("write_text", "write_json", "mkdir", "makedirs", "promote"):
        assert not re.search(rf"\b{writer}\b", code), (
            f"the assembly layer calls {writer}"
        )
    assert not _canonical_root().exists()
    assert sorted(tmp_path.iterdir()) == []


def test_no_collaborator_changes_authorization_state(tmp_path):
    before = CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED
    A.assemble_oof_state_columns(columns=_columns(), selections=_selections())
    A.assemble_challenge(t2_identity=_identity_file(tmp_path))(oof_columns=_columns())
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is before
    code = _code_only()
    assert "T1_EXECUTION_SPECIFICATION_AUTHORIZED" not in code


def test_no_retry_force_reset_or_seed_override_is_introduced():
    code = _code_only()
    import re

    for forbidden in ("retry", "force", "reset", "overwrite", "fresh_seed"):
        assert not re.search(rf"\b{forbidden}\b", code), f"introduces {forbidden}"
    # The bootstrap seed is the frozen constant, never a parameter.
    assert "T1_BOOTSTRAP_SEED" in code
    assert "seed=" not in code


def test_the_canonical_attempt_is_untouched():
    assert not _canonical_root().exists()


# ---------------------------------------------------------------------------
# 7-8. Nothing upstream moved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,digest",
    [
        (
            "t1_protocol.py",
            "b0df6ea2ade450037e94e5ab3b193694fea980337851a2458b3f43873450b192",
        ),
        (
            "t1_execution_spec.py",
            "edb0cbf1afe43dee48b5d2d0ed190e0939530fc026fd2f09d3312b929ab1fbe3",
        ),
        (
            "t1_development_run.py",
            "ad08035d33a1f421cf5a6a18df33e9a7ed55fad29074e7581bbe3ba796b90a8e",
        ),
        (
            "t1_evidence_store.py",
            "464ca1607191aa02042a6dcbb8cfeda4d4f3aced1eae2e29ae4b77be8cf6d39c",
        ),
        (
            "t1_persistence.py",
            "77c0e0a40efa7056777ef8d3bb13983ae4cd1bb9493d3c6c7eb11c7faebd68ad",
        ),
    ],
)
def test_this_pr_modifies_no_frozen_source(name, digest):
    import hashlib

    path = REPOSITORY_ROOT / "src" / "cardiosentinel" / "neural" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_the_upstream_contracts_are_unchanged():
    assert SPEC.T1_REQUIRED_M2_RETAINED_ARM == "M2-G"
    assert SPEC.T1_REQUIRED_U1_FAMILY == "platt_logistic_on_recovered_logit"
    assert SPEC.T1_REQUIRED_T2_RETAINED_ARM == "causal_s4d_longitudinal_v1"
    assert SPEC.T1_U1_REFIT_PERMITTED is False
    assert SPEC.T1_FOLD_COUNT == 12


# ---------------------------------------------------------------------------
# The pre-claim capability gate
# ---------------------------------------------------------------------------


def _bound_collaborators():
    from cardiosentinel.neural import t1_canonical_driver as D
    from cardiosentinel.neural import t1_fold_evaluation as E

    return D.T1ExecutionCollaborators(
        m2_row_evidence=Path("/nonexistent/m2.npz"),
        t2_identity=Path("/nonexistent/t2_outer_row_identity.npz"),
        t2_selected_scores=Path("/nonexistent/s.npz"),
        calibrators={"ltstdb:s2004": object()},
        target_source=_Source(),
        subject_of_record=A.subject_of_record(),
        evaluate_fold=E.T1NonExecutingFoldEvaluator(),
        assemble_oof_state_columns=A.assemble_oof_state_columns,
        assemble_oof_result=A.assemble_oof_result(
            episode_evidence={
                "reference_episodes": 6,
                "predicted_event_runs": 4,
                "matched_episodes": 3,
            },
            onset_latency_seconds=[5.0],
            primary_confusion={
                "true_positive": 3,
                "false_positive": 2,
                "false_negative": 3,
                "true_negative": 16,
            },
        ),
        assemble_subject_evidence=A.assemble_subject_evidence(
            held_out_traces=_held_out_traces()
        ),
        assemble_bootstrap=A.assemble_bootstrap(held_out_traces=_held_out_traces()),
        assemble_challenge=A.assemble_challenge(t2_identity=IDENTITY_PATH),
        assemble_final_configuration=A.assemble_final_configuration(
            configuration=dict.fromkeys(A.FINAL_CONFIGURATION_FIELDS, 0.5),
            oof_result_promoted=True,
        ),
    )


def test_every_assembly_collaborator_satisfies_the_pre_claim_gate():
    """The gate is an allowlist: silence is refused, so each must attest."""
    from cardiosentinel.neural import t1_capability_gate as G

    report = G.capability_report(_bound_collaborators())
    for role, entry in report["roles"].items():
        if role == "evaluate_fold":
            continue
        assert entry["executes"] is True, f"{role} does not attest capability"


def test_the_signatures_match_the_calls_the_driver_actually_makes():
    """The defect the gate exists to catch: bound, callable, wrong shape."""
    import inspect

    from cardiosentinel.neural.t1_capability_gate import CAPABILITY_CALL_CONTRACT

    collaborators = _bound_collaborators()
    for role, (positional, keywords) in CAPABILITY_CALL_CONTRACT.items():
        if role == "evaluate_fold":
            continue
        target = getattr(collaborators, role)
        signature = inspect.signature(target)
        signature.bind(
            *[object()] * positional, **{name: object() for name in keywords}
        )


def test_the_only_role_that_cannot_execute_is_the_fold_evaluator():
    from cardiosentinel.neural import t1_capability_gate as G

    report = G.capability_report(_bound_collaborators())
    unable = sorted(
        role for role, entry in report["roles"].items() if not entry["executes"]
    )
    assert unable == ["evaluate_fold"]
    assert report["execution_graph_complete"] is False


def test_the_incomplete_graph_is_refused_before_the_claim():
    from cardiosentinel.neural import t1_capability_gate as G

    with pytest.raises(G.T1CapabilityError, match="evaluate_fold"):
        G.require_executable_capability(_bound_collaborators())
    assert not _canonical_root().exists()
