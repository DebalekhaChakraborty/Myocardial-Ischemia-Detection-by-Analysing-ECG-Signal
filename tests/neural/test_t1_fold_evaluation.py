"""Tests for the controlled T1 fold evaluation capability.

The capability is complete and disabled. These tests exercise the guards and
the contract; none of them runs a fold, reads a validation label, generates a
prediction, computes a metric, produces OOF evidence, selects a policy,
reaches TEST or creates the canonical attempt directory.

The label-bearing reader is written and never called: every refusal below
fires before an array is opened, and one test measures that by pointing a
source at a path that does not exist and proving the refusal arrives anyway.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_canonical_driver as D
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_fold_authority as A
from cardiosentinel.neural import t1_fold_evaluation as E
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.t1_protocol import t1_folds

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IDENTITY = Path("/nonexistent/t2_outer_row_identity.npz")


class _Source:
    """Minimal target source: the protocol shape and nothing wider."""

    def read_subject_targets(self, subject_id, *, partition):  # pragma: no cover
        raise AssertionError("no test here opens targets")


def _source() -> E.T1CorpusTargetSource:
    return E.T1CorpusTargetSource(IDENTITY)


def _fold_state() -> dict[str, object]:
    return {
        "selection_promoted": True,
        "selection_digest_verified": True,
        SPEC.T1_HELD_OUT_ACCESS_FLAG: True,
        "selection_sha256": "0" * 64,
    }


def _request(index: int = 0) -> E.T1FoldEvaluationRequest:
    fold = t1_folds()[index]
    source = _source()
    return E.build_fold_request(
        A.fit_evaluation_authority(fold, source=source),
        A.held_out_evaluation_authority(fold, _fold_state(), source=source),
    )


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _code_only(module) -> str:
    """Source with docstrings and prose literals stripped."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
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
# T1TargetSource: reachable only through the authority
# ---------------------------------------------------------------------------


def test_a_source_called_directly_is_refused():
    """The security property. Without it, the authority is decorative."""
    with pytest.raises(A.T1FoldAuthorityError, match="No fold authority is sponsoring"):
        _source().read_subject_targets("ltstdb:s2004", partition="validation")


def test_the_refusal_arrives_before_the_archive_is_opened():
    """Measured: the path does not exist, and the error is never about that."""
    assert not IDENTITY.exists()
    with pytest.raises(A.T1FoldAuthorityError) as caught:
        _source().read_subject_targets("ltstdb:s2004", partition="validation")
    message = str(caught.value)
    assert "No such file" not in message
    assert "FileNotFoundError" not in message


def test_a_stale_sponsorship_cannot_be_reused():
    """The ticket is live for one delegated read and is reset afterwards.

    Uses a source that records the ticket it saw and then raises, so the reset
    is proved without opening anything: a test that reached the archive would
    be measuring the filesystem instead of the boundary.
    """

    class Sentinel(RuntimeError):
        pass

    seen: list[A.ScopedTargetRequest | None] = []

    class RecordingSource:
        def read_subject_targets(self, subject_id: str, *, partition: str):
            seen.append(A.active_scoped_request())
            raise Sentinel("stopped before any read")

    fold = t1_folds()[0]
    authority = A.fit_evaluation_authority(fold, source=RecordingSource())
    assert A.active_scoped_request() is None
    with pytest.raises(Sentinel):
        authority.targets_for_subject(fold.fit_subjects[0])
    assert len(seen) == 1
    assert seen[0] is not None
    assert seen[0].subject_id == fold.fit_subjects[0]
    assert seen[0].fold_index == fold.fold_index
    assert A.active_scoped_request() is None, "the authorization outlived the read"


def test_a_sponsorship_for_another_subject_is_refused():
    request = A.ScopedTargetRequest(
        fold_index=0,
        subject_id="ltstdb:s2004",
        scope=A.SCOPE_FIT,
        partition="validation",
    )
    with A._authorized_request(request):
        A.require_active_scoped_request("ltstdb:s2004", "validation")
        with pytest.raises(A.T1FoldAuthorityError, match="authorized"):
            A.require_active_scoped_request("ltstdb:s2005", "validation")


def test_unauthorized_subject_access_fails():
    fold = t1_folds()[0]
    source = _source()
    authority = A.fit_evaluation_authority(fold, source=source)
    from cardiosentinel.neural.t1_development_run import T1DevelopmentError

    with pytest.raises(T1DevelopmentError, match="not in this authority"):
        E.authorized_targets(authority, fold.held_out_subject)


def test_invalid_fold_identity_fails():
    from cardiosentinel.neural.t1_protocol import T1Fold

    frozen = t1_folds()[0]
    forged = T1Fold(
        fold_index=frozen.fold_index,
        held_out_subject=frozen.held_out_subject,
        fit_subjects=frozen.fit_subjects[:-1],
    )
    with pytest.raises(A.T1FoldAuthorityError, match="does not match the frozen"):
        A.fit_evaluation_authority(forged, source=_source())


@pytest.mark.parametrize("sealed", A.T1_SEALED_PARTITIONS)
def test_test_cannot_be_represented(sealed):
    with pytest.raises((SPEC.T1ExecutionSpecError, A.T1FoldAuthorityError)):
        _source().read_subject_targets("ltstdb:s2004", partition=sealed)


def test_a_test_identity_path_cannot_build_a_source():
    for bad in (
        Path("cardiosentinel-runs/test/t2_outer_row_identity.npz"),
        Path("/data/test.npz"),
    ):
        with pytest.raises((SPEC.T1ExecutionSpecError, E.T1FoldEvaluationError)):
            E.T1CorpusTargetSource(bad)


def test_the_source_reads_only_the_row_identity():
    with pytest.raises(E.T1FoldEvaluationError, match="row identity"):
        E.T1CorpusTargetSource(Path("/data/some_other_archive.npz"))


@pytest.mark.parametrize("member", E.T1_TARGET_MEMBERS_REFUSED)
def test_evaluation_annotation_members_are_refused(member):
    with pytest.raises(E.T1FoldEvaluationError, match="refused"):
        E.T1CorpusTargetSource(IDENTITY, _members=(member,))
    with pytest.raises(E.T1FoldEvaluationError, match="refused"):
        E.target_member_plan((member,))


def test_global_target_access_api_does_not_exist():
    for forbidden in (
        "get_all_labels",
        "get_validation_labels",
        "get_test_labels",
        "all_targets",
        "labels",
        "dataframe",
        "as_frame",
        "to_pandas",
        "__getitem__",
        "__iter__",
    ):
        assert not hasattr(E.T1CorpusTargetSource, forbidden), forbidden
        assert not hasattr(_source(), forbidden), forbidden


def test_arbitrary_subject_lookup_is_impossible():
    with pytest.raises(A.T1FoldAuthorityError, match="not in the frozen"):
        _source().read_subject_targets("ltstdb:sZZZZ", partition="validation")


def test_construction_does_not_consume_labels(tmp_path):
    before = sorted(tmp_path.iterdir())
    for _ in range(5):
        E.T1CorpusTargetSource(IDENTITY)
    assert sorted(tmp_path.iterdir()) == before
    assert E.evaluation_capability()["construction_reads_targets"] is False


def test_the_source_provenance_carries_no_target():
    provenance = _source().as_dict()
    assert provenance["reachable_without_an_authority"] is False
    assert provenance["test_accessed"] is False
    # Member names may be described; no target value may be carried.
    assert set(provenance) == {
        "source",
        "identity_file",
        "members",
        "refused_members",
        "reachable_without_an_authority",
        "partition",
        "test_accessed",
    }
    assert provenance["members"] == list(E.T1_TARGET_MEMBERS)
    assert "label" in provenance["members"], "the reader is deliberately label-bearing"
    assert all(isinstance(value, str) for value in provenance["members"])


def test_the_label_bearing_read_is_the_only_place_an_array_opens():
    code = _code_only(E)
    assert code.count("np.load") == 1
    tree = ast.parse(Path(E.__file__).read_text(encoding="utf-8"))
    loaders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Attribute) and inner.attr == "load"
            for inner in ast.walk(node)
        )
    ]
    assert [fn.name for fn in loaders] == ["_read_one_subject"]


def test_the_label_blind_reader_was_not_widened():
    """Reaching labels required a new door, never a relaxed old one."""
    from cardiosentinel.neural import t1_evidence_store as STORE

    for member in ("label", "primary_mask", "target_family"):
        with pytest.raises(Exception):
            STORE.read_t2_identity_members(IDENTITY, members=(member,))


# ---------------------------------------------------------------------------
# T1FoldEvaluator: the contract
# ---------------------------------------------------------------------------


def test_the_evaluator_accepts_only_authorized_fold_inputs():
    evaluator = E.T1NonExecutingFoldEvaluator()
    for bogus in (0, "fold-0", None, {"fold_index": 0}):
        with pytest.raises(E.T1FoldEvaluationError, match="authorized"):
            evaluator.evaluate(bogus)


def test_a_request_pairs_two_authorities_of_the_same_fold():
    request = _request(0)
    assert request.fold_index == 0
    assert request.fit.scope == A.SCOPE_FIT
    assert request.held_out.scope == A.SCOPE_HELD_OUT
    assert request.held_out.authorized_subjects == (request.fit.held_out_subject,)


def test_mismatched_folds_cannot_form_a_request():
    source = _source()
    fit = A.fit_evaluation_authority(t1_folds()[0], source=source)
    held = A.held_out_evaluation_authority(t1_folds()[1], _fold_state(), source=source)
    with pytest.raises(E.T1FoldEvaluationError, match="different folds"):
        E.build_fold_request(fit, held)


def test_swapped_scopes_cannot_form_a_request():
    source = _source()
    fold = t1_folds()[0]
    fit = A.fit_evaluation_authority(fold, source=source)
    held = A.held_out_evaluation_authority(fold, _fold_state(), source=source)
    with pytest.raises(E.T1FoldEvaluationError, match="selection authority"):
        E.build_fold_request(held, fit)


def test_a_request_cannot_be_built_from_anything_else():
    with pytest.raises(E.T1FoldEvaluationError, match="two FoldScoped"):
        E.build_fold_request("fit", "held_out")


def test_the_evaluator_contract_exposes_one_method_taking_a_request():
    import inspect

    parameters = list(
        inspect.signature(E.T1NonExecutingFoldEvaluator.evaluate).parameters
    )
    assert parameters == ["self", "request"]
    for forbidden in ("path", "frame", "dataset", "labels", "partition", "subjects"):
        assert forbidden not in parameters


def test_the_evaluator_refuses_with_the_capability_message():
    with pytest.raises(E.T1FoldEvaluationError) as caught:
        E.T1NonExecutingFoldEvaluator().evaluate(_request())
    message = str(caught.value)
    assert E.EVALUATION_DISABLED_MESSAGE in message
    assert "no attempt has been consumed" in message


def test_the_evaluator_returns_nothing_rather_than_a_plausible_empty_result():
    """A stub returning {} is indistinguishable from a run that found nothing."""
    evaluator = E.T1NonExecutingFoldEvaluator()
    with pytest.raises(E.T1FoldEvaluationError):
        evaluator.evaluate(_request())


def test_the_evaluator_carries_no_independent_access():
    evaluator = E.require_no_independent_access(E.T1NonExecutingFoldEvaluator())
    assert evaluator.as_dict()["reads_datasets_independently"] is False
    assert evaluator.as_dict()["reads_labels_independently"] is False


def test_an_evaluator_holding_its_own_path_is_refused():
    class Leaky:
        def __init__(self):
            self.corpus = Path("/data/corpus")

        def evaluate(self, request):  # pragma: no cover - never called
            return {}

    with pytest.raises(E.T1FoldEvaluationError, match="is a Path"):
        E.require_no_independent_access(Leaky())


def test_an_evaluator_holding_a_source_directly_is_refused():
    class Bypassing:
        def __init__(self):
            self.source = E.T1CorpusTargetSource(IDENTITY)

        def evaluate(self, request):  # pragma: no cover - never called
            return {}

    with pytest.raises(E.T1FoldEvaluationError, match="bypassing"):
        E.require_no_independent_access(Bypassing())


def test_the_evaluator_contains_no_model_selection_logic():
    code = _code_only(E)
    for frozen in (
        "select_policy",
        "score_policy",
        "generate_thresholds",
        "run_policy_over_streams",
        "episode_f1",
        "window_mcc",
        "next_state",
        "policy_sort_key",
        "subject_bootstrap_indices",
    ):
        assert frozen not in code, f"the evaluation layer calls {frozen}"


def test_the_held_out_barrier_is_preserved():
    source = _source()
    fold = t1_folds()[0]
    for broken in ({}, {"selection_promoted": True}):
        with pytest.raises(SPEC.T1ExecutionSpecError):
            A.held_out_evaluation_authority(fold, broken, source=source)


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------


def _collaborators() -> D.T1ExecutionCollaborators:
    return D.T1ExecutionCollaborators(
        m2_row_evidence=Path("/nonexistent/row_evidence.npz"),
        t2_identity=IDENTITY,
        t2_selected_scores=Path("/nonexistent/scores.npz"),
        calibrators={"ltstdb:s2004": object()},
        target_source=_Source(),
        subject_of_record=lambda record: "ltstdb:s2004",
        evaluate_fold=E.T1NonExecutingFoldEvaluator(),
        assemble_oof_state_columns=lambda **k: {},
        assemble_oof_result=lambda **k: {},
        assemble_subject_evidence=lambda **k: {},
        assemble_bootstrap=lambda **k: {},
        assemble_challenge=lambda **k: {},
        assemble_final_configuration=lambda **k: {},
    )


def test_the_complete_collaborator_graph_exists():
    report = D.T1CanonicalDevelopmentExecutor.verify_collaborators(_collaborators())
    assert report["collaborators_complete"] is True
    assert all(report["capabilities_present"].values()), report["capabilities_present"]


def test_verification_runs_nothing_and_consumes_nothing():
    """Verification reports permission; it does not act on it.

    `execution_authorized` now reports True because it mirrors the repository
    constant. The properties that matter are the other three: verifying a
    graph must not enable, execute or consume anything.
    """
    report = D.T1CanonicalDevelopmentExecutor.verify_collaborators(_collaborators())
    assert report["execution_authorized"] is (
        CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED
    )
    assert report["execution_enabled"] is False
    assert report["executed"] is False
    assert report["attempt_consumed"] is False


def test_verification_opens_no_labels_and_runs_no_fold():
    report = D.T1CanonicalDevelopmentExecutor.verify_collaborators(_collaborators())
    assert report["labels_opened"] is False
    assert report["folds_run"] is False


def test_a_complete_graph_refuses_when_permission_is_withdrawn(monkeypatch):
    """Capability is complete; withdrawing permission still stops it dead.

    Permission is withdrawn on both modules -- the driver holds its own
    imported reference -- so the refusal observed here is the permission one
    rather than a later check firing for an unrelated reason.
    """
    monkeypatch.setattr(CFG, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    executor = D.T1CanonicalDevelopmentExecutor(
        run=D.T1DevelopmentRun(authorized_git_sha="0" * 40)
    )
    with pytest.raises(D.T1DriverError, match="human authorization is not granted"):
        executor.execute(_collaborators())
    assert executor.run.stages.entered == []
    assert not _canonical_root().exists()


# ---------------------------------------------------------------------------
# Nothing was produced, nothing moved
# ---------------------------------------------------------------------------


def test_no_run_directory_is_created():
    assert not _canonical_root().exists()
    for index in range(len(t1_folds())):
        _request(index)
    assert not _canonical_root().exists()


def test_no_metrics_are_generated():
    code = _code_only(E)
    for metric in ("auprc", "f1", "mcc", "sensitivity", "specificity", "bootstrap"):
        assert metric not in code.lower(), f"the evaluation layer computes {metric}"


def test_no_scientific_artifact_is_written():
    code = _code_only(E)
    for forbidden in ("write_text", "write_json", "mkdir", "makedirs", "promote"):
        assert forbidden not in code, f"the evaluation layer calls {forbidden}"
    assert not any((REPOSITORY_ROOT / "cardiosentinel-runs").glob("phase9-*/T1_*.json"))


def test_test_stays_sealed():
    assert SPEC.T1_TEST_ACCESSED is False
    assert SPEC.T1_SEALED_TEST_STATE == "unopened"
    assert E.evaluation_capability()["test_accessed"] is False
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()


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


def test_the_experiment_configuration_is_untouched():
    import hashlib

    config = REPOSITORY_ROOT / "configs" / "t1_episode.yaml"
    assert (
        hashlib.sha256(config.read_bytes()).hexdigest()
        == "d5ec66fad71c77edc26cf30329b27459eba770f1f0b66b42dd6dfb1006284e60"
    )


def test_the_upstream_contracts_are_unchanged():
    assert SPEC.T1_REQUIRED_M2_RETAINED_ARM == "M2-G"
    assert SPEC.T1_REQUIRED_U1_FAMILY == "platt_logistic_on_recovered_logit"
    assert SPEC.T1_REQUIRED_T2_RETAINED_ARM == "causal_s4d_longitudinal_v1"
    assert SPEC.T1_U1_REFIT_PERMITTED is False
    assert SPEC.T1_CANDIDATE_POLICIES_PER_FOLD == 12
    assert SPEC.T1_HELD_OUT_POLICY_RUNS_PER_FOLD == 1
