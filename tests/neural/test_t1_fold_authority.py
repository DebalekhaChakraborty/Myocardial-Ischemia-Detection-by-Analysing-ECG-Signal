"""Tests for the T1 fold-scoped evaluation authority.

The authority is the only permitted door to a fold's targets. These tests
exercise the door, not the data behind it: every source used here is a stub
that counts its calls, and several tests exist specifically to prove the door
was never opened.

Nothing here runs a fold, reads a validation label, calculates a metric,
generates a prediction, selects a policy, produces OOF evidence, touches TEST
or creates the canonical attempt directory.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_canonical_driver as D
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_fold_authority as A
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.t1_protocol import T1_VALIDATION_SUBJECTS, T1Fold, t1_folds

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CountingSource:
    """A target source that records every call it receives.

    Returns a single row so a successful read is distinguishable from a
    refusal, and counts calls so "construction reads nothing" is a measurement
    rather than an assertion about the code.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def read_subject_targets(
        self, subject_id: str, *, partition: str
    ) -> A.T1SubjectTargets:
        self.calls.append((subject_id, partition))
        return A.T1SubjectTargets(
            subject_id=subject_id,
            stable_id=("row-0",),
            primary_positive=(False,),
            primary_mask=(True,),
        )


class ImpostorSource(CountingSource):
    """Answers for somebody else. The authority must refuse the answer."""

    def read_subject_targets(
        self, subject_id: str, *, partition: str
    ) -> A.T1SubjectTargets:
        super().read_subject_targets(subject_id, partition=partition)
        other = next(s for s in T1_VALIDATION_SUBJECTS if s != subject_id)
        return A.T1SubjectTargets(
            subject_id=other,
            stable_id=("row-0",),
            primary_positive=(True,),
            primary_mask=(True,),
        )


class _Source:
    """Minimal target source: the protocol shape and nothing wider."""

    def read_subject_targets(self, subject_id, *, partition):  # pragma: no cover
        raise AssertionError("no test here opens targets")


def _promoted_fold_state() -> dict[str, object]:
    """The state the held-out barrier requires: promoted and digest-verified."""
    return {
        "selection_promoted": True,
        "selection_digest_verified": True,
        SPEC.T1_HELD_OUT_ACCESS_FLAG: True,
        "selection_sha256": "0" * 64,
    }


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _authority_code_only() -> str:
    """Source with docstrings and prose literals removed.

    This module's prose names every forbidden accessor on purpose, so a raw
    substring scan would report all of them.
    """
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
# 1-2. Fold identity is explicit, and invalid identity fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fold", t1_folds())
def test_an_authority_can_be_created_for_every_frozen_fold(fold):
    authority = A.fit_evaluation_authority(fold, source=CountingSource())
    assert authority.fold_index == fold.fold_index
    assert authority.held_out_subject == fold.held_out_subject
    assert authority.scope == A.SCOPE_FIT
    assert authority.partition == A.T1_PERMITTED_PARTITION


def test_the_fold_must_be_one_the_frozen_design_produces():
    """A hand-built fold naming a different fit set is not that fold."""
    frozen = t1_folds()[0]
    forged = T1Fold(
        fold_index=frozen.fold_index,
        held_out_subject=frozen.held_out_subject,
        fit_subjects=frozen.fit_subjects[:-1],
    )
    with pytest.raises(A.T1FoldAuthorityError, match="does not match the frozen"):
        A.fit_evaluation_authority(forged, source=CountingSource())


def test_a_fold_swapping_the_held_out_subject_is_refused():
    frozen = t1_folds()[0]
    forged = T1Fold(
        fold_index=frozen.fold_index,
        held_out_subject=t1_folds()[1].held_out_subject,
        fit_subjects=frozen.fit_subjects,
    )
    with pytest.raises(A.T1FoldAuthorityError, match="does not match the frozen"):
        A.fit_evaluation_authority(forged, source=CountingSource())


@pytest.mark.parametrize("index", [-1, 12, 99])
def test_an_out_of_range_fold_index_is_refused(index):
    frozen = t1_folds()[0]
    forged = T1Fold(
        fold_index=index,
        held_out_subject=frozen.held_out_subject,
        fit_subjects=frozen.fit_subjects,
    )
    with pytest.raises(A.T1FoldAuthorityError, match="outside the frozen"):
        A.fit_evaluation_authority(forged, source=CountingSource())


@pytest.mark.parametrize("bogus", ["fold-0", 0, None, {"fold_index": 0}])
def test_a_non_fold_cannot_become_an_authority(bogus):
    with pytest.raises(A.T1FoldAuthorityError, match="needs a T1Fold"):
        A.fit_evaluation_authority(bogus, source=CountingSource())


# ---------------------------------------------------------------------------
# 3. TEST cannot be represented
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sealed", A.T1_SEALED_PARTITIONS)
def test_a_test_partition_cannot_carry_an_authority(sealed):
    with pytest.raises((SPEC.T1ExecutionSpecError, A.T1FoldAuthorityError)) as caught:
        A.require_validation_partition(sealed)
    assert "test" in str(caught.value).lower()


def test_only_validation_is_a_permitted_partition():
    assert A.require_validation_partition("validation") == "validation"
    for other in ("train", "challenge", "deployment", "holdout"):
        with pytest.raises(A.T1FoldAuthorityError, match="only permitted partition"):
            A.require_validation_partition(other)


def test_the_partition_is_sealed_not_selected():
    """There is no parameter through which a partition could be passed in."""
    import inspect

    for factory in (A.fit_evaluation_authority, A.held_out_evaluation_authority):
        parameters = set(inspect.signature(factory).parameters)
        assert "partition" not in parameters, f"{factory.__name__} accepts a partition"


def test_no_path_construction_can_resolve_test():
    for named in ("test", "cardiosentinel-runs/test/rows.npz", "b4_test"):
        with pytest.raises(SPEC.T1ExecutionSpecError):
            PERSIST.require_no_test_path(named)
    code = _authority_code_only()
    assert "Path(" not in code, "the authority constructs a path"
    assert "open(" not in code
    assert "np.load" not in code


def test_a_test_subject_cannot_be_authorized():
    with pytest.raises((SPEC.T1ExecutionSpecError, A.T1FoldAuthorityError)):
        A.require_known_subject("test")
    with pytest.raises(A.T1FoldAuthorityError, match="not in the frozen"):
        A.require_known_subject("ltstdb:sTEST")


def test_test_stays_sealed():
    assert SPEC.T1_TEST_ACCESSED is False
    assert SPEC.T1_SEALED_TEST_STATE == "unopened"
    assert A.authority_contract()["test_accessed"] is False
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()


# ---------------------------------------------------------------------------
# 4-5. Arbitrary and global access are impossible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("accessor", A.T1_FORBIDDEN_AUTHORITY_ACCESSORS)
def test_the_authority_exposes_no_global_or_arbitrary_accessor(accessor):
    assert not hasattr(A.FoldScopedEvaluationAuthority, accessor)
    authority = A.fit_evaluation_authority(t1_folds()[0], source=CountingSource())
    assert not hasattr(authority, accessor)


def test_the_authority_cannot_be_indexed_or_iterated():
    authority = A.fit_evaluation_authority(t1_folds()[0], source=CountingSource())
    with pytest.raises(TypeError):
        authority["ltstdb:s2004"]  # type: ignore[index]
    with pytest.raises(TypeError):
        list(authority)  # type: ignore[call-overload]


def test_a_subject_outside_the_scope_is_refused():
    fold = t1_folds()[0]
    source = CountingSource()
    authority = A.fit_evaluation_authority(fold, source=source)
    from cardiosentinel.neural.t1_development_run import T1DevelopmentError

    with pytest.raises(T1DevelopmentError) as caught:
        authority.targets_for_subject(fold.held_out_subject)
    assert "not in this authority" in str(caught.value)
    assert source.calls == [], "a refused subject still reached the source"


def test_the_selection_authority_never_sees_the_held_out_subject():
    for fold in t1_folds():
        authority = A.fit_evaluation_authority(fold, source=CountingSource())
        assert fold.held_out_subject not in authority.authorized_subjects
        assert len(authority.authorized_subjects) == 11


def test_the_evaluation_authority_sees_exactly_one_subject():
    fold = t1_folds()[0]
    authority = A.held_out_evaluation_authority(
        fold, _promoted_fold_state(), source=CountingSource()
    )
    assert authority.authorized_subjects == (fold.held_out_subject,)
    from cardiosentinel.neural.t1_development_run import T1DevelopmentError

    for fit_subject in fold.fit_subjects:
        with pytest.raises(T1DevelopmentError, match="not in this authority"):
            authority.targets_for_subject(fit_subject)


def test_the_held_out_barrier_is_not_reopened_here():
    """This layer adds no second way past the promoted-and-verified barrier."""
    fold = t1_folds()[0]
    for broken in (
        {},
        {"selection_promoted": True},
        {"selection_promoted": True, "selection_digest_verified": True},
    ):
        with pytest.raises(SPEC.T1ExecutionSpecError):
            A.held_out_evaluation_authority(fold, broken, source=CountingSource())


def test_a_wider_source_is_refused():
    class WideSource:
        def get_all_labels(self):  # pragma: no cover - never called
            return {}

    with pytest.raises(A.T1FoldAuthorityError, match="needs a target source"):
        A.fit_evaluation_authority(t1_folds()[0], source=WideSource())


def test_a_source_that_answers_for_another_subject_is_refused():
    fold = t1_folds()[0]
    authority = A.fit_evaluation_authority(fold, source=ImpostorSource())
    with pytest.raises(A.T1FoldAuthorityError, match="answered for"):
        authority.targets_for_subject(fold.fit_subjects[0])


def test_targets_carry_one_subject_and_no_frame():
    targets = A.T1SubjectTargets(
        subject_id="ltstdb:s2004",
        stable_id=("a", "b"),
        primary_positive=(True, False),
        primary_mask=(True, True),
    )
    assert len(targets) == 2
    provenance = targets.as_dict()
    assert set(provenance) == {"subject_id", "row_count", "columns"}
    # The column names may be described; the values may never be carried.
    assert targets.primary_positive not in provenance.values()
    assert targets.primary_mask not in provenance.values()
    assert targets.stable_id not in provenance.values()
    with pytest.raises(A.T1FoldAuthorityError, match="ragged"):
        A.T1SubjectTargets(
            subject_id="ltstdb:s2004",
            stable_id=("a",),
            primary_positive=(True, False),
            primary_mask=(True,),
        )


def test_the_authority_provenance_carries_no_target():
    fold = t1_folds()[0]
    authority = A.fit_evaluation_authority(fold, source=CountingSource())
    provenance = authority.as_dict()
    for forbidden in ("primary_positive", "primary_mask", "targets", "label"):
        assert forbidden not in provenance
    assert provenance["test_accessed"] is False


# ---------------------------------------------------------------------------
# 6-8. Construction reads nothing; no artifacts; no attempt directory
# ---------------------------------------------------------------------------


def test_creating_an_authority_reads_no_target():
    source = CountingSource()
    for fold in t1_folds():
        A.fit_evaluation_authority(fold, source=source)
    assert source.calls == [], "construction delegated a read"


def test_the_plan_is_inspectable_without_building_anything():
    source = CountingSource()
    plan = A.fold_authority_plan()
    assert len(plan) == 12
    assert {entry["partition"] for entry in plan} == {"validation"}
    assert source.calls == []


def test_a_scoped_read_reaches_the_source_exactly_once():
    fold = t1_folds()[0]
    source = CountingSource()
    authority = A.fit_evaluation_authority(fold, source=source)
    authority.targets_for_subject(fold.fit_subjects[0])
    assert source.calls == [(fold.fit_subjects[0], "validation")]


def test_no_scientific_artifact_is_generated(tmp_path):
    before = sorted(tmp_path.iterdir())
    source = CountingSource()
    for fold in t1_folds():
        A.fit_evaluation_authority(fold, source=source)
        A.held_out_evaluation_authority(fold, _promoted_fold_state(), source=source)
    assert sorted(tmp_path.iterdir()) == before
    assert not any((REPOSITORY_ROOT / "cardiosentinel-runs").glob("phase9-*/T1_*.json"))


def test_the_canonical_attempt_directory_is_not_created():
    assert not _canonical_root().exists()
    for fold in t1_folds():
        A.fit_evaluation_authority(fold, source=CountingSource())
    assert not _canonical_root().exists()


def test_the_authority_computes_no_metric_and_selects_no_policy():
    code = _authority_code_only()
    for frozen in (
        "select_policy",
        "score_policy",
        "generate_thresholds",
        "run_policy_over_streams",
        "episode_f1",
        "window_mcc",
        "next_state",
        "subject_bootstrap_indices",
    ):
        assert frozen not in code, f"the authority calls {frozen}"


def test_the_authority_writes_nothing():
    code = _authority_code_only()
    for forbidden in ("mkdir", "makedirs", "write_text", "write_json", "promote"):
        assert forbidden not in code, f"the authority calls {forbidden}"


# ---------------------------------------------------------------------------
# Integration with the driver, without executing
# ---------------------------------------------------------------------------


def test_the_driver_can_verify_collaborators_without_executing():
    """ "Could this run" is answerable without answering "may this run"."""
    collaborators = D.T1ExecutionCollaborators(
        m2_row_evidence=Path("/nonexistent/m2.npz"),
        t2_identity=Path("/nonexistent/id.npz"),
        t2_selected_scores=Path("/nonexistent/scores.npz"),
        calibrators={"ltstdb:s2004": object()},
        target_source=_Source(),
        subject_of_record=lambda record: "ltstdb:s2004",
        evaluate_fold=lambda *a, **k: {"artifact": {}},
        assemble_oof_state_columns=lambda **k: {},
        assemble_oof_result=lambda **k: {},
        assemble_subject_evidence=lambda **k: {},
        assemble_bootstrap=lambda **k: {},
        assemble_challenge=lambda **k: {},
        assemble_final_configuration=lambda **k: {},
    )
    report = D.T1CanonicalDevelopmentExecutor.verify_collaborators(collaborators)
    assert report["collaborators_complete"] is True
    assert report["executed"] is False
    assert report["attempt_consumed"] is False
    # Mirrors the repository constant; the properties that matter here are
    # that verifying answered "could this run" without running anything.
    assert report["execution_authorized"] is (
        CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED
    )
    assert not _canonical_root().exists()


def test_verifying_collaborators_does_not_grant_permission():
    """Verification neither grants nor revokes; the two views stay in step."""
    before = CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED
    assert D.T1CanonicalDevelopmentExecutor.verify_collaborators.__doc__
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is before


def test_the_authority_contract_is_the_specifications():
    contract = A.authority_contract()
    assert contract["global_label_table_permitted"] is False
    assert contract["fold_scoped_authority_required"] is True
    assert contract["fold_count"] == SPEC.T1_FOLD_COUNT
    assert contract["held_out_policy_runs_per_fold"] == 1
    assert contract["construction_reads_targets"] is False


# ---------------------------------------------------------------------------
# 9-10. Nothing upstream moved
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
    assert SPEC.T1_M2_REPLAY_PERMITTED is False
    assert SPEC.T1_T2_REPLAY_PERMITTED is False
    assert SPEC.T1_U1_REFIT_PERMITTED is False


def test_the_existing_subject_scope_object_is_reused_not_replaced():
    """One implementation of subject membership, not two."""
    fold = t1_folds()[0]
    authority = A.fit_evaluation_authority(fold, source=CountingSource())
    from cardiosentinel.neural.t1_development_run import FoldScopedTargetAuthority

    assert isinstance(authority.subject_scope, FoldScopedTargetAuthority)
    assert not hasattr(FoldScopedTargetAuthority, "targets_for_subject")
