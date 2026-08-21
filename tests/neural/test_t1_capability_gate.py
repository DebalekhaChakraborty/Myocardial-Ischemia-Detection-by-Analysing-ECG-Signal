"""Tests for the pre-claim capability gate.

The defect these tests close: `require_complete` proved every collaborator was
*bound*, and ``callable`` is true of an object whose entire body is a refusal.
A graph containing `T1NonExecutingFoldEvaluator` therefore passed the binding
check, claimed the single canonical attempt at stage 10, and refused at stage
17 -- consuming the attempt to discover something knowable before stage 1.

Every test below runs with authorization closed or monkeypatched open in
isolation. None of them runs a fold, reads a validation label, computes a
metric, reaches TEST, or creates the canonical attempt directory; several
assert that last point directly.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_canonical_driver as D
from cardiosentinel.neural import t1_capability_gate as G
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_development_run as R
from cardiosentinel.neural import t1_evidence_store as STORE
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_fold_evaluation as E
from cardiosentinel.neural import t1_persistence as PERSIST

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _code_only(module) -> str:
    """Source with docstrings, comments and prose literals stripped."""
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


def _execute_body() -> list[ast.stmt]:
    """The statements of `execute`, docstring dropped."""
    tree = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
    execute = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    body = list(execute.body)
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return body


# ---------------------------------------------------------------------------
# A graph whose every role is attested, so one role can be broken in isolation
# ---------------------------------------------------------------------------


def _attested(role: str, function):
    """One attested collaborator per role.

    A fresh wrapper per role on purpose: `declare_execution_capability` sets an
    attribute on the object it is given, so attesting one shared function for
    three roles would leave it carrying only the last declaration.
    """

    @functools.wraps(function)
    def bound(*args, **kwargs):
        return function(*args, **kwargs)

    return G.declare_execution_capability(
        role, executes=True, reason="Test double standing in for a frozen step."
    )(bound)


def subject_of_record(record):
    return "ltstdb:s2004"


def evaluate_fold(fold, authority):
    return {"artifact": {}}


def assemble_two(*, oof_columns=None, columns=None, selections=None):
    return {}


def assemble_one(*, oof_columns=None):
    return {}


def _complete(**overrides) -> D.T1ExecutionCollaborators:
    """Every collaborator bound, attested, and shaped for the driver's call."""
    fields = {
        "m2_row_evidence": Path("/nonexistent/m2_row_evidence.npz"),
        "t2_identity": Path("/nonexistent/t2_outer_row_identity.npz"),
        "t2_selected_scores": Path("/nonexistent/t2_selected_scores.npz"),
        "calibrators": {"ltstdb:s2004": object()},
        "subject_of_record": _attested("subject_of_record", subject_of_record),
        "evaluate_fold": _attested("evaluate_fold", evaluate_fold),
        "assemble_oof_state_columns": _attested(
            "assemble_oof_state_columns", assemble_two
        ),
        "assemble_oof_result": _attested("assemble_oof_result", assemble_two),
        "assemble_subject_evidence": _attested(
            "assemble_subject_evidence", assemble_one
        ),
        "assemble_bootstrap": _attested("assemble_bootstrap", assemble_one),
        "assemble_challenge": _attested("assemble_challenge", assemble_one),
        "assemble_final_configuration": _attested(
            "assemble_final_configuration", assemble_two
        ),
    }
    fields.update(overrides)
    return D.T1ExecutionCollaborators(**fields)


def _executor() -> D.T1CanonicalDevelopmentExecutor:
    return D.T1CanonicalDevelopmentExecutor(
        run=R.T1DevelopmentRun(authorized_git_sha="0" * 40)
    )


@pytest.fixture
def authorized(monkeypatch):
    """Permission open, so the capability gate is the thing under test.

    Patched on both modules and only for the duration of one test. The frozen
    constant itself is never written; a test at the bottom of this file proves
    it is still False afterwards.
    """
    monkeypatch.setattr(CFG, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", True)
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", True)
    return True


# ---------------------------------------------------------------------------
# 1. A callable refusal-only evaluator is rejected
# ---------------------------------------------------------------------------


def test_the_non_executing_evaluator_is_callable():
    """The premise of the defect: the old check would have passed it."""
    evaluator = E.T1NonExecutingFoldEvaluator()
    assert callable(evaluator)
    _complete(evaluate_fold=evaluator).require_complete()


def test_a_callable_refusal_evaluator_is_rejected_by_the_gate():
    evaluator = E.T1NonExecutingFoldEvaluator()
    with pytest.raises(G.T1CapabilityError) as caught:
        G.require_executable_capability(_complete(evaluate_fold=evaluator))
    message = str(caught.value)
    assert "evaluate_fold" in message
    assert "T1NonExecutingFoldEvaluator" in message
    assert "cannot complete" in message
    assert "nothing was consumed" in message.lower()


def test_the_evaluator_declares_its_own_inability():
    attestation = E.T1NonExecutingFoldEvaluator().t1_execution_capability()
    assert attestation.role == "evaluate_fold"
    assert attestation.executes is False
    assert attestation.reason


def test_a_refusal_names_the_stage_that_would_have_failed():
    with pytest.raises(G.T1CapabilityError) as caught:
        G.require_executable_capability(
            _complete(evaluate_fold=E.T1NonExecutingFoldEvaluator())
        )
    message = str(caught.value)
    assert SPEC.STAGE_FOLD_RUN_CANDIDATES in message
    assert SPEC.STAGE_CLAIM in message


def test_an_undeclared_collaborator_is_refused_rather_than_assumed():
    """Silence is a refusal. The allowlist is the point."""
    with pytest.raises(G.T1CapabilityError, match="declares no execution capability"):
        G.require_executable_capability(
            _complete(evaluate_fold=lambda fold, authority: {"artifact": {}})
        )


def test_an_attestation_cannot_be_moved_to_another_role():
    borrowed = _attested("assemble_bootstrap", assemble_one)
    with pytest.raises(G.T1CapabilityError, match="carries an attestation for"):
        G.require_completable("assemble_challenge", borrowed)


# ---------------------------------------------------------------------------
# 2. A body that cannot return is refused even when it claims otherwise
# ---------------------------------------------------------------------------


@G.declare_execution_capability(
    "assemble_bootstrap", executes=True, reason="Claims to execute and does not."
)
def _lying_collaborator(*, oof_columns=None):
    raise RuntimeError("this body can only raise")


def test_an_attestation_does_not_override_a_body_that_cannot_return():
    with pytest.raises(G.T1CapabilityError) as caught:
        G.require_completable("assemble_bootstrap", _lying_collaborator)
    message = str(caught.value)
    assert "no reachable return" in message
    assert "the proof wins" in message


def test_the_lying_collaborator_fails_the_whole_gate():
    with pytest.raises(G.T1CapabilityError):
        G.require_executable_capability(
            _complete(assemble_bootstrap=_lying_collaborator)
        )


def test_a_return_inside_a_closure_does_not_count_as_the_bodys_return():
    """The driver calls the outer function; a closure's return is not its own."""

    @G.declare_execution_capability(
        "assemble_challenge", executes=True, reason="Returns only from a closure."
    )
    def outer(*, oof_columns=None):
        def inner():
            return {}

        raise RuntimeError("outer still cannot return")

    with pytest.raises(G.T1CapabilityError, match="no reachable return"):
        G.require_completable("assemble_challenge", outer)


def test_a_body_that_returns_is_accepted():
    assert (
        G.require_completable(
            "assemble_challenge", _attested("assemble_challenge", assemble_one)
        ).executes
        is True
    )


def test_unreadable_source_is_not_treated_as_proof_of_anything():
    """A builtin has no source; absence of proof is not proof of absence."""
    assert G._returns_a_value(len) is None


# ---------------------------------------------------------------------------
# 3. A collaborator that could not accept the driver's call is refused
# ---------------------------------------------------------------------------


def test_a_wrongly_shaped_collaborator_is_refused_before_the_claim():
    @G.declare_execution_capability(
        "assemble_challenge", executes=True, reason="Wrong signature."
    )
    def wrong(unexpected_parameter):
        return {}

    with pytest.raises(G.T1CapabilityError, match="cannot accept the call"):
        G.require_accepts_call("assemble_challenge", wrong)


def test_the_shape_check_never_invokes_the_collaborator():
    calls: list[int] = []

    @G.declare_execution_capability(
        "assemble_bootstrap", executes=True, reason="Records invocation."
    )
    def counted(*, oof_columns=None):
        calls.append(1)
        return {}

    G.require_accepts_call("assemble_bootstrap", counted)
    G.require_completable("assemble_bootstrap", counted)
    assert calls == []


def test_the_call_contract_matches_the_drivers_actual_call_sites():
    """The contract is transcribed; this is what stops it drifting."""
    driver = Path(D.__file__).read_text(encoding="utf-8")
    harness = Path(R.__file__).read_text(encoding="utf-8")
    assert "subject_of_record=collaborators.subject_of_record" in driver
    assert "evaluate_fold=collaborators.evaluate_fold" in driver
    assert "evaluate_fold(fold, authority)" in harness
    assert "subject_of_record(str(record))" in harness
    for role, (_, keywords) in G.CAPABILITY_CALL_CONTRACT.items():
        for keyword in keywords:
            assert f"{keyword}=" in driver, (role, keyword)


# ---------------------------------------------------------------------------
# 4. An incomplete graph fails before the claim
# ---------------------------------------------------------------------------


def test_an_incomplete_graph_fails_before_the_claim(authorized):
    executor = _executor()
    with pytest.raises(G.T1CapabilityError):
        executor.execute(_complete(evaluate_fold=E.T1NonExecutingFoldEvaluator()))
    assert executor.run.stages.entered == []
    assert executor.run.claimed is None
    assert executor.receipts == []


def test_an_unbound_collaborator_still_fails_before_the_claim(authorized):
    executor = _executor()
    with pytest.raises(D.T1DriverError, match="is not bound"):
        executor.execute(_complete(evaluate_fold=None))
    assert executor.run.stages.entered == []
    assert executor.run.claimed is None


def test_the_gate_runs_before_preflight_and_before_the_claim():
    """Structural: the capability gate is the second statement in `execute`."""
    body = _execute_body()
    first, second = body[0], body[1]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert first.value.func.id == "require_canonical_execution_capability"
    assert isinstance(second, ast.Expr) and isinstance(second.value, ast.Call)
    assert second.value.func.id == "require_executable_capability"

    entered = [
        index
        for index, node in enumerate(body)
        for call in ast.walk(node)
        if isinstance(call, ast.Attribute) and call.attr.startswith("stage_")
    ]
    assert entered, "no stage is entered at all"
    assert min(entered) > 1, "a stage is entered before the capability gate"


def test_the_gate_does_not_read_the_permission_constant():
    """Could-this-finish must stay answerable without asking may-this-run."""
    code = _code_only(G)
    assert "T1_EXECUTION_SPECIFICATION_AUTHORIZED" not in code
    assert "t1_config" not in code


def test_every_threaded_collaborator_has_a_pre_claim_binding():
    assert set(D.REQUIRED_COLLABORATOR_CALLABLES) == set(G.CAPABILITY_STAGE_BINDINGS)
    assert D._UNGATED == []


def test_the_roles_are_verified_in_the_order_the_driver_reaches_them():
    indices = [
        SPEC.T1_STAGE_ORDER.index(G.CAPABILITY_STAGE_BINDINGS[role])
        for role in G.REQUIRED_CAPABILITY_ROLES
    ]
    assert indices == sorted(indices)


def test_every_bound_stage_is_after_the_claim():
    claim = SPEC.T1_STAGE_ORDER.index(SPEC.STAGE_CLAIM)
    for role, stage in G.CAPABILITY_STAGE_BINDINGS.items():
        assert SPEC.T1_STAGE_ORDER.index(stage) > claim, role


# ---------------------------------------------------------------------------
# 5. A complete graph passes the gate and still cannot run
# ---------------------------------------------------------------------------


def test_a_complete_attested_graph_passes_the_gate():
    receipt = G.require_executable_capability(_complete())
    assert receipt["execution_graph_complete"] is True
    assert receipt["verified_before_claim"] is True
    assert receipt["attempt_consumed"] is False
    assert receipt["run_directory_created"] is False
    assert len(receipt["attestations"]) == len(G.REQUIRED_CAPABILITY_ROLES)


def test_passing_the_capability_gate_is_not_permission():
    G.require_executable_capability(_complete())
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is False
    with pytest.raises(D.T1DriverError, match="human authorization is not granted"):
        _executor().execute(_complete())


def test_the_report_separates_bound_from_executable():
    report = D.T1CanonicalDevelopmentExecutor.verify_collaborators(
        _complete(evaluate_fold=E.T1NonExecutingFoldEvaluator())
    )
    assert report["collaborators_complete"] is True
    assert report["execution_graph_complete"] is False
    assert report["capability_gate"]["roles"]["evaluate_fold"]["executes"] is False
    assert report["attempt_consumed"] is False


def test_the_report_never_raises_on_an_incomplete_graph():
    report = G.capability_report(_complete(evaluate_fold=None))
    assert report["execution_graph_complete"] is False
    assert "refusal" in report["roles"]["evaluate_fold"]


# ---------------------------------------------------------------------------
# 6. Nothing was created, opened, claimed or authorized
# ---------------------------------------------------------------------------


def test_no_run_directory_is_created_on_capability_failure(authorized):
    assert not _canonical_root().exists()
    with pytest.raises(G.T1CapabilityError):
        _executor().execute(_complete(evaluate_fold=E.T1NonExecutingFoldEvaluator()))
    assert not _canonical_root().exists()


def test_the_gate_alone_creates_nothing(authorized):
    assert not _canonical_root().exists()
    with pytest.raises(G.T1CapabilityError):
        G.require_executable_capability(
            _complete(evaluate_fold=E.T1NonExecutingFoldEvaluator())
        )
    G.require_executable_capability(_complete())
    G.capability_report(_complete())
    assert not _canonical_root().exists()


def test_the_gate_module_never_writes_or_opens_anything():
    code = _code_only(G)
    for forbidden in (
        "mkdir",
        "makedirs",
        "write_text",
        "write_bytes",
        "np.load",
        "open(",
        "shutil",
        "os.remove",
    ):
        assert forbidden not in code, f"the gate calls {forbidden}"


def test_authorization_remains_false_after_every_check():
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is False
    assert D.T1_EXECUTION_SPECIFICATION_AUTHORIZED is False
    assert CFG.T1_EXECUTION_SPECIFICATION_EXISTS is True
    assert CFG.T1_CANONICAL_DEVELOPMENT_HARNESS_EXISTS is True


def test_the_canonical_attempt_is_still_unconsumed():
    assert not _canonical_root().exists()


# ---------------------------------------------------------------------------
# 7. The firewalls this PR must not have touched
# ---------------------------------------------------------------------------


def test_the_test_firewall_is_unchanged():
    """The same helper, the same verdicts, untouched by this PR."""
    assert SPEC.require_no_test_access("validation") == "validation"
    for name in ("test", "TEST", " Test "):
        with pytest.raises(SPEC.T1ExecutionSpecError, match="TEST is sealed"):
            SPEC.require_no_test_access(name)


def test_the_gate_can_reach_no_evidence_at_all():
    """An import-surface check, not a substring scan.

    The gate's receipts legitimately contain the word "label" -- they assert
    ``labels_opened: False`` -- so scanning for it would fail on the very
    statement that makes the guarantee. What actually matters is narrower and
    provable: the module imports nothing that can open evidence.
    """
    tree = ast.parse(Path(G.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    package = {name for name in imported if name.startswith("cardiosentinel")}
    assert package == {"cardiosentinel.neural.t1_execution_spec"}, package
    assert not {"numpy", "pandas", "joblib", "torch"} & imported
    assert "cardiosentinel.neural.t1_evidence_store" not in imported
    assert "cardiosentinel.neural.t1_fold_evaluation" not in imported


def test_the_upstream_contracts_are_unchanged():
    assert SPEC.T1_REQUIRED_M2_RETAINED_ARM == "M2-G"
    assert SPEC.T1_REQUIRED_U1_FAMILY == "platt_logistic_on_recovered_logit"
    assert SPEC.T1_REQUIRED_T2_RETAINED_ARM == "causal_s4d_longitudinal_v1"
    assert SPEC.T1_REQUIRED_T2_SCORE_SEMANTICS == "uncalibrated_temporal_model_score"
    assert SPEC.T1_M2_REPLAY_PERMITTED is False
    assert SPEC.T1_T2_REPLAY_PERMITTED is False
    assert SPEC.T1_U1_REFIT_PERMITTED is False


def test_the_evidence_store_contract_is_unchanged():
    assert (
        STORE.forbidden_members() == SPEC.T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND
    )
    assert set(SPEC.T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS) >= {
        "label",
        "target_family",
        "primary_mask",
    }


def test_the_frozen_stage_order_is_unchanged():
    assert len(SPEC.T1_STAGE_ORDER) == 29
    assert SPEC.T1_STAGE_ORDER.index(SPEC.STAGE_CLAIM) == 9
    assert len(D.CANONICAL_EXECUTION_PLAN) == 29
