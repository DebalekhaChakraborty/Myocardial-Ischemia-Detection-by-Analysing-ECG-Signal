"""Structural tests for the canonical T1-v1 development driver.

The driver is the orchestration layer that was missing: `t1_development_run`
implemented twenty-nine stages and nothing sequenced them. These tests prove
the sequence exists, that there is exactly one of it, and that building it
authorized nothing.

Nothing here executes the science. No canonical run is started, no attempt
directory is created, no VALIDATION row is read, no label is opened, no OOF
evidence is generated and TEST stays sealed. Most of these tests exist
specifically to prove those things about the driver rather than to trust them.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest
from _attempt_guard import assert_attempt_unconsumed

from cardiosentinel.neural import t1_canonical_driver as D
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_development_run as R
from cardiosentinel.neural import t1_evidence_store as STORE
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

FROZEN_SOURCE_DIGESTS = {
    "t1_protocol.py": (
        "b0df6ea2ade450037e94e5ab3b193694fea980337851a2458b3f43873450b192"
    ),
    "t1_execution_spec.py": (
        "edb0cbf1afe43dee48b5d2d0ed190e0939530fc026fd2f09d3312b929ab1fbe3"
    ),
    "t1_development_run.py": (
        "ad08035d33a1f421cf5a6a18df33e9a7ed55fad29074e7581bbe3ba796b90a8e"
    ),
    "t1_evidence_store.py": (
        "464ca1607191aa02042a6dcbb8cfeda4d4f3aced1eae2e29ae4b77be8cf6d39c"
    ),
    "t1_persistence.py": (
        "77c0e0a40efa7056777ef8d3bb13983ae4cd1bb9493d3c6c7eb11c7faebd68ad"
    ),
}


class _Source:
    """Minimal target source: the protocol shape and nothing wider."""

    def read_subject_targets(self, subject_id, *, partition):  # pragma: no cover
        raise AssertionError("no test here opens targets")


def _driver_source() -> str:
    return Path(D.__file__).read_text(encoding="utf-8")


def _driver_code_only() -> str:
    """The driver with comments and docstrings removed.

    A naive substring scan over the raw file reports every word the prose
    uses. This module's docstrings say "no retry" and "labels" precisely
    because it does neither, so the scans below must read the code.
    """
    tree = ast.parse(_driver_source())
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
        """Blank out prose literals; a column name never contains a space."""

        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            if isinstance(node.value, str) and " " in node.value:
                return ast.Constant(value="")
            return node

    return ast.unparse(_DropProse().visit(tree))


def _names_word(haystack: str, needle: str) -> bool:
    """Whole-word match, so `label` never matches `labels` or `label_blind`."""
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


def _executor(sha: str = "0" * 40) -> D.T1CanonicalDevelopmentExecutor:
    return D.T1CanonicalDevelopmentExecutor(
        run=R.T1DevelopmentRun(authorized_git_sha=sha)
    )


def _collaborators(**overrides) -> D.T1ExecutionCollaborators:
    """A fully bound set pointing at paths that do not exist.

    Deliberately unreadable: if a refusal ever stopped working, the next thing
    the driver did would fail on a missing file rather than open real evidence.
    """
    fields = {
        "m2_row_evidence": Path("/nonexistent/m2_row_evidence.npz"),
        "t2_identity": Path("/nonexistent/t2_outer_row_identity.npz"),
        "t2_selected_scores": Path("/nonexistent/t2_selected_scores.npz"),
        "calibrators": {"s20221": object()},
        "target_source": _Source(),
        "subject_of_record": lambda record: "s20221",
        "evaluate_fold": lambda *a, **k: {"artifact": {}},
        "assemble_oof_state_columns": lambda **k: {},
        "assemble_oof_result": lambda **k: {},
        "assemble_subject_evidence": lambda **k: {},
        "assemble_bootstrap": lambda **k: {},
        "assemble_challenge": lambda **k: {},
        "assemble_final_configuration": lambda **k: {},
    }
    fields.update(overrides)
    return D.T1ExecutionCollaborators(**fields)


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


# ---------------------------------------------------------------------------
# 1. The driver exists
# ---------------------------------------------------------------------------


def test_the_driver_exists():
    assert hasattr(D, "T1CanonicalDevelopmentExecutor")
    assert D.DRIVER_NAME == "T1CanonicalDevelopmentExecutor"
    assert callable(D.T1CanonicalDevelopmentExecutor.execute)


def test_importing_the_driver_creates_nothing_and_emits_nothing(tmp_path, capsys):
    """Import must do no filesystem work and take no upstream reading."""
    import importlib

    before = sorted(tmp_path.iterdir())
    importlib.reload(D)
    assert sorted(tmp_path.iterdir()) == before
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert_attempt_unconsumed()


def test_the_plan_is_readable_without_authorization(monkeypatch):
    """Auditing a choreography is not permission to run it."""
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    assert len(D.T1CanonicalDevelopmentExecutor.plan()) == 29
    assert len(D.T1CanonicalDevelopmentExecutor.stage_receipts()) == 29


# ---------------------------------------------------------------------------
# 2. Exactly one canonical stage ordering
# ---------------------------------------------------------------------------


def test_the_driver_sequences_the_frozen_stage_order_exactly():
    ordering = tuple(step.stage for step in D.CANONICAL_EXECUTION_PLAN)
    assert ordering == tuple(SPEC.T1_STAGE_ORDER)
    assert len(ordering) == 29
    assert len(set(ordering)) == 29


def test_the_ordering_is_derived_from_the_specification_not_retyped():
    """A second hand-written list beside the frozen one could drift from it."""
    tree = ast.parse(_driver_source())
    literal_stage_lists = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.List, ast.Tuple))
        and len(node.elts) >= 10
        and all(isinstance(element, ast.Constant) for element in node.elts)
    ]
    assert literal_stage_lists == [], (
        "the driver re-types a long literal sequence; the ordering must be "
        "generated from T1_STAGE_ORDER"
    )
    assert "T1_STAGE_ORDER" in _driver_source()


def test_there_is_exactly_one_execution_path():
    source = _driver_source()
    executes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    ]
    assert len(executes) == 1, "more than one execute path exists"


def test_every_frozen_stage_is_bound_to_a_real_harness_method():
    for step in D.CANONICAL_EXECUTION_PLAN:
        assert hasattr(R.T1DevelopmentRun, step.binding), step.stage
        assert callable(getattr(R.T1DevelopmentRun, step.binding))


def test_a_stage_without_a_binding_is_refused_rather_than_skipped(monkeypatch):
    monkeypatch.setattr(
        SPEC, "T1_STAGE_ORDER", SPEC.T1_STAGE_ORDER + ("a_new_unbound_stage",)
    )
    monkeypatch.setattr(D, "T1_STAGE_ORDER", SPEC.T1_STAGE_ORDER)
    with pytest.raises(D.T1DriverError, match="cannot enter"):
        D._build_plan()


def test_the_claim_is_the_boundary_the_attempt_is_spent_at():
    by_stage = {step.stage: step for step in D.CANONICAL_EXECUTION_PLAN}
    assert by_stage[SPEC.STAGE_START].consumes_attempt is False
    assert by_stage[SPEC.STAGE_PROVE_ATTEMPT_ABSENT].consumes_attempt is False
    assert by_stage[SPEC.STAGE_CLAIM].consumes_attempt is True
    assert by_stage[SPEC.STAGE_COMPLETION].consumes_attempt is True
    assert SPEC.T1_PRE_CLAIM_REFUSAL_CONSUMES_ATTEMPT is False
    assert SPEC.T1_POST_CLAIM_FAILURE_CONSUMES_ATTEMPT is True


# ---------------------------------------------------------------------------
# 3. The driver cannot execute while authorization is false
# ---------------------------------------------------------------------------


def test_authorization_is_open_on_this_branch():
    """The tripwire this test was fired to protect has done its job.

    It previously asserted the constant was False and instructed the separate
    authorization PR to update it here, in the same change that flips it. This
    is that change. All three facts now read True, which is exactly the state
    in which a check derived from the first two would be indistinguishable
    from a deliberate one -- so the behavioural tests around this one drive the
    gate explicitly rather than relying on the repository constant.
    """
    assert CFG.T1_EXECUTION_SPECIFICATION_EXISTS is True
    assert CFG.T1_CANONICAL_DEVELOPMENT_HARNESS_EXISTS is True
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is True


def test_the_driver_refuses_to_execute(monkeypatch):
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    with pytest.raises(D.T1DriverError) as caught:
        _executor().execute(_collaborators())
    message = str(caught.value)
    assert (
        "canonical execution capability exists, but human authorization is "
        "not granted" in message
    )
    assert "none of them a permission" in message


def test_the_refusal_happens_before_any_stage_is_entered(monkeypatch):
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    executor = _executor()
    with pytest.raises(D.T1DriverError):
        executor.execute(_collaborators())
    assert executor.run.stages.entered == []
    assert executor.receipts == []
    assert executor.run.claimed is None
    assert executor.run.upstream == {}


def test_the_gate_is_asked_before_anything_else_in_execute():
    """Structural: the permission call is the first statement in the body."""
    execute = next(
        node
        for node in ast.walk(ast.parse(_driver_source()))
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    body = list(execute.body)
    # Drop the docstring; the gate must be the first thing that runs.
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    first = body[0]
    assert isinstance(first, ast.Expr), "the first statement is not a bare call"
    assert isinstance(first.value, ast.Call)
    assert first.value.func.id == "require_canonical_execution_capability"


def test_granting_permission_is_not_something_the_driver_can_do():
    """The driver reads the constant and never writes it."""
    source = _driver_source()
    assert "T1_EXECUTION_SPECIFICATION_AUTHORIZED = " not in source
    assert "T1_EXECUTION_SPECIFICATION_AUTHORIZED =True" not in source
    writes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
        and target.id == "T1_EXECUTION_SPECIFICATION_AUTHORIZED"
    ]
    assert writes == []


# ---------------------------------------------------------------------------
# 4. No canonical attempt directory
# ---------------------------------------------------------------------------


def test_the_canonical_attempt_directory_is_absent_and_stays_absent(monkeypatch):
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    assert_attempt_unconsumed()
    with pytest.raises(D.T1DriverError):
        _executor().execute(_collaborators())
    assert_attempt_unconsumed()


def test_the_driver_never_creates_a_directory():
    source = _driver_code_only()
    for forbidden in ("mkdir", "makedirs", "touch(", "open(", "write_text"):
        assert forbidden not in source, f"the driver calls {forbidden}"


def test_the_driver_names_no_alternate_run_root():
    source = _driver_source()
    assert "phase9-t1-development-v1" not in source
    assert SPEC.T1_ALTERNATE_RUN_ROOT_PERMITTED is False


# ---------------------------------------------------------------------------
# 5-6. VALIDATION and TEST stay closed
# ---------------------------------------------------------------------------


def test_the_driver_reads_no_validation_row():
    """It threads paths; it never opens a store itself."""
    source = _driver_code_only()
    for reader in (
        "read_store",
        "read_m2g_row_evidence",
        "read_t2_identity_members",
        "read_t2_selected_scores",
        "np.load",
        "load(",
    ):
        assert reader not in source, f"the driver calls {reader}"


def test_the_driver_opens_no_labels():
    """It never names a forbidden evidence column or identity member."""
    code = _driver_code_only()
    forbidden = set(SPEC.T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS) | set(
        SPEC.T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND
    )
    forbidden |= {"primary_positive", "held_out_labels"}
    named = sorted(name for name in forbidden if _names_word(code, name))
    assert named == [], f"the driver names label-bearing columns {named}"


def test_the_driver_generates_no_oof_evidence():
    source = _driver_source()
    assert "write_oof_state_evidence" not in source
    assert "write_input_evidence" not in source
    assert not any(
        (REPOSITORY_ROOT / "cardiosentinel-runs").glob("phase9-*/T1_OOF_*.json")
    )


def test_the_driver_selects_no_policy_and_computes_no_metric():
    """Selection and metrics stay frozen upstream; the driver only threads."""
    source = _driver_code_only()
    for frozen in (
        "select_policy",
        "score_policy",
        "generate_thresholds",
        "run_policy_over_streams",
        "episode_f1",
        "window_mcc",
        "subject_bootstrap_indices",
        "next_state",
        "policy_sort_key",
    ):
        assert frozen not in source, f"the driver reimplements or calls {frozen}"


def test_test_stays_sealed():
    assert SPEC.T1_TEST_ACCESSED is False
    assert SPEC.T1_SEALED_TEST_STATE == "unopened"
    source = _driver_source()
    assert "--test" not in source
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()


def test_the_artifact_plan_validation_refuses_a_test_artifact(monkeypatch):
    monkeypatch.setattr(D, "T1_PLANNED_ARTIFACTS", ("T1_RESULT.json", "test.json"))
    with pytest.raises(SPEC.T1ExecutionSpecError, match="TEST is sealed"):
        D.T1CanonicalDevelopmentExecutor.validate_artifact_plan()


def test_the_artifact_plan_is_the_specifications():
    assert D.T1CanonicalDevelopmentExecutor.planned_artifacts() == tuple(
        SPEC.T1_PLANNED_ARTIFACTS
    )
    assert D.T1CanonicalDevelopmentExecutor.validate_artifact_plan() == tuple(
        SPEC.T1_PLANNED_ARTIFACTS
    )


# ---------------------------------------------------------------------------
# 7. No retry or recovery path
# ---------------------------------------------------------------------------


def test_the_driver_has_no_retry_or_recovery_path():
    """No loop, and no handler that can end without re-raising.

    This test used to assert that `execute` contained no handler at all, which
    was true of the code it was written against and is no longer the property
    worth holding. The specification requires a post-claim failure to leave a
    receipt, and writing one means catching the exception that produced it.

    What must never happen is a handler that *ends* -- one that returns,
    continues, passes or raises something else, any of which turns a consumed
    attempt into a second one. So the assertion moved from "no handler" to
    "every handler re-raises the exception it caught", which is the honest
    version of the same guarantee and would still catch a retry hidden in an
    except block.
    """
    execute = next(
        node
        for node in ast.walk(ast.parse(_driver_source()))
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    loops = [n for n in ast.walk(execute) if isinstance(n, (ast.While, ast.For))]
    assert loops == [], "the driver loops over stages; a retry could hide there"

    handlers = [n for n in ast.walk(execute) if isinstance(n, ast.ExceptHandler)]
    assert handlers, "the post-claim failure receipt needs a handler to be written"
    for handler in handlers:
        last = handler.body[-1]
        assert isinstance(last, ast.Raise), (
            "an except block in the driver ends without raising; a swallowed "
            "refusal is how a consumed attempt turns into a second one"
        )
        assert last.exc is None, (
            "the driver raises a new exception instead of re-raising the one "
            "that failed; the original failure is the evidence"
        )
        for node in ast.walk(handler):
            assert not isinstance(node, (ast.While, ast.For)), (
                "a loop inside the failure handler could retry the attempt"
            )


def test_the_failure_handler_only_writes_a_receipt():
    """The handler records; it must not decide, repair or re-enter a stage."""
    execute = next(
        node
        for node in ast.walk(ast.parse(_driver_source()))
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    for handler in [n for n in ast.walk(execute) if isinstance(n, ast.ExceptHandler)]:
        called = {
            node.func.attr
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert called == {"_receipt_on_failure"}, (
            f"the failure handler calls {sorted(called)}; it may only write the "
            "receipt before re-raising"
        )
        assert not any(name.startswith("stage_") for name in called), (
            "the failure handler re-enters a stage"
        )


def test_the_driver_exposes_no_override_of_any_kind():
    """Whole-word, code-only: the docstrings say "no retry" on purpose."""
    code = _driver_code_only()
    exposed = sorted(
        name
        for name in (
            "retry",
            "recovery",
            "resume",
            "force",
            "reset",
            "overwrite",
            "fresh_seed",
            "seed",
            "fold_override",
            "subject_override",
        )
        if _names_word(code, name)
    )
    assert exposed == [], f"the driver exposes {exposed}"


def test_the_frozen_no_retry_constants_are_unchanged():
    assert SPEC.T1_AUTOMATIC_RETRY_PERMITTED is False
    assert SPEC.T1_RECOVERY_IDENTITY_PREDECLARED is False
    assert SPEC.T1_FOLD_RETRY_PERMITTED is False
    assert SPEC.T1_ATTEMPT_NAME_CARRIES_TIMESTAMP is False
    assert SPEC.T1_ATTEMPT_NAME_CARRIES_UUID is False


# ---------------------------------------------------------------------------
# 8. No additional CLI surface
# ---------------------------------------------------------------------------


def test_the_driver_adds_no_cli_argument():
    source = _driver_code_only()
    assert "argparse" not in source
    assert "add_argument" not in source
    assert "ArgumentParser" not in source


def test_the_cli_surface_is_unchanged():
    assert R.registered_options() == (
        "--execute-canonical-development",
        "--expected-git-sha",
    )
    for forbidden in SPEC.T1_FORBIDDEN_CLI_OPTIONS:
        assert forbidden not in R.registered_options()


# ---------------------------------------------------------------------------
# 9. The frozen files are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,digest", sorted(FROZEN_SOURCE_DIGESTS.items()))
def test_the_driver_pr_modifies_no_frozen_source(name, digest):
    import hashlib

    path = REPOSITORY_ROOT / "src" / "cardiosentinel" / "neural" / name
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == digest, f"{name} changed; this PR must not touch it"


def test_the_frozen_documents_are_untouched():
    import hashlib

    for name, digest in (
        (
            "T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1.md",
            "ef044754020b1756ea7aae5fa1b747c5ba6fc0c8cd70d52e73185555897d70d4",
        ),
        (
            "T1_CANONICAL_DEVELOPMENT_EXECUTION_SPEC_V1.md",
            "11b6a9aff2f1d928a9f33516db2ea764cf0553a949cd79c14562bafe34f090bf",
        ),
    ):
        path = REPOSITORY_ROOT / "docs" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


# ---------------------------------------------------------------------------
# The collaborator contract
# ---------------------------------------------------------------------------


def test_an_unbound_collaborator_is_refused_before_the_claim(monkeypatch):
    """Discovering a missing capability must not cost the attempt."""
    monkeypatch.setattr(CFG, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", True)
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", True)
    executor = _executor()
    with pytest.raises(D.T1DriverError, match="is not bound"):
        executor.execute(_collaborators(evaluate_fold=None))
    assert executor.run.stages.entered == []
    assert_attempt_unconsumed()


def test_every_label_bearing_step_is_a_required_collaborator():
    required = set(D.REQUIRED_COLLABORATOR_CALLABLES)
    assert "evaluate_fold" in required
    assert "subject_of_record" in required
    assert len(required) == 8


def test_the_collaborators_carry_no_configuration_value():
    """Paths and callables only -- never a threshold, quantile or profile."""
    annotations = inspect.get_annotations(D.T1ExecutionCollaborators)
    for name in annotations:
        assert not name.startswith("q_")
        assert "threshold" not in name
        assert "profile" not in name
        assert "seed" not in name


def test_a_calibrator_set_is_required_and_never_refitted():
    with pytest.raises(D.T1DriverError, match="never refitted"):
        _collaborators(calibrators={}).require_complete()
    assert SPEC.T1_U1_REFIT_PERMITTED is False


# ---------------------------------------------------------------------------
# 10. Upstream regression is untouched
# ---------------------------------------------------------------------------


def test_the_upstream_contracts_are_unchanged():
    assert SPEC.T1_REQUIRED_M2_RETAINED_ARM == "M2-G"
    assert SPEC.T1_REQUIRED_U1_FAMILY == "platt_logistic_on_recovered_logit"
    assert SPEC.T1_REQUIRED_T2_RETAINED_ARM == "causal_s4d_longitudinal_v1"
    assert SPEC.T1_REQUIRED_T2_SCORE_SEMANTICS == "uncalibrated_temporal_model_score"
    assert SPEC.T1_M2_REPLAY_PERMITTED is False
    assert SPEC.T1_T2_REPLAY_PERMITTED is False


def test_the_evidence_store_contract_is_unchanged():
    assert (
        STORE.forbidden_members() == SPEC.T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND
    )
    assert set(SPEC.T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS) >= {
        "label",
        "target_family",
        "primary_mask",
    }
