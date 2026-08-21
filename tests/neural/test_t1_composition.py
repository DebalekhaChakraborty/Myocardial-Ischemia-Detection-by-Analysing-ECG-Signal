"""Tests for the canonical composition root.

The composition root resolves frozen artifacts, reconstructs the frozen U1
fits, binds the collaborators and delegates. These tests build the complete
graph against the real canonical artifacts on disk and prove that authorization
is the only thing still refusing.

Nothing here authorizes execution, claims the canonical attempt, creates the
canonical run directory or reaches TEST.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_canonical_driver as D
from cardiosentinel.neural import t1_capability_gate as G
from cardiosentinel.neural import t1_composition as C
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_development_run as R
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.t1_protocol import T1_VALIDATION_SUBJECTS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# The canonical artifacts are gitignored, so CI has none of them. Every test
# that needs one skips there rather than asserting a repository property that
# only holds on the scientific machine.
_ARTIFACTS_PRESENT = True
try:
    C.canonical_artifact_paths(REPOSITORY_ROOT)
except Exception:  # pragma: no cover - environment dependent
    _ARTIFACTS_PRESENT = False

needs_artifacts = pytest.mark.skipif(
    not _ARTIFACTS_PRESENT, reason="canonical upstream artifacts are not present"
)


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _run() -> R.T1DevelopmentRun:
    return R.T1DevelopmentRun(authorized_git_sha="0" * 40)


def _code_only() -> str:
    tree = ast.parse(Path(C.__file__).read_text(encoding="utf-8"))
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
        def visit_Constant(self, node):
            if isinstance(node.value, str) and " " in node.value:
                return ast.Constant(value="")
            return node

    return ast.unparse(_DropProse().visit(tree))


# ---------------------------------------------------------------------------
# 1. The complete frozen dependency graph
# ---------------------------------------------------------------------------


@needs_artifacts
def test_every_frozen_artifact_resolves():
    paths = C.canonical_artifact_paths(REPOSITORY_ROOT)
    assert set(paths) == {
        "m2_row_evidence",
        "t2_identity",
        "t2_selected_scores",
        "u1_fold_manifest",
    }
    for path in paths.values():
        assert path.exists()


@needs_artifacts
def test_the_artifacts_are_named_by_the_retention_decisions():
    paths = C.canonical_artifact_paths(REPOSITORY_ROOT)
    assert SPEC.T1_REQUIRED_M2_RETAINED_ARM in str(paths["m2_row_evidence"])
    assert paths["m2_row_evidence"].name == SPEC.T1_M2_ROW_EVIDENCE_NAME
    assert paths["t2_identity"].name == SPEC.T1_T2_IDENTITY_NAME
    assert (
        paths["t2_selected_scores"].name
        == (C.T2_SCORE_FILE_BY_ARM[SPEC.T1_REQUIRED_T2_RETAINED_ARM])
    )
    assert paths["u1_fold_manifest"].name == SPEC.T1_U1_FOLD_MANIFEST_NAME


@needs_artifacts
def test_twelve_out_of_fold_calibrators_are_reconstructed():
    paths = C.canonical_artifact_paths(REPOSITORY_ROOT)
    calibrators = C.load_oof_calibrators(paths["u1_fold_manifest"])
    assert sorted(calibrators) == sorted(T1_VALIDATION_SUBJECTS)
    assert len(calibrators) == SPEC.T1_U1_FOLD_COUNT == 12
    for subject, calibrator in calibrators.items():
        assert calibrator.family == SPEC.T1_REQUIRED_U1_FAMILY
        assert subject not in calibrator.fit_subjects
        assert len(calibrator.fit_subjects) == SPEC.T1_U1_FIT_SUBJECTS_PER_FOLD


@needs_artifacts
def test_the_complete_graph_passes_the_capability_gate():
    run = _run()
    composition = C.resolve_canonical_composition(REPOSITORY_ROOT)
    collaborators = C.build_canonical_collaborators(run, composition)
    receipt = G.require_executable_capability(collaborators)
    assert receipt["execution_graph_complete"] is True
    assert receipt["attempt_consumed"] is False
    assert receipt["run_directory_created"] is False


@needs_artifacts
def test_permission_is_the_only_remaining_blocker(monkeypatch):
    """The real graph is complete, and withdrawing permission still stops it.

    This is the one test in the suite that builds the complete graph against
    the real canonical artifacts, so it is also the one that most needs
    permission withdrawn explicitly rather than inherited from the repository
    constant. Both modules are patched because the driver holds its own
    imported reference. `_run()` additionally carries a dummy authorized SHA,
    so the commit check would refuse this invocation even if the gate did not.
    """
    monkeypatch.setattr(CFG, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    run = _run()
    composition = C.resolve_canonical_composition(REPOSITORY_ROOT)
    collaborators = C.build_canonical_collaborators(run, composition)
    G.require_executable_capability(collaborators)
    with pytest.raises(D.T1DriverError, match="human authorization is not granted"):
        D.T1CanonicalDevelopmentExecutor(run=run).execute(collaborators)
    assert run.stages.entered == []
    assert run.claimed is None
    assert not _canonical_root().exists()


# ---------------------------------------------------------------------------
# 2. A missing artifact fails before the claim
# ---------------------------------------------------------------------------


def test_a_missing_artifact_fails_closed(tmp_path):
    with pytest.raises(C.T1CompositionError, match="is absent"):
        C.canonical_artifact_paths(tmp_path)
    assert not _canonical_root().exists()


def test_a_missing_artifact_refusal_names_the_artifact(tmp_path):
    with pytest.raises(C.T1CompositionError) as caught:
        C.resolve_canonical_composition(tmp_path)
    message = str(caught.value)
    assert "does not search for an alternative" in message
    assert "create one" in message


def test_a_short_fold_manifest_fails_closed(tmp_path):
    import json

    path = tmp_path / SPEC.T1_U1_FOLD_MANIFEST_NAME
    path.write_text(json.dumps({"folds": []}), encoding="utf-8")
    with pytest.raises(C.T1CompositionError, match="not 12"):
        C.load_oof_calibrators(path)


# ---------------------------------------------------------------------------
# 3. It resolves; it does not compute
# ---------------------------------------------------------------------------


def test_the_composition_root_performs_no_science():
    code = _code_only()
    for forbidden in (
        "generate_thresholds",
        "run_policy_over_streams",
        "score_policy",
        "select_policy",
        "policy_sort_key",
        "next_state",
        "empirical_order_statistic",
        "episode_f1",
        "window_mcc",
    ):
        assert forbidden not in code, f"the composition root calls {forbidden}"
    assert C.composition_capability()["performs_scientific_computation"] is False


def test_no_calibrator_is_refitted():
    code = _code_only()
    for forbidden in SPEC.T1_U1_FORBIDDEN_FITTING_CALLABLES:
        assert forbidden not in code, f"the composition root calls {forbidden}"
    assert SPEC.T1_U1_REFIT_PERMITTED is False
    assert C.composition_capability()["refit_performed"] is False


def test_no_artifact_root_can_be_supplied():
    """A run that could be pointed elsewhere has provenance as an argument."""
    import inspect

    parameters = inspect.signature(C.canonical_artifact_paths).parameters
    assert list(parameters) == ["repository_root"]
    code = _code_only()
    for forbidden in ("glob", "rglob", "iterdir", "latest", "sorted(Path"):
        assert forbidden not in code, f"the composition root discovers via {forbidden}"


def test_the_composition_root_never_reads_the_permission_constant():
    code = _code_only()
    assert "T1_EXECUTION_SPECIFICATION_AUTHORIZED" not in code
    assert C.composition_capability()["authorizes_execution"] is False


def test_the_composition_root_creates_nothing():
    code = _code_only()
    for forbidden in ("mkdir", "makedirs", "write_text", "write_bytes", "shutil"):
        assert forbidden not in code, f"the composition root calls {forbidden}"


# ---------------------------------------------------------------------------
# 4. The entrypoint order
# ---------------------------------------------------------------------------


def test_main_asks_permission_before_it_resolves_anything():
    source = Path(R.__file__).read_text(encoding="utf-8")
    body = source[source.index("def main(") :]
    gate = body.index("require_canonical_execution_capability()")
    for later in (
        "resolve_canonical_composition(",
        "build_canonical_collaborators(",
        "executor.execute(",
    ):
        assert body.index(later) > gate, f"{later} runs before the permission gate"


def test_main_delegates_to_the_one_driver():
    source = Path(R.__file__).read_text(encoding="utf-8")
    body = source[source.index("def main(") :]
    assert "T1CanonicalDevelopmentExecutor" in body
    assert "executor.execute(collaborators)" in body
    assert "stage_claim" not in body


# ---------------------------------------------------------------------------
# 5. Nothing authorized, claimed, created or opened
# ---------------------------------------------------------------------------


def test_composing_the_graph_does_not_change_authorization():
    """Resolving artifacts is not a permission event.

    Agreement is the invariant that survives the flip: a divergent copy of the
    constant is how a gate opens on one code path and not another.
    """
    assert D.T1_EXECUTION_SPECIFICATION_AUTHORIZED is (
        CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED
    )


def test_the_canonical_attempt_is_untouched():
    assert not _canonical_root().exists()


def test_no_test_access_occurs():
    code = _code_only()
    assert "require_no_test_access" in code
    assert C.composition_capability()["test_accessed"] is False
    with pytest.raises(SPEC.T1ExecutionSpecError, match="TEST is sealed"):
        SPEC.require_no_test_access("test")


def test_the_frozen_sources_are_byte_identical():
    import hashlib

    frozen = {
        "t1_protocol.py": (
            "b0df6ea2ade450037e94e5ab3b193694fea980337851a2458b3f43873450b192"
        ),
        "t1_execution_spec.py": (
            "edb0cbf1afe43dee48b5d2d0ed190e0939530fc026fd2f09d3312b929ab1fbe3"
        ),
        "t1_evidence_store.py": (
            "464ca1607191aa02042a6dcbb8cfeda4d4f3aced1eae2e29ae4b77be8cf6d39c"
        ),
    }
    for name, digest in frozen.items():
        path = REPOSITORY_ROOT / "src" / "cardiosentinel" / "neural" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
