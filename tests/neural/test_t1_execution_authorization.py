"""Tests for the T1 canonical execution authorization gate.

The enabling change flipped `T1_EXECUTION_SPECIFICATION_AUTHORIZED` and
nothing else. These tests prove that granting permission weakened no other
check: every invocation still has to prove the authorized commit against a
clean HEAD, the runtime identity against the frozen dependency digest, that
TEST is unopened, and that the canonical attempt does not already exist.

The gate itself is unchanged and singular --
`t1_canonical_driver.require_canonical_execution_capability` -- and this file
adds no second permission function, no duplicate check and no alternate path.

Nothing here runs the science. No canonical run is started, no OOF evidence is
generated, no VALIDATION row is read and TEST stays sealed. Every test that
reaches the entry point carries a deliberately wrong authorized SHA, so the
commit check refuses even where the gate does not; several tests exist
specifically to prove a refusal leaves no directory behind.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_canonical_driver as D
from cardiosentinel.neural import t1_composition as C
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_development_run as R
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.runtime_sentinel import (
    RuntimeIntegrityError,
    RuntimeIntegrityRecord,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# The commit this change was cut from, where the constant was still False.
PRE_AUTHORIZATION_SHA = "064fe5e06857af032375554bfc10fa0ad17effff"

# A SHA that names no commit, used wherever the entry point is reached. The
# commit check refuses it independently of the gate, so no test in this file
# can execute even if permission is open.
WRONG_SHA = "0" * 40


def _observed_dependency_digest() -> str:
    from cardiosentinel.neural.provenance import dependency_environment

    return str(dependency_environment()["installed_packages_sha256"])


def _frozen_dependency_digest() -> str:
    from cardiosentinel.neural.p1_experiment import FROZEN_DEPENDENCY_DIGEST

    return str(FROZEN_DEPENDENCY_DIGEST)


# `stage_preflight` opens with the runtime-identity check, so any test that
# needs to reach the commit check behind it requires the frozen scientific
# interpreter. CI installs a different set and is refused at stage 1, which is
# correct behaviour; `test_a_foreign_runtime_is_refused_first` covers that
# directly and runs everywhere.
ON_FROZEN_INTERPRETER = _observed_dependency_digest() == _frozen_dependency_digest()
requires_frozen_runtime = pytest.mark.skipif(
    not ON_FROZEN_INTERPRETER,
    reason=(
        "reaching the commit check requires the frozen scientific interpreter; "
        "this environment reports a different installed-package digest"
    ),
)

# The canonical artifacts are gitignored, so CI has none of them and the
# composition root refuses before the commit check is reached.
_ARTIFACTS_PRESENT = True
try:
    C.canonical_artifact_paths(REPOSITORY_ROOT)
except Exception:  # pragma: no cover - environment dependent
    _ARTIFACTS_PRESENT = False

needs_artifacts = pytest.mark.skipif(
    not _ARTIFACTS_PRESENT, reason="canonical upstream artifacts are not present"
)


def _argv(sha: str) -> list[str]:
    return [SPEC.T1_CANONICAL_EXECUTION_FLAG, f"{SPEC.T1_EXPECTED_GIT_SHA_FLAG}={sha}"]


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _withdraw(monkeypatch) -> None:
    """Close the gate on both modules.

    The driver holds its own imported reference to the constant, so patching
    the config module alone would leave the gate open on the path that matters.
    """
    monkeypatch.setattr(CFG, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    monkeypatch.setattr(D, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)


# ---------------------------------------------------------------------------
# What the change actually changed
# ---------------------------------------------------------------------------


def test_authorization_was_false_before_this_change():
    """The gate this change opened was genuinely shut beforehand.

    Read from git rather than asserted from memory: if the constant had always
    been True the enabling change would be a no-op, and every refusal test
    below would be proving nothing.
    """
    reachable = subprocess.run(
        ["git", "cat-file", "-e", f"{PRE_AUTHORIZATION_SHA}^{{commit}}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if reachable.returncode != 0:
        pytest.skip(
            f"{PRE_AUTHORIZATION_SHA[:12]} is not in this clone (shallow "
            "checkout); the proof runs where the history is complete"
        )
    blob = subprocess.run(
        [
            "git",
            "show",
            f"{PRE_AUTHORIZATION_SHA}:src/cardiosentinel/neural/t1_config.py",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    values = {
        node.target.id: node.value.value
        for node in ast.parse(blob).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert values["T1_EXECUTION_SPECIFICATION_AUTHORIZED"] is False
    assert values["T1_CANONICAL_DEVELOPMENT_HARNESS_EXISTS"] is True
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is True


def test_the_permission_constant_is_read_in_exactly_one_place_in_config():
    """One place asks the question, so it cannot be answered two ways."""
    tree = ast.parse(Path(CFG.__file__).read_text(encoding="utf-8"))
    readers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "T1_EXECUTION_SPECIFICATION_AUTHORIZED"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(readers) == 1, "the authorization constant is read twice in config"


def test_the_gate_has_exactly_one_implementation():
    """Permission is asked through one function, in one module."""
    driver = Path(D.__file__).read_text(encoding="utf-8")
    definitions = [
        node
        for node in ast.walk(ast.parse(driver))
        if isinstance(node, ast.FunctionDef)
        and node.name == "require_canonical_execution_capability"
    ]
    assert len(definitions) == 1
    assert not hasattr(CFG, "require_canonical_execution_authorized"), (
        "a second permission function was introduced alongside the gate"
    )


def test_the_config_no_longer_states_the_gate_is_closed():
    """The prose next to the constant must track the constant."""
    lowered = Path(CFG.__file__).read_text(encoding="utf-8").lower()
    for stale in (
        "the canonical development harness is implemented      -> false",
        "canonical scientific execution is authorized          -> false",
    ):
        assert stale not in lowered, f"stale wording is back: {stale!r}"
    assert "neither is a permission" in lowered


# ---------------------------------------------------------------------------
# Permission remains a real gate
# ---------------------------------------------------------------------------


def test_the_gate_refuses_when_authorization_is_withdrawn(monkeypatch):
    """Withdrawing permission stops the run before anything is resolved."""
    _withdraw(monkeypatch)
    with pytest.raises(D.T1DriverError) as caught:
        D.require_canonical_execution_capability()
    message = str(caught.value)
    assert "human authorization is not granted" in message
    assert "none of them a permission" in message
    assert not _canonical_root().exists()


@needs_artifacts
def test_the_entry_point_refuses_when_authorization_is_withdrawn(monkeypatch):
    """The gate is reached from `main` before any artifact is resolved."""
    _withdraw(monkeypatch)
    opened: list[str] = []
    monkeypatch.setattr(
        C,
        "resolve_canonical_composition",
        lambda root: opened.append("composition"),
    )
    with pytest.raises(D.T1DriverError, match="human authorization is not granted"):
        R.main(_argv(WRONG_SHA))
    assert opened == [], "artifacts were resolved despite a closed gate"
    assert not _canonical_root().exists()


def test_granting_permission_is_not_something_the_driver_can_do():
    """The driver reads the constant and never writes it."""
    source = Path(D.__file__).read_text(encoding="utf-8")
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
# The SHA is the authorization mechanism
# ---------------------------------------------------------------------------


def test_the_expected_sha_is_required():
    """There is no default commit and no way to omit the flag."""
    with pytest.raises(SystemExit):
        R.main([SPEC.T1_CANONICAL_EXECUTION_FLAG])
    assert not _canonical_root().exists()


def test_the_execution_flag_is_required():
    with pytest.raises(SystemExit):
        R.main([f"{SPEC.T1_EXPECTED_GIT_SHA_FLAG}={WRONG_SHA}"])
    assert not _canonical_root().exists()


@requires_frozen_runtime
@needs_artifacts
def test_a_wrong_sha_is_refused(monkeypatch):
    """HEAD is re-read and compared; the authorization names one commit."""
    monkeypatch.setattr(
        PERSIST,
        "git_provenance",
        lambda root: {"git_sha": PRE_AUTHORIZATION_SHA, "git_dirty": False},
    )
    with pytest.raises(PERSIST.T1PersistenceError) as caught:
        R.main(_argv(WRONG_SHA))
    assert WRONG_SHA in str(caught.value)
    assert not _canonical_root().exists()


@requires_frozen_runtime
@needs_artifacts
def test_a_dirty_tree_is_refused(monkeypatch):
    """Canonical evidence requires a clean checkout, and nothing is repaired."""
    monkeypatch.setattr(
        PERSIST,
        "git_provenance",
        lambda root: {"git_sha": WRONG_SHA, "git_dirty": True},
    )
    with pytest.raises(PERSIST.T1PersistenceError) as caught:
        R.main(_argv(WRONG_SHA))
    assert "dirty" in str(caught.value).lower()
    assert not _canonical_root().exists()


def test_a_foreign_runtime_is_refused_first():
    """The runtime identity is stage 1, ahead of the commit check.

    Runs everywhere: the record is given an expected digest matching no
    environment, so the refusal is a property of the check rather than of the
    machine.
    """
    run = R.T1DevelopmentRun(
        authorized_git_sha=WRONG_SHA,
        runtime=RuntimeIntegrityRecord(expected_digest="0" * 64),
    )
    with pytest.raises(RuntimeIntegrityError):
        run.stage_preflight()
    assert run.stages.entered == [SPEC.STAGE_START]
    assert not _canonical_root().exists()


# ---------------------------------------------------------------------------
# TEST stays sealed, no bypass, attempt unconsumed
# ---------------------------------------------------------------------------


def test_test_remains_unopened_through_the_entry_point():
    assert SPEC.T1_TEST_ACCESSED is False
    assert SPEC.T1_SEALED_TEST_STATE == "unopened"
    with pytest.raises(SPEC.T1ExecutionSpecError):
        SPEC.require_no_test_access("test")
    for named in ("test", "cardiosentinel-runs/test/rows.npz", "b4_test"):
        with pytest.raises(SPEC.T1ExecutionSpecError):
            PERSIST.require_no_test_path(named)
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()


def test_the_entry_point_registers_no_test_option():
    assert "--test" not in R.registered_options()
    assert "--test" in SPEC.T1_FORBIDDEN_CLI_OPTIONS


def test_no_retry_or_recovery_path_exists():
    """Authorization did not introduce one, and the constants still forbid it."""
    assert SPEC.T1_AUTOMATIC_RETRY_PERMITTED is False
    assert SPEC.T1_RECOVERY_IDENTITY_PREDECLARED is False
    assert SPEC.T1_ALTERNATE_RUN_ROOT_PERMITTED is False
    for option in ("--retry", "--force", "--reset", "--fresh-seed", "--overwrite"):
        assert option not in R.registered_options()


def test_the_entry_point_never_loops_over_attempts():
    """No retry structure was introduced into the entry point."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    main_fn = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    handlers = [n for n in ast.walk(main_fn) if isinstance(n, (ast.Try, ast.While))]
    assert handlers == [], "the entry point retries or swallows a refusal"


def test_the_cli_surface_is_still_exactly_two_options():
    assert R.registered_options() == (
        SPEC.T1_CANONICAL_EXECUTION_FLAG,
        SPEC.T1_EXPECTED_GIT_SHA_FLAG,
    )


def test_the_canonical_attempt_is_still_absent():
    """The single canonical attempt has not been consumed by these tests."""
    assert not _canonical_root().exists()
    assert PERSIST.require_unclaimed_canonical_attempt(REPOSITORY_ROOT) == {
        "attempt_id": SPEC.T1_DEVELOPMENT_ATTEMPT_ID,
        "existing_run_directory": False,
        "automatic_retry_permitted": False,
        "automatic_alternate_name_permitted": False,
        "recovery_identity_predeclared": False,
    }


def test_an_existing_attempt_refuses_rather_than_re_rooting(tmp_path):
    """A claimed attempt is not deleted, renamed, re-rooted or retried."""
    occupied = PERSIST.canonical_run_directory(tmp_path)
    occupied.mkdir(parents=True)
    with pytest.raises(PERSIST.T1PersistenceError) as caught:
        PERSIST.require_unclaimed_canonical_attempt(tmp_path)
    message = str(caught.value).lower()
    assert "already claimed" in message
    assert "no automatic retry" in message
