"""Tests for the T1 canonical execution authorization gate.

The reviewed enabling change flipped `T1_EXECUTION_SPECIFICATION_AUTHORIZED`
and replaced the unconditional refusal in `t1_development_run.main` with the
frozen pre-claim verification. These tests prove that granting permission
weakened nothing: every invocation still has to prove the authorized commit
against a clean HEAD, the runtime identity against the frozen dependency
digest, the upstream chain, that TEST is unopened and that the canonical
attempt does not exist.

Nothing here runs the science. No canonical run is started, no OOF evidence is
generated, no VALIDATION row is read and TEST stays sealed. Several tests exist
specifically to prove that a refused invocation leaves no directory behind.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_development_run as R
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.runtime_sentinel import (
    RuntimeIntegrityError,
    RuntimeIntegrityRecord,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SHA = "5804e66668dda062a7dfe2d3a5bb3a43bff7ee5e"


def _observed_dependency_digest() -> str:
    from cardiosentinel.neural.provenance import dependency_environment

    return str(dependency_environment()["installed_packages_sha256"])


def _frozen_dependency_digest() -> str:
    from cardiosentinel.neural.p1_experiment import FROZEN_DEPENDENCY_DIGEST

    return str(FROZEN_DEPENDENCY_DIGEST)


# `stage_preflight` opens with the runtime-identity check, so any test that
# needs to reach the commit check behind it requires the frozen scientific
# interpreter. CI installs a different set and is refused at stage 1 -- which
# is correct behaviour, and `test_a_foreign_runtime_is_refused_first` covers
# it directly and runs everywhere. Same convention as the sibling harness
# suite.
ON_FROZEN_INTERPRETER = _observed_dependency_digest() == _frozen_dependency_digest()
requires_frozen_runtime = pytest.mark.skipif(
    not ON_FROZEN_INTERPRETER,
    reason=(
        "reaching the commit check requires the frozen scientific interpreter; "
        "this environment reports a different installed-package digest"
    ),
)


def _argv(sha: str) -> list[str]:
    return [SPEC.T1_CANONICAL_EXECUTION_FLAG, f"{SPEC.T1_EXPECTED_GIT_SHA_FLAG}={sha}"]


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


# ---------------------------------------------------------------------------
# What the change actually changed
# ---------------------------------------------------------------------------


def test_the_three_facts_stay_distinct():
    """Specification, capability and permission are still three constants.

    Collapsing any two of them is the failure this file guards. All three
    happen to be True now, which is exactly when a derived check would be
    indistinguishable from a deliberate one.
    """
    assert CFG.T1_EXECUTION_SPECIFICATION_EXISTS is True
    assert CFG.T1_CANONICAL_DEVELOPMENT_HARNESS_EXISTS is True
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is True


def test_authorization_was_false_before_this_change():
    """The gate this change opened was genuinely shut in the merged harness.

    Read from git rather than asserted from memory: if the constant had always
    been True, the enabling change would have been a no-op and every refusal
    test below would be proving nothing.
    """
    import subprocess

    reachable = subprocess.run(
        ["git", "cat-file", "-e", f"{CANONICAL_SHA}^{{commit}}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
    )
    if reachable.returncode != 0:
        pytest.skip(
            f"{CANONICAL_SHA[:12]} is not in this clone (shallow checkout); the "
            "proof runs where the history is complete"
        )
    blob = subprocess.run(
        ["git", "show", f"{CANONICAL_SHA}:src/cardiosentinel/neural/t1_config.py"],
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


def test_the_permission_gate_has_exactly_one_implementation():
    """One place asks the question, so it cannot be answered two ways."""
    source = Path(CFG.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    readers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "T1_EXECUTION_SPECIFICATION_AUTHORIZED"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(readers) == 1, (
        "the authorization constant is read in more than one place"
    )


def test_the_config_no_longer_claims_the_harness_is_missing():
    """The stale wording this change removed must not come back."""
    source = Path(CFG.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    for stale in (
        "what does not exist\n# is the canonical development harness",
        "the canonical development harness is implemented      -> false",
        "no such specification exists yet",
    ):
        assert stale not in lowered, f"stale wording is back: {stale!r}"
    assert "neither is a permission" in lowered
    assert "a harness is a capability" in lowered


def test_main_no_longer_refuses_unconditionally():
    """The old body raised before constructing anything; the new one verifies.

    Proven structurally: `main` now reaches `stage_preflight`, which is what
    carries the six checks. A body that raised on permission alone could not.
    """
    source = Path(R.__file__).read_text(encoding="utf-8")
    main_fn = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    called = {
        node.func.attr
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "require_canonical_execution_authorized" in called
    assert "stage_preflight" in called


# ---------------------------------------------------------------------------
# Permission remains a real gate
# ---------------------------------------------------------------------------


def test_the_entry_point_refuses_when_authorization_is_withdrawn(monkeypatch):
    """Withdrawing permission stops the run before any verification happens."""
    monkeypatch.setattr(CFG, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    with pytest.raises(CFG.T1ConfigError) as caught:
        R.main(_argv(CANONICAL_SHA))
    message = str(caught.value).lower()
    assert "not authorized" in message
    assert "neither is a permission" in message
    assert not _canonical_root().exists()


def test_a_canonical_config_refuses_when_authorization_is_withdrawn(monkeypatch):
    """The config loader asks the same single question the entry point does."""
    monkeypatch.setattr(CFG, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    with pytest.raises(CFG.T1ConfigError) as caught:
        CFG.require_canonical_execution_authorized()
    assert "contract" in str(caught.value).lower()


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
        R.main([f"{SPEC.T1_EXPECTED_GIT_SHA_FLAG}={CANONICAL_SHA}"])
    assert not _canonical_root().exists()


@requires_frozen_runtime
def test_a_wrong_sha_is_refused(monkeypatch):
    """HEAD is re-read and compared; the authorization names one commit."""
    monkeypatch.setattr(
        PERSIST,
        "git_provenance",
        lambda root: {"git_sha": CANONICAL_SHA, "git_dirty": False},
    )
    with pytest.raises(PERSIST.T1PersistenceError) as caught:
        R.main(_argv("0" * 40))
    message = str(caught.value)
    assert "authorized for" in message
    assert "0" * 40 in message
    assert not _canonical_root().exists()


@requires_frozen_runtime
def test_a_dirty_tree_is_refused(monkeypatch):
    """Canonical evidence requires a clean checkout, and nothing is repaired."""
    monkeypatch.setattr(
        PERSIST,
        "git_provenance",
        lambda root: {"git_sha": CANONICAL_SHA, "git_dirty": True},
    )
    with pytest.raises(PERSIST.T1PersistenceError) as caught:
        R.main(_argv(CANONICAL_SHA))
    message = str(caught.value).lower()
    assert "dirty" in message
    assert "retried" in message
    assert not _canonical_root().exists()


@requires_frozen_runtime
def test_the_sha_is_checked_before_any_upstream_or_row_access(monkeypatch):
    """A wrong SHA stops at stage 2, before the upstream chain is even read."""
    opened: list[str] = []
    monkeypatch.setattr(
        PERSIST,
        "git_provenance",
        lambda root: {"git_sha": CANONICAL_SHA, "git_dirty": False},
    )
    monkeypatch.setattr(
        R.T1DevelopmentRun,
        "_validate_m2",
        lambda self: opened.append("m2") or {},
    )
    with pytest.raises(PERSIST.T1PersistenceError):
        R.main(_argv("1" * 40))
    assert opened == [], "upstream evidence was read despite a wrong SHA"


# ---------------------------------------------------------------------------
# TEST stays sealed and the attempt stays unconsumed
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
    assert SPEC.T1_ATTEMPT_NAME_CARRIES_TIMESTAMP is False
    assert SPEC.T1_ATTEMPT_NAME_CARRIES_UUID is False
    assert SPEC.T1_ATTEMPT_NAME_CARRIES_RANDOM_SUFFIX is False
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


@pytest.mark.parametrize(
    "argv",
    [
        _argv("0" * 40),
        _argv(""),
        [SPEC.T1_CANONICAL_EXECUTION_FLAG],
    ],
)
def test_no_run_directory_is_created_on_any_refusal_path(argv, monkeypatch):
    """The attempt is a directory; a refusal must not leave one behind."""
    monkeypatch.setattr(
        PERSIST,
        "git_provenance",
        lambda root: {"git_sha": CANONICAL_SHA, "git_dirty": False},
    )
    existed = _canonical_root().exists()
    with pytest.raises(
        (
            PERSIST.T1PersistenceError,
            RuntimeIntegrityError,
            SystemExit,
            CFG.T1ConfigError,
        )
    ):
        R.main(argv)
    assert _canonical_root().exists() is existed
    assert not _canonical_root().exists()


def test_a_foreign_runtime_is_refused_first():
    """The runtime identity is stage 1, ahead of the commit check.

    Runs everywhere: the record is given an expected digest that matches no
    environment, so the refusal is a property of the check rather than of the
    machine. This is what the three commit-check tests above skip behind on a
    non-frozen interpreter.
    """
    run = R.T1DevelopmentRun(
        authorized_git_sha=CANONICAL_SHA,
        runtime=RuntimeIntegrityRecord(expected_digest="0" * 64),
    )
    with pytest.raises(RuntimeIntegrityError) as caught:
        run.stage_preflight()
    message = str(caught.value)
    assert "start" in message
    assert "NOT repaired" in message
    assert run.stages.entered == [SPEC.STAGE_START]
    assert not _canonical_root().exists()


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
    assert "human review" in message
