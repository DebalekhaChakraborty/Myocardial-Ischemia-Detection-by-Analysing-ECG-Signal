"""The guard that replaced thirty-eight stale assertions, and its tripwire.

Until the canonical attempt ran, eleven T1 suites proved they consumed nothing
by asserting the canonical run directory did not exist. That assertion was true
by accident of history and became permanently false the moment the science ran:
the run directory is now immutable evidence.

`_attempt_guard.assert_attempt_unconsumed` states the property those assertions
were written for -- the canonical attempt is exactly as this session found it --
and `conftest.py` applies it after every test in this package, including the
tests nobody thought to guard.

This file proves the guard actually detects a change, and that the stale pattern
cannot come back.
"""

from __future__ import annotations

import ast
from pathlib import Path

import _attempt_guard as GUARD
import pytest
from _attempt_guard import assert_attempt_unconsumed

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. The guard detects what it claims to detect
# ---------------------------------------------------------------------------


def test_the_guard_passes_when_nothing_changed():
    assert_attempt_unconsumed()


def test_the_guard_fails_when_the_attempt_gains_a_file(monkeypatch, tmp_path):
    """A test that wrote into the canonical attempt must not pass quietly."""
    attempt = tmp_path / "t1-v1-development"
    attempt.mkdir()
    (attempt / "T1_RESULT.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(GUARD, "CANONICAL_ATTEMPT", attempt)
    monkeypatch.setattr(GUARD, "SESSION_BASELINE", ((False, ()), (False, ())))
    with pytest.raises(AssertionError, match="canonical T1 attempt changed"):
        GUARD.assert_attempt_unconsumed()


def test_the_guard_fails_when_a_promoted_artifact_changes_size(monkeypatch, tmp_path):
    attempt = tmp_path / "t1-v1-development"
    attempt.mkdir()
    artifact = attempt / "T1_OOF_STATE_EVIDENCE.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(GUARD, "CANONICAL_ATTEMPT", attempt)
    monkeypatch.setattr(GUARD, "SESSION_BASELINE", GUARD.attempt_fingerprint())
    artifact.write_text('{"rewritten": true}', encoding="utf-8")
    with pytest.raises(AssertionError, match="canonical T1 attempt changed"):
        GUARD.assert_attempt_unconsumed()


def test_the_guard_watches_the_continuation_namespace_too(monkeypatch, tmp_path):
    """The continuation attempt does not exist yet and must not appear unnoticed."""
    continuation = tmp_path / "phase9-t1-continuation-v1"
    monkeypatch.setattr(GUARD, "CONTINUATION_ROOT", continuation)
    monkeypatch.setattr(GUARD, "SESSION_BASELINE", GUARD.attempt_fingerprint())
    (continuation / "t1-v1-measurement-continuation").mkdir(parents=True)
    with pytest.raises(AssertionError, match="continuation run directory changed"):
        GUARD.assert_attempt_unconsumed()


def test_the_fingerprint_covers_both_namespaces():
    """The fingerprint reports both namespaces, whichever exists here.

    The consumed attempt is gitignored and local-only: present on the frozen
    interpreter where the science ran, absent on CI. Asserting it is present
    would be this module's own mistake facing the other way, so the assertion
    is on the shape of the answer and on the namespace that must be empty
    everywhere.
    """
    canonical, continuation = GUARD.attempt_fingerprint()
    assert canonical[0] is GUARD.ATTEMPT_PRESENT
    assert isinstance(canonical[1], tuple)
    assert continuation[0] is False, "a continuation run directory exists"
    assert continuation[1] == ()


def test_the_guard_is_applied_to_every_test_in_this_package():
    """The fixture is autouse, so no suite has to remember to call it."""
    conftest = ast.parse((TEST_DIR / "conftest.py").read_text(encoding="utf-8"))
    fixtures = [
        node
        for node in ast.walk(conftest)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and any(
            keyword.arg == "autouse" and keyword.value.value is True
            for keyword in decorator.keywords
        )
    ]
    assert len(fixtures) == 1, "the package guard is not a single autouse fixture"
    called = {
        node.func.id
        for node in ast.walk(fixtures[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "assert_attempt_unconsumed" in called


# ---------------------------------------------------------------------------
# 2. The stale pattern cannot come back
# ---------------------------------------------------------------------------


def _asserts_bare_absence(path: Path) -> list[int]:
    """Line numbers of `assert not <...canonical...>.exists()`.

    Syntax tree rather than text: the prose in these suites discusses the
    canonical run directory constantly, and a substring scan reports every
    sentence that mentions it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
            continue
        call = test.operand
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "exists"
        ):
            continue
        receiver = ast.dump(call.func.value)
        if "canonical_root" in receiver or "canonical_run_directory" in receiver:
            found.append(node.lineno)
    return found


@pytest.mark.parametrize(
    "path", sorted(TEST_DIR.glob("test_t1_*.py")), ids=lambda p: p.name
)
def test_no_suite_asserts_the_canonical_attempt_is_absent(path):
    """It is not absent. It ran, it failed after the claim, and it is evidence.

    A suite that asserts otherwise fails on the one machine where the science
    actually happened, and passes on CI only because the run directory is
    gitignored -- a test that is green remotely and red locally teaches people
    to ignore the suite.
    """
    offending = _asserts_bare_absence(path)
    assert offending == [], (
        f"{path.name} asserts the canonical attempt does not exist at lines "
        f"{offending}. Use assert_attempt_unconsumed(): it proves this test "
        "consumed nothing, which is what the assertion was written for and "
        "stays true now that the attempt has been consumed."
    )


def test_every_suite_that_guards_the_attempt_imports_the_guard():
    users = [
        path
        for path in sorted(TEST_DIR.glob("test_t1_*.py"))
        if "assert_attempt_unconsumed(" in path.read_text(encoding="utf-8")
    ]
    assert len(users) >= 11, "the guard is used by fewer suites than expected"
    for path in users:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "_attempt_guard"
            for alias in node.names
        }
        assert "assert_attempt_unconsumed" in imported, path.name
