"""Proof that a test consumed no canonical T1 attempt.

Until 2026-08-21 the T1 suites proved this by asserting the canonical run
directory did not exist. That was true when those tests were written and it is
permanently false now: the canonical attempt ran, failed at stage 24, and its
directory is immutable evidence that must never be deleted.

The assertion conflated two different claims -- "no run has ever happened" and
"this test consumed nothing" -- and only the second was ever the point. A test
that asserts the first is a test that must fail on a machine where the science
actually ran, which is precisely the machine whose results matter.

So the invariant here is the honest one: **the canonical attempt is exactly as
this session found it.** That holds before the run and after it, on the frozen
interpreter and on CI, and it still fails loudly the moment a test creates,
extends or disturbs a canonical attempt -- which is the tripwire the original
assertions were fired to be.

The fingerprint is deliberately cheap: existence, and the name and size of every
file. It runs after every test in this package via the autouse fixture in
``conftest.py``, so a test does not have to remember to check.
"""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# The two canonical namespaces. The second does not exist yet; naming it here
# means a continuation attempt cannot be created by a test unnoticed either.
CANONICAL_ATTEMPT = (
    REPOSITORY_ROOT
    / "cardiosentinel-runs"
    / "phase9-t1-development-v1"
    / "t1-v1-development"
)
CONTINUATION_ROOT = (
    REPOSITORY_ROOT / "cardiosentinel-runs" / "phase9-t1-continuation-v1"
)


def _fingerprint(root: Path) -> tuple:
    """Existence, plus every file's path and size. Cheap enough to run per test."""
    if not root.exists():
        return (False, ())
    return (
        True,
        tuple(
            sorted(
                (str(path.relative_to(root)), path.stat().st_size)
                for path in root.rglob("*")
                if path.is_file()
            )
        ),
    )


def attempt_fingerprint() -> tuple:
    return (_fingerprint(CANONICAL_ATTEMPT), _fingerprint(CONTINUATION_ROOT))


# Captured once, at import, before any test in this package has run.
SESSION_BASELINE = attempt_fingerprint()


def assert_attempt_unconsumed() -> None:
    """The canonical attempt is exactly as this session found it.

    Replaces `assert not canonical_run_directory().exists()`. It proves the
    property that assertion was written for, and keeps proving it after the
    canonical attempt has been consumed.
    """
    observed = attempt_fingerprint()
    if observed == SESSION_BASELINE:
        return
    canonical_before, continuation_before = SESSION_BASELINE
    canonical_now, continuation_now = observed
    if canonical_now != canonical_before:
        raise AssertionError(
            "the canonical T1 attempt changed during this test. It is "
            "claim-bearing evidence: not created, extended, rewritten or "
            f"deleted by any test.\n  before: {canonical_before}\n  after:  "
            f"{canonical_now}"
        )
    raise AssertionError(
        "a T1 continuation run directory changed during this test.\n"
        f"  before: {continuation_before}\n  after:  {continuation_now}"
    )
