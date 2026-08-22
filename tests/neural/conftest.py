"""Package-wide guard: no test consumes or disturbs a canonical T1 attempt.

Thirty-eight assertions across eleven suites used to check this individually,
each by asserting the canonical run directory did not exist. That check stopped
being honest the moment the canonical attempt ran, and it only ever covered the
tests that remembered to write it.

One autouse fixture covers every test in this package instead, including the
ones nobody thought to guard, and states the invariant that survives execution:
the canonical attempt is exactly as this session found it.
"""

from __future__ import annotations

import pytest
from _attempt_guard import assert_attempt_unconsumed

from cardiosentinel.neural import t1_continuation_spec


@pytest.fixture(autouse=True, scope="session")
def continuation_is_disarmed_for_the_test_session():
    """Pytest may never execute the continuation, however the flag is committed.

    Arming `T1_CONTINUATION_AUTHORIZED` is an operator decision. Once it is True
    on disk, the refusal tests stop refusing at stage 1 and a runner call in a
    fresh interpreter could walk on toward the claim -- so a routine `pytest`
    could consume the single authorized attempt.

    The `_attempt_guard` fixture would notice afterwards, which is exactly the
    wrong time. So the flag is forced False for the session and restored after.
    Tests that need it armed patch it locally; the test asserting the repository
    is armed reads the committed source rather than this process's value.
    """
    original = t1_continuation_spec.T1_CONTINUATION_AUTHORIZED
    t1_continuation_spec.T1_CONTINUATION_AUTHORIZED = False
    try:
        yield
    finally:
        t1_continuation_spec.T1_CONTINUATION_AUTHORIZED = original


@pytest.fixture(autouse=True)
def canonical_attempt_is_not_consumed():
    """Fail any test that creates, extends, rewrites or deletes the attempt."""
    yield
    assert_attempt_unconsumed()
