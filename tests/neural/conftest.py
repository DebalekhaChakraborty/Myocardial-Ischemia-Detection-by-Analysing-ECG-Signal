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


@pytest.fixture(autouse=True)
def canonical_attempt_is_not_consumed():
    """Fail any test that creates, extends, rewrites or deletes the attempt."""
    yield
    assert_attempt_unconsumed()
