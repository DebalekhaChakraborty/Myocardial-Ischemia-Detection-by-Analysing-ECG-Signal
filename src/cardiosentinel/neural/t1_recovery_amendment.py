"""Provenance binding for the T1 Execution Recovery Amendment V1.1.

The amendment is a human governance decision. It permits **one** measurement
continuation of the canonical T1 attempt that was consumed by a post-claim
failure at stage 24 of 29, and it amends exactly three clauses to allow it:
specification §1 (one alternate run root, for one named identity), specification
§17 and protocol §14 ("once" and "exactly once" constrain decision-informing
evaluation, not evidence persistence).

**This module authorizes nothing and executes nothing.** It binds the document's
digest so that a continuation cannot run against an amendment that has moved,
and it holds the identity that amendment §7 names. There is no continuation
capability here, no gate, no runner and no attestation; those are a separate
reviewed change and this module must not grow into one.

**Why a module of its own.** The amendment's digest has to live in code before a
continuation may execute -- acceptance criterion §13.1 -- and the two modules
that already carry document digests, `t1_protocol.py` and `t1_execution_spec.py`,
are byte-frozen and pinned in seven suites each. Placing a constant in either
would spend the invariant that proves the frozen science has not changed, one
change before the run that depends on that proof. So the amendment's provenance
lives in a file named for the amendment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]

RECOVERY_AMENDMENT_VERSION: Final = "V1.1"
RECOVERY_AMENDMENT_NAME: Final = "T1_EXECUTION_RECOVERY_AMENDMENT_V1_1"
RECOVERY_AMENDMENT_PATH: Final = (
    REPOSITORY_ROOT / "docs" / f"{RECOVERY_AMENDMENT_NAME}.md"
)
RECOVERY_AMENDMENT_SHA256: Final = (
    "d3ea7734c93be8f59796e03e8c0210778716327f7adc033cb2d3dcfff7f92c96"
)

# The documents the amendment amends, and the clauses it touches. Bound here so
# a reader of the code can see the scope without opening the document, and so a
# future change to that scope is a change to this file.
AMENDED_CLAUSES: Final = {
    "T1_CANONICAL_DEVELOPMENT_EXECUTION_SPEC_V1": ("1", "17"),
    "T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1": ("14",),
}

# The identity amendment §7 names, and the attempt it continues. Neither exists
# on disk: naming them is not creating them, and this module creates nothing.
CONTINUATION_RUN_CLASS: Final = "t1_continuation_measurement"
CONTINUATION_ATTEMPT_ID: Final = "t1-v1-measurement-continuation"
CONTINUATION_RUN_ROOT_RELATIVE: Final = Path(
    "cardiosentinel-runs/phase9-t1-continuation-v1"
)
CONTINUED_ATTEMPT_ID: Final = "t1-v1-development"
CONTINUED_AUTHORIZED_GIT_SHA: Final = "c538181eb93884f4583a8bd328e50573efbcf3df"

# What the amendment narrowed, from §9.1 and §13.6. Recorded as data because the
# continuation must prove each of them zero, and a constraint that lives only in
# prose is a constraint the code can forget.
CONTINUATION_ZERO_COUNTERS: Final = (
    "state_machine_invocations",
    "threshold_generation_calls",
    "policy_selection_calls",
    "fold_evaluations",
)


class T1RecoveryAmendmentError(RuntimeError):
    """Raised when the frozen recovery amendment is missing or has moved."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_recovery_amendment_document(path: Path | None = None) -> str:
    """Verify the frozen recovery amendment byte-for-byte.

    The path is resolved at CALL time rather than bound as a default argument,
    for the same reason the sibling document validators do it: a default bound
    at definition time can never be reached by monkeypatching the module
    constant, which has caused real confusion in this repository before.

    A continuation that ran against an amendment which had moved would be a run
    whose permission is an argument rather than a fact, so this is a refusal
    rather than a warning.
    """
    document = Path(path) if path is not None else RECOVERY_AMENDMENT_PATH
    if not document.is_file():
        raise T1RecoveryAmendmentError(
            f"The T1 execution recovery amendment is missing at {document}. "
            "Continuation is permitted by a frozen human decision; without the "
            "document there is no permission to read."
        )
    digest = _sha256_file(document)
    if digest != RECOVERY_AMENDMENT_SHA256:
        raise T1RecoveryAmendmentError(
            f"The recovery amendment digest {digest} differs from the frozen "
            f"{RECOVERY_AMENDMENT_SHA256}. The amendment is immutable: it "
            "records a decision that was taken, and a decision that can be "
            "edited afterwards is not a decision."
        )
    return digest
