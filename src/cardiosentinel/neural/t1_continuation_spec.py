"""Identity, permission and predecessor bindings for the T1 measurement continuation.

The consumed canonical attempt `t1-v1-development` failed post-claim at stage 24
of 29 with a key-vocabulary defect, after twelve folds of held-out measurement
had already completed in memory. Ten of twelve scientific components survive on
disk; three label-derived quantities do not. This module carries the identity
and the bindings under which one measurement continuation may recover them.

**What a continuation is, and is not.** It is not a retry: a retry would re-enter
the frozen state machine and produce a second trace that could differ from the
first. It is not a new experiment: a new experiment would select its own policy.
It is a *measurement* over evidence that already exists --

    frozen predictions + held-out labels -> measurement

-- which is why amendment §9.1 narrows the exercise below what §9 would permit,
and why §13.6 makes that narrowing mechanically provable rather than a
convention. The scientific claim continues to rest on the original immutable
state trace, so no question of determinism or floating-point reproducibility
arises: there is no second trace.

**Separate governance.** `T1_CONTINUATION_AUTHORIZED` is deliberately *not*
`t1_config.T1_EXECUTION_SPECIFICATION_AUTHORIZED`. Permission to run the
canonical development experiment and permission to measure over its remains are
different human decisions, taken at different times on different evidence, and a
continuation that read the canonical flag would inherit an authorization nobody
granted it. The canonical flag is still `True` on master; if the continuation
inherited it, the continuation would already be armed.

This module binds and refuses. It executes nothing, creates no directory and
opens no evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from cardiosentinel.neural.t1_recovery_amendment import (
    CONTINUATION_ATTEMPT_ID,
    CONTINUATION_RUN_CLASS,
    CONTINUATION_RUN_ROOT_RELATIVE,
    CONTINUED_ATTEMPT_ID,
    CONTINUED_AUTHORIZED_GIT_SHA,
    RECOVERY_AMENDMENT_NAME,
    RECOVERY_AMENDMENT_SHA256,
)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Permission -- its own decision, never the canonical one
# ---------------------------------------------------------------------------

#: Whether a human has authorized the continuation to execute.
#:
#: False here means the continuation refuses at its first gate, having touched
#: nothing. Flipping it is a governance act on the same footing as the amendment
#: itself, and it is deliberately the only edit that arms this path.
T1_CONTINUATION_AUTHORIZED: Final = False

#: The continuation is authorized once. There is no second attempt predeclared,
#: and no identity is reserved for one (amendment §14).
T1_CONTINUATION_ATTEMPTS_AUTHORIZED: Final = 1
T1_CONTINUATION_AUTOMATIC_RETRY_PERMITTED: Final = False
T1_CONTINUATION_MAY_BE_DELETED_OR_REWRITTEN: Final = False

# ---------------------------------------------------------------------------
# Identity (amendment §7)
# ---------------------------------------------------------------------------

CONTINUATION_RUN_ROOT: Final = REPOSITORY_ROOT / CONTINUATION_RUN_ROOT_RELATIVE
CONTINUATION_ATTEMPT_DIR_RELATIVE: Final = (
    CONTINUATION_RUN_ROOT_RELATIVE / CONTINUATION_ATTEMPT_ID
)

#: Prefixes the continuation identity may never begin with, matched
#: case-insensitively. `t1-v1-development-continuation` reads as a natural name
#: and is exactly the one that must be refused: a name that begins with the
#: consumed attempt's identity claims to *be* that attempt continued in place,
#: and the whole point of a separate run root is that it is not.
RESERVED_IDENTITY_PREFIXES: Final = (
    "t1-v1-development",
    "phase9-t1-development-v1",
)


class T1ContinuationIdentityError(RuntimeError):
    """Raised when a continuation identity collides with a reserved prefix."""


class T1ContinuationPermissionError(RuntimeError):
    """Raised when the continuation is not authorized to execute.

    Distinct from the identity error and from the capability error on purpose.
    "You may not run", "you are not who you say you are" and "this graph could
    not finish" are three different refusals, and a caller that conflated them
    would read a withheld permission as a broken implementation.
    """


def require_continuation_identity(attempt_id: str, run_root: str | Path) -> str:
    """Refuse any identity that reaches into canonical reserved namespace.

    Checked on the attempt id *and* the run root, because a continuation writing
    into the canonical root under a fresh attempt name would be just as much an
    extension of the consumed attempt as one reusing its name.
    """
    lowered = str(attempt_id).strip().lower()
    for prefix in RESERVED_IDENTITY_PREFIXES:
        if lowered.startswith(prefix):
            raise T1ContinuationIdentityError(
                f"Continuation attempt id {attempt_id!r} begins with the "
                f"reserved canonical prefix {prefix!r}. The continuation has "
                f"its own identity, {CONTINUATION_ATTEMPT_ID!r}: a name that "
                "extends the consumed attempt's name claims to be that attempt "
                "carried on, and it is not."
            )
    root = Path(run_root)
    for part in root.parts:
        if part.lower().startswith("phase9-t1-development-v1"):
            raise T1ContinuationIdentityError(
                f"Continuation run root {root} reaches into the canonical "
                "development run root. The consumed attempt directory is "
                "immutable and is not extended."
            )
    return str(attempt_id)


def require_continuation_authorized() -> None:
    """Refuse unless a human has armed the continuation. Touches nothing."""
    if not T1_CONTINUATION_AUTHORIZED:
        raise T1ContinuationPermissionError(
            "The T1 measurement continuation is not authorized. "
            f"{RECOVERY_AMENDMENT_NAME} permits one continuation; arming it is "
            "a separate, deliberate governance act recorded by setting "
            "T1_CONTINUATION_AUTHORIZED. This refusal happens before any "
            "artifact is resolved, any directory is created and any evidence "
            "is opened."
        )


# ---------------------------------------------------------------------------
# Predecessor bindings (amendment §1.3 and §1.4)
# ---------------------------------------------------------------------------

CONSUMED_RUN_ROOT_RELATIVE: Final = Path("cardiosentinel-runs/phase9-t1-development-v1")
CONSUMED_ATTEMPT_DIR_RELATIVE: Final = CONSUMED_RUN_ROOT_RELATIVE / CONTINUED_ATTEMPT_ID
CONSUMED_ATTEMPT_DIR: Final = REPOSITORY_ROOT / CONSUMED_ATTEMPT_DIR_RELATIVE

#: §1.3 file digests. `T1_RUN_STATUS.json` is bound too: it reads `STARTED` with
#: every label-access flag false, which was true at the claim and never became
#: true. It is left exactly as the run wrote it, so its digest is stable and
#: binding it proves the attempt was not tidied.
PREDECESSOR_FILE_DIGESTS: Final = {
    "T1_PREFLIGHT.json": (
        "917b5421c9c7731eb185821ed279564c65fed5737153316cfa410811ea4f25da"
    ),
    "T1_RUN_STATUS.json": (
        "f305da7ad3d465c4500124fe4d4422dfc471580a01afe7b9d424e866e9e2c59d"
    ),
    "T1_INPUT_LINEAGE.json": (
        "e307bdd3ad244f6440ad437f66d5f7b4e2af3072b6b1833e74552095ede3c555"
    ),
    "T1_INPUT_EVIDENCE.json": (
        "bf36ac0e538b0cee61a97109de413c52ec942356d974930e5de64bc32b86423b"
    ),
    "t1_input_evidence.npz": (
        "4391b4e7cda5ac5d70c93663563cc37954afdfc7b28092ef65c2d351006c2f5c"
    ),
    "T1_FOLD_SELECTIONS.json": (
        "71e0da62ad2a86fd6bb2561137e0a152df2d5b894bd9fecfb67ad762a5682f6d"
    ),
    "T1_OOF_STATE_EVIDENCE.json": (
        "aefc922a5224b7c857b9bf99b12441e55e46fdc71def373c043ffb112e5e2405"
    ),
    "t1_oof_state_evidence.npz": (
        "72f13a8b29eafdd99801bb64dbf8b61f19717f3d7af777d74f21c9709dd28232"
    ),
}

#: Canonical payload self-digests. These are digests of the canonical JSON of a
#: store's *contents*, not of the file on disk, and they differ from the file
#: digests by design. Conflating the two has cost this programme time before.
PREDECESSOR_CONTENT_DIGESTS: Final = {
    "input_evidence": (
        "57d434d9b4eee9fa3d37f581397d89aca6a5bbd3188aa35f907801766be6a8ac"
    ),
    "oof_state_evidence": (
        "cf74f00a6eb38471e80ce008dc6b88d16aa5c36b110bce87c7c37dba6d7d835f"
    ),
    "oof_fold_selection_binding": (
        "32bab16ca6ec4d8ab7d3b6f2d9a3c8782ae97f3e58a84eb900357df1d881451d"
    ),
}

#: The array digest the measured trace must carry (§13.6 Layer 3).
PREDECESSOR_OOF_ARRAY_SHA256: Final = PREDECESSOR_FILE_DIGESTS[
    "t1_oof_state_evidence.npz"
]
PREDECESSOR_OOF_CONTENT_SHA256: Final = PREDECESSOR_CONTENT_DIGESTS[
    "oof_state_evidence"
]

#: §1.4, fold index -> (held-out subject, selected policy, selection file digest).
PREDECESSOR_FOLD_SELECTIONS: Final = {
    0: (
        "ltstdb:s2004",
        "qw0.9_qe0.99_FAST",
        "02ffccd4eb546a7d07017f7234aec9f3c3f189819f4f90ca5663e1d4cf11467c",
    ),
    1: (
        "ltstdb:s2005",
        "qw0.9_qe0.99_BALANCED",
        "f08799a205200c7b1d22a26f1d8354848149828c4eb6e68beb51c6eebde5a786",
    ),
    2: (
        "ltstdb:s2019",
        "qw0.9_qe0.99_FAST",
        "e5d4967a45eb891a11294d640ab6a5e5de77cffdb60cf0c1338be5ff8e3558a1",
    ),
    3: (
        "ltstdb:s2020",
        "qw0.9_qe0.99_FAST",
        "daa0e1def15d45cc826516b8478369c92755ec77634429014580161ed7d6d7ed",
    ),
    4: (
        "ltstdb:s2023",
        "qw0.9_qe0.99_FAST",
        "fa3cec3519513d7681100bb701f38d988af3bacb504cb9dc4702bd6432559dc0",
    ),
    5: (
        "ltstdb:s2031",
        "qw0.9_qe0.99_FAST",
        "6c07098b90548fe03eddf8437ac56bedccd3b1a39abdaa90aa077694b2fb0d0f",
    ),
    6: (
        "ltstdb:s2057",
        "qw0.9_qe0.99_FAST",
        "c9c1b0fb345693ff07f95073d92afb3cacaa72cfb05e290c36dd07ee2c5a6c9a",
    ),
    7: (
        "ltstdb:s2058",
        "qw0.9_qe0.99_FAST",
        "9a6ee4e4e33372e5208234c38c7830015ddccb23ee166140a1a78c83eb68a72d",
    ),
    8: (
        "ltstdb:s2059",
        "qw0.9_qe0.99_FAST",
        "602c9d1f09a3af4f46b234e483592ccb3eb56a9f78251f95f030ce150630a07e",
    ),
    9: (
        "ltstdb:s3068",
        "qw0.9_qe0.99_FAST",
        "31bca60e10377c7bfc77f9fb6a9c54340b6b91e9c048b006f8e198781df99961",
    ),
    10: (
        "ltstdb:s3072",
        "qw0.9_qe0.99_FAST",
        "696e99527caf33c9798721dffd92d12a5d98ef98720579f006c9a96aae4c26a8",
    ),
    11: (
        "ltstdb:s3073",
        "qw0.9_qe0.99_FAST",
        "3384a1261c7e069d8276eb5fe35a66dd7589c953fd37d3ae902ab0e496e03050",
    ),
}

PREDECESSOR_FOLD_COUNT: Final = 12


def fold_selection_relative_path(fold_index: int) -> Path:
    return Path("fold_selections") / f"T1_FOLD_{int(fold_index):02d}_SELECTION.json"


# ---------------------------------------------------------------------------
# The attestation contract (amendment §13.7)
# ---------------------------------------------------------------------------

CONTINUATION_ATTESTATION_NAME: Final = "T1_V1_CONTINUATION_EXECUTION_ATTESTATION.json"
CONTINUATION_ATTESTATION_CLASS: Final = "t1_v1_continuation_execution_attestation"

#: The four counters §13.6 requires to read zero, in the amendment's own
#: vocabulary. The names are the contract: an attestation whose keys do not
#: match what a reader looks for is exactly the defect that consumed the
#: canonical attempt -- `KeyError: 'true_positive'` at stage 24, after the
#: claim, because a producer said `tp` and a consumer said `true_positive`.
#: These names are re-exported from the amendment module rather than retyped.
CONTINUATION_ZERO_COUNTERS: Final = (
    "state_machine_invocations",
    "threshold_generation_calls",
    "policy_selection_calls",
    "fold_evaluations",
)

#: Every field §13.7 requires the attestation to carry, at minimum.
CONTINUATION_ATTESTATION_REQUIRED_FIELDS: Final = (
    "artifact_class",
    *CONTINUATION_ZERO_COUNTERS,
    "state_trace_source",
    "state_trace_content_sha256",
    "state_trace_array_sha256",
    "selection_performed_here",
    "thresholds_generated_here",
    "state_transitions_regenerated",
    "predecessor_digests_verified",
    "test_accessed",
    "sealed_test_state",
)

CONTINUATION_STATE_TRACE_SOURCE: Final = "predecessor_oof_state_evidence"

#: Fields whose value is fixed by the amendment, checked on promotion.
CONTINUATION_ATTESTATION_FIXED_VALUES: Final = {
    "artifact_class": CONTINUATION_ATTESTATION_CLASS,
    "state_machine_invocations": 0,
    "threshold_generation_calls": 0,
    "policy_selection_calls": 0,
    "fold_evaluations": 0,
    "state_trace_source": CONTINUATION_STATE_TRACE_SOURCE,
    "state_trace_content_sha256": PREDECESSOR_OOF_CONTENT_SHA256,
    "state_trace_array_sha256": PREDECESSOR_OOF_ARRAY_SHA256,
    "selection_performed_here": False,
    "thresholds_generated_here": False,
    "state_transitions_regenerated": False,
    "predecessor_digests_verified": True,
    "test_accessed": False,
    "sealed_test_state": "unopened",
}

#: A counter no continuation artifact may carry, because no policy was run.
#: Named so a test can prove its absence rather than trusting it.
FORBIDDEN_CONTINUATION_FIELDS: Final = ("policy_runs",)


def continuation_identity() -> dict[str, object]:
    """The identity block every continuation artifact carries. Data, not action."""
    return {
        "run_class": CONTINUATION_RUN_CLASS,
        "attempt_id": CONTINUATION_ATTEMPT_ID,
        "run_root": str(CONTINUATION_RUN_ROOT_RELATIVE),
        "continues_attempt_id": CONTINUED_ATTEMPT_ID,
        "continues_authorized_git_sha": CONTINUED_AUTHORIZED_GIT_SHA,
        "governing_amendment": RECOVERY_AMENDMENT_NAME,
        "governing_amendment_sha256": RECOVERY_AMENDMENT_SHA256,
        "attempts_authorized": T1_CONTINUATION_ATTEMPTS_AUTHORIZED,
        "automatic_retry_permitted": T1_CONTINUATION_AUTOMATIC_RETRY_PERMITTED,
        "is_continuation_artifact": True,
        "is_recovery_artifact": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
