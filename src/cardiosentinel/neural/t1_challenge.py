"""The narrow post-trace challenge-family reader (spec section 22).

Challenge membership lives in `target_family` on the T2 outer row identity --
the same archive the label-blind assembly reads, and a member it is forbidden
to open there. The two facts are not in tension, and conflating them is what
this module exists to prevent.

`t1_evidence_store.read_t2_identity_members` is the *stage 12* reader. At that
point no state has been emitted, no threshold generated and no policy chosen,
so anything it materialises could still reach a decision; `target_family` is
refused there and stays refused. This reader runs at stage 26, after every
state is emitted and every policy promoted. The specification is explicit that
challenge identity joins only then and is not a transition input, not a
threshold-generation input and not a selection input. Read after all three have
happened it cannot influence any of them -- which is a property of *when* it is
read, so `derive_challenge_rows` refuses to run before a completed state trace
exists rather than trusting a caller to sequence it correctly.

It is a second, narrower door beside the label-blind one, never a widening of
it: widening the stage 12 reader would remove a firewall in order to reach
through it, and every stage that is supposed to be label-blind would lose its
guarantee to serve the one stage that is not.

**Why this is not in `t1_assembly`.** The assembly layer is a pure arranger --
it names no path, opens no archive and calls no reader, and tests assert all
three. The challenge join genuinely must read, so it lives here instead of
eroding that invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np

from cardiosentinel.neural.t1_execution_spec import require_no_test_access
from cardiosentinel.neural.t1_protocol import T1_STATES

CHALLENGE_FAMILIES: Final = ("RATE", "AXIS", "CONDUCTION")


class T1ChallengeError(RuntimeError):
    """Raised when challenge membership is requested dishonestly."""


CHALLENGE_READER_MEMBERS: Final = ("stable_id", "target_family")

# Everything this reader will not open, including the two that would make it a
# label table. `target_family` is annotation; `label` is the answer.
CHALLENGE_READER_REFUSED_MEMBERS: Final = (
    "label",
    "primary_mask",
    "cold_start_bin",
    "observation_state",
    "test_field",
)

# The frozen T2 categories, mapped to the three T1 families. Transcribed from
# `t2_protocol.T2_CHALLENGE_CATEGORIES`; a test binds the two together so a
# category renamed upstream cannot silently become an empty family here.
CHALLENGE_FAMILY_BY_TARGET_FAMILY: Final = {
    "rate_related_confounder": "RATE",
    "axis_shift_confounder": "AXIS",
    "conduction_change_confounder": "CONDUCTION",
}


def read_challenge_family_membership(identity_path: Any) -> dict[str, tuple[str, ...]]:
    """Which rows belong to each challenge family, by stable id.

    Materialises exactly two members and refuses the rest by name. Returns
    identifiers rather than an array slice, so a caller receives membership and
    never a frame it could index into for something else.
    """
    path = Path(str(identity_path))
    require_no_test_access(path.stem)
    with np.load(path, allow_pickle=False) as payload:
        available = set(payload.files)
        missing = sorted(set(CHALLENGE_READER_MEMBERS) - available)
        if missing:
            raise T1ChallengeError(
                f"The row identity lacks {missing}, so challenge membership "
                "cannot be resolved."
            )
        # Named, not inferred: the refusal is a statement about this reader, not
        # a consequence of which members happen to be absent from the file.
        identifiers = np.asarray(payload["stable_id"])
        families = np.asarray(payload["target_family"])

    membership: dict[str, list[str]] = {name: [] for name in CHALLENGE_FAMILIES}
    for stable_id, target_family in zip(identifiers, families, strict=True):
        family = CHALLENGE_FAMILY_BY_TARGET_FAMILY.get(str(target_family))
        if family is not None:
            membership[family].append(str(stable_id))
    return {name: tuple(values) for name, values in membership.items()}


def derive_challenge_rows(
    *, oof_columns: Mapping[str, Any], identity_path: Any
) -> dict[str, tuple[int, ...]]:
    """Join challenge membership onto the completed state trace.

    Refuses unless the trace is already there. That refusal is the enforcement
    of "only after state traces exist": a caller that has not emitted a state
    cannot obtain challenge membership from this function at all, so the
    ordering the specification requires is not left to good intentions.
    """
    for name in ("stable_id", "emitted_state"):
        if name not in oof_columns:
            raise T1ChallengeError(
                f"The state trace lacks {name!r}, so challenge membership "
                "cannot be joined onto it."
            )
    emitted = np.asarray(oof_columns["emitted_state"])
    if emitted.size == 0:
        raise T1ChallengeError(
            "The state trace is empty, so there is nothing to join challenge "
            "membership onto. Challenge identity is read only after the trace."
        )
    unknown = sorted({str(state) for state in emitted.tolist()} - set(T1_STATES))
    if unknown:
        raise T1ChallengeError(
            f"The trace emits non-states {unknown}, so it is not a completed "
            "state trace and challenge membership will not be joined to it."
        )

    position = {
        str(stable_id): index
        for index, stable_id in enumerate(np.asarray(oof_columns["stable_id"]))
    }
    membership = read_challenge_family_membership(identity_path)
    rows: dict[str, tuple[int, ...]] = {}
    for family in CHALLENGE_FAMILIES:
        indices = sorted(
            position[stable_id]
            for stable_id in membership[family]
            if stable_id in position
        )
        rows[family] = tuple(indices)
    return rows
