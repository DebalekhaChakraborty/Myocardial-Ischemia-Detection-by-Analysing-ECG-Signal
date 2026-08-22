"""Continuation-specific evidence promotion, kept separate from the canonical path.

The canonical §17 path in `t1_persistence.promote_held_out_evaluation` **requires**
a `policy_runs` counter. Amendment §13.6 Layer 3 states that

    no continuation artifact carries a `policy_runs` counter, because no policy
    was run

so the two contracts are not merely different, they are mutually unsatisfiable:
supply the field and the artifact contradicts the amendment; omit it and the
canonical promoter refuses. A continuation could not persist a single fold
through the canonical path.

That is the same shape as the defect that consumed the canonical attempt -- a
producer and a consumer disagreeing about a key -- and it is resolved here in
the direction that changes nothing already frozen. `t1_persistence.py` is
**not modified**: it is digest-pinned in four suites, and re-cutting those pins
immediately before the single authorized run would spend an invariant to save a
branch. The continuation gets its own promoter instead, and the two contracts
stay visibly distinct, which is what they are.

**What is shared and what is not.** The mechanics are shared: atomic write,
re-read verification, digest return. The *contract* is not. A canonical artifact
answers "what did this policy run produce"; a continuation artifact answers "what
does the persisted trace measure against these labels", and it must additionally
name what it continues and what it consumed.

This module writes only into the continuation run root. It cannot address the
consumed attempt directory, and a path that resolves outside the continuation
root is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Mapping

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.t1_continuation_spec import (
    CONSUMED_ATTEMPT_DIR_RELATIVE,
    CONTINUATION_ATTEMPT_ID,
    CONTINUATION_RUN_ROOT,
    FORBIDDEN_CONTINUATION_FIELDS,
    continuation_identity,
)

HELD_OUT_DIR: Final = "held_out_evaluations"

CONTINUATION_HELD_OUT_CLASS: Final = "t1_v1_continuation_held_out_evaluation_evidence"

#: The canonical §17 field set, minus `policy_runs`, plus the two provenance
#: blocks amendment §8 requires. Written out in full rather than derived from
#: the canonical tuple: deriving it would couple the continuation contract to a
#: frozen one, so a later change to either would silently move the other.
CONTINUATION_HELD_OUT_REQUIRED_FIELDS: Final = (
    "artifact_class",
    "attempt_id",
    "authorized_git_sha",
    "fold_index",
    "held_out_subject",
    "selected_policy_id",
    "policy",
    "thresholds",
    "primary_confusion",
    "episode_evidence",
    "onset_latency_seconds",
    "fold_selection_sha256",
    "generated_during_canonical_execution",
    "is_recovery_artifact",
    "is_continuation_artifact",
    "test_accessed",
    "continues",
    "consumed_evidence",
)

#: Required inside the `continues` block, so a reader can reconstruct the input
#: set without trusting this run's summary of it.
CONTINUES_REQUIRED_KEYS: Final = (
    "predecessor_run",
    "predecessor_digest",
    "governing_amendment",
    "governing_amendment_sha256",
)


class T1ContinuationPersistenceError(RuntimeError):
    """Raised when continuation evidence is incomplete, untrue or misplaced."""


def _require_inside_continuation_root(path: Path) -> Path:
    """Refuse any write that resolves outside the continuation run root.

    The consumed attempt is immutable and the canonical run root is not extended.
    Checked on the resolved path rather than the string, so a traversal cannot
    walk out through `..`.
    """
    resolved = path.resolve()
    root = CONTINUATION_RUN_ROOT.resolve()
    if root != resolved and root not in resolved.parents:
        raise T1ContinuationPersistenceError(
            f"{resolved} is outside the continuation run root {root}. The "
            "continuation writes only into its own namespace; the consumed "
            "attempt is immutable and is not extended."
        )
    if str(CONSUMED_ATTEMPT_DIR_RELATIVE) in str(resolved):
        raise T1ContinuationPersistenceError(
            f"{resolved} reaches into the consumed canonical attempt."
        )
    return resolved


def validate_continuation_held_out_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse evidence that is incomplete, carries `policy_runs`, or lies.

    Three distinct refusals, deliberately not collapsed into one:

    * **missing** a required field -- the artifact cannot answer §17;
    * carrying **`policy_runs`** -- it reports a quantity that cannot exist,
      because no policy was run here. §13.6 Layer 3 names that absence as
      evidence, so the key's presence contradicts the claim even at zero;
    * a **false identity flag** -- an artifact that says it was generated during
      canonical execution, or that it is not a continuation artifact, is
      describing a different run.
    """
    missing = [
        field for field in CONTINUATION_HELD_OUT_REQUIRED_FIELDS if field not in payload
    ]
    if missing:
        raise T1ContinuationPersistenceError(
            f"Continuation held-out evidence is missing {missing}. Incomplete "
            "evidence is refused rather than promoted: an artifact that exists "
            "but cannot answer the specification is worse than one that is absent."
        )

    present = [field for field in FORBIDDEN_CONTINUATION_FIELDS if field in payload]
    if present:
        raise T1ContinuationPersistenceError(
            f"Continuation held-out evidence carries {present}. No policy was "
            "run here, so that counter has no meaning; amendment §13.6 Layer 3 "
            "requires its absence, not a zero."
        )

    if payload["is_continuation_artifact"] is not True:
        raise T1ContinuationPersistenceError(
            "Continuation evidence must record is_continuation_artifact: true."
        )
    if payload["generated_during_canonical_execution"] is not False:
        raise T1ContinuationPersistenceError(
            "Continuation evidence was not generated during canonical execution."
        )
    if payload["test_accessed"] is not False:
        raise T1ContinuationPersistenceError(
            "Continuation evidence must record test_accessed: false."
        )

    continues = payload["continues"]
    if not isinstance(continues, Mapping):
        raise T1ContinuationPersistenceError("`continues` must be a mapping.")
    absent = [key for key in CONTINUES_REQUIRED_KEYS if key not in continues]
    if absent:
        raise T1ContinuationPersistenceError(
            f"The `continues` block is missing {absent}. A continuation that "
            "cannot name what it continues is a new experiment."
        )

    consumed = payload["consumed_evidence"]
    if not isinstance(consumed, (list, tuple)) or not consumed:
        raise T1ContinuationPersistenceError(
            "`consumed_evidence` must be a non-empty sequence: a continuation "
            "that names no consumed artifact read nothing it can be checked on."
        )
    for entry in consumed:
        if not isinstance(entry, Mapping) or {"artifact", "sha256"} - set(entry):
            raise T1ContinuationPersistenceError(
                f"Malformed consumed_evidence entry {entry!r}; each needs "
                "`artifact` and `sha256`."
            )
    return dict(payload)


def build_continuation_held_out_evidence(
    measurement: Mapping[str, Any],
    *,
    authorized_git_sha: str,
    fold_selection_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one fold's continuation evidence from a measurement result.

    `measurement` is `FoldMeasurement.as_dict()`. Nothing is computed here; this
    only names what was measured, under the identity the amendment fixed.
    """
    identity = continuation_identity()
    payload: dict[str, Any] = {
        "artifact_class": CONTINUATION_HELD_OUT_CLASS,
        "attempt_id": CONTINUATION_ATTEMPT_ID,
        "authorized_git_sha": str(authorized_git_sha),
        "fold_index": int(measurement["fold_index"]),
        "held_out_subject": measurement["held_out_subject"],
        "selected_policy_id": measurement["selected_policy_id"],
        "policy": measurement["selected_policy_id"],
        "thresholds": dict(measurement["thresholds"]),
        "primary_confusion": dict(measurement["primary_confusion"]),
        "episode_evidence": dict(measurement["episode_evidence"]),
        "onset_latency_seconds": list(measurement["onset_latency_seconds"]),
        "stream_count": int(measurement["stream_count"]),
        "fold_selection_sha256": str(fold_selection_sha256),
        "generated_during_canonical_execution": False,
        "is_recovery_artifact": True,
        "is_continuation_artifact": True,
        "test_accessed": False,
        "sealed_test_state": identity["sealed_test_state"],
        "run_class": identity["run_class"],
        "continues": dict(provenance["continues"]),
        "consumed_evidence": [dict(e) for e in provenance["consumed_evidence"]],
    }
    return validate_continuation_held_out_evidence(payload)


def promote_continuation_held_out_evaluation(
    attempt_dir: Path, fold_index: int, payload: Mapping[str, Any]
) -> str:
    """Promote one fold's continuation evidence, once, into the continuation root.

    Written per fold the moment the measurement returns, for the reason the
    canonical §17 path was changed in the first place: evidence that exists only
    after the last stage succeeds is evidence that a failure in any stage can
    erase. Twelve folds of completed measurement were lost that way.

    The re-read verifies that what was written is what was meant, exactly as the
    selection barrier does. Nothing here measures, selects or derives.
    """
    validate_continuation_held_out_evidence(payload)
    directory = _require_inside_continuation_root(Path(attempt_dir) / HELD_OUT_DIR)
    path = directory / f"T1_CONTINUATION_FOLD_{int(fold_index):02d}_HELD_OUT.json"
    if path.exists():
        raise T1ContinuationPersistenceError(
            f"Fold {fold_index} continuation evidence is already promoted. It is "
            "not overwritten, and a second measurement of a held-out subject "
            "does not exist."
        )
    # Every refusal above happens before the directory exists, so a refused
    # promotion leaves no trace of having been attempted.
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, dict(payload))
    reread = json.loads(path.read_text(encoding="utf-8"))
    if canonical_sha256(reread) != canonical_sha256(dict(payload)):
        raise T1ContinuationPersistenceError(
            f"Fold {fold_index} continuation evidence did not read back as written."
        )
    return sha256_file(path)


def read_continuation_held_out_evaluations(
    attempt_dir: Path,
) -> dict[int, dict[str, Any]]:
    """Every fold's continuation evidence that survives on disk, by fold index."""
    directory = Path(attempt_dir) / HELD_OUT_DIR
    if not directory.is_dir():
        return {}
    found: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob("T1_CONTINUATION_FOLD_*_HELD_OUT.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        found[int(payload["fold_index"])] = payload
    return found
