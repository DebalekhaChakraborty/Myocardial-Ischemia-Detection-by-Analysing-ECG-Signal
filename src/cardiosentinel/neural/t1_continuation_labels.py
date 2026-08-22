"""Held-out label access for the continuation, through the existing §16 authority.

The continuation opens held-out labels one fold at a time, under the selection
already promoted for that fold. It does **not** get its own label reader:
`t1_fold_evaluation.T1CorpusTargetSource` is the authority that already governs
which members may be read and which are refused, and
`t1_fold_authority.FoldScopedEvaluationAuthority` is the only door through which
a read may pass. A second reader would be a second opinion about the same
barrier.

**What this module got wrong before PR #56.** The first version called
`read_subject_targets(subject)` directly on a source constructed from a corpus
*directory*. Three things were wrong with that and all three were invisible
until runtime:

1. the source takes the path of one identity artifact, `t2_outer_row_identity.npz`,
   not a corpus root;
2. `read_subject_targets` takes a required keyword-only `partition`;
3. an unsponsored read is refused outright -- `require_active_scoped_request`
   admits only a read a `FoldScopedEvaluationAuthority` is currently sponsoring.

Stage 8 would therefore have failed *after* the claim, consuming the single
authorized attempt to discover a wiring error. The fix is to stop reaching past
the authority and go through it, which is what it was built for.

**The barrier state is earned, not asserted.** `held_out_evaluation_authority`
refuses unless the fold's selection artifact was promoted and its digest
re-verified. The continuation can answer both truthfully: the twelve selections
were promoted by the consumed attempt and are bound by amendment §1.4, and the
runner re-verifies that fold's digest immediately before this module is asked
for its labels. `continuation_fold_state` therefore takes the digest that was
just checked, rather than a boolean somebody set.

**Why this module is not in the proven continuation graph.** Reaching the
authority imports `t1_fold_evaluation` -> `t1_fold_authority` ->
`t1_development_run`, whose forbidden entry points carry real call counters for
exactly this reason. The genuinely dangerous module, `t1_fold_evaluator`, stays
unloaded. Every project import here is deferred into a function, so importing
this module dirties nothing.

Labels are read, never used to choose anything. They score states that were
emitted before any label was opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping

from cardiosentinel.neural.t1_continuation_spec import PREDECESSOR_FOLD_SELECTIONS

#: The identity artifact the target source reads. Named here so a caller does
#: not have to know the authority's file convention, and so a wrong path is a
#: refusal at construction rather than a surprise at read time.
T1_T2_IDENTITY_ARTIFACT: Final = "t2_outer_row_identity.npz"

#: Where the promoted T2 outer-validation row identity lives, relative to the
#: repository root. The continuation reads targets from the same artifact the
#: canonical run's evaluation did; a different file would be a second provenance
#: for one truth.
DEFAULT_IDENTITY_RELATIVE: Final = (
    Path(
        "cardiosentinel-runs/phase8-t2-development-v1/t2-v1-outer-validation"
        "/row_evidence"
    )
    / T1_T2_IDENTITY_ARTIFACT
)

#: The two label members the continuation consumes. `label` and `target_family`
#: are never materialised: the measurement scores PRIMARY rows against PRIMARY
#: positives and needs nothing else.
CONTINUATION_LABEL_MEMBERS: Final = ("primary_mask", "primary_positive")

#: The three facts `require_held_out_access_authorized` checks. Named as data so
#: a test can prove the continuation supplies exactly these and nothing wider.
BARRIER_STATE_KEYS: Final = (
    "selection_promoted",
    "selection_digest_verified",
    "held_out_label_access_authorized_for_this_fold",
)


class T1ContinuationLabelError(RuntimeError):
    """Raised when held-out labels cannot be obtained for exactly one fold."""


def continuation_identity_path(repository_root: Path) -> Path:
    """The promoted T2 outer row-identity artifact this corpus reads from."""
    return Path(repository_root) / DEFAULT_IDENTITY_RELATIVE


def continuation_target_source(identity_path: Path) -> Any:
    """Build the existing §16 target source. Imported late, on purpose.

    Takes the path of the identity **artifact**, not a corpus directory: the
    source refuses any other filename, and a refusal at construction is cheaper
    than one at stage 8. The import is deferred so that merely importing this
    module does not pull `t1_development_run` into the process.
    """
    from cardiosentinel.neural.t1_fold_evaluation import T1CorpusTargetSource

    path = Path(identity_path)
    if path.name != T1_T2_IDENTITY_ARTIFACT:
        raise T1ContinuationLabelError(
            f"A continuation target source reads {T1_T2_IDENTITY_ARTIFACT}, not "
            f"{path.name!r}. Pass the identity artifact, not the directory that "
            "holds it."
        )
    return T1CorpusTargetSource(path)


def continuation_fold_state(
    fold_index: int, verified_selection_sha256: str
) -> dict[str, Any]:
    """The barrier state for one fold, from a digest that was just re-verified.

    `require_held_out_access_authorized` asks three questions, and the
    continuation can answer all three truthfully rather than by assertion: the
    selection was promoted by the consumed attempt, its digest has just been
    re-checked against the amendment's §1.4 binding, and access follows from
    those two rather than from a flag somebody set.

    Passing the digest rather than a boolean is the point. A caller that has not
    verified anything has nothing to pass.
    """
    if fold_index not in PREDECESSOR_FOLD_SELECTIONS:
        raise T1ContinuationLabelError(
            f"Fold {fold_index} is not one of the twelve promoted folds."
        )
    _subject, _policy, expected = PREDECESSOR_FOLD_SELECTIONS[fold_index]
    if verified_selection_sha256 != expected:
        raise T1ContinuationLabelError(
            f"Fold {fold_index} selection digest {verified_selection_sha256} is "
            f"not the promoted {expected}. Held-out labels are not opened "
            "against a selection that has moved."
        )
    return {
        "fold_index": int(fold_index),
        "selection_promoted": True,
        "selection_digest_verified": True,
        "held_out_label_access_authorized_for_this_fold": True,
        "selection_sha256": expected,
        "authorized_by": "t1_continuation_runner._reverify_fold_selection",
    }


def continuation_held_out_authority(
    fold_index: int, source: Any, *, verified_selection_sha256: str
) -> Any:
    """One fold's held-out authority: one subject, after the barrier.

    The authority is the door. Building one is what makes a subsequent read
    legitimate, and it is refused unless the barrier state above holds.
    """
    from cardiosentinel.neural.t1_fold_authority import held_out_evaluation_authority
    from cardiosentinel.neural.t1_protocol import t1_folds

    state = continuation_fold_state(fold_index, verified_selection_sha256)
    folds = t1_folds()
    fold = next((f for f in folds if f.fold_index == fold_index), None)
    if fold is None:
        raise T1ContinuationLabelError(
            f"The frozen fold table has no fold {fold_index}."
        )
    subject, _policy, _digest = PREDECESSOR_FOLD_SELECTIONS[fold_index]
    if fold.held_out_subject != subject:
        raise T1ContinuationLabelError(
            f"Fold {fold_index} holds out {fold.held_out_subject!r} in the frozen "
            f"table but {subject!r} in the promoted selection. The two views of "
            "one fold must describe the same subject."
        )
    return held_out_evaluation_authority(fold, state, source=source)


def held_out_labels_for_fold(
    authority: Any, fold_index: int
) -> dict[str, dict[str, bool]]:
    """Open one fold's held-out labels through its authority, keyed by `stable_id`.

    The subject comes from the amendment's §1.4 binding rather than from an
    argument, so a caller cannot ask this for a subject that fold never held
    out. One fold, one subject, one call.

    Returns plain dictionaries. Nothing here retains the authority, so the
    measurement layer receives data rather than a live label reader.
    """
    if fold_index not in PREDECESSOR_FOLD_SELECTIONS:
        raise T1ContinuationLabelError(
            f"Fold {fold_index} is not one of the twelve promoted folds."
        )
    subject, _policy, _digest = PREDECESSOR_FOLD_SELECTIONS[fold_index]

    targets = authority.targets_for_subject(subject)
    if targets.subject_id != subject:
        raise T1ContinuationLabelError(
            f"The authority answered for {targets.subject_id!r} when {subject!r} "
            "was asked for."
        )
    stable_ids = [str(value) for value in targets.stable_id]
    masks = [bool(value) for value in targets.primary_mask]
    positives = [bool(value) for value in targets.primary_positive]

    if not (len(stable_ids) == len(masks) == len(positives)):
        raise T1ContinuationLabelError(
            f"Fold {fold_index} targets for {subject!r} are ragged."
        )
    if not stable_ids:
        raise T1ContinuationLabelError(
            f"The authority returned no rows for {subject!r}."
        )
    return {
        "primary_mask": dict(zip(stable_ids, masks)),
        "primary_positive": dict(zip(stable_ids, positives)),
    }


def require_labels_cover_trace(
    labels: Mapping[str, Mapping[str, bool]], stable_ids: list[str], fold_index: int
) -> None:
    """Every trace row must have a label, or the two views describe different rows.

    The canonical evaluator proved the same property from the other direction --
    it refused a row the authority named that the timeline did not contain. Here
    the trace is fixed and the labels arrive second, so the check runs the other
    way round.
    """
    for member in CONTINUATION_LABEL_MEMBERS:
        if member not in labels:
            raise T1ContinuationLabelError(f"Labels are missing {member!r}.")
    known = set(labels["primary_mask"])
    unlabelled = [sid for sid in stable_ids if sid not in known]
    if unlabelled:
        raise T1ContinuationLabelError(
            f"Fold {fold_index}: {len(unlabelled)} trace rows have no label "
            f"(first: {unlabelled[:3]}). The persisted trace and the held-out "
            "labels must describe the same rows."
        )
