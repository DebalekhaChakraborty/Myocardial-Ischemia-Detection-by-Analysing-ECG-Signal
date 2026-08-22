"""Held-out label access for the continuation, through the existing §16 authority.

The continuation opens held-out labels for one fold at a time, under the
selection already promoted for that fold. It does **not** get its own label
reader: `t1_fold_evaluation.T1CorpusTargetSource` is the authority that already
governs which members may be read (`stable_id`, `subject_id`, `label`,
`primary_mask`) and which are refused, and a second reader would be a second
opinion about the same barrier.

**Why this module is not part of the proven continuation graph.**
`t1_fold_evaluation` imports `t1_fold_authority`, which imports
`t1_development_run` -- so reaching the label authority necessarily loads a
module holding `generate_thresholds`, `select_policy` and
`run_policy_over_streams`. The negative capability gate therefore treats
`t1_development_run` the way it already treats `t1_protocol`: a module that must
be loaded, whose forbidden entry points earn real call counters rather than an
absence argument. The genuinely dangerous module, `t1_fold_evaluator`, stays
unloaded and keeps the stronger never-imported proof.

This module is the boundary where that happens. It is deliberately thin, it is
excluded from the gate's proven module set, and the labels leave it as plain
data so `t1_continuation_measurement` never has to import anything to obtain
them.

Labels are read, never used to choose anything. They score states that were
emitted before any label was opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping

from cardiosentinel.neural.t1_continuation_spec import PREDECESSOR_FOLD_SELECTIONS

#: The two label members the continuation needs, and the only two it takes.
#: `label` and `target_family` are never materialised here: the measurement
#: scores PRIMARY rows against PRIMARY positives and needs nothing else.
CONTINUATION_LABEL_MEMBERS: Final = ("primary_mask", "primary_positive")


class T1ContinuationLabelError(RuntimeError):
    """Raised when held-out labels cannot be obtained for exactly one fold."""


def held_out_labels_for_fold(
    target_source: Any, fold_index: int
) -> dict[str, dict[str, bool]]:
    """Open one fold's held-out labels, keyed by `stable_id`.

    The subject is taken from the amendment's §1.4 binding rather than from an
    argument, so a caller cannot ask this for a subject that fold never held
    out. One fold, one subject, one call.

    Returns plain dictionaries. Nothing here retains the authority, so the
    measurement layer receives data rather than a live label reader.
    """
    if fold_index not in PREDECESSOR_FOLD_SELECTIONS:
        raise T1ContinuationLabelError(
            f"Fold {fold_index} is not one of the twelve promoted folds."
        )
    subject, _policy_id, _digest = PREDECESSOR_FOLD_SELECTIONS[fold_index]

    targets = target_source.read_subject_targets(subject)
    stable_ids = [str(value) for value in targets.stable_id]
    masks = [bool(value) for value in targets.primary_mask]
    positives = [bool(value) for value in targets.primary_positive]

    if not (len(stable_ids) == len(masks) == len(positives)):
        raise T1ContinuationLabelError(
            f"Fold {fold_index} targets for {subject!r} are ragged: "
            f"{len(stable_ids)} ids, {len(masks)} masks, {len(positives)} positives."
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


def continuation_target_source(corpus_root: Path) -> Any:
    """Build the existing §16 target authority. Imported late, on purpose.

    The import happens inside the function so that merely importing this module
    does not load `t1_development_run` transitively. A caller that never asks for
    labels never dirties its interpreter, which keeps the gate's clean-interpreter
    proof available to anything that does not need labels.
    """
    from cardiosentinel.neural.t1_fold_evaluation import T1CorpusTargetSource

    return T1CorpusTargetSource(Path(corpus_root))
