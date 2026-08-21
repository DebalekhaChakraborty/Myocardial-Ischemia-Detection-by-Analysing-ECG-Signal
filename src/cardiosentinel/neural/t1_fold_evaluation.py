"""The controlled T1 fold evaluation capability.

Two pieces complete the execution architecture without arming it.

`T1CorpusTargetSource` is the concrete target source the fold authority
delegates to. It is the only implementation of `T1TargetSource` in the
repository, and it refuses to answer anything that is not a live, authority-
sponsored request for one authorized subject in VALIDATION. It resolves nothing
and reads nothing at construction.

`T1NonExecutingFoldEvaluator` satisfies the driver's `evaluate_fold`
collaborator so the collaborator graph is complete, and refuses to run. That is
not a placeholder standing in for missing work: the evaluator's scientific body
is the step that runs twelve candidate policies and scores them, and enabling
it is a separate decision from wiring it. Capability and permission stay
separate here exactly as they do at the driver and the config gate.

It also declares its own inability to the pre-claim capability gate, through
`t1_execution_capability`. Being callable was once enough to reach the claim,
which meant this object could have cost the single canonical attempt at stage
17; the declaration is what moves that refusal to before stage 1.

**What neither piece does.** No fold is run, no validation label is read, no
prediction is generated, no metric is computed, no OOF evidence is produced, no
policy is selected, TEST is unreachable and no canonical directory is created.
The label-bearing reader is written and never called: this module's tests
exercise its refusals, all of which fire before an array is opened.

**Why the reader lives here and not in the evidence store.**
`t1_evidence_store.read_t2_identity_members` is the *label-blind* reader and
refuses `label`, `target_family` and `primary_mask` by design. The targets a
fold evaluator needs are exactly those members, so reading them requires a
deliberately separate, deliberately narrower door rather than a relaxation of
the existing one. Widening the label-blind reader would have removed a firewall
to reach through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, Sequence, runtime_checkable

import numpy as np

from cardiosentinel.neural.t1_capability_gate import (
    T1CapabilityAttestation,
    attest,
)
from cardiosentinel.neural.t1_execution_spec import (
    T1_CANDIDATE_POLICIES_PER_FOLD,
    T1_HELD_OUT_POLICY_RUNS_PER_FOLD,
    T1_T2_IDENTITY_NAME,
    require_no_test_access,
)
from cardiosentinel.neural.t1_fold_authority import (
    SCOPE_FIT,
    SCOPE_HELD_OUT,
    T1_PERMITTED_PARTITION,
    FoldScopedEvaluationAuthority,
    T1FoldAuthorityError,
    T1SubjectTargets,
    require_active_scoped_request,
    require_known_subject,
    require_validation_partition,
)
from cardiosentinel.neural.t1_persistence import require_no_test_path

# The members a target read materialises, and nothing else. `target_family` and
# `cold_start_bin` are evaluation annotation the selection metrics never
# consume, so they are not listed and asking for them is refused.
T1_TARGET_MEMBERS: Final = ("stable_id", "subject_id", "label", "primary_mask")
T1_TARGET_MEMBERS_REFUSED: Final = (
    "target_family",
    "cold_start_bin",
    "episode_identity",
    "challenge_identity",
    "test_field",
)

EVALUATION_DISABLED_MESSAGE: Final = (
    "Evaluation capability exists, but scientific execution is not enabled."
)


class T1FoldEvaluationError(RuntimeError):
    """Raised when a fold evaluation is attempted or requested dishonestly."""


# ---------------------------------------------------------------------------
# The concrete target source
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T1CorpusTargetSource:
    """The one target source, reachable only through a fold authority.

    Holds a path and nothing else: no cached frame, no subject index, no label
    table. Every read re-proves the authorization, the partition and the
    subject before an array is opened, so a source reference that leaks to a
    caller with no authority is inert rather than dangerous.
    """

    identity_path: Path
    _members: tuple[str, ...] = field(default=T1_TARGET_MEMBERS, repr=False)

    def __post_init__(self) -> None:
        # Validation only. Nothing is opened, and the file need not exist yet.
        name = str(self.identity_path)
        require_no_test_access(Path(name).stem)
        require_no_test_path(name)
        if Path(name).name != T1_T2_IDENTITY_NAME:
            raise T1FoldEvaluationError(
                f"A target source reads {T1_T2_IDENTITY_NAME}, not "
                f"{Path(name).name!r}. The targets live beside the row identity "
                "the label-blind assembly already binds; a different file would "
                "be a second provenance for the same truth."
            )
        for member in self._members:
            if member in T1_TARGET_MEMBERS_REFUSED:
                raise T1FoldEvaluationError(
                    f"Member {member!r} is evaluation annotation the selection "
                    "metrics never consume, and is refused."
                )

    def read_subject_targets(
        self, subject_id: str, *, partition: str
    ) -> T1SubjectTargets:
        """One authorized subject's targets. Refused unless an authority asked.

        The order matters and is the point: the sponsoring authorization is
        proved first, then the partition, then the subject, and only then is
        the archive opened. Every refusal below happens before a read.
        """
        partition = require_validation_partition(partition)
        subject = require_known_subject(subject_id)
        require_active_scoped_request(subject, partition)
        return self._read_one_subject(subject)

    def _read_one_subject(self, subject: str) -> T1SubjectTargets:
        """The only place an array is opened, and only for one subject."""
        with np.load(Path(self.identity_path), allow_pickle=False) as payload:
            missing = sorted(set(self._members) - set(payload.files))
            if missing:
                raise T1FoldEvaluationError(
                    f"The row identity lacks members {missing}."
                )
            subjects = np.asarray(payload["subject_id"])
            rows = subjects == subject
            if not np.any(rows):
                raise T1FoldEvaluationError(
                    f"The row identity carries no rows for {subject!r}."
                )
            return T1SubjectTargets(
                subject_id=subject,
                stable_id=tuple(str(v) for v in np.asarray(payload["stable_id"])[rows]),
                primary_positive=tuple(
                    bool(v) for v in np.asarray(payload["label"])[rows]
                ),
                primary_mask=tuple(
                    bool(v) for v in np.asarray(payload["primary_mask"])[rows]
                ),
            )

    def as_dict(self) -> dict[str, Any]:
        """Provenance. Names the members, never their values."""
        return {
            "source": type(self).__name__,
            "identity_file": Path(self.identity_path).name,
            "members": list(self._members),
            "refused_members": list(T1_TARGET_MEMBERS_REFUSED),
            "reachable_without_an_authority": False,
            "partition": T1_PERMITTED_PARTITION,
            "test_accessed": False,
        }


# ---------------------------------------------------------------------------
# The fold evaluator contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T1FoldEvaluationRequest:
    """Everything an evaluator may see, and it is all already authorized.

    Deliberately carries two authorities rather than a fold identity and a
    dataset: an evaluator that received a fold index would have to resolve its
    own subjects, and one that received a path would have to open it. Both are
    exactly the independence this contract removes.
    """

    fit: FoldScopedEvaluationAuthority
    held_out: FoldScopedEvaluationAuthority

    def __post_init__(self) -> None:
        if self.fit.scope != SCOPE_FIT:
            raise T1FoldEvaluationError(
                f"The selection authority must be {SCOPE_FIT!r}, not "
                f"{self.fit.scope!r}."
            )
        if self.held_out.scope != SCOPE_HELD_OUT:
            raise T1FoldEvaluationError(
                f"The evaluation authority must be {SCOPE_HELD_OUT!r}, not "
                f"{self.held_out.scope!r}."
            )
        if self.fit.fold_index != self.held_out.fold_index:
            raise T1FoldEvaluationError(
                f"The two authorities describe different folds "
                f"({self.fit.fold_index} and {self.held_out.fold_index}). One "
                "request is one fold."
            )
        if self.held_out.authorized_subjects != (self.fit.held_out_subject,):
            raise T1FoldEvaluationError(
                "The evaluation authority does not scope this fold's held-out "
                "subject. The two authorities must belong to the same fold."
            )

    @property
    def fold_index(self) -> int:
        return self.fit.fold_index

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "selection_scope": self.fit.as_dict(),
            "evaluation_scope": self.held_out.as_dict(),
            "candidate_policies": T1_CANDIDATE_POLICIES_PER_FOLD,
            "held_out_policy_runs": T1_HELD_OUT_POLICY_RUNS_PER_FOLD,
        }


@runtime_checkable
class T1FoldEvaluator(Protocol):
    """The contract a future scientific evaluator satisfies.

    One method, taking an already-authorized request. There is no parameter
    through which a path, a frame, a subject list or a partition could arrive,
    so an implementation has no way to reach a dataset except back through the
    authorities it was handed.
    """

    def evaluate(self, request: T1FoldEvaluationRequest) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class T1NonExecutingFoldEvaluator:
    """Completes the collaborator graph and refuses to run.

    This is the shape a scientific evaluator must take, with the scientific
    body absent rather than stubbed: it returns no artifact, no metric and no
    selection, and it raises instead of producing an empty one. A stub that
    returned a plausible-looking result would be indistinguishable from a run
    that found nothing.
    """

    def evaluate(self, request: T1FoldEvaluationRequest) -> dict[str, Any]:
        if not isinstance(request, T1FoldEvaluationRequest):
            raise T1FoldEvaluationError(
                "A fold evaluation takes an authorized T1FoldEvaluationRequest, "
                f"not {type(request).__name__}."
            )
        raise T1FoldEvaluationError(
            f"{EVALUATION_DISABLED_MESSAGE} Fold {request.fold_index} is fully "
            "authorized at the label boundary and the request is well formed, "
            "but the twelve-candidate selection and the held-out trace are not "
            "implemented here. Enabling them is a separate decision from wiring "
            "them, and no attempt has been consumed."
        )

    def __call__(self, fold: Any, authority: Any) -> dict[str, Any]:  # pragma: no cover
        """The driver's `evaluate_fold` shape, which also refuses."""
        raise T1FoldEvaluationError(EVALUATION_DISABLED_MESSAGE)

    def t1_execution_capability(self) -> T1CapabilityAttestation:
        """Declare, to the pre-claim gate, that this cannot finish a run.

        Being callable is not the same as being able to complete, and the
        driver claims the attempt seven stages before it first calls a fold
        evaluator. Saying so here is what keeps that claim from happening: the
        gate refuses this graph before stage 1 rather than after stage 10.
        """
        return attest(
            "evaluate_fold",
            provider=type(self).__name__,
            executes=False,
            reason=(
                "The twelve-candidate selection and the held-out trace are not "
                "implemented; this object exists to complete the collaborator "
                "graph for review, not to run science."
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluator": type(self).__name__,
            "capability_present": True,
            "execution_enabled": False,
            "reads_datasets_independently": False,
            "reads_labels_independently": False,
            "contains_model_selection_logic": False,
        }


# ---------------------------------------------------------------------------
# Capability reporting
# ---------------------------------------------------------------------------


def evaluation_capability() -> dict[str, Any]:
    """What this layer provides, as data a receipt can carry."""
    return {
        "target_source": T1CorpusTargetSource.__name__,
        "evaluator_contract": T1FoldEvaluator.__name__,
        "evaluator_implementation": T1NonExecutingFoldEvaluator.__name__,
        "execution_enabled": False,
        "declares_capability_to_the_pre_claim_gate": True,
        "target_members": list(T1_TARGET_MEMBERS),
        "refused_members": list(T1_TARGET_MEMBERS_REFUSED),
        "reads_require_an_active_authority": True,
        "construction_reads_targets": False,
        "test_accessed": False,
    }


def build_fold_request(
    fit: FoldScopedEvaluationAuthority,
    held_out: FoldScopedEvaluationAuthority,
) -> T1FoldEvaluationRequest:
    """Pair two authorities into one fold request, validated on construction."""
    for authority in (fit, held_out):
        if not isinstance(authority, FoldScopedEvaluationAuthority):
            raise T1FoldEvaluationError(
                "A fold request is built from two FoldScopedEvaluationAuthority "
                f"objects, not {type(authority).__name__}. An evaluator that "
                "could be handed anything else would not be scoped."
            )
    return T1FoldEvaluationRequest(fit=fit, held_out=held_out)


def require_no_independent_access(evaluator: Any) -> Any:
    """Refuse an evaluator that carries its own way to the data.

    An evaluator holding a path, a frame or a source has a route around the
    authorities, which is the one thing this contract exists to prevent.
    """
    for attribute in dir(evaluator):
        if attribute.startswith("__"):
            continue
        try:
            value = getattr(evaluator, attribute)
        except AttributeError:  # pragma: no cover - defensive
            continue
        if isinstance(value, Path):
            raise T1FoldEvaluationError(
                f"Evaluator attribute {attribute!r} is a Path. An evaluator "
                "reaches data only through the authorities it is handed."
            )
        if isinstance(value, T1CorpusTargetSource):
            raise T1FoldEvaluationError(
                f"Evaluator attribute {attribute!r} holds a target source "
                "directly, bypassing the fold authority."
            )
        if isinstance(value, np.ndarray):
            raise T1FoldEvaluationError(
                f"Evaluator attribute {attribute!r} holds an array. An "
                "evaluator carries no dataset of its own."
            )
    return evaluator


def authorized_targets(
    authority: FoldScopedEvaluationAuthority, subject_id: str
) -> T1SubjectTargets:
    """The only supported way to obtain targets, spelled out in one place.

    A thin pass-through on purpose: it exists so callers have an obvious
    correct route, not so the authority can be skipped. Skipping it and calling
    a source directly still fails, because the source proves the sponsoring
    authorization itself.
    """
    if not isinstance(authority, FoldScopedEvaluationAuthority):
        raise T1FoldAuthorityError(
            "Targets come from a FoldScopedEvaluationAuthority, not "
            f"{type(authority).__name__}."
        )
    return authority.targets_for_subject(subject_id)


def target_member_plan(members: Sequence[str] = T1_TARGET_MEMBERS) -> dict[str, Any]:
    """Which members a read would materialise, described without reading."""
    requested = tuple(members)
    refused = sorted(set(requested) & set(T1_TARGET_MEMBERS_REFUSED))
    if refused:
        raise T1FoldEvaluationError(f"Members {refused} are refused.")
    return {
        "members": list(requested),
        "materialises_whole_archive": False,
        "subject_scoped": True,
        "partition": T1_PERMITTED_PARTITION,
    }
