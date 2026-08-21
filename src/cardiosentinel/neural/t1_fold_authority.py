"""The fold-scoped evaluation authority: the only door to a fold's targets.

`t1_development_run.FoldScopedTargetAuthority` answers one question -- "is this
subject inside my scope?" -- and deliberately has no method that returns
labels. That absence is the fold firewall, and it is why no fold evaluator
could be written against it: there was no permitted way to obtain a target at
all, only a permitted way to be told no.

This module supplies the missing door, and makes it the only one. An authority
binds a fold identity, a scope, an explicit subject set and a sealed partition
at construction time, and exposes exactly one accessor, for exactly one subject
at a time, which it refuses unless that subject is inside the scope it was
built for. There is no method that returns every label, no method that returns
a partition, no frame, no index, no iterator and no mapping protocol -- and
tests assert each of those absences rather than trusting them.

**Construction reads nothing.** Building an authority resolves no path, opens
no store and touches no target. It is a permission object that knows how to
delegate a single scoped read, not a loaded label table.

**The partition is sealed, not selected.** VALIDATION is the only partition a
T1 development authority can be built for, and it is hard-coded rather than
passed. TEST cannot be requested because there is no parameter that could
carry it, and the guards are still applied on the way through so that a future
caller who invents one is refused before any read is delegated.

**This is a security boundary, not a convenience wrapper.** The narrowness is
the feature: a caller that wants a second subject must ask a second authority
that someone had to be authorized to build.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final, Iterator, Protocol, Sequence, runtime_checkable

from cardiosentinel.neural.t1_development_run import (
    TARGET_AUTHORITY,
    FoldScopedTargetAuthority,
    fit_authority,
    held_out_authority,
)
from cardiosentinel.neural.t1_execution_spec import (
    T1_FOLD_COUNT,
    T1_FOLD_SCOPED_TARGET_AUTHORITY_REQUIRED,
    T1_GLOBAL_LABEL_TABLE_PERMITTED,
    T1_HELD_OUT_POLICY_RUNS_PER_FOLD,
    require_no_test_access,
)
from cardiosentinel.neural.t1_persistence import require_no_test_path
from cardiosentinel.neural.t1_protocol import (
    T1_VALIDATION_SUBJECTS,
    T1Fold,
    t1_folds,
)

# The partition a T1 development authority may be built for. Hard-coded, never
# a parameter: a partition that can be passed is a partition that can be got
# wrong, and there is exactly one right answer here.
T1_PERMITTED_PARTITION: Final = "validation"

# Names that must never resolve to an authority. Held as data so the refusal is
# testable, not as a branch that could be reordered away.
T1_SEALED_PARTITIONS: Final = ("test", "TEST", "locked_test", "holdout_test")

SCOPE_FIT: Final = "fit_subjects_only"
SCOPE_HELD_OUT: Final = "held_out_subject_only"
T1_AUTHORITY_SCOPES: Final = (SCOPE_FIT, SCOPE_HELD_OUT)

# Accessors that must not exist on the authority. A reviewer should be able to
# read this tuple and know what was deliberately not built.
T1_FORBIDDEN_AUTHORITY_ACCESSORS: Final = (
    "get_all_labels",
    "get_validation_labels",
    "get_test_labels",
    "all_labels",
    "labels",
    "label_table",
    "dataframe",
    "as_frame",
    "to_pandas",
    "subjects_labels",
    "__getitem__",
    "__iter__",
    "keys",
    "items",
    "values",
)


class T1FoldAuthorityError(RuntimeError):
    """Raised when a fold-scoped target request is outside its authority."""


# ---------------------------------------------------------------------------
# What a scoped read may return
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T1SubjectTargets:
    """The targets of exactly one subject, and nothing else.

    Not a frame and not a slice of one. There is no other subject reachable
    from this object, no index into a larger table, and no column beyond the
    two the frozen selection metrics consume.
    """

    subject_id: str
    stable_id: tuple[str, ...]
    primary_positive: tuple[bool, ...]
    primary_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        widths = {
            len(self.stable_id),
            len(self.primary_positive),
            len(self.primary_mask),
        }
        if len(widths) != 1:
            raise T1FoldAuthorityError(
                f"Targets for {self.subject_id!r} are ragged: stable_id, "
                "primary_positive and primary_mask must describe the same rows."
            )

    def __len__(self) -> int:
        return len(self.stable_id)

    def as_dict(self) -> dict[str, Any]:
        """Provenance only. Never the targets themselves."""
        return {
            "subject_id": self.subject_id,
            "row_count": len(self),
            "columns": ("stable_id", "primary_positive", "primary_mask"),
        }


@dataclass(frozen=True, slots=True)
class ScopedTargetRequest:
    """Proof that an authority authorized this exact subject, just now.

    Minted inside `targets_for_subject` and visible only for the duration of
    that one delegated read. A target source that finds no request in scope was
    called directly, which is the case this exists to refuse: without it,
    "reached only through the authority" would be a convention that any caller
    holding a source reference could ignore.
    """

    fold_index: int
    subject_id: str
    scope: str
    partition: str


_ACTIVE_REQUEST: Final[ContextVar[ScopedTargetRequest | None]] = ContextVar(
    "t1_active_scoped_target_request", default=None
)


@contextmanager
def _authorized_request(request: ScopedTargetRequest) -> Iterator[None]:
    token = _ACTIVE_REQUEST.set(request)
    try:
        yield
    finally:
        _ACTIVE_REQUEST.reset(token)


def active_scoped_request() -> ScopedTargetRequest | None:
    """The authorization in scope right now, if a read is being delegated."""
    return _ACTIVE_REQUEST.get()


def require_active_scoped_request(
    subject_id: str, partition: str
) -> ScopedTargetRequest:
    """Refuse a read that no authority is currently sponsoring.

    Checked by the target source rather than by the authority, because the
    authority cannot police a caller that never asked it anything.
    """
    request = _ACTIVE_REQUEST.get()
    if request is None:
        raise T1FoldAuthorityError(
            "No fold authority is sponsoring this read. Targets are reachable "
            "only through FoldScopedEvaluationAuthority.targets_for_subject; a "
            "source called directly is a global label table being read by "
            "another name."
        )
    if request.subject_id != str(subject_id) or request.partition != str(partition):
        raise T1FoldAuthorityError(
            f"The sponsoring authority authorized {request.subject_id!r} in "
            f"{request.partition!r}, but the read asked for {subject_id!r} in "
            f"{partition!r}."
        )
    return request


@runtime_checkable
class T1TargetSource(Protocol):
    """The narrowest possible contract a target store may satisfy.

    One subject, one partition, one call. A source that could be asked for
    everything would make the authority decorative, so the protocol has no
    method that takes a subject list, a partition list, a filter or a slice.
    """

    def read_subject_targets(
        self, subject_id: str, *, partition: str
    ) -> T1SubjectTargets: ...


# ---------------------------------------------------------------------------
# Refusals that run before anything is delegated
# ---------------------------------------------------------------------------


def require_validation_partition(partition: str) -> str:
    """VALIDATION is the only partition; TEST is refused by name and by path."""
    name = str(partition).strip()
    require_no_test_access(name)
    require_no_test_path(name)
    if name.lower() != T1_PERMITTED_PARTITION:
        raise T1FoldAuthorityError(
            f"Partition {partition!r} cannot carry a T1 development authority. "
            f"The only permitted partition is {T1_PERMITTED_PARTITION!r}; TEST "
            "is sealed and no other partition is defined for this protocol."
        )
    return T1_PERMITTED_PARTITION


def require_known_fold(fold: T1Fold) -> T1Fold:
    """The fold must be one the frozen design actually produces.

    Identity is checked against `t1_folds()` rather than merely type-checked,
    so a hand-built fold naming a different held-out subject, a different fit
    set or an out-of-range index cannot become an authority.
    """
    if not isinstance(fold, T1Fold):
        raise T1FoldAuthorityError(
            f"A fold authority needs a T1Fold, got {type(fold).__name__}."
        )
    if not 0 <= fold.fold_index < T1_FOLD_COUNT:
        raise T1FoldAuthorityError(
            f"Fold index {fold.fold_index!r} is outside the frozen "
            f"0..{T1_FOLD_COUNT - 1} design."
        )
    frozen = t1_folds()[fold.fold_index]
    if fold != frozen:
        raise T1FoldAuthorityError(
            f"Fold {fold.fold_index} does not match the frozen leave-one-subject"
            f"-out design. Frozen: held out {frozen.held_out_subject!r} over "
            f"{len(frozen.fit_subjects)} fit subjects. Given: held out "
            f"{fold.held_out_subject!r} over {len(fold.fit_subjects)}. Fold "
            "membership depends on subject identity alone and is not negotiable."
        )
    return fold


def require_known_subject(subject_id: str) -> str:
    """Subjects come from the frozen validation roster, never from a caller."""
    name = str(subject_id).strip()
    require_no_test_access(name)
    require_no_test_path(name)
    if name not in T1_VALIDATION_SUBJECTS:
        raise T1FoldAuthorityError(
            f"Subject {subject_id!r} is not in the frozen T1 validation roster. "
            "An authority cannot be built for a subject the protocol does not "
            "name."
        )
    return name


# ---------------------------------------------------------------------------
# The authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldScopedEvaluationAuthority:
    """One fold, one scope, one subject at a time.

    Wraps the existing `FoldScopedTargetAuthority` rather than replacing it:
    that object already decides subject membership, and a second implementation
    of the same question is a second answer waiting to disagree. What this adds
    is the fold identity, the sealed partition and the single scoped accessor
    that the older object deliberately never had.
    """

    fold_index: int
    held_out_subject: str
    scope: str
    partition: str
    subject_scope: FoldScopedTargetAuthority
    source: T1TargetSource = field(repr=False, compare=False)

    @property
    def authority(self) -> str:
        return self.subject_scope.authority

    @property
    def authorized_subjects(self) -> tuple[str, ...]:
        return self.subject_scope.authorized_subjects

    def require_authorized(self, subject_id: str) -> str:
        """Delegated, so membership has exactly one implementation."""
        return self.subject_scope.require_authorized(str(subject_id))

    def targets_for_subject(self, subject_id: str) -> T1SubjectTargets:
        """The only accessor. One subject, checked before anything is read.

        The subject is authorized first, the partition is re-proved second, and
        only then is the read delegated. The returned targets are checked to be
        the subject that was asked for: a source that answered with someone
        else's rows would have crossed the boundary from the far side, and that
        is refused here rather than trusted.
        """
        authorized = self.require_authorized(subject_id)
        partition = require_validation_partition(self.partition)
        request = ScopedTargetRequest(
            fold_index=self.fold_index,
            subject_id=authorized,
            scope=self.scope,
            partition=partition,
        )
        with _authorized_request(request):
            targets = self.source.read_subject_targets(authorized, partition=partition)
        if not isinstance(targets, T1SubjectTargets):
            raise T1FoldAuthorityError(
                "A target source must return T1SubjectTargets, not "
                f"{type(targets).__name__}. A looser return type is how a frame "
                "gets back through a scoped door."
            )
        if targets.subject_id != authorized:
            raise T1FoldAuthorityError(
                f"The target source answered for {targets.subject_id!r} when "
                f"{authorized!r} was authorized. The authority refuses rows it "
                "did not ask for."
            )
        return targets

    def as_dict(self) -> dict[str, Any]:
        """Provenance for the fold selection artifact. Carries no target."""
        return {
            "fold_index": self.fold_index,
            "held_out_subject": self.held_out_subject,
            "scope": self.scope,
            "partition": self.partition,
            "test_accessed": False,
            **self.subject_scope.as_dict(),
        }


def _build(
    fold: T1Fold, *, scope: str, subject_scope: FoldScopedTargetAuthority, source: Any
) -> FoldScopedEvaluationAuthority:
    if scope not in T1_AUTHORITY_SCOPES:
        raise T1FoldAuthorityError(
            f"Scope {scope!r} is not one of {T1_AUTHORITY_SCOPES}."
        )
    if not isinstance(source, T1TargetSource):
        raise T1FoldAuthorityError(
            "A fold authority needs a target source exposing "
            "read_subject_targets(subject_id, *, partition). Nothing wider is "
            "accepted, because a wider source is a global label table with a "
            "narrow door painted on it."
        )
    for subject in subject_scope.authorized_subjects:
        require_known_subject(subject)
    return FoldScopedEvaluationAuthority(
        fold_index=fold.fold_index,
        held_out_subject=fold.held_out_subject,
        scope=scope,
        partition=require_validation_partition(T1_PERMITTED_PARTITION),
        subject_scope=subject_scope,
        source=source,
    )


def fit_evaluation_authority(
    fold: T1Fold, *, source: T1TargetSource
) -> FoldScopedEvaluationAuthority:
    """Selection-time authority: the eleven fit subjects, never the held-out one."""
    known = require_known_fold(fold)
    return _build(
        known,
        scope=SCOPE_FIT,
        subject_scope=fit_authority(known.fit_subjects),
        source=source,
    )


def held_out_evaluation_authority(
    fold: T1Fold, fold_state: dict[str, Any], *, source: T1TargetSource
) -> FoldScopedEvaluationAuthority:
    """Evaluation-time authority: one subject, and only after the barrier.

    `held_out_authority` refuses unless this fold's selection artifact has been
    promoted and re-read with a verified digest, so the barrier stays exactly
    where it already was and this layer adds no second way through it.
    """
    known = require_known_fold(fold)
    return _build(
        known,
        scope=SCOPE_HELD_OUT,
        subject_scope=held_out_authority(known.held_out_subject, fold_state),
        source=source,
    )


def authority_contract() -> dict[str, Any]:
    """What this layer guarantees, as data a receipt can carry."""
    return {
        "authority": TARGET_AUTHORITY,
        "permitted_partition": T1_PERMITTED_PARTITION,
        "sealed_partitions": list(T1_SEALED_PARTITIONS),
        "scopes": list(T1_AUTHORITY_SCOPES),
        "fold_count": T1_FOLD_COUNT,
        "held_out_policy_runs_per_fold": T1_HELD_OUT_POLICY_RUNS_PER_FOLD,
        "global_label_table_permitted": T1_GLOBAL_LABEL_TABLE_PERMITTED,
        "fold_scoped_authority_required": T1_FOLD_SCOPED_TARGET_AUTHORITY_REQUIRED,
        "forbidden_accessors": list(T1_FORBIDDEN_AUTHORITY_ACCESSORS),
        "construction_reads_targets": False,
        "reads_require_an_active_authority": True,
        "test_accessed": False,
    }


def fold_authority_plan(
    subjects: Sequence[str] = T1_VALIDATION_SUBJECTS,
) -> tuple[dict[str, Any], ...]:
    """The authorities a run would build, described without building them.

    Inspectable before authorization: knowing which subject each fold would be
    allowed to see is not the same as being allowed to see it.
    """
    return tuple(
        {
            "fold_index": fold.fold_index,
            "held_out_subject": fold.held_out_subject,
            "fit_subject_count": len(fold.fit_subjects),
            "selection_scope": SCOPE_FIT,
            "evaluation_scope": SCOPE_HELD_OUT,
            "partition": T1_PERMITTED_PARTITION,
        }
        for fold in t1_folds(subjects)
    )
