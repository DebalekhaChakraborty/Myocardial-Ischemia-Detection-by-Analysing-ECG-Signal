"""The final all-VALIDATION configuration selection (spec section 23).

**This is a separate scientific selection event, not a summary of the folds.**
The twelve fold selections each chose a policy from eleven subjects to evaluate
on a twelfth. This chooses one policy from all twelve, using the same candidate
set, the same threshold rules, the same transition logic and the same
lexicographic order. Its answer may coincide with a fold's and it is not
derived from one; nothing here reads a fold selection, a fold trace or the OOF
result.

**What it is for, and what it is not.** The specification is explicit: this is
deployment and test configuration only. Its in-sample all-VALIDATION
performance is *not* T1 development evidence, and it must never replace or
overwrite the OOF result. So this module returns the seven frozen configuration
fields and deliberately returns no metric alongside them -- a performance
number emitted here would eventually be quoted as though it were development
evidence, and the cheapest way to prevent that is to never produce it.

**Why a new authority type rather than a new scope.**
`t1_fold_authority.T1_AUTHORITY_SCOPES` is exactly ``(fit, held_out)`` and a
test asserts it. Adding a third would let a fold-path caller construct an
all-twelve authority, and the fold firewall is precisely the guarantee that no
such object is reachable from there. `FinalValidationAuthority` is therefore a
distinct type: the fold evaluator refuses it because it is not a
`FoldScopedEvaluationAuthority`, and this module refuses a fold authority
because it is not a `FinalValidationAuthority`. Neither can be used where the
other belongs, and that is enforced by type rather than by a scope string.

**It cannot represent TEST.** The partition is not a parameter. It is fixed to
the one permitted value, re-proved on every read, and the sealed names are
refused by the same helper the fold path uses.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final, Iterator, Mapping

from cardiosentinel.neural.t1_development_run import (
    generate_thresholds,
    run_policy_over_streams,
    score_policy,
    select_policy,
)
from cardiosentinel.neural.t1_execution_spec import (
    T1_CANDIDATE_POLICIES_PER_FOLD,
    require_no_test_access,
)
from cardiosentinel.neural.t1_fold_authority import (
    T1_PERMITTED_PARTITION,
    T1_SEALED_PARTITIONS,
    ScopedTargetRequest,
    T1SubjectTargets,
    T1TargetSource,
    _authorized_request,
    require_validation_partition,
)
from cardiosentinel.neural.t1_fold_evaluator import (
    _build_fold_view,
    background_negative_population,
)
from cardiosentinel.neural.t1_protocol import (
    T1_VALIDATION_SUBJECTS,
    candidate_policies,
)

SCOPE_FINAL_VALIDATION: Final = "all_validation_subjects_final_configuration_only"

# The seven fields section 23 names, in the order it names them. Nothing else is
# persisted from this selection.
FINAL_CONFIGURATION_FIELDS: Final = (
    "q_watch",
    "q_event",
    "p_watch",
    "s_watch",
    "p_event",
    "s_event",
    "persistence_profile",
)

# Stated as data so a test can assert it rather than infer it from absence.
FINAL_CONFIGURATION_IS_DEVELOPMENT_EVIDENCE: Final = False
FINAL_CONFIGURATION_REPORTS_PERFORMANCE: Final = False


class T1FinalConfigurationError(RuntimeError):
    """Raised when the final selection cannot proceed honestly."""


@dataclass(frozen=True, slots=True)
class FinalValidationAuthority:
    """All twelve VALIDATION subjects, for one purpose, once.

    Deliberately not a fold authority and deliberately not constructible from
    one. It carries the complete frozen roster because section 23 requires all
    twelve, and it carries nothing else: there is no fold index, no held-out
    subject and no method that returns more than one subject's targets.
    """

    subjects: tuple[str, ...]
    source: T1TargetSource = field(repr=False, compare=False)
    partition: str = T1_PERMITTED_PARTITION
    scope: str = SCOPE_FINAL_VALIDATION

    def __post_init__(self) -> None:
        if self.scope != SCOPE_FINAL_VALIDATION:
            raise T1FinalConfigurationError(
                f"This authority has one scope, {SCOPE_FINAL_VALIDATION!r}, "
                f"not {self.scope!r}."
            )
        require_validation_partition(self.partition)
        require_no_test_access(self.partition)
        for sealed in T1_SEALED_PARTITIONS:
            if str(self.partition).strip().lower() == sealed.strip().lower():
                raise T1FinalConfigurationError(  # pragma: no cover - belt and braces
                    f"{sealed!r} is sealed and resolves to no authority."
                )
        observed = tuple(self.subjects)
        if len(set(observed)) != len(observed):
            raise T1FinalConfigurationError(
                "A subject appears more than once in the final roster; each of "
                "the twelve is included exactly once."
            )
        if sorted(observed) != sorted(T1_VALIDATION_SUBJECTS):
            missing = sorted(set(T1_VALIDATION_SUBJECTS) - set(observed))
            extra = sorted(set(observed) - set(T1_VALIDATION_SUBJECTS))
            raise T1FinalConfigurationError(
                "The final configuration selection is over all twelve "
                f"VALIDATION subjects. Missing {missing}; unexpected {extra}. "
                "A partial roster would be a thirteenth fold wearing this "
                "selection's name."
            )
        if not isinstance(self.source, T1TargetSource):
            raise T1FinalConfigurationError(
                "A final authority needs a target source exposing "
                "read_subject_targets(subject_id, *, partition)."
            )

    @property
    def authorized_subjects(self) -> tuple[str, ...]:
        return tuple(sorted(self.subjects))

    def require_authorized(self, subject_id: str) -> str:
        subject = str(subject_id)
        if subject not in set(self.subjects):
            raise T1FinalConfigurationError(
                f"{subject!r} is not a VALIDATION subject and is refused."
            )
        return subject

    def targets_for_subject(self, subject_id: str) -> T1SubjectTargets:
        """One subject at a time, even here.

        The roster is complete but the accessor is not widened: there is no
        method that returns every subject's targets at once, for the same
        reason the fold authority has none.
        """
        authorized = self.require_authorized(subject_id)
        partition = require_validation_partition(self.partition)
        request = ScopedTargetRequest(
            fold_index=-1,
            subject_id=authorized,
            scope=self.scope,
            partition=partition,
        )
        with _authorized_request(request):
            targets = self.source.read_subject_targets(authorized, partition=partition)
        if not isinstance(targets, T1SubjectTargets):
            raise T1FinalConfigurationError(
                "A target source must return T1SubjectTargets, not "
                f"{type(targets).__name__}."
            )
        if targets.subject_id != authorized:
            raise T1FinalConfigurationError(
                f"The target source answered for {targets.subject_id!r} when "
                f"{authorized!r} was authorized."
            )
        return targets

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "partition": self.partition,
            "subject_count": len(self.subjects),
            "subjects": list(self.authorized_subjects),
            "is_a_fold_authority": False,
            "fold_index": None,
            "held_out_subject": None,
            "test_accessed": False,
        }


def final_validation_authority(*, source: T1TargetSource) -> FinalValidationAuthority:
    """The one constructor. Always the complete frozen roster.

    The subject list is not a parameter: a caller that could choose it could
    choose eleven, and eleven is a fold.
    """
    return FinalValidationAuthority(
        subjects=tuple(T1_VALIDATION_SUBJECTS), source=source
    )


@contextmanager
def _no_fold_authority_in_scope() -> Iterator[None]:
    """Documentation as code: this selection borrows nothing from a fold."""
    yield


def select_final_validation_configuration(
    *, columns: Mapping[str, Any], authority: FinalValidationAuthority
) -> dict[str, Any]:
    """The thirteenth selection: all twelve subjects, the same frozen rules.

    Same candidate enumeration, same order-statistic threshold rule, same
    transition function, same lexicographic `policy_sort_key`. The only thing
    that differs from a fold is the population, which is the whole point.

    Returns the seven configuration fields and no performance number.
    """
    if not isinstance(authority, FinalValidationAuthority):
        raise T1FinalConfigurationError(
            "The final selection takes a FinalValidationAuthority, not "
            f"{type(authority).__name__}. A fold authority scopes eleven "
            "subjects or one, and section 23 is over all twelve."
        )
    require_no_test_access(authority.partition)

    view = _build_fold_view(columns, authority, authority.authorized_subjects)
    if len(view.subjects) != len(T1_VALIDATION_SUBJECTS):  # pragma: no cover
        raise T1FinalConfigurationError(
            f"The assembled view covers {len(view.subjects)} subjects, not "
            f"{len(T1_VALIDATION_SUBJECTS)}."
        )
    background = background_negative_population(view)

    rows_by_stream = {key: view.streams[key].rows for key in view.streams}
    start_samples = {key: view.streams[key].start_samples for key in view.streams}
    positives = {key: view.streams[key].primary_positive for key in view.streams}
    masks = {key: view.streams[key].primary_mask for key in view.streams}

    policies = candidate_policies()
    if len(policies) != T1_CANDIDATE_POLICIES_PER_FOLD:  # pragma: no cover
        raise T1FinalConfigurationError(
            f"{len(policies)} candidates enumerated; the design is "
            f"{T1_CANDIDATE_POLICIES_PER_FOLD}."
        )

    thresholds_by_name = {}
    scored: dict[str, dict[str, Any]] = {}
    for policy in policies:
        thresholds = generate_thresholds(
            policy,
            background_p=background["background_p"],
            background_s=background["background_s"],
            stable_ids=background["stable_ids"],
        )
        thresholds_by_name[policy.name] = thresholds
        scored[policy.name] = score_policy(
            run_policy_over_streams(rows_by_stream, thresholds, policy),
            start_samples=start_samples,
            primary_positive=positives,
            primary_mask=masks,
        )

    selected = select_policy(scored, policies)
    chosen = thresholds_by_name[selected.name]
    return {
        "artifact_class": "t1_v1_final_validation_configuration",
        "selection_scope": authority.as_dict(),
        "selected_policy_id": selected.name,
        "candidate_count": len(policies),
        "candidate_order": [policy.name for policy in policies],
        "threshold_population": background["population"],
        "threshold_population_row_count": background["row_count"],
        "configuration": {
            "q_watch": float(selected.q_watch),
            "q_event": float(selected.q_event),
            "p_watch": float(chosen.p_watch),
            "s_watch": float(chosen.s_watch),
            "p_event": float(chosen.p_event),
            "s_event": float(chosen.s_event),
            "persistence_profile": selected.profile.name,
        },
        # Said out loud, in the artifact, so a reader cannot mistake it.
        "is_development_evidence": FINAL_CONFIGURATION_IS_DEVELOPMENT_EVIDENCE,
        "reports_performance": FINAL_CONFIGURATION_REPORTS_PERFORMANCE,
        "derived_from_fold_selections": False,
        "test_accessed": False,
    }


def final_configuration_capability() -> dict[str, Any]:
    """What this layer provides, as data a receipt can carry."""
    return {
        "scope": SCOPE_FINAL_VALIDATION,
        "subject_count": len(T1_VALIDATION_SUBJECTS),
        "candidate_policies": T1_CANDIDATE_POLICIES_PER_FOLD,
        "fields": list(FINAL_CONFIGURATION_FIELDS),
        "is_development_evidence": FINAL_CONFIGURATION_IS_DEVELOPMENT_EVIDENCE,
        "reports_performance": FINAL_CONFIGURATION_REPORTS_PERFORMANCE,
        "test_accessed": False,
    }
