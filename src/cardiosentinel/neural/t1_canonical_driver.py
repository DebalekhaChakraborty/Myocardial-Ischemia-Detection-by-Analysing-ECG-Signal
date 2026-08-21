"""The canonical T1-v1 development driver: the orchestration layer.

`t1_development_run` implements the twenty-nine frozen stages as individually
verifiable methods, and until now nothing sequenced them. That gap was not
cosmetic. A stage recorder proves *an* order was respected; only a driver fixes
*which* order runs, and a choreography that exists as twenty-nine callable
fragments has no single answer to "what would executing this actually do".

This module supplies that answer, and only that answer. It owns the ordering,
binds each frozen stage to the method that enters it, threads each stage's
output into the next, and refuses to begin at all while human authorization is
closed. It computes nothing: every scientific quantity comes from a frozen
component in `t1_protocol` or `t1_development_run`, reached by composition.

**The driver is a capability, not a permission.** Building it changes nothing
about whether the canonical attempt may be consumed. `execute` asks
`t1_config.T1_EXECUTION_SPECIFICATION_AUTHORIZED` before it touches anything,
and while that constant is False the refusal happens before a path is
resolved, a row is read, a label is opened or a directory is created.

**Why the scientific collaborators are injected rather than defined here.**
The label-blind assembly, the per-fold evaluation and the reporting joins need
inputs this module deliberately does not manufacture: evidence-store paths,
the frozen U1 calibrators, the record-to-subject authority, and a fold
evaluator. The last of those is the load-bearing one.
`FoldScopedTargetAuthority` is a permission object with no method that returns
labels -- that absence is the fold firewall, and a test asserts it -- so any
fold evaluator must open labels through a path that does not exist yet.
Defining it here would mean opening labels, which is exactly what a driver PR
must not do. `T1ExecutionCollaborators` therefore states the contract as a
type, and `require_complete` refuses an execution whose collaborators are not
all bound. The gap is enforced and visible instead of latent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from cardiosentinel.neural.t1_config import (
    T1_CANONICAL_DEVELOPMENT_HARNESS_MODULE,
    T1_EXECUTION_SPECIFICATION_AUTHORIZED,
)
from cardiosentinel.neural.t1_development_run import T1DevelopmentRun
from cardiosentinel.neural.t1_execution_spec import (
    STAGE_ASSEMBLE_LABEL_BLIND,
    STAGE_BOOTSTRAP,
    STAGE_CHALLENGE,
    STAGE_CLAIM,
    STAGE_COMPLETION,
    STAGE_EXPERIMENT_LOCK,
    STAGE_FINAL_CONFIGURATION,
    STAGE_FOLD_AUTHORIZE_HELD_OUT,
    STAGE_FOLD_GENERATE_THRESHOLDS,
    STAGE_FOLD_OPEN_FIT_LABELS,
    STAGE_FOLD_OPEN_HELD_OUT_LABELS,
    STAGE_FOLD_PROMOTE_HELD_OUT,
    STAGE_FOLD_PROMOTE_SELECTION,
    STAGE_FOLD_RUN_CANDIDATES,
    STAGE_FOLD_RUN_SELECTED,
    STAGE_FOLD_SELECT,
    STAGE_OOF_RESULT,
    STAGE_OOF_STATE_EVIDENCE,
    STAGE_PROMOTE_INPUT_EVIDENCE,
    STAGE_PROVE_ATTEMPT_ABSENT,
    STAGE_PROVE_TEST_UNOPENED,
    STAGE_START,
    STAGE_VALIDATE_M2,
    STAGE_VALIDATE_PROTOCOL,
    STAGE_VALIDATE_SPEC,
    STAGE_VALIDATE_T2,
    STAGE_VALIDATE_U1,
    STAGE_VERIFY_GIT,
    STAGE_VERIFY_UPSTREAM,
    T1_EXECUTION_SPEC_NAME,
    T1_EXECUTION_SPEC_SHA256,
    T1_PLANNED_ARTIFACTS,
    T1_STAGE_ORDER,
    require_no_test_access,
)

DRIVER_NAME: Final = "T1CanonicalDevelopmentExecutor"

PHASE_PRE_CLAIM: Final = "pre_claim"
PHASE_CLAIM: Final = "claim"
PHASE_POST_CLAIM: Final = "post_claim"


class T1DriverError(RuntimeError):
    """Raised when the canonical driver cannot proceed honestly."""


# ---------------------------------------------------------------------------
# The one canonical ordering
# ---------------------------------------------------------------------------

# Each frozen stage, and the `T1DevelopmentRun` method that enters it. Several
# stages share a method: `stage_preflight` walks the nine pre-claim stages and
# `stage_folds` walks the nine per-fold stages, because each of those sequences
# is indivisible -- a caller that could enter them separately could interleave
# something between a fold's selection promotion and its held-out barrier.
_STAGE_BINDINGS: Final = {
    STAGE_START: "stage_preflight",
    STAGE_VERIFY_GIT: "stage_preflight",
    STAGE_VALIDATE_PROTOCOL: "stage_preflight",
    STAGE_VALIDATE_SPEC: "stage_preflight",
    STAGE_VALIDATE_M2: "stage_preflight",
    STAGE_VALIDATE_U1: "stage_preflight",
    STAGE_VALIDATE_T2: "stage_preflight",
    STAGE_PROVE_TEST_UNOPENED: "stage_preflight",
    STAGE_PROVE_ATTEMPT_ABSENT: "stage_preflight",
    STAGE_CLAIM: "stage_claim",
    STAGE_VERIFY_UPSTREAM: "stage_verify_upstream",
    STAGE_ASSEMBLE_LABEL_BLIND: "stage_assemble_label_blind",
    STAGE_PROMOTE_INPUT_EVIDENCE: "stage_promote_input_evidence",
    STAGE_FOLD_OPEN_FIT_LABELS: "stage_folds",
    STAGE_FOLD_GENERATE_THRESHOLDS: "stage_folds",
    STAGE_FOLD_RUN_CANDIDATES: "stage_folds",
    STAGE_FOLD_SELECT: "stage_folds",
    STAGE_FOLD_PROMOTE_SELECTION: "stage_folds",
    STAGE_FOLD_AUTHORIZE_HELD_OUT: "stage_folds",
    STAGE_FOLD_OPEN_HELD_OUT_LABELS: "stage_folds",
    STAGE_FOLD_RUN_SELECTED: "stage_folds",
    STAGE_FOLD_PROMOTE_HELD_OUT: "stage_folds",
    STAGE_OOF_STATE_EVIDENCE: "stage_oof_state_evidence",
    STAGE_OOF_RESULT: "stage_oof_result",
    STAGE_BOOTSTRAP: "stage_subject_evidence_and_bootstrap",
    STAGE_CHALLENGE: "stage_challenge",
    STAGE_FINAL_CONFIGURATION: "stage_final_configuration",
    STAGE_EXPERIMENT_LOCK: "stage_experiment_lock",
    STAGE_COMPLETION: "stage_completion",
}


@dataclass(frozen=True, slots=True)
class T1ExecutionStep:
    """One frozen stage, its position, and what entering it would cost."""

    index: int
    stage: str
    phase: str
    binding: str
    consumes_attempt: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "stage": self.stage,
            "phase": self.phase,
            "binding": self.binding,
            "consumes_attempt": self.consumes_attempt,
        }


def _build_plan() -> tuple[T1ExecutionStep, ...]:
    """Derived from the frozen order, never re-typed beside it.

    The plan cannot drift from `T1_STAGE_ORDER` because it is generated from
    it. A stage added to the specification without a binding here is a hard
    error at import rather than a stage the driver would silently skip.
    """
    missing = [stage for stage in T1_STAGE_ORDER if stage not in _STAGE_BINDINGS]
    if missing:
        raise T1DriverError(
            f"The frozen stage order contains stages this driver cannot enter: "
            f"{missing}. A driver that skipped them would run a different "
            "choreography than the one the specification froze."
        )
    unknown = sorted(set(_STAGE_BINDINGS) - set(T1_STAGE_ORDER))
    if unknown:
        raise T1DriverError(
            f"This driver binds stages the frozen order does not contain: {unknown}."
        )

    steps: list[T1ExecutionStep] = []
    for index, stage in enumerate(T1_STAGE_ORDER, start=1):
        if stage == STAGE_CLAIM:
            phase = PHASE_CLAIM
        elif index < T1_STAGE_ORDER.index(STAGE_CLAIM) + 1:
            phase = PHASE_PRE_CLAIM
        else:
            phase = PHASE_POST_CLAIM
        steps.append(
            T1ExecutionStep(
                index=index,
                stage=stage,
                phase=phase,
                binding=_STAGE_BINDINGS[stage],
                # The claim is the scientific claim: everything from it onwards
                # consumes the one attempt, and a pre-claim refusal does not.
                consumes_attempt=phase != PHASE_PRE_CLAIM,
            )
        )
    return tuple(steps)


CANONICAL_EXECUTION_PLAN: Final = _build_plan()


# ---------------------------------------------------------------------------
# The permission gate
# ---------------------------------------------------------------------------


def require_canonical_execution_capability() -> None:
    """Refuse while human authorization is closed.

    Asked before anything is resolved, read, opened or created, so a refusal
    leaves the single canonical attempt exactly as it was.
    """
    if not T1_EXECUTION_SPECIFICATION_AUTHORIZED:
        raise T1DriverError(
            "canonical execution capability exists, but human authorization is "
            f"not granted. The specification {T1_EXECUTION_SPEC_NAME} (digest "
            f"{T1_EXECUTION_SPEC_SHA256}) is frozen, the harness "
            f"{T1_CANONICAL_DEVELOPMENT_HARNESS_MODULE} is implemented, and "
            f"{DRIVER_NAME} now sequences all {len(CANONICAL_EXECUTION_PLAN)} "
            "frozen stages -- three capabilities, none of them a permission. "
            "Authorization is a separate human decision, taken by naming the "
            "merged commit, and it has not been taken."
        )


# ---------------------------------------------------------------------------
# What the driver composes but does not manufacture
# ---------------------------------------------------------------------------


# The label-bearing steps the driver threads but is not permitted to define.
# Module scope, not a class attribute: a `Final` inside a `slots=True`
# dataclass becomes a slot descriptor rather than a readable constant.
REQUIRED_COLLABORATOR_CALLABLES: Final = (
    "subject_of_record",
    "evaluate_fold",
    "assemble_oof_state_columns",
    "assemble_oof_result",
    "assemble_subject_evidence",
    "assemble_bootstrap",
    "assemble_challenge",
    "assemble_final_configuration",
)

REQUIRED_COLLABORATOR_PATHS: Final = (
    "m2_row_evidence",
    "t2_identity",
    "t2_selected_scores",
)


@dataclass(frozen=True, slots=True)
class T1ExecutionCollaborators:
    """The frozen components the driver threads together.

    Every field is something that already exists elsewhere or must be supplied
    by a caller that is allowed to open it. None of them is a configuration
    value, a tuning knob or a scientific choice: the paths name canonical
    upstream artifacts, the calibrators are the frozen U1 fits applied without
    refitting, and the callables are the label-bearing steps this module is
    not permitted to define.
    """

    m2_row_evidence: Path
    t2_identity: Path
    t2_selected_scores: Path
    calibrators: Mapping[str, Any]
    subject_of_record: Callable[[str], str]
    evaluate_fold: Callable[..., dict[str, Any]]
    assemble_oof_state_columns: Callable[..., dict[str, Any]]
    assemble_oof_result: Callable[..., dict[str, Any]]
    assemble_subject_evidence: Callable[..., dict[str, Any]]
    assemble_bootstrap: Callable[..., dict[str, Any]]
    assemble_challenge: Callable[..., dict[str, Any]]
    assemble_final_configuration: Callable[..., dict[str, Any]]

    def require_complete(self) -> None:
        """Refuse an execution whose collaborators are not all bound.

        Checked before the claim, so a driver that could not finish never
        starts. An unbound collaborator discovered after the claim would have
        consumed the attempt to learn something knowable in advance.
        """
        for name in REQUIRED_COLLABORATOR_CALLABLES:
            bound = getattr(self, name, None)
            if not callable(bound):
                raise T1DriverError(
                    f"Collaborator {name!r} is not bound. The driver composes "
                    "frozen components and manufactures none of them; a "
                    "label-bearing step it cannot reach is a missing "
                    "capability, and the attempt is not spent discovering it."
                )
        for name in REQUIRED_COLLABORATOR_PATHS:
            if not isinstance(getattr(self, name), Path):
                raise T1DriverError(f"Collaborator {name!r} must be a Path.")
        if not self.calibrators:
            raise T1DriverError(
                "No U1 calibrators were supplied. They are applied as frozen "
                "fits and are never refitted here."
            )


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class T1CanonicalDevelopmentExecutor:
    """The single canonical execution path.

    Owns the stage ordering, calls the existing stage methods in that order,
    threads each result into the next, and records a deterministic receipt per
    step. There is exactly one path through `execute`: no retry, no recovery,
    no resume, no alternate run root, no seed, fold or subject override and no
    TEST option. A failure at any stage propagates unchanged -- the driver has
    no exception handler, because swallowing a refusal is how a consumed
    attempt turns into a second one.
    """

    run: T1DevelopmentRun
    receipts: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.receipts is None:
            self.receipts = []

    # -- inspection, available without authorization -------------------------

    @staticmethod
    def plan() -> tuple[T1ExecutionStep, ...]:
        """The canonical ordering. Pure, side-effect free, always available.

        Readable without authorization on purpose: knowing what an execution
        would do is not permission to do it, and a reviewer must be able to
        audit the choreography without arming it.
        """
        return CANONICAL_EXECUTION_PLAN

    @staticmethod
    def stage_receipts() -> tuple[dict[str, Any], ...]:
        """A deterministic receipt per frozen stage, in order."""
        return tuple(step.as_dict() for step in CANONICAL_EXECUTION_PLAN)

    @staticmethod
    def planned_artifacts() -> tuple[str, ...]:
        """The artifacts a completed run would have promoted, in frozen order."""
        return tuple(T1_PLANNED_ARTIFACTS)

    @staticmethod
    def verify_collaborators(
        collaborators: T1ExecutionCollaborators,
    ) -> dict[str, Any]:
        """Prove the execution collaborators are all bound, without executing.

        Deliberately available without authorization and outside `execute`:
        "could this run" and "may this run" are different questions, and a
        reviewer must be able to ask the first without answering the second.
        Nothing is claimed, read, opened or created, and the permission gate is
        not consulted because no permission is being exercised.
        """
        collaborators.require_complete()
        return {
            "collaborators_complete": True,
            "required_callables": list(REQUIRED_COLLABORATOR_CALLABLES),
            "required_paths": list(REQUIRED_COLLABORATOR_PATHS),
            "execution_authorized": bool(T1_EXECUTION_SPECIFICATION_AUTHORIZED),
            "executed": False,
            "attempt_consumed": False,
        }

    @staticmethod
    def validate_artifact_plan() -> tuple[str, ...]:
        """Stage 25: the artifact plan is the specification's, not the driver's."""
        planned = tuple(T1_PLANNED_ARTIFACTS)
        if len(set(planned)) != len(planned):
            raise T1DriverError(f"The artifact plan repeats a name: {planned}.")
        for name in planned:
            require_no_test_access(Path(name).stem.lower())
        return planned

    # -- the one execution path ----------------------------------------------

    def execute(self, collaborators: T1ExecutionCollaborators) -> dict[str, Any]:
        """Walk the twenty-nine frozen stages once, in order.

        Refuses before anything is resolved while authorization is closed, so
        today this function never reaches a filesystem, a row or a label.
        """
        require_canonical_execution_capability()
        collaborators.require_complete()
        self.validate_artifact_plan()

        # -- stages 1-9: pre-claim verification ------------------------------
        preflight = self.run.stage_preflight()
        self._record(STAGE_PROVE_ATTEMPT_ABSENT, "pre_claim_verification_passed")

        # -- stage 10: the claim; the attempt is spent from here -------------
        claimed = self.run.stage_claim(preflight)
        self._record(STAGE_CLAIM, str(claimed.run_dir))

        # -- stage 11: upstream identity must not have moved -----------------
        self.run.stage_verify_upstream()
        self._record(STAGE_VERIFY_UPSTREAM, "upstream_identity_stable")

        # -- stages 12-13: the label-blind timeline --------------------------
        columns = self.run.stage_assemble_label_blind(
            m2_row_evidence=collaborators.m2_row_evidence,
            t2_identity=collaborators.t2_identity,
            t2_scores=collaborators.t2_selected_scores,
            calibrators=collaborators.calibrators,
            subject_of_record=collaborators.subject_of_record,
        )
        self._record(STAGE_ASSEMBLE_LABEL_BLIND, "label_blind_timeline_assembled")

        input_manifest = self.run.stage_promote_input_evidence(columns)
        self._record(
            STAGE_PROMOTE_INPUT_EVIDENCE, str(input_manifest["content_sha256"])
        )

        # -- stages 14-22: twelve folds behind twelve label barriers ---------
        selections = self.run.stage_folds(evaluate_fold=collaborators.evaluate_fold)
        self._record(STAGE_FOLD_PROMOTE_HELD_OUT, f"folds_completed={len(selections)}")

        # -- stages 23-24: cross-fitted evidence and its result --------------
        oof_columns = collaborators.assemble_oof_state_columns(
            columns=columns, selections=selections
        )
        oof_manifest = self.run.stage_oof_state_evidence(
            oof_columns,
            fold_selection_sha256=_fold_selection_digest(selections),
        )
        self._record(STAGE_OOF_STATE_EVIDENCE, str(oof_manifest["content_sha256"]))

        oof_digest = self.run.stage_oof_result(
            collaborators.assemble_oof_result(
                oof_columns=oof_columns, selections=selections
            )
        )
        self._record(STAGE_OOF_RESULT, oof_digest)

        # -- stage 25: subject evidence and the frozen bootstrap -------------
        subject_digest, bootstrap_digest = (
            self.run.stage_subject_evidence_and_bootstrap(
                subject_evidence=collaborators.assemble_subject_evidence(
                    oof_columns=oof_columns
                ),
                bootstrap=collaborators.assemble_bootstrap(oof_columns=oof_columns),
            )
        )
        self._record(STAGE_BOOTSTRAP, f"{subject_digest}:{bootstrap_digest}")

        # -- stage 26: challenge annotation, joined after the state trace ----
        challenge_digest = self.run.stage_challenge(
            collaborators.assemble_challenge(oof_columns=oof_columns)
        )
        self._record(STAGE_CHALLENGE, challenge_digest)

        # -- stage 27: deployment configuration, never development evidence --
        configuration_digest = self.run.stage_final_configuration(
            collaborators.assemble_final_configuration(
                oof_columns=oof_columns, selections=selections
            )
        )
        self._record(STAGE_FINAL_CONFIGURATION, configuration_digest)

        # -- stages 28-29: lock and completion -------------------------------
        lock_digest = self.run.stage_experiment_lock()
        self._record(STAGE_EXPERIMENT_LOCK, lock_digest)

        completion = self.run.stage_completion()
        self._record(STAGE_COMPLETION, "run_complete")

        return {
            "driver": DRIVER_NAME,
            "stages_entered": list(self.run.stages.entered),
            "stage_count": len(CANONICAL_EXECUTION_PLAN),
            "receipts": list(self.receipts),
            "input_evidence_sha256": input_manifest["content_sha256"],
            "oof_state_evidence_sha256": oof_manifest["content_sha256"],
            "oof_result_sha256": oof_digest,
            "experiment_lock_sha256": lock_digest,
            "completion": completion,
            "retry_performed": False,
            "recovery_performed": False,
            "test_accessed": False,
        }

    def _record(self, stage: str, detail: str) -> None:
        self.receipts.append(
            {"index": len(self.receipts) + 1, "stage": stage, "detail": detail}
        )


def _fold_selection_digest(selections: Sequence[dict[str, Any]]) -> str:
    """The per-fold selection digests, bound in the frozen fold order."""
    from cardiosentinel.neural.integrity import canonical_sha256

    return canonical_sha256(
        [str(selection["selection_sha256"]) for selection in selections]
    )
