"""Claim, stage ordering and artifact promotion for the canonical T1 run.

The run directory is the scientific claim. Everything here exists so that a
claim, once made, describes exactly one attempt honestly: promoted artifacts
are immutable and digest-bound, the stage order is enforced by index rather
than by convention, a failure is recorded additively rather than cleaned up,
and there is no path by which a second attempt can be reached automatically.

This module writes files. It computes no score, generates no threshold, runs
no state machine and reads no upstream evidence -- that is
``t1_development_run``'s work, and it calls in here to record what it did.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.m2_persistence import require_frozen_runtime_record
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.provenance import dependency_environment
from cardiosentinel.neural.runtime_sentinel import (
    EnforcementPoint,
    RuntimeIntegrityRecord,
    require_runtime_identity,
)
from cardiosentinel.neural.t1_execution_spec import (
    T1_DEVELOPMENT_ATTEMPT_ID,
    T1_EXECUTION_SPEC_NAME,
    T1_EXECUTION_SPEC_SHA256,
    T1_EXPERIMENT_IDENTITY,
    T1_FAILURE_RECEIPT_FIELDS,
    T1_PLANNED_ARTIFACTS,
    T1_RUN_ROOT_RELATIVE,
    T1_SEALED_TEST_STATE,
    T1_STAGE_ORDER,
    T1_TEST_ACCESSED,
    T1ExecutionSpecError,
    require_stage_known,
    stage_index,
    validate_t1_execution_spec_document,
)
from cardiosentinel.neural.t1_protocol import (
    T1_PROTOCOL_NAME,
    T1_PROTOCOL_SHA256,
    validate_t1_protocol_document,
)

STATUS_STARTED: Final = "STARTED"
STATUS_COMPLETE: Final = "COMPLETE"
STATUS_FAILED: Final = "FAILED_OR_INTERRUPTED"

# Operational checkpoints between the claim and a terminal status. They exist
# so that an interrupted run says how far it got, which a status written once
# at the claim cannot: a file reading STARTED ten minutes after the process
# died reports a state that did not hold.
#
# These are operational metadata and nothing else. No count, digest, threshold
# or per-row value belongs here -- the promoted artifacts are the evidence, and
# a second place to read a number is a second number to disagree with.
STATUS_PREFLIGHT_COMPLETE: Final = "PREFLIGHT_COMPLETE"
STATUS_LABEL_BLIND_EVIDENCE_COMPLETE: Final = "LABEL_BLIND_EVIDENCE_COMPLETE"
STATUS_FOLDS_COMPLETE: Final = "FOLDS_COMPLETE"
STATUS_OOF_STATE_COMPLETE: Final = "OOF_STATE_COMPLETE"

STATUS_CHECKPOINTS: Final = (
    STATUS_PREFLIGHT_COMPLETE,
    STATUS_LABEL_BLIND_EVIDENCE_COMPLETE,
    STATUS_FOLDS_COMPLETE,
    STATUS_OOF_STATE_COMPLETE,
)
STATUS_SEQUENCE: Final = (
    STATUS_STARTED,
    *STATUS_CHECKPOINTS,
    STATUS_COMPLETE,
)

RUN_STATUS_NAME: Final = "T1_RUN_STATUS.json"
PREFLIGHT_NAME: Final = "T1_PREFLIGHT.json"
INPUT_LINEAGE_NAME: Final = "T1_INPUT_LINEAGE.json"
FOLD_SELECTIONS_NAME: Final = "T1_FOLD_SELECTIONS.json"
OOF_RESULT_NAME: Final = "T1_OOF_RESULT.json"
SUBJECT_EVIDENCE_NAME: Final = "T1_SUBJECT_EVIDENCE.json"
BOOTSTRAP_NAME: Final = "T1_BOOTSTRAP.json"
CHALLENGE_EVIDENCE_NAME: Final = "T1_CHALLENGE_EVIDENCE.json"
FINAL_CONFIGURATION_NAME: Final = "T1_FINAL_CONFIGURATION.json"
RESULT_NAME: Final = "T1_RESULT.json"
EXPERIMENT_LOCK_NAME: Final = "T1_EXPERIMENT_LOCK.json"
FAILURE_RECEIPT_NAME: Final = "T1_FAILURE_RECEIPT.json"

FOLD_SELECTION_DIR: Final = "fold_selections"
HELD_OUT_TRACE_DIR: Final = "held_out_traces"

RESULT_CLASS: Final = "t1_v1_canonical_development_result"
LOCK_CLASS: Final = "t1_v1_canonical_development_experiment_lock"

# Names that must never be reachable. There is no recovery identity, and none
# is predeclared: predeclaring one is how a second attempt becomes reachable
# without a human deciding it should be.
FORBIDDEN_ATTEMPT_SUFFIXES: Final = (
    "-recovery",
    "-recovery1",
    "-recovery2",
    "-retry",
    "-rerun",
    "-attempt2",
    "-v2",
    "-fresh",
    "-reseed",
)


class T1PersistenceError(RuntimeError):
    """Raised when a claim, a stage order or a promotion is not honest."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_run_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    return Path(repository_root) / T1_RUN_ROOT_RELATIVE


def canonical_run_directory(repository_root: Path = REPOSITORY_ROOT) -> Path:
    return canonical_run_root(repository_root) / T1_DEVELOPMENT_ATTEMPT_ID


def require_canonical_attempt_id(attempt_id: str) -> str:
    """The attempt name is frozen and deterministic. Nothing else is claimable."""
    if attempt_id != T1_DEVELOPMENT_ATTEMPT_ID:
        raise T1PersistenceError(
            f"{attempt_id!r} is not the frozen canonical T1 attempt "
            f"{T1_DEVELOPMENT_ATTEMPT_ID!r}. There is exactly one canonical "
            "attempt name: no timestamp, no uuid, no suffix, no alternate."
        )
    lowered = attempt_id.lower()
    for suffix in FORBIDDEN_ATTEMPT_SUFFIXES:
        if lowered.endswith(suffix):
            raise T1PersistenceError(  # pragma: no cover - frozen id has no suffix
                f"{attempt_id!r} looks like a retry identity, which does not exist."
            )
    return attempt_id


def require_unclaimed_canonical_attempt(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Prove nothing from this attempt exists. Run BEFORE any per-row access."""
    run_dir = canonical_run_directory(repository_root)
    occupied = [str(path) for path in (run_dir,) if path.exists()]
    if occupied:
        raise T1PersistenceError(
            f"The canonical T1 attempt is already claimed: {occupied}. It is not "
            "deleted, reset, renamed, re-rooted or reseeded, no alternate name is "
            "chosen automatically, and no automatic retry exists. This requires "
            "documented human review."
        )
    return {
        "attempt_id": T1_DEVELOPMENT_ATTEMPT_ID,
        "existing_run_directory": False,
        "automatic_retry_permitted": False,
        "automatic_alternate_name_permitted": False,
        "recovery_identity_predeclared": False,
    }


def require_authorized_git_identity(authorized_git_sha: str) -> dict[str, Any]:
    """Re-read HEAD and prove it is still the commit the human authorized.

    Called immediately before a claim-bearing promotion. A result written at
    commit A beside a lock written at commit B would be two well-formed
    artifacts describing different code.
    """
    git = git_provenance(REPOSITORY_ROOT)
    if git["git_dirty"] is not False:
        raise T1PersistenceError(
            "The working tree is dirty. Canonical T1 evidence requires a clean "
            "checkout; the attempt is consumed and nothing is repaired or retried."
        )
    if git["git_sha"] != authorized_git_sha:
        raise T1PersistenceError(
            f"HEAD is {git['git_sha']}, but the attempt was authorized for "
            f"{authorized_git_sha}. The authorization names one commit."
        )
    return {
        "authorized_git_sha": authorized_git_sha,
        "git_sha": str(git["git_sha"]),
        "git_dirty": False,
        "git_identity_verified": True,
    }


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class T1StageRecorder:
    """Enforces the frozen stage order by index, and records what was reached.

    A stage may not repeat and may not precede a stage already entered. This is
    what makes "the claim happens before any per-row access" a mechanism rather
    than a promise.
    """

    entered: list[str] = field(default_factory=list)

    def enter(self, stage: str) -> str:
        require_stage_known(stage)
        if stage in self.entered:
            raise T1PersistenceError(
                f"Stage {stage!r} was already entered. A canonical stage runs once; "
                "re-entering one is a retry under another name."
            )
        if self.entered and stage_index(stage) <= stage_index(self.entered[-1]):
            raise T1PersistenceError(
                f"Stage {stage!r} may not follow {self.entered[-1]!r}: the frozen "
                "execution order is not a suggestion."
            )
        self.entered.append(stage)
        return stage

    def require_reached(self, stage: str) -> None:
        require_stage_known(stage)
        if stage not in self.entered:
            raise T1PersistenceError(f"Stage {stage!r} has not been reached.")

    @property
    def current(self) -> str | None:
        return self.entered[-1] if self.entered else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages_entered": list(self.entered),
            "stage_count": len(self.entered),
            "frozen_stage_order": list(T1_STAGE_ORDER),
            "stage_order_enforced": True,
        }


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class T1ClaimedRun:
    """A claimed canonical attempt and the artifacts promoted into it."""

    run_dir: Path
    attempt_id: str
    started_at: str
    authorized_git_sha: str
    runtime: RuntimeIntegrityRecord
    stages: T1StageRecorder
    promoted: dict[str, str] = field(default_factory=dict)

    @property
    def fold_selection_dir(self) -> Path:
        return self.run_dir / FOLD_SELECTION_DIR

    @property
    def held_out_dir(self) -> Path:
        return self.run_dir / HELD_OUT_TRACE_DIR


def claim_canonical_run(
    *,
    authorized_git_sha: str,
    runtime: RuntimeIntegrityRecord,
    stages: T1StageRecorder,
    repository_root: Path = REPOSITORY_ROOT,
) -> T1ClaimedRun:
    """Atomically claim the one canonical T1 attempt. The directory IS the claim."""
    require_frozen_runtime_record(runtime)
    require_canonical_attempt_id(T1_DEVELOPMENT_ATTEMPT_ID)
    run_dir = canonical_run_directory(repository_root)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise T1PersistenceError(
            f"The canonical T1 attempt at {run_dir} is already claimed. Automatic "
            "rerun, retry, selective rerun and fresh-seed restart are prohibited."
        ) from error
    claimed = T1ClaimedRun(
        run_dir=run_dir,
        attempt_id=T1_DEVELOPMENT_ATTEMPT_ID,
        started_at=_now(),
        authorized_git_sha=authorized_git_sha,
        runtime=runtime,
        stages=stages,
    )
    write_json_atomic(
        run_dir / RUN_STATUS_NAME,
        {
            "attempt_id": claimed.attempt_id,
            "experiment_identity": T1_EXPERIMENT_IDENTITY,
            "status": STATUS_STARTED,
            "started_at": claimed.started_at,
            "updated_at": claimed.started_at,
            "authorized_git_sha": authorized_git_sha,
            "label_blind_input_opened": False,
            "held_out_labels_opened_for_folds": [],
            "oof_evidence_promoted": False,
            "final_configuration_completed": False,
            "test_accessed": T1_TEST_ACCESSED,
            "sealed_test_state": T1_SEALED_TEST_STATE,
        },
    )
    return claimed


def read_run_status(claimed: T1ClaimedRun) -> dict[str, Any]:
    """The status as it currently stands on disk."""
    path = claimed.run_dir / RUN_STATUS_NAME
    if not path.is_file():
        raise T1PersistenceError(f"No run status at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_status_checkpoint(
    claimed: T1ClaimedRun | None,
    status: str,
    *,
    stage: str,
    progress: dict[str, Any] | None = None,
) -> Path | None:
    """Record how far a claimed run has got. Operational metadata only.

    Nothing scientific passes through here. The promoted artifacts are the
    evidence and a second place to read a count is a second count to disagree
    with, so a checkpoint carries a status, a stage name, a timestamp and the
    same progress booleans the claim wrote -- and no metric, digest or row.

    Refuses an unknown status, and refuses to move backwards through the frozen
    sequence. A checkpoint that regresses means the stages ran out of order,
    which is a defect worth a refusal rather than a quietly rewritten file.
    Nothing is deleted or repaired: this file is operational, and it is the one
    artifact in the run directory that is meant to be rewritten.
    """
    if claimed is None:
        return None
    if status not in STATUS_SEQUENCE:
        raise T1PersistenceError(
            f"{status!r} is not a T1 run status. Known statuses are "
            f"{list(STATUS_SEQUENCE)}."
        )
    existing = read_run_status(claimed)
    previous = str(existing.get("status", STATUS_STARTED))
    if previous in STATUS_SEQUENCE and STATUS_SEQUENCE.index(
        status
    ) <= STATUS_SEQUENCE.index(previous):
        raise T1PersistenceError(
            f"Run status cannot move from {previous!r} to {status!r}. The "
            "checkpoint sequence is frozen and does not repeat or regress."
        )
    updated = {
        "attempt_id": claimed.attempt_id,
        "experiment_identity": T1_EXPERIMENT_IDENTITY,
        "status": status,
        "stage": stage,
        "started_at": claimed.started_at,
        "updated_at": _now(),
        "authorized_git_sha": claimed.authorized_git_sha,
        "automatic_retry_permitted": False,
        "test_accessed": T1_TEST_ACCESSED,
        "sealed_test_state": T1_SEALED_TEST_STATE,
    }
    for field_name in (
        "label_blind_input_opened",
        "held_out_labels_opened_for_folds",
        "oof_evidence_promoted",
        "final_configuration_completed",
    ):
        updated[field_name] = (progress or {}).get(field_name, existing.get(field_name))
    path = claimed.run_dir / RUN_STATUS_NAME
    write_json_atomic(path, updated)
    return path


def promote(
    claimed: T1ClaimedRun,
    name: str,
    payload: dict[str, Any],
    *,
    point: EnforcementPoint = EnforcementPoint.PRE_PROMOTION,
    detail: str | None = None,
) -> str:
    """Promote one artifact under its own runtime observation. Immutable."""
    if name not in T1_PLANNED_ARTIFACTS:
        raise T1PersistenceError(f"{name!r} is not a planned T1 artifact.")
    if name in claimed.promoted:
        raise T1PersistenceError(f"{name} was already promoted; it is immutable.")
    require_runtime_identity(point, record=claimed.runtime, detail=detail or name)
    path = claimed.run_dir / name
    write_json_atomic(path, payload)
    digest = sha256_file(path)
    claimed.promoted[name] = digest
    return digest


def promote_fold_selection(
    claimed: T1ClaimedRun, fold_index: int, payload: dict[str, Any]
) -> str:
    """Promote one fold's selection, then re-read it and verify its digest.

    The re-read is the point. An artifact that was written but never read back
    is not proof that it was written, and this artifact is the barrier that
    stands between a fold's policy and its held-out labels.
    """
    claimed.fold_selection_dir.mkdir(parents=True, exist_ok=True)
    path = claimed.fold_selection_dir / f"T1_FOLD_{fold_index:02d}_SELECTION.json"
    if path.exists():
        raise T1PersistenceError(f"Fold {fold_index} selection already promoted.")
    require_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        record=claimed.runtime,
        detail=f"fold_selection:{fold_index}",
    )
    write_json_atomic(path, payload)
    digest = sha256_file(path)
    reread = json.loads(path.read_text(encoding="utf-8"))
    if canonical_sha256(reread) != canonical_sha256(payload):
        raise T1PersistenceError(
            f"Fold {fold_index} selection did not read back as written."
        )
    if sha256_file(path) != digest:
        raise T1PersistenceError(  # pragma: no cover - filesystem race
            f"Fold {fold_index} selection changed between promotion and re-read."
        )
    return digest


def write_failure_receipt(
    claimed: T1ClaimedRun | None,
    error: BaseException,
    *,
    state: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> Path | None:
    """Additive receipt. Nothing is deleted, repaired, retried or made to look clean."""
    if claimed is None:
        return None
    receipt = {field: state.get(field) for field in T1_FAILURE_RECEIPT_FIELDS}
    receipt.update(
        {
            "attempt_id": claimed.attempt_id,
            "failed_at": _now(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-8000:],
            "stages_entered": list(claimed.stages.entered),
            "automatic_retry_permitted": False,
            "attempt_consumed": True,
            "test_state": T1_SEALED_TEST_STATE,
            "runtime_integrity_state": claimed.runtime.as_dict(),
        }
    )
    path = claimed.run_dir / FAILURE_RECEIPT_NAME
    write_json_atomic(path, receipt)
    failed_at = str(receipt["failed_at"])
    # The status names the failing stage, the exception type and when it
    # happened, so the operational file answers "what broke and where" without
    # opening the receipt. A status left reading STARTED after a failure
    # reports a state that did not hold, which is worse than no status at all.
    write_json_atomic(
        claimed.run_dir / RUN_STATUS_NAME,
        {
            "attempt_id": claimed.attempt_id,
            "experiment_identity": T1_EXPERIMENT_IDENTITY,
            "status": STATUS_FAILED,
            "stage": receipt.get("stage"),
            "started_at": claimed.started_at,
            "updated_at": failed_at,
            "failed_at": failed_at,
            "exception_type": type(error).__name__,
            "authorized_git_sha": claimed.authorized_git_sha,
            "failure_receipt": FAILURE_RECEIPT_NAME,
            "attempt_consumed": True,
            "automatic_retry_permitted": False,
            "test_accessed": T1_TEST_ACCESSED,
            "sealed_test_state": T1_SEALED_TEST_STATE,
        },
    )
    return path


def build_preflight(
    *, authorized_git_sha: str, upstream: dict[str, Any]
) -> dict[str, Any]:
    """What was proved before the claim, recorded as the claim's first artifact."""
    environment = dependency_environment()
    return {
        "artifact_class": "t1_v1_preflight",
        "attempt_id": T1_DEVELOPMENT_ATTEMPT_ID,
        "experiment_identity": T1_EXPERIMENT_IDENTITY,
        "authorized_git_sha": authorized_git_sha,
        "protocol": {
            "name": T1_PROTOCOL_NAME,
            "document_sha256": validate_t1_protocol_document(),
        },
        "execution_specification": {
            "name": T1_EXECUTION_SPEC_NAME,
            "document_sha256": validate_t1_execution_spec_document(),
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "installed_packages_sha256": environment["installed_packages_sha256"],
            "installed_package_count": environment["installed_package_count"],
        },
        "upstream": dict(upstream),
        "per_row_evidence_opened_before_claim": False,
        "test_accessed": T1_TEST_ACCESSED,
        "sealed_test_state": T1_SEALED_TEST_STATE,
    }


def build_experiment_lock(
    claimed: T1ClaimedRun,
    *,
    artifact_digests: dict[str, str],
    upstream: dict[str, Any],
) -> dict[str, Any]:
    """The single artifact that binds everything this attempt rests on."""
    return {
        "artifact_class": LOCK_CLASS,
        "attempt_id": claimed.attempt_id,
        "experiment_identity": T1_EXPERIMENT_IDENTITY,
        "run_root": str(T1_RUN_ROOT_RELATIVE),
        "authorized_git_sha": claimed.authorized_git_sha,
        "protocol_document_sha256": T1_PROTOCOL_SHA256,
        "execution_spec_document_sha256": T1_EXECUTION_SPEC_SHA256,
        "upstream": dict(upstream),
        "artifact_sha256": dict(sorted(artifact_digests.items())),
        "stage_record": claimed.stages.as_dict(),
        "runtime_identity_checks": claimed.runtime.as_dict(),
        "automatic_retry_performed": False,
        "fold_retry_performed": False,
        "test_accessed": T1_TEST_ACCESSED,
        "sealed_test_state": T1_SEALED_TEST_STATE,
        "routing_defined": False,
        "is_unseen_generalization": False,
    }


def complete_run(
    claimed: T1ClaimedRun, *, result_digests: dict[str, str]
) -> dict[str, Any]:
    """Terminal status. Only reachable once every stage has been entered."""
    finished_at = _now()
    status = {
        "attempt_id": claimed.attempt_id,
        "experiment_identity": T1_EXPERIMENT_IDENTITY,
        "status": STATUS_COMPLETE,
        "started_at": claimed.started_at,
        "updated_at": finished_at,
        "authorized_git_sha": claimed.authorized_git_sha,
        "promoted_artifact_sha256": dict(sorted(result_digests.items())),
        "stages_entered": list(claimed.stages.entered),
        "test_accessed": T1_TEST_ACCESSED,
        "sealed_test_state": T1_SEALED_TEST_STATE,
    }
    write_json_atomic(claimed.run_dir / RUN_STATUS_NAME, status)
    return status


def read_artifact(run_dir: Path, name: str) -> dict[str, Any]:
    path = Path(run_dir) / name
    if not path.is_file():
        raise T1PersistenceError(f"Artifact {name} is missing at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def planned_artifacts() -> Sequence[str]:
    return T1_PLANNED_ARTIFACTS


def require_no_test_path(value: Any) -> Any:
    """Refuse anything that names TEST, before any path is resolved."""
    text = str(value).strip().lower()
    if text == "test" or "/test/" in text or text.endswith("_test"):
        raise T1ExecutionSpecError(
            "TEST is sealed. The canonical T1 development package resolves no "
            "TEST path, reads no TEST metadata and computes no TEST metric."
        )
    return value
