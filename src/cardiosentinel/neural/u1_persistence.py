"""Claim-bearing U1 persistence: one attempt, one lock, one result.

This follows the M2 convention rather than inventing a second provenance
system. `require_frozen_runtime_record` is imported from `m2_persistence`
unchanged, so a U1 claim rests on exactly the same frozen-digest invariant a
canonical M2 claim did, and the runtime-integrity sentinel is the existing one.

**The directory is the claim.** `u1-v1-development` is created with
`exist_ok=False`; if it exists in ANY state the attempt is consumed and the run
STOPS. Nothing is deleted, reset, renamed, re-rooted, reseeded or retried, and
there is no alternate suite name to fall back to.

**Every claim-bearing artifact takes its own PRE_PROMOTION observation**, as
design §3 of the sentinel requires -- the component results, the top-level
result and the lock alike. The lock's observation is taken BEFORE the lock is
built, so its `runtime_identity_checks` genuinely contains it and its
self-digest is computed only afterwards.

**Two distinct non-completion paths, both additive and both preserved.**

* A saturation census outside its frozen bound is a protocol STOP, not a
  failure: the census is real evidence and is promoted, the receipt proves that
  nothing was fitted, and the status becomes `STOPPED_FOR_HUMAN_REVIEW`.
* An exception after the claim writes a forensic failure receipt beside the
  claim directory -- never inside it, because the original status file is
  historical evidence and is not rewritten to look cleaner.

A raised REPORTING GUARD is neither of those: it is a scientific outcome, the
complete result is promoted normally, and only `human_review_required` and the
flags record it.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.m2_persistence import require_frozen_runtime_record
from cardiosentinel.neural.m2_selection import M2_RETENTION_DECISION_SHA256
from cardiosentinel.neural.p1_experiment import FROZEN_DEPENDENCY_DIGEST
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.provenance import dependency_environment
from cardiosentinel.neural.runtime_sentinel import (
    EnforcementPoint,
    RuntimeIntegrityError,
    RuntimeIntegrityRecord,
    observe_runtime_identity,
    require_runtime_identity,
)
from cardiosentinel.neural.u1_protocol import (
    U1_CLASSIFICATION_THRESHOLD,
    U1_M2_SUITE_SHA256,
    U1_M2G_ARM_RESULT_SHA256,
    U1_M2G_LOCK_SHA256,
    U1_PROTOCOL_SHA256,
    U1_SPLIT_SHA256,
)

STATUS_STARTED: Final = "STARTED"
STATUS_COMPLETE: Final = "COMPLETE"
STATUS_FAILED: Final = "FAILED_OR_INTERRUPTED"
STATUS_STOPPED: Final = "STOPPED_FOR_HUMAN_REVIEW"

RUN_STATUS_NAME: Final = "U1_RUN_STATUS.json"
SATURATION_CENSUS_NAME: Final = "U1_SATURATION_CENSUS.json"
FOLD_MANIFEST_NAME: Final = "U1_FOLD_MANIFEST.json"
OOF_CALIBRATION_NAME: Final = "U1_OOF_CALIBRATION.json"
FAMILY_SELECTION_NAME: Final = "U1_FAMILY_SELECTION.json"
OOF_RESULT_NAME: Final = "U1_OOF_RESULT.json"
DEPLOYMENT_CALIBRATOR_NAME: Final = "U1_DEPLOYMENT_CALIBRATOR.json"
RESULT_NAME: Final = "U1_RESULT.json"
EXPERIMENT_LOCK_NAME: Final = "U1_EXPERIMENT_LOCK.json"
RUNTIME_FAILURE_NAME: Final = "U1_RUNTIME_INTEGRITY_FAILURE.json"

SATURATION_STOP_RECEIPT_NAME: Final = "U1_SATURATION_STOP_RECEIPT.json"
ATTEMPT_FAILURE_RECEIPT_NAME: Final = "U1_ATTEMPT_FAILURE_RECEIPT.json"

STAGING_PREFIX: Final = ".staging-"
ID_SEPARATOR: Final = "__"
EVIDENCE_SUFFIX: Final = f"{ID_SEPARATOR}evidence"
REVIEW_SUFFIX: Final = f"{ID_SEPARATOR}review"

COMPONENT_ARTIFACTS: Final = (
    SATURATION_CENSUS_NAME,
    FOLD_MANIFEST_NAME,
    OOF_CALIBRATION_NAME,
    FAMILY_SELECTION_NAME,
    OOF_RESULT_NAME,
    DEPLOYMENT_CALIBRATOR_NAME,
)

RESULT_CLASS: Final = "u1_v1_canonical_development_result"
LOCK_CLASS: Final = "u1_v1_canonical_development_run_lock"

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")


class U1PersistenceError(RuntimeError):
    """Raised when a claim-bearing U1 artifact cannot be persisted safely."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _require_sha256(label: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.match(value):
        raise U1PersistenceError(f"{label} is not a SHA-256 digest: {value!r}.")
    return value


# --------------------------------------------------------------------------
# Deterministic paths -- no timestamp, no uuid, no random suffix
# --------------------------------------------------------------------------


def u1_run_directory(run_root: Path, experiment_id: str) -> Path:
    if ID_SEPARATOR in str(experiment_id):
        raise U1PersistenceError(
            f"A U1 experiment id may not contain {ID_SEPARATOR!r}; it would make "
            "the evidence and review directories ambiguous."
        )
    return Path(run_root) / str(experiment_id)


def u1_evidence_workspace(run_root: Path, experiment_id: str) -> Path:
    return Path(run_root) / f"{experiment_id}{EVIDENCE_SUFFIX}"


def u1_review_directory(run_root: Path, experiment_id: str) -> Path:
    """Where additive, non-claim-bearing receipts live, OUTSIDE the claim."""
    return Path(run_root) / f"{experiment_id}{REVIEW_SUFFIX}"


def require_unclaimed_u1_attempt(run_root: Path, experiment_id: str) -> dict[str, Any]:
    """Prove nothing from this attempt exists. Run BEFORE any per-window access.

    A pre-existing run directory, staging directory, evidence workspace or
    review directory means the one attempt is consumed. It is never deleted,
    reset, renamed, re-rooted, reseeded or automatically retried, and no
    alternate identity is generated.
    """
    run_dir = u1_run_directory(run_root, experiment_id)
    occupied = [
        str(path)
        for path in (
            run_dir,
            run_dir.parent / f"{STAGING_PREFIX}{run_dir.name}",
            u1_evidence_workspace(run_root, experiment_id),
            u1_review_directory(run_root, experiment_id),
        )
        if path.exists()
    ]
    if occupied:
        raise U1PersistenceError(
            f"Canonical U1 attempt {experiment_id} is already claimed; these "
            f"paths exist: {sorted(occupied)}. The attempt is consumed. Nothing "
            "is deleted, reset, renamed, re-rooted or reseeded, no automatic "
            "retry or alternate name is permitted, and this requires documented "
            "human review."
        )
    return {
        "experiment_id": str(experiment_id),
        "existing_run_directory": False,
        "existing_evidence_workspace": False,
        "existing_review_directory": False,
        "automatic_alternate_name_permitted": False,
        "automatic_retry_permitted": False,
    }


# --------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------


@dataclass(slots=True)
class U1RunDirectory:
    """A claimed canonical U1 attempt and the artifacts promoted into it."""

    run_dir: Path
    experiment_id: str
    started_at: str
    promoted: dict[str, str] = field(default_factory=dict)

    @property
    def staging_dir(self) -> Path:
        return self.run_dir.parent / f"{STAGING_PREFIX}{self.run_dir.name}"


def claim_u1_run_directory(
    run_root: Path, experiment_id: str, *, runtime: RuntimeIntegrityRecord
) -> U1RunDirectory:
    """Atomically claim the one canonical U1 attempt. The directory IS the claim.

    The frozen-digest invariant is checked here, not deferred to final
    validation, because creating the directory is itself claim-bearing.
    """
    require_frozen_runtime_record(runtime)
    require_runtime_identity(
        EnforcementPoint.PRE_PROMOTION, record=runtime, detail="u1_claim_directory"
    )
    run_dir = u1_run_directory(run_root, experiment_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise U1PersistenceError(
            f"Canonical U1 attempt {experiment_id} is already claimed at "
            f"{run_dir}. Automatic rerun, retry, selective rerun and fresh-seed "
            "restart are prohibited and require documented human review."
        ) from error
    claimed = U1RunDirectory(
        run_dir=run_dir, experiment_id=str(experiment_id), started_at=_now()
    )
    write_json_atomic(
        run_dir / RUN_STATUS_NAME,
        {
            "experiment_id": claimed.experiment_id,
            "status": STATUS_STARTED,
            "claim_bearing_result_promoted": False,
            "calibrator_fitting_started": False,
            "started_at": claimed.started_at,
            "updated_at": claimed.started_at,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        },
    )
    return claimed


def promote_component(
    claimed: U1RunDirectory,
    name: str,
    payload: dict[str, Any],
    *,
    runtime: RuntimeIntegrityRecord,
) -> str:
    """Promote one component artifact under its own PRE_PROMOTION observation."""
    if name not in COMPONENT_ARTIFACTS:
        raise U1PersistenceError(f"{name!r} is not a U1 component artifact.")
    if name in claimed.promoted:
        raise U1PersistenceError(f"{name} was already promoted; it is immutable.")
    require_frozen_runtime_record(runtime)
    check = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        expected_digest=runtime.expected_digest,
        detail=f"promote:{name}",
    )
    runtime.record(check)
    if not check.matches:
        raise RuntimeIntegrityError(
            f"Runtime identity differed before promoting {name}. It was NOT "
            "promoted, the environment was NOT repaired and nothing is retried."
        )
    path = claimed.run_dir / name
    write_json_atomic(path, payload)
    digest = sha256_file(path)
    claimed.promoted[name] = digest
    return digest


# --------------------------------------------------------------------------
# The canonical lock -- the ONE provenance construction path
# --------------------------------------------------------------------------

REQUIRED_LOCK_FIELDS: Final = (
    "git_sha",
    "git_dirty",
    "interpreter",
    "package_count",
    "dependency_digest",
    "u1_protocol_sha256",
    "m2_retention_decision_sha256",
    "m2_suite_sha256",
    "m2g_arm_result_sha256",
    "m2g_lock_sha256",
    "split_sha256",
    "primary_population_identity",
    "challenge_population_identity",
    "full_replay_population_identity",
    "fold_assignment_sha256",
    "fold_parameters",
    "pooled_oof_nll",
    "selected_family",
    "oof_evidence_store_sha256",
    "u_star_dev",
    "final_deployment_calibrator_sha256",
    "u_star_deploy",
    "artifact_sha256",
)

_FROZEN_LOCK_EXPECTATIONS: Final = {
    "u1_protocol_sha256": U1_PROTOCOL_SHA256,
    "m2_retention_decision_sha256": M2_RETENTION_DECISION_SHA256,
    "m2_suite_sha256": U1_M2_SUITE_SHA256,
    "m2g_arm_result_sha256": U1_M2G_ARM_RESULT_SHA256,
    "m2g_lock_sha256": U1_M2G_LOCK_SHA256,
    "split_sha256": U1_SPLIT_SHA256,
}


def runtime_provenance() -> dict[str, Any]:
    """Interpreter and dependency facts §25 requires, without importing Torch."""
    environment = dependency_environment()
    return {
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "package_count": int(environment["installed_package_count"]),
        "dependency_digest": str(environment["installed_packages_sha256"]),
    }


def build_u1_run_lock(
    *,
    experiment_id: str,
    runtime: RuntimeIntegrityRecord,
    provenance: dict[str, Any],
    started_at: str,
    completed_at: str,
    artifact_sha256: dict[str, str],
) -> dict[str, Any]:
    """Construct the one canonical U1 run lock. The only identity assembly."""
    git = git_provenance(REPOSITORY_ROOT)
    environment = runtime_provenance()
    lock: dict[str, Any] = {
        "lock_class": LOCK_CLASS,
        "experiment_id": str(experiment_id),
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        **environment,
        "u1_protocol_sha256": U1_PROTOCOL_SHA256,
        "m2_retention_decision_sha256": M2_RETENTION_DECISION_SHA256,
        "m2_suite_sha256": U1_M2_SUITE_SHA256,
        "m2g_arm_result_sha256": U1_M2G_ARM_RESULT_SHA256,
        "m2g_lock_sha256": U1_M2G_LOCK_SHA256,
        "split_sha256": U1_SPLIT_SHA256,
        "classification_threshold": U1_CLASSIFICATION_THRESHOLD,
        "classification_threshold_selected_here": False,
        "m2_replay_invoked": False,
        "m2_rerun_performed": False,
        "primary_population_identity": provenance["primary_population_identity"],
        "challenge_population_identity": provenance["challenge_population_identity"],
        "full_replay_population_identity": provenance[
            "full_replay_population_identity"
        ],
        "fold_assignment_sha256": provenance["fold_assignment_sha256"],
        "fold_parameters": provenance["fold_parameters"],
        "pooled_oof_nll": provenance["pooled_oof_nll"],
        "selected_family": provenance["selected_family"],
        "oof_evidence_store_sha256": provenance["oof_evidence_store_sha256"],
        "u_star_dev": provenance["u_star_dev"],
        "final_deployment_calibrator_sha256": provenance[
            "final_deployment_calibrator_sha256"
        ],
        "u_star_deploy": provenance["u_star_deploy"],
        "runtime_dependency_digest_start": runtime.digest_at(EnforcementPoint.START),
        "runtime_dependency_digest_pre_promotion": runtime.digest_at(
            EnforcementPoint.PRE_PROMOTION
        ),
        "runtime_dependency_digest_end": runtime.digest_at(EnforcementPoint.COMPLETION),
        "runtime_identity_checks": runtime.as_dict(),
        "partition_accessed": "validation",
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "started_at": started_at,
        "completed_at": completed_at,
        "artifact_sha256": dict(artifact_sha256),
        "automatic_retry_performed": False,
        "repeat_attempt_permitted": False,
        "automatic_retention": False,
        "automatic_u2_transition": False,
    }
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    return lock


def validate_u1_run_lock(
    lock: dict[str, Any], *, run_dir: Path | None = None
) -> dict[str, Any]:
    """Validate the ACTUAL values of a U1 lock, not merely the presence of keys."""
    recorded = lock.get("experiment_lock_sha256")
    body = {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise U1PersistenceError("The U1 experiment lock failed digest validation.")
    if lock.get("lock_class") != LOCK_CLASS:
        raise U1PersistenceError(f"Unknown U1 lock class {lock.get('lock_class')!r}.")

    missing = [field_ for field_ in REQUIRED_LOCK_FIELDS if field_ not in lock]
    if missing:
        raise U1PersistenceError(f"The canonical U1 lock is missing {missing}.")
    empty = [
        field_
        for field_ in REQUIRED_LOCK_FIELDS
        if lock[field_] is None and field_ != "git_dirty"
    ]
    if empty:
        raise U1PersistenceError(
            f"The canonical U1 lock carries None where a real identity is "
            f"required: {empty}."
        )

    if not isinstance(lock["git_sha"], str) or not _GIT_SHA_PATTERN.match(
        lock["git_sha"]
    ):
        raise U1PersistenceError(f"git_sha is malformed: {lock['git_sha']!r}.")
    if lock["git_dirty"] is not False:
        raise U1PersistenceError(
            "Canonical U1 evidence requires a clean Git checkout, matching the "
            "existing P1/M1/M2 convention."
        )
    for field_, expected in _FROZEN_LOCK_EXPECTATIONS.items():
        _require_sha256(field_, lock[field_])
        if lock[field_] != expected:
            raise U1PersistenceError(
                f"{field_} is {lock[field_]!r}, expected the frozen {expected!r}."
            )
    _require_sha256("oof_evidence_store_sha256", lock["oof_evidence_store_sha256"])
    _require_sha256("fold_assignment_sha256", lock["fold_assignment_sha256"])
    _require_sha256(
        "final_deployment_calibrator_sha256",
        lock["final_deployment_calibrator_sha256"],
    )
    if lock["classification_threshold"] != U1_CLASSIFICATION_THRESHOLD:
        raise U1PersistenceError("The lock does not bind the frozen M2 threshold.")
    for flag in (
        "classification_threshold_selected_here",
        "m2_replay_invoked",
        "m2_rerun_performed",
        "test_accessed",
        "automatic_retry_performed",
        "repeat_attempt_permitted",
        "automatic_retention",
        "automatic_u2_transition",
    ):
        if lock[flag] is not False:
            raise U1PersistenceError(f"A canonical U1 lock must record {flag}=false.")
    if lock["validation_accessed"] is not True:
        raise U1PersistenceError(
            "Canonical U1 development evidence is computed on VALIDATION."
        )
    if lock["sealed_test_state"] != "unopened":
        raise U1PersistenceError("The B4 sealed test must remain unopened.")

    for label in ("start", "pre_promotion", "end"):
        digest = lock[f"runtime_dependency_digest_{label}"]
        _require_sha256(f"runtime_dependency_digest_{label}", digest)
        if digest != FROZEN_DEPENDENCY_DIGEST:
            raise U1PersistenceError(
                f"runtime_dependency_digest_{label} is not the frozen identity."
            )
    if lock["runtime_identity_checks"].get("all_observations_matched") is not True:
        raise U1PersistenceError(
            "Canonical U1 evidence requires every runtime observation to match."
        )
    if lock["package_count"] != int(runtime_provenance()["package_count"]):
        raise U1PersistenceError(
            "The lock's package count differs from the observed environment."
        )

    artifacts = dict(lock["artifact_sha256"])
    if not artifacts:
        raise U1PersistenceError("A canonical U1 lock binds no artifact hash.")
    for name, digest in artifacts.items():
        _require_sha256(f"artifact_sha256[{name}]", digest)
        if run_dir is not None:
            path = Path(run_dir) / name
            if not path.is_file() or sha256_file(path) != digest:
                raise U1PersistenceError(
                    f"U1 artifact {name} does not match its lock digest."
                )
    if run_dir is not None:
        # The per-row binary evidence is part of the canonical artifact, not an
        # optional extra. A caller verifying a U1 attempt must not have to
        # remember a second validator, so the sibling workspace is derived
        # deterministically from the experiment id and verified here.
        validate_u1_evidence_binding(lock, run_dir=Path(run_dir))
    return lock


def validate_u1_evidence_binding(
    lock: dict[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Verify the sibling OOF evidence workspace against the promoted lock.

    Mutating a `.npz` after the lock was written must break canonical U1
    validation, not merely a separately-invoked store validator. The workspace
    path is derived from `experiment_id`, so no caller supplies it and none can
    point the check at a different directory.
    """
    from cardiosentinel.neural.u1_evidence_store import (
        U1_STORE_MANIFEST_NAME,
        U1EvidenceStoreError,
        validate_u1_evidence_store,
    )

    workspace = u1_evidence_workspace(Path(run_dir).parent, str(lock["experiment_id"]))
    manifest_path = workspace / U1_STORE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise U1PersistenceError(
            f"The canonical U1 attempt binds an OOF evidence store but none is "
            f"present at {manifest_path}."
        )
    try:
        manifest = validate_u1_evidence_store(
            json.loads(manifest_path.read_text()), root=workspace
        )
    except U1EvidenceStoreError as error:
        raise U1PersistenceError(
            f"The U1 OOF evidence store bound by this lock is not intact: {error}"
        ) from error

    if manifest["content_sha256"] != lock["oof_evidence_store_sha256"]:
        raise U1PersistenceError(
            f"The OOF evidence store digests to {manifest['content_sha256']!r}, "
            f"but the lock binds {lock['oof_evidence_store_sha256']!r}. The "
            "per-row evidence has changed since the lock was written."
        )
    if manifest["selected_family"] != lock["selected_family"]:
        raise U1PersistenceError(
            f"The evidence store records selected family "
            f"{manifest['selected_family']!r}, the lock {lock['selected_family']!r}."
        )
    if manifest["fold_assignment_sha256"] != lock["fold_assignment_sha256"]:
        raise U1PersistenceError(
            "The evidence store's fold assignment differs from the lock's."
        )
    if sorted(manifest["row_groups"]) != ["challenge_metric", "primary_metric"]:
        raise U1PersistenceError(
            f"The evidence store carries row groups "
            f"{sorted(manifest['row_groups'])}; PRIMARY and CHALLENGE are "
            "separate groups and neither is optional."
        )
    return {
        "binding_class": "u1_oof_evidence_binding",
        "workspace": str(workspace),
        "content_sha256": manifest["content_sha256"],
        "row_group_sha256": {
            name: entry["sha256"] for name, entry in manifest["row_groups"].items()
        },
        "matches_lock": True,
    }


def validate_canonical_u1_attempt(run_root: Path, experiment_id: str) -> dict[str, Any]:
    """The one complete canonical verification of a promoted U1 attempt.

    Result payload, every component digest, the lock and its provenance, and
    the sibling per-row evidence store -- in one call, so a verifier cannot
    accidentally check only part of the artifact.
    """
    run_dir = u1_run_directory(run_root, experiment_id)
    result_path = run_dir / RESULT_NAME
    lock_path = run_dir / EXPERIMENT_LOCK_NAME
    for path in (result_path, lock_path):
        if not path.is_file():
            raise U1PersistenceError(f"No canonical U1 artifact at {path}.")

    result = json.loads(result_path.read_text())
    validate_u1_result_payload(result)
    for name, digest in result["component_sha256"].items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise U1PersistenceError(
                f"U1 component {name} does not match the digest the result binds."
            )
    lock = validate_u1_run_lock(json.loads(lock_path.read_text()), run_dir=run_dir)
    if (
        result["oof_evidence_store"]["content_sha256"]
        != (lock["oof_evidence_store_sha256"])
    ):
        raise U1PersistenceError(
            "The result and the lock disagree about the OOF evidence store."
        )
    return {
        "verification_class": "u1_canonical_attempt_verification",
        "experiment_id": str(experiment_id),
        "result_sha256": sha256_file(result_path),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "component_sha256": dict(result["component_sha256"]),
        "oof_evidence_store_sha256": lock["oof_evidence_store_sha256"],
        "verified": True,
    }


# --------------------------------------------------------------------------
# The top-level result and its promotion
# --------------------------------------------------------------------------


def validate_u1_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    """A complete U1 result in its own right, before anything is hashed."""
    if result.get("artifact_class") != RESULT_CLASS:
        raise U1PersistenceError(
            f"Unknown U1 result class {result.get('artifact_class')!r}."
        )
    missing = [
        field_
        for field_ in (
            "experiment_id",
            "component_sha256",
            "oof_evidence_store",
            "selected_family",
            "u_star_dev",
            "u_star_deploy",
            "routing_guards",
            "human_review_required",
            "automatic_retention",
            "development_evidence_source",
        )
        if field_ not in result
    ]
    if missing:
        raise U1PersistenceError(f"The U1 result is missing {missing}.")
    components = dict(result["component_sha256"])
    absent = [name for name in COMPONENT_ARTIFACTS if name not in components]
    if absent:
        raise U1PersistenceError(
            f"The U1 result does not bind every component artifact: {absent}."
        )
    for name, digest in components.items():
        _require_sha256(f"component_sha256[{name}]", digest)
    if result["automatic_retention"] is not False:
        raise U1PersistenceError("U1 produces no automatic retention.")
    if result["human_review_required"] is not True:
        raise U1PersistenceError(
            "A U1 result always stops for the human retention review."
        )
    if result.get("development_evidence_source") != "u1_oof_development_calibration":
        raise U1PersistenceError(
            "Every U1 DEVELOPMENT performance number comes from the OOF artifact."
        )
    if result.get("deployment_calibrator_in_sample_performance_reported") is not False:
        raise U1PersistenceError(
            "The final all-validation fit is parameterisation, not evaluation; "
            "its in-sample performance is never a U1 result."
        )
    if result.get("test_accessed") is not False:
        raise U1PersistenceError("A U1 result must record test_accessed=false.")
    if result.get("sealed_test_state") != "unopened":
        raise U1PersistenceError("The B4 sealed test must remain unopened.")
    return result


def finalize_and_promote_u1_result(
    claimed: U1RunDirectory,
    *,
    result: dict[str, Any],
    provenance: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
) -> dict[str, Any]:
    """Validate, gate, promote and lock the one canonical U1 result.

    A raised reporting guard does NOT divert this path: the evidence exists and
    is persisted in full, with `human_review_required` and the flags recording
    the scientific outcome.
    """
    require_frozen_runtime_record(runtime)
    validate_u1_result_payload(result)
    for name in COMPONENT_ARTIFACTS:
        if claimed.promoted.get(name) != result["component_sha256"].get(name):
            raise U1PersistenceError(
                f"The result's digest for {name} differs from the promoted bytes."
            )

    completion = observe_runtime_identity(
        EnforcementPoint.COMPLETION,
        expected_digest=runtime.expected_digest,
        detail=RESULT_NAME,
    )
    runtime.record(completion)
    if not completion.matches:
        _retain_runtime_failure(claimed, completion)
        raise RuntimeIntegrityError(
            "Runtime identity differed at COMPLETION. Canonical promotion is "
            "INVALIDATED: the U1 result was NOT promoted, the environment was "
            "NOT repaired and already-promoted components are retained as "
            "non-claim-bearing forensic material."
        )

    result_check = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        expected_digest=runtime.expected_digest,
        detail=f"promote:{RESULT_NAME}",
    )
    runtime.record(result_check)
    if not result_check.matches:
        _retain_runtime_failure(claimed, result_check)
        raise RuntimeIntegrityError(
            "Runtime identity differed before the U1 result promotion; neither "
            "the result nor the lock was written."
        )
    result_path = claimed.run_dir / RESULT_NAME
    write_json_atomic(result_path, result)
    result_digest = sha256_file(result_path)

    lock_check = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        expected_digest=runtime.expected_digest,
        detail=f"promote:{EXPERIMENT_LOCK_NAME}",
    )
    runtime.record(lock_check)
    if not lock_check.matches:
        _retain_runtime_failure(claimed, lock_check)
        raise RuntimeIntegrityError(
            "Runtime identity differed before the U1 experiment-lock promotion. "
            "The run is NOT COMPLETE and NOT canonical. Already-promoted "
            "evidence is preserved for human review, never deleted or blessed."
        )
    lock = build_u1_run_lock(
        experiment_id=claimed.experiment_id,
        runtime=runtime,
        provenance=provenance,
        started_at=claimed.started_at,
        completed_at=_now(),
        artifact_sha256={
            **{name: digest for name, digest in claimed.promoted.items()},
            RESULT_NAME: result_digest,
        },
    )
    validate_u1_run_lock(lock)
    write_json_atomic(claimed.run_dir / EXPERIMENT_LOCK_NAME, lock)
    validate_u1_run_lock(lock, run_dir=claimed.run_dir)

    status = {
        "experiment_id": claimed.experiment_id,
        "status": STATUS_COMPLETE,
        "claim_bearing_result_promoted": True,
        "canonical": True,
        "started_at": claimed.started_at,
        "updated_at": _now(),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "artifact_sha256": dict(lock["artifact_sha256"]),
        "human_review_required": True,
        "automatic_retention": False,
        "automatic_u2_transition": False,
        "routing_guard_flags_raised": list(
            result["routing_guards"].get("flags_raised", [])
        ),
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "runtime_identity_checks": runtime.as_dict(),
    }
    write_json_atomic(claimed.run_dir / RUN_STATUS_NAME, status)
    return {"result": result, "lock": lock, "status": status}


def _retain_runtime_failure(claimed: U1RunDirectory, check: Any) -> None:
    from cardiosentinel.neural.runtime_sentinel import runtime_failure_record

    git = git_provenance(REPOSITORY_ROOT)
    write_json_atomic(
        claimed.run_dir / RUNTIME_FAILURE_NAME,
        runtime_failure_record(
            check,
            git_sha=str(git["git_sha"]),
            git_dirty=bool(git["git_dirty"]),
            experiment_id=claimed.experiment_id,
        ),
    )


# --------------------------------------------------------------------------
# The two non-completion paths, both additive
# --------------------------------------------------------------------------


def record_saturation_stop(
    run_root: Path,
    claimed: U1RunDirectory,
    *,
    census: dict[str, Any],
    census_sha256: str,
) -> dict[str, Any]:
    """The §3.1 STOP: real evidence, no fitting, no fallback, no M2 rerun.

    This is not a failure receipt. The census is a promoted artifact and the
    receipt proves what did NOT happen: nothing was fitted, no U1 metric
    exists, no clamp was widened and TEST stayed shut.
    """
    receipt = {
        "receipt_class": "u1_saturation_stop_receipt",
        "claim_bearing": False,
        "experiment_id": claimed.experiment_id,
        "claim_state": STATUS_STOPPED,
        "stop_reason": "saturated_fraction_exceeds_frozen_review_bound",
        "saturation_census": dict(census),
        "saturation_census_sha256": _require_sha256("census_sha256", census_sha256),
        "calibrator_fitting_started": False,
        "calibrators_fitted": 0,
        "u1_metrics_produced": False,
        "routing_threshold_derived": False,
        "deployment_calibrator_fitted": False,
        "clamp_widened": False,
        "fallback_calibrator_used": False,
        "m2_rerun_performed": False,
        "automatic_retry_performed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "human_review_required": True,
        "recorded_at": _now(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    directory = u1_review_directory(run_root, claimed.experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / SATURATION_STOP_RECEIPT_NAME, receipt)
    write_json_atomic(
        claimed.run_dir / RUN_STATUS_NAME,
        {
            "experiment_id": claimed.experiment_id,
            "status": STATUS_STOPPED,
            "claim_bearing_result_promoted": False,
            "canonical": False,
            "calibrator_fitting_started": False,
            "stop_reason": receipt["stop_reason"],
            "saturation_stop_receipt_sha256": receipt["receipt_sha256"],
            "started_at": claimed.started_at,
            "updated_at": _now(),
            "human_review_required": True,
            "repeat_attempt_permitted": False,
            "automatic_retry_performed": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        },
    )
    return receipt


def record_u1_attempt_failure(
    run_root: Path,
    claimed: U1RunDirectory,
    *,
    exception: BaseException,
    stage: str,
    exposure: dict[str, Any],
    runtime: RuntimeIntegrityRecord | None = None,
) -> dict[str, Any]:
    """One additive forensic receipt for a post-claim failure.

    The receipt reports the REAL exposure the run reached; it never copies a
    flattering template from an earlier attempt, and it re-reads promotion state
    from the filesystem rather than trusting what the run believed it wrote.
    """
    git = git_provenance(REPOSITORY_ROOT)
    promoted = {
        name: sha256_file(claimed.run_dir / name)
        for name in (*COMPONENT_ARTIFACTS, RESULT_NAME, EXPERIMENT_LOCK_NAME)
        if (claimed.run_dir / name).is_file()
    }
    receipt = {
        "receipt_class": "u1_attempt_failure_receipt",
        "claim_bearing": False,
        "experiment_id": claimed.experiment_id,
        "failed_stage": str(stage),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback_tail": "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )[-4000:],
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "exposure": dict(exposure),
        "promotion_state_source": "filesystem",
        "promoted_artifacts": promoted,
        "runtime_identity_checks": runtime.as_dict() if runtime is not None else None,
        "automatic_retry_performed": False,
        "repeat_attempt_permitted": False,
        "attempt_consumed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "human_review_required": True,
        "recorded_at": _now(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    directory = u1_review_directory(run_root, claimed.experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / ATTEMPT_FAILURE_RECEIPT_NAME, receipt)
    write_json_atomic(
        claimed.run_dir / RUN_STATUS_NAME,
        {
            "experiment_id": claimed.experiment_id,
            "status": STATUS_FAILED,
            "claim_bearing_result_promoted": (claimed.run_dir / RESULT_NAME).is_file(),
            "experiment_lock_promoted": (
                claimed.run_dir / EXPERIMENT_LOCK_NAME
            ).is_file(),
            "promotion_state_source": "filesystem",
            "canonical": False,
            "failed_stage": str(stage),
            "attempt_failure_receipt_sha256": receipt["receipt_sha256"],
            "started_at": claimed.started_at,
            "updated_at": _now(),
            "human_review_required": True,
            "repeat_attempt_permitted": False,
            "automatic_retry_performed": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        },
    )
    return receipt
