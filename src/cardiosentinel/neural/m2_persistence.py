"""Claim-bearing M2 result/run-lock persistence with staged-then-promote safety.

A partial or interrupted run must never be able to look complete. Canonical
artifacts are written to a staging directory first and promoted only after
every gate below passes, in this order:

1. computation finished successfully;
2. required artifacts are internally valid;
3. the runtime-integrity PRE_PROMOTION check is GREEN;
4. the forbidden-partition audit is GREEN;
5. output hashes are computed.

After promotion the COMPLETION runtime check is taken. If it differs from the
frozen identity the attempt is recorded as non-promotable/non-canonical per the
runtime-integrity policy -- it is never silently blessed.

Run status follows the repository's existing convention: `STARTED`,
`COMPLETE`, `FAILED_OR_INTERRUPTED`. A claim is expressed only by a promoted
`COMPLETE` artifact.

**The runner never selects an arm.** A two-arm M2 suite carries
`memory_selection_performed: false` and `memory_selected: null` until a human
review that this authorization explicitly does not perform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.neural import m2_gate as GATE
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.m2_execution import (
    FORBIDDEN_PARTITIONS,
    M2ExecutionError,
    require_permitted_partition,
)
from cardiosentinel.neural.m2_policy import M2_ARMS
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.runtime_sentinel import (
    EnforcementPoint,
    RuntimeIntegrityError,
    RuntimeIntegrityRecord,
    observe_runtime_identity,
    require_runtime_identity,
    runtime_failure_record,
)

STATUS_STARTED: Final = "STARTED"
STATUS_COMPLETE: Final = "COMPLETE"
STATUS_FAILED: Final = "FAILED_OR_INTERRUPTED"

RUN_STATUS_NAME: Final = "M2_RUN_STATUS.json"
ARM_RESULT_NAME: Final = "M2_ARM_RESULT.json"
SUITE_RESULT_NAME: Final = "M2_SUITE_RESULT.json"
RUNTIME_FAILURE_NAME: Final = "M2_RUNTIME_INTEGRITY_FAILURE.json"
STAGING_PREFIX: Final = ".staging-"


class M2PersistenceError(RuntimeError):
    """Raised when a claim-bearing M2 artifact cannot be persisted safely."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# Provenance binding required of every canonical run
# --------------------------------------------------------------------------

REQUIRED_PROVENANCE_FIELDS: Final = (
    "experiment_id",
    "arm",
    "git_sha",
    "git_dirty",
    "m2_protocol_sha256",
    "m2_gate_receipt_sha256",
    "m1_retention_decision_sha256",
    "retained_m1l_lock_sha256",
    "retained_m1l_checkpoint_sha256",
    "p1b_lock_sha256",
    "b4b_checkpoint_sha256",
    "distance_standardizer_sha256",
    "split_sha256",
    "feature_corpus_sha256",
    "ordered_chronology_sha256",
    "stream_cache_sha256",
    "signal_v1_schema_sha256",
    "morphology_v1_schema_sha256",
    "combined_v1_schema_sha256",
    "runtime_dependency_digest_start",
    "runtime_dependency_digest_pre_promotion",
    "runtime_dependency_digest_end",
    "partition_accessed",
    "validation_accessed",
    "test_accessed",
    "sealed_test_state",
    "m1l_classification_threshold",
    "normal_evidence_threshold",
    "started_at",
    "completed_at",
    "artifact_sha256",
)


def build_run_provenance(
    *,
    experiment_id: str,
    arm: str,
    execution_identity: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
    started_at: str,
    completed_at: str,
    artifact_sha256: dict[str, str],
) -> dict[str, Any]:
    """Bind every identity a canonical M2 run must carry."""
    from cardiosentinel.features.schema import COMBINED_V1, MORPHOLOGY_V1, SIGNAL_V1
    from cardiosentinel.neural.m2_scorer import (
        M1L_CLASSIFICATION_THRESHOLD,
        NORMAL_EVIDENCE_THRESHOLD,
    )

    if arm not in M2_ARMS:
        raise M2PersistenceError(f"Unknown M2 arm {arm!r}.")
    provenance = git_provenance(REPOSITORY_ROOT)
    inputs = execution_identity["input_identity"]
    scorer = execution_identity["scorer_identity"]
    partition = require_permitted_partition(inputs["partition"])

    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "arm": arm,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "m2_protocol_sha256": GATE.M2_PROTOCOL_SHA256,
        "m2_gate_receipt_sha256": GATE.M2_GATE_RECEIPT_SHA256,
        "m1_retention_decision_sha256": sha256_file(
            REPOSITORY_ROOT / "docs" / "M1_MEMORY_RETENTION_DECISION_V1.md"
        ),
        "retained_m1l_lock_sha256": scorer["retained_lock_sha256"],
        "retained_m1l_checkpoint_sha256": scorer["retained_checkpoint_sha256"],
        "p1b_lock_sha256": scorer["p1b_lock_sha256"],
        "b4b_checkpoint_sha256": scorer["b4b_checkpoint_sha256"],
        "distance_standardizer_sha256": inputs["distance_standardizer_sha256"],
        "split_sha256": inputs["split_sha256"],
        "feature_corpus_sha256": inputs["feature_corpus_sha256"],
        "ordered_chronology_sha256": inputs["ordered_chronology_sha256"],
        "stream_cache_sha256": inputs["stream_cache_sha256"],
        "signal_v1_schema_sha256": SIGNAL_V1.sha256,
        "morphology_v1_schema_sha256": MORPHOLOGY_V1.sha256,
        "combined_v1_schema_sha256": COMBINED_V1.sha256,
        "runtime_dependency_digest_start": runtime.digest_at(EnforcementPoint.START),
        "runtime_dependency_digest_pre_promotion": runtime.digest_at(
            EnforcementPoint.PRE_PROMOTION
        ),
        "runtime_dependency_digest_end": runtime.digest_at(EnforcementPoint.COMPLETION),
        "runtime_identity_checks": runtime.as_dict(),
        "partition_accessed": partition,
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "m1l_classification_threshold": M1L_CLASSIFICATION_THRESHOLD,
        "normal_evidence_threshold": NORMAL_EVIDENCE_THRESHOLD,
        "classification_threshold_used_for_admission": False,
        "classifier_retrained": False,
        "threshold_selected_during_run": False,
        "rollback": False,
        "started_at": started_at,
        "completed_at": completed_at,
        "artifact_sha256": dict(artifact_sha256),
        "automatic_retry_performed": False,
        "repeat_attempt_permitted": False,
    }
    missing = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in payload]
    if missing:
        raise M2PersistenceError(f"Run provenance is missing {missing}.")
    return payload


def build_suite_result(
    *,
    suite_id: str,
    arm_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """A two-arm suite that expresses no retention decision.

    `memory_selection_performed` stays false and `memory_selected` stays null
    until a human review. The runner never automatically prefers M2-G.
    """
    if set(arm_results) != set(M2_ARMS):
        raise M2PersistenceError(
            f"An M2 suite binds exactly {M2_ARMS}; received {sorted(arm_results)}."
        )
    payload: dict[str, Any] = {
        "suite_id": suite_id,
        "suite_class": "m2_v1_two_arm_suite",
        "arms": list(M2_ARMS),
        "arm_results": arm_results,
        "memory_selection_performed": False,
        "memory_selected": None,
        "automatic_arm_preference_applied": False,
        "human_review_required": True,
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "rollback_evaluated": False,
    }
    payload["m2_suite_sha256"] = canonical_sha256(payload)
    return payload


# --------------------------------------------------------------------------
# Staged-then-promote persistence
# --------------------------------------------------------------------------


@dataclass(slots=True)
class M2RunDirectory:
    """A claimed canonical run directory with a staging area."""

    run_dir: Path
    experiment_id: str
    arm: str
    started_at: str

    @property
    def staging_dir(self) -> Path:
        return self.run_dir.parent / f"{STAGING_PREFIX}{self.run_dir.name}"


def claim_run_directory(run_root: Path, experiment_id: str, arm: str) -> M2RunDirectory:
    """Atomically claim the one canonical attempt. The directory IS the claim.

    There is no force, overwrite, retry, reseed or delete path. If the
    directory already exists in any state the attempt is consumed and this
    refuses, exactly as the M1 convention does.
    """
    if arm not in M2_ARMS:
        raise M2PersistenceError(f"Unknown M2 arm {arm!r}.")
    run_dir = Path(run_root) / experiment_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise M2PersistenceError(
            f"Canonical M2 run {experiment_id} is already claimed at {run_dir}. "
            "Automatic rerun, retry, selective rerun and fresh-seed restart are "
            "prohibited and require documented human review."
        ) from error
    claimed = M2RunDirectory(
        run_dir=run_dir, experiment_id=experiment_id, arm=arm, started_at=_now()
    )
    write_json_atomic(
        run_dir / RUN_STATUS_NAME,
        {
            "experiment_id": experiment_id,
            "arm": arm,
            "status": STATUS_STARTED,
            "claim_bearing_result_promoted": False,
            "started_at": claimed.started_at,
            "updated_at": claimed.started_at,
        },
    )
    return claimed


def record_failure(
    claimed: M2RunDirectory, reason: str, *, runtime_check: Any | None = None
) -> dict[str, Any]:
    """Mark an attempt FAILED_OR_INTERRUPTED. The claim is never released."""
    provenance = git_provenance(REPOSITORY_ROOT)
    payload = {
        "experiment_id": claimed.experiment_id,
        "arm": claimed.arm,
        "status": STATUS_FAILED,
        "claim_bearing_result_promoted": False,
        "reason": reason,
        "started_at": claimed.started_at,
        "updated_at": _now(),
        "human_review_required": True,
        "repeat_attempt_permitted": False,
        "automatic_retry_performed": False,
    }
    write_json_atomic(claimed.run_dir / RUN_STATUS_NAME, payload)
    if runtime_check is not None:
        write_json_atomic(
            claimed.run_dir / RUNTIME_FAILURE_NAME,
            runtime_failure_record(
                runtime_check,
                git_sha=str(provenance["git_sha"]),
                git_dirty=bool(provenance["git_dirty"]),
                experiment_id=claimed.experiment_id,
            ),
        )
    return payload


def audit_forbidden_partitions(execution_identity: dict[str, Any]) -> None:
    """Refuse promotion of anything that touched a forbidden partition."""
    partition = str(execution_identity.get("partition_accessed", "")).lower()
    if partition in FORBIDDEN_PARTITIONS:
        raise M2ExecutionError(
            f"Forbidden partition {partition!r} reached the promotion gate."
        )
    for flag in ("validation_accessed", "test_accessed"):
        if execution_identity.get(flag) is not False:
            raise M2PersistenceError(
                f"A claim-bearing M2 artifact must record {flag}=false."
            )
    if execution_identity.get("sealed_test_state") != "unopened":
        raise M2PersistenceError("The B4 sealed test must remain unopened.")


def promote_arm_result(
    claimed: M2RunDirectory,
    *,
    result: dict[str, Any],
    execution_identity: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
) -> dict[str, Any]:
    """Stage, gate, then atomically promote one arm's claim-bearing result.

    Every gate must pass before anything reaches the canonical location. A
    mismatch at PRE_PROMOTION refuses the promotion, records a
    non-claim-bearing failure artifact and leaves the run FAILED_OR_INTERRUPTED.
    """
    staging = claimed.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    staged_path = staging / ARM_RESULT_NAME

    # 1-2. computation finished and the artifact is internally valid.
    if result.get("arm") != claimed.arm:
        raise M2PersistenceError("The staged result's arm disagrees with the claim.")
    write_json_atomic(staged_path, result)

    # 3. runtime-integrity pre-promotion check.
    check = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION, detail=ARM_RESULT_NAME
    )
    runtime.record(check)
    if not check.matches:
        record_failure(
            claimed,
            "runtime identity differed immediately before promotion",
            runtime_check=check,
        )
        raise RuntimeIntegrityError(
            "Runtime identity differed at PRE_PROMOTION; the claim-bearing "
            "result was NOT promoted, the environment was NOT repaired and the "
            "attempt is consumed."
        )

    # 4. forbidden-partition audit.
    audit_forbidden_partitions(execution_identity)

    # 5. output hashes, then atomic promotion.
    artifact_sha256 = {ARM_RESULT_NAME: sha256_file(staged_path)}
    promoted = dict(result)
    promoted["artifact_sha256"] = artifact_sha256
    write_json_atomic(claimed.run_dir / ARM_RESULT_NAME, promoted)
    staged_path.unlink()
    staging.rmdir()

    # END check, taken after promotion and recorded either way.
    end_check = observe_runtime_identity(
        EnforcementPoint.COMPLETION, detail=ARM_RESULT_NAME
    )
    runtime.record(end_check)
    status = {
        "experiment_id": claimed.experiment_id,
        "arm": claimed.arm,
        "status": STATUS_COMPLETE if end_check.matches else STATUS_FAILED,
        "claim_bearing_result_promoted": True,
        "canonical": end_check.matches,
        "started_at": claimed.started_at,
        "updated_at": _now(),
        "runtime_identity_checks": runtime.as_dict(),
    }
    if not end_check.matches:
        # Promoted, but the runtime moved before completion. Per the frozen
        # policy this is recorded as non-canonical rather than silently blessed.
        status["non_canonical_reason"] = (
            "runtime identity differed at COMPLETION; the attempt is marked "
            "non-promotable/non-canonical and requires human review"
        )
        status["human_review_required"] = True
        write_json_atomic(
            claimed.run_dir / RUNTIME_FAILURE_NAME,
            runtime_failure_record(
                end_check,
                git_sha=str(git_provenance(REPOSITORY_ROOT)["git_sha"]),
                git_dirty=bool(git_provenance(REPOSITORY_ROOT)["git_dirty"]),
                experiment_id=claimed.experiment_id,
            ),
        )
    write_json_atomic(claimed.run_dir / RUN_STATUS_NAME, status)
    return status


def require_runtime_start(runtime: RuntimeIntegrityRecord, detail: str) -> None:
    """The START enforcement point, before any scientific input is opened."""
    require_runtime_identity(EnforcementPoint.START, record=runtime, detail=detail)
