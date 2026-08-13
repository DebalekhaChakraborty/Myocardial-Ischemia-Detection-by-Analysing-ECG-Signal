"""Claim-bearing M2 persistence: one canonical provenance path, one lock.

A partial or interrupted run must never be able to look complete, and a
canonical claim must never rest on anything but the frozen scientific runtime.

**The frozen digest is a production invariant.** Every canonical claim boundary
requires `RuntimeIntegrityRecord.expected_digest == FROZEN_DEPENDENCY_DIGEST`
and requires the recorded START observation to have expected, observed and
matched that exact identity. A record configured to expect some other digest --
useful for isolated mechanism tests -- can never produce a canonical claim, and
the failure is detected at the claim boundary rather than deferred to final
validation. The claim directory itself is claim-bearing.

**One provenance path.** `build_canonical_run_lock()` is the only construction
of canonical identity, and `finalize_and_promote_arm_result()` calls it
directly; there is no second, weaker route by which a minimal result can become
canonical.

**The repository's existing lock convention.** Following the M1/P1/B4 pattern
exactly: the claim-bearing result file holds no hash of itself; a separate
immutable `M2_EXPERIMENT_LOCK.json` binds `artifact_sha256` for the promoted
result plus the complete provenance, and carries its own
`experiment_lock_sha256 = canonical_sha256(body)` over everything but that
field. No second locking system is invented for M2.

Ordering, unchanged from the accepted correction: stage -> audit -> COMPLETION
-> final PRE_PROMOTION -> finalize -> hash the promoted bytes -> atomic promote
-> COMPLETE. A COMPLETION mismatch invalidates canonical standing; the staged
result is retained as non-claim-bearing forensic material.

**The runner never selects an arm.**
"""

from __future__ import annotations

import re
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
from cardiosentinel.neural.m2_scorer import (
    FROZEN_B4B_CHECKPOINT_SHA256,
    FROZEN_P1B_LOCK_SHA256,
    M1L_CLASSIFICATION_THRESHOLD,
    NORMAL_EVIDENCE_THRESHOLD,
    RETAINED_M1L_CHECKPOINT_SHA256,
    RETAINED_M1L_LOCK_SHA256,
)
from cardiosentinel.neural.p1_experiment import FROZEN_DEPENDENCY_DIGEST
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
EXPERIMENT_LOCK_NAME: Final = "M2_EXPERIMENT_LOCK.json"
SUITE_RESULT_NAME: Final = "M2_SUITE_RESULT.json"
RUNTIME_FAILURE_NAME: Final = "M2_RUNTIME_INTEGRITY_FAILURE.json"
STAGING_PREFIX: Final = ".staging-"

CLAIM_DIRECTORY_PROMOTION_DETAIL: Final = "arm_claim_directory"

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")


class M2PersistenceError(RuntimeError):
    """Raised when a claim-bearing M2 artifact cannot be persisted safely."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _require_sha256(label: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.match(value):
        raise M2PersistenceError(f"{label} is not a SHA-256 digest: {value!r}.")
    return value


# --------------------------------------------------------------------------
# The frozen-digest invariant for every canonical claim boundary
# --------------------------------------------------------------------------


def require_frozen_runtime_record(runtime: RuntimeIntegrityRecord) -> None:
    """A canonical claim may only rest on the frozen scientific identity.

    A `RuntimeIntegrityRecord` may be configured to expect any digest, which is
    useful for isolated mechanism tests. Production canonical claims must never
    treat a matching non-frozen digest as sufficient, so this refuses anything
    but the frozen identity -- and refuses a START observation that expected,
    observed or matched anything else.
    """
    if runtime.expected_digest != FROZEN_DEPENDENCY_DIGEST:
        raise M2PersistenceError(
            f"A canonical M2 claim requires the frozen scientific identity "
            f"{FROZEN_DEPENDENCY_DIGEST!r}; this record expects "
            f"{runtime.expected_digest!r}. A matching non-frozen digest is "
            "never sufficient for a canonical claim."
        )
    starts = [
        check
        for check in runtime.checks
        if check.enforcement_point == EnforcementPoint.START.value
    ]
    if not starts:
        raise M2PersistenceError(
            "A canonical M2 claim requires a successful START runtime check "
            "before any scientific input is opened; none is recorded."
        )
    for check in starts:
        if check.expected_digest != FROZEN_DEPENDENCY_DIGEST:
            raise M2PersistenceError(
                "The recorded START check expected "
                f"{check.expected_digest!r}, not the frozen identity."
            )
        if check.observed_digest != FROZEN_DEPENDENCY_DIGEST or not check.matches:
            raise M2PersistenceError(
                "The recorded START check did not observe the frozen "
                f"identity (observed {check.observed_digest!r}); no canonical "
                "claim may be created."
            )


# --------------------------------------------------------------------------
# Canonical run lock -- the ONE provenance construction path
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
    "evaluated_population_identity",
    "m1l_classification_threshold",
    "normal_evidence_threshold",
    "runtime_dependency_digest_start",
    "runtime_dependency_digest_pre_promotion",
    "runtime_dependency_digest_end",
    "runtime_identity_checks",
    "partition_accessed",
    "validation_accessed",
    "test_accessed",
    "sealed_test_state",
    "started_at",
    "completed_at",
    "artifact_sha256",
    "automatic_retry_performed",
    "repeat_attempt_permitted",
    "rollback",
    "memory_selection_performed",
    "memory_selected",
)

_REQUIRED_SHA_FIELDS: Final = (
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
)

_FROZEN_IDENTITY_EXPECTATIONS: Final = {
    "m2_protocol_sha256": GATE.M2_PROTOCOL_SHA256,
    "m2_gate_receipt_sha256": GATE.M2_GATE_RECEIPT_SHA256,
    "retained_m1l_lock_sha256": RETAINED_M1L_LOCK_SHA256,
    "retained_m1l_checkpoint_sha256": RETAINED_M1L_CHECKPOINT_SHA256,
    "p1b_lock_sha256": FROZEN_P1B_LOCK_SHA256,
    "b4b_checkpoint_sha256": FROZEN_B4B_CHECKPOINT_SHA256,
}


ARM_RESULT_CLASS: Final = "m2_v1_canonical_arm_result"

MANDATORY_RESULT_SECTIONS: Final = (
    "policy_evidence",
    "window_evidence",
    "false_alarm_evidence",
    "cold_start_evidence",
    "contamination_evidence",
)

REQUIRED_RESULT_FIELDS: Final = (
    "artifact_class",
    "arm",
    "scientific_computation_completed",
    "label_blind_replay_completed",
    "m1l_classification_threshold",
    "threshold_selected_here",
    "classifier_retrained",
    "memory_selection_performed",
    "memory_selected",
    "rollback",
    "partition_accessed",
    "evaluated_population_identity",
    "validation_accessed",
    "test_accessed",
    "sealed_test_state",
    *MANDATORY_RESULT_SECTIONS,
)


def validate_claim_bearing_arm_result_payload(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Validate the RESULT PAYLOAD itself, not merely its lock.

    The experiment lock proves provenance; it cannot vouch for what the result
    actually contains. A canonical completed arm result must carry every
    section the frozen protocol requires, so a minimal `{"arm": ...}` object
    can never be hashed and promoted as canonical evidence.

    Sections are required to be PRESENT and structured. This validator computes
    no metric and invents none: a section whose evidence is legitimately
    unavailable must say so explicitly through a protocol-valid exclusion
    structure rather than being omitted.
    """
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in result]
    if missing:
        raise M2PersistenceError(
            f"Claim-bearing M2 arm result is missing required fields: {missing}."
        )
    if result["artifact_class"] != ARM_RESULT_CLASS:
        raise M2PersistenceError(
            f"artifact_class must be {ARM_RESULT_CLASS!r}; received "
            f"{result['artifact_class']!r}."
        )
    if result["arm"] not in M2_ARMS:
        raise M2PersistenceError(f"Unknown M2 arm {result['arm']!r}.")
    for flag in ("scientific_computation_completed", "label_blind_replay_completed"):
        if result[flag] is not True:
            raise M2PersistenceError(
                f"A canonical M2 arm result must record {flag}=true."
            )
    if result["m1l_classification_threshold"] != M1L_CLASSIFICATION_THRESHOLD:
        raise M2PersistenceError(
            "The result does not bind the frozen M1L classification threshold."
        )
    for flag in (
        "threshold_selected_here",
        "classifier_retrained",
        "memory_selection_performed",
        "rollback",
        "validation_accessed",
        "test_accessed",
    ):
        if result[flag] is not False:
            raise M2PersistenceError(
                f"A canonical M2 arm result must record {flag}=false."
            )
    if result["memory_selected"] is not None:
        raise M2PersistenceError("A canonical M2 arm result selects no arm.")
    if result["sealed_test_state"] != "unopened":
        raise M2PersistenceError("The B4 sealed test must remain unopened.")
    require_permitted_partition(result["partition_accessed"])

    for section in MANDATORY_RESULT_SECTIONS:
        payload = result[section]
        if payload is None or payload == {}:
            raise M2PersistenceError(
                f"Mandatory result section {section!r} is empty. Evidence that "
                "is legitimately unavailable must be recorded as an explicit "
                "protocol-valid exclusion, never omitted."
            )
        if not isinstance(payload, dict):
            raise M2PersistenceError(
                f"Result section {section!r} must be a structured object."
            )

    population = validate_evaluated_population_identity(
        result["evaluated_population_identity"]
    )

    # Every headline section that carries a population identity must agree with
    # the arm result's. A disagreement means the metrics and the declared
    # population describe different row sets, which is fatal.
    for section in ("window_evidence", "false_alarm_evidence", "cold_start_evidence"):
        embedded = result[section].get("population_identity")
        if embedded is None:
            continue
        if embedded != population:
            raise M2PersistenceError(
                f"{section} declares a population identity that differs from "
                "the arm result's evaluated population. The metrics and the "
                "declared population must describe the same rows."
            )
    return result


def build_canonical_run_lock(
    *,
    experiment_id: str,
    arm: str,
    execution_identity: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
    evaluated_population_identity: dict[str, Any] | None,
    started_at: str,
    completed_at: str,
    artifact_sha256: dict[str, str],
) -> dict[str, Any]:
    """Construct the one canonical M2 run lock.

    This is the only place canonical identity is assembled. Finalization calls
    it directly, so a minimal result cannot become canonical by bypassing it.
    """
    from cardiosentinel.features.schema import COMBINED_V1, MORPHOLOGY_V1, SIGNAL_V1

    if arm not in M2_ARMS:
        raise M2PersistenceError(f"Unknown M2 arm {arm!r}.")
    provenance = git_provenance(REPOSITORY_ROOT)
    inputs = execution_identity["input_identity"]
    scorer = execution_identity["scorer_identity"]
    partition = require_permitted_partition(inputs["partition"])

    lock: dict[str, Any] = {
        "lock_class": "m2_v1_canonical_arm_run_lock",
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
        "evaluated_population_identity": evaluated_population_identity,
        "m1l_classification_threshold": M1L_CLASSIFICATION_THRESHOLD,
        "normal_evidence_threshold": NORMAL_EVIDENCE_THRESHOLD,
        "classification_threshold_used_for_admission": False,
        "classifier_retrained": False,
        "threshold_selected_during_run": False,
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
        "started_at": started_at,
        "completed_at": completed_at,
        "artifact_sha256": dict(artifact_sha256),
        "automatic_retry_performed": False,
        "repeat_attempt_permitted": False,
        "rollback": False,
        "memory_selection_performed": False,
        "memory_selected": None,
    }
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    return lock


def validate_canonical_run_lock(
    lock: dict[str, Any],
    *,
    run_dir: Path | None = None,
    requires_evaluation: bool = True,
) -> dict[str, Any]:
    """Validate the ACTUAL values of a canonical lock, not merely their keys."""
    recorded = lock.get("experiment_lock_sha256")
    body = {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise M2PersistenceError("M2 experiment lock failed digest validation.")

    missing = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in lock]
    if missing:
        raise M2PersistenceError(f"Canonical M2 lock is missing {missing}.")
    # `memory_selected` is null by design; `evaluated_population_identity` is
    # governed by `requires_evaluation` below, so a run that produced no
    # label-joined evaluation is not forced to invent one.
    _nullable = {"memory_selected", "evaluated_population_identity"}
    empty = [
        field
        for field in REQUIRED_PROVENANCE_FIELDS
        if lock[field] is None and field not in _nullable
    ]
    if empty:
        raise M2PersistenceError(
            f"Canonical M2 lock carries None where a real identity is "
            f"required: {empty}."
        )

    if lock["arm"] not in M2_ARMS:
        raise M2PersistenceError(f"Unknown M2 arm {lock['arm']!r}.")
    if not isinstance(lock["git_sha"], str) or not _GIT_SHA_PATTERN.match(
        lock["git_sha"]
    ):
        raise M2PersistenceError(f"git_sha is malformed: {lock['git_sha']!r}.")
    if lock["git_dirty"] is not False:
        raise M2PersistenceError(
            "Canonical M2 evidence requires a clean Git checkout, matching the "
            "existing P1/M1 convention."
        )
    for field in _REQUIRED_SHA_FIELDS:
        _require_sha256(field, lock[field])
    for field, expected in _FROZEN_IDENTITY_EXPECTATIONS.items():
        if lock[field] != expected:
            raise M2PersistenceError(
                f"{field} is {lock[field]!r}, expected the frozen {expected!r}."
            )
    if lock["m1l_classification_threshold"] != M1L_CLASSIFICATION_THRESHOLD:
        raise M2PersistenceError("The lock does not bind the frozen M1L threshold.")
    if lock["normal_evidence_threshold"] != NORMAL_EVIDENCE_THRESHOLD:
        raise M2PersistenceError("The lock does not bind the frozen M2 margin.")
    if lock.get("classification_threshold_used_for_admission") is not False:
        raise M2PersistenceError(
            "The classification threshold must never gate memory admission."
        )

    require_permitted_partition(lock["partition_accessed"])
    for flag in ("validation_accessed", "test_accessed"):
        if lock[flag] is not False:
            raise M2PersistenceError(f"A canonical M2 lock must record {flag}=false.")
    if lock["sealed_test_state"] != "unopened":
        raise M2PersistenceError("The B4 sealed test must remain unopened.")
    for flag in ("automatic_retry_performed", "repeat_attempt_permitted", "rollback"):
        if lock[flag] is not False:
            raise M2PersistenceError(f"A canonical M2 lock must record {flag}=false.")
    if lock["memory_selection_performed"] is not False:
        raise M2PersistenceError("A canonical M2 run performs no arm selection.")
    if lock["memory_selected"] is not None:
        raise M2PersistenceError("A canonical M2 run selects no arm automatically.")

    for label in ("start", "pre_promotion", "end"):
        _require_sha256(
            f"runtime_dependency_digest_{label}",
            lock[f"runtime_dependency_digest_{label}"],
        )
        if lock[f"runtime_dependency_digest_{label}"] != FROZEN_DEPENDENCY_DIGEST:
            raise M2PersistenceError(
                f"runtime_dependency_digest_{label} is not the frozen identity."
            )
    checks = lock["runtime_identity_checks"]
    if checks.get("all_observations_matched") is not True:
        raise M2PersistenceError(
            "Canonical evidence requires every runtime observation to match."
        )

    if requires_evaluation:
        validate_evaluated_population_identity(lock["evaluated_population_identity"])

    artifacts = dict(lock["artifact_sha256"])
    if not artifacts:
        raise M2PersistenceError("A canonical M2 lock binds no artifact hash.")
    for name, digest in artifacts.items():
        _require_sha256(f"artifact_sha256[{name}]", digest)
        if run_dir is not None:
            path = Path(run_dir) / name
            if not path.is_file() or sha256_file(path) != digest:
                raise M2PersistenceError(
                    f"M2 artifact {name} does not match its lock digest."
                )
    return lock


def validate_evaluated_population_identity(
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Require a real, verified, full-scope evaluated-population identity."""
    from cardiosentinel.neural.m2_evaluation import POPULATION_SCOPE_FULL

    if not identity:
        raise M2PersistenceError(
            "A claim-bearing M2 result containing label-joined evaluation must "
            "bind evaluated_population_identity; None or {} is refused."
        )
    rows = identity.get("evaluated_rows")
    if not isinstance(rows, int) or rows <= 0:
        raise M2PersistenceError(
            f"evaluated_rows must be a positive integer; received {rows!r}."
        )
    _require_sha256(
        "evaluated_ordered_stable_id_sha256",
        identity.get("evaluated_ordered_stable_id_sha256"),
    )
    if not identity.get("identity_key"):
        raise M2PersistenceError("evaluated population identity names no identity key.")
    if identity.get("positional_join_used") is not False:
        raise M2PersistenceError(
            "A claim-bearing evaluated population must record "
            "positional_join_used=false."
        )
    if identity.get("population_scope") != POPULATION_SCOPE_FULL:
        raise M2PersistenceError(
            f"Headline claim-bearing evaluation requires "
            f"{POPULATION_SCOPE_FULL!r}; received "
            f"{identity.get('population_scope')!r}."
        )
    if identity.get("population_verified_against_canonical_replay") is not True:
        raise M2PersistenceError(
            "A claim-bearing full population must be VERIFIED against the "
            "canonical replay population digest, not merely self-consistent."
        )
    return dict(identity)


def validate_complete_runtime_identity(
    runtime: RuntimeIntegrityRecord,
) -> dict[str, Any]:
    """Require a COMPLETE, GREEN, frozen runtime block -- not a present field."""
    require_frozen_runtime_record(runtime)
    points = {check.enforcement_point for check in runtime.checks}
    for required in (
        EnforcementPoint.START,
        EnforcementPoint.PRE_PROMOTION,
        EnforcementPoint.COMPLETION,
    ):
        if required.value not in points:
            raise M2PersistenceError(
                f"Canonical COMPLETE evidence requires a {required.value!r} "
                "runtime observation; none is recorded."
            )
    for label, digest in (
        ("start", runtime.digest_at(EnforcementPoint.START)),
        ("completion", runtime.digest_at(EnforcementPoint.COMPLETION)),
    ):
        if digest is None:
            raise M2PersistenceError(
                f"runtime_dependency_digest_{label} is None; canonical COMPLETE "
                "evidence may never carry an unobserved digest."
            )
    if not runtime.all_matched:
        mismatch = runtime.first_mismatch()
        raise M2PersistenceError(
            "Canonical COMPLETE evidence requires every runtime observation to "
            f"match; {mismatch.enforcement_point!r} observed "
            f"{mismatch.observed_digest}."
        )
    return runtime.as_dict()


def build_suite_result(
    *, suite_id: str, arm_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """A two-arm suite that expresses no retention decision."""
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
# Claim + staged-then-promote
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


def claim_run_directory(
    run_root: Path,
    experiment_id: str,
    arm: str,
    *,
    runtime: RuntimeIntegrityRecord,
) -> M2RunDirectory:
    """Atomically claim the one canonical attempt. The directory IS the claim.

    The claim boundary requires the frozen scientific identity (record
    expectation AND the recorded START observation), then takes its own
    PRE_PROMOTION observation against that same frozen identity, and only then
    creates the directory. A non-frozen record is refused here rather than at
    final validation, because the directory itself is claim-bearing.
    """
    if arm not in M2_ARMS:
        raise M2PersistenceError(f"Unknown M2 arm {arm!r}.")
    require_frozen_runtime_record(runtime)
    require_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        record=runtime,
        detail=CLAIM_DIRECTORY_PROMOTION_DETAIL,
    )

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
        "canonical": False,
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


def _retain_forensic_failure(claimed: M2RunDirectory, check: Any, reason: str) -> None:
    """Record a refused promotion. Staged evidence is retained, never deleted."""
    provenance = git_provenance(REPOSITORY_ROOT)
    record_failure(claimed, reason, runtime_check=check)
    write_json_atomic(
        claimed.run_dir / RUNTIME_FAILURE_NAME,
        {
            **runtime_failure_record(
                check,
                git_sha=str(provenance["git_sha"]),
                git_dirty=bool(provenance["git_dirty"]),
                experiment_id=claimed.experiment_id,
            ),
            "staged_result_retained_as_forensic_material": True,
            "canonical_promotion_invalidated": True,
        },
    )


def finalize_and_promote_arm_result(
    claimed: M2RunDirectory,
    *,
    result: dict[str, Any],
    execution_identity: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
    requires_evaluation: bool = True,
) -> dict[str, Any]:
    """Validate, stage, gate, finalize and atomically promote one arm's result.

    The RESULT is validated in its own right before anything is hashed, so a
    minimal payload can never reach the canonical location. The evaluated
    population identity is then EXTRACTED FROM THE VALIDATED RESULT rather
    than accepted as a free-standing argument, so the result and the lock
    cannot disagree about which rows were evaluated.

    `M2_ARM_RESULT.json` and `M2_EXPERIMENT_LOCK.json` are two separate
    claim-bearing artifacts, so each gets its own PRE_PROMOTION observation.
    The lock's observation is taken BEFORE the lock is built, so the lock's
    `runtime_identity_checks` genuinely contains it and its self-digest is
    computed only afterwards -- the observation is never fabricated after the
    fact.
    """
    require_frozen_runtime_record(runtime)
    if result.get("arm") != claimed.arm:
        raise M2PersistenceError("The result's arm disagrees with the claim.")

    # §5: the payload must be a complete canonical result in its own right.
    if requires_evaluation:
        validate_claim_bearing_arm_result_payload(result)
        population = dict(result["evaluated_population_identity"])
    else:
        population = result.get("evaluated_population_identity")

    staging = claimed.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    staged_path = staging / ARM_RESULT_NAME
    write_json_atomic(staged_path, result)

    audit_forbidden_partitions(execution_identity)

    completion = observe_runtime_identity(
        EnforcementPoint.COMPLETION,
        expected_digest=runtime.expected_digest,
        detail=ARM_RESULT_NAME,
    )
    runtime.record(completion)
    if not completion.matches:
        _retain_forensic_failure(
            claimed,
            completion,
            "runtime identity differed at COMPLETION; canonical standing is "
            "invalidated and nothing was promoted",
        )
        raise RuntimeIntegrityError(
            "Runtime identity differed at COMPLETION. Canonical promotion is "
            "INVALIDATED: the result was NOT promoted, the run is not COMPLETE, "
            "the environment was NOT repaired and the staged result is retained "
            "as non-claim-bearing forensic material."
        )

    # Promotion check for the RESULT artifact.
    result_check = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        expected_digest=runtime.expected_digest,
        detail=f"promote:{ARM_RESULT_NAME}",
    )
    runtime.record(result_check)
    if not result_check.matches:
        _retain_forensic_failure(
            claimed, result_check, "runtime identity differed before result promotion"
        )
        raise RuntimeIntegrityError(
            "Runtime identity differed before the result promotion; neither the "
            "canonical result nor the canonical lock was written."
        )

    artifact_digest = sha256_file(staged_path)
    canonical_path = claimed.run_dir / ARM_RESULT_NAME
    write_json_atomic(canonical_path, result)
    if sha256_file(canonical_path) != artifact_digest:
        raise M2PersistenceError(
            "The promoted canonical artifact does not match the hashed bytes."
        )

    # Separate promotion check for the LOCK artifact, taken before the lock is
    # built so the lock can bind this very observation.
    lock_check = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        expected_digest=runtime.expected_digest,
        detail=f"promote:{EXPERIMENT_LOCK_NAME}",
    )
    runtime.record(lock_check)
    if not lock_check.matches:
        _retain_forensic_failure(
            claimed,
            lock_check,
            "runtime identity differed before the experiment-lock promotion; "
            "the result was already promoted and is retained for human review",
        )
        raise RuntimeIntegrityError(
            "Runtime identity differed before the experiment-lock promotion. "
            "The run is NOT COMPLETE and NOT canonical. Already-promoted "
            "evidence is preserved for human review, never deleted or blessed, "
            "and nothing is retried automatically."
        )

    validate_complete_runtime_identity(runtime)
    lock = build_canonical_run_lock(
        experiment_id=claimed.experiment_id,
        arm=claimed.arm,
        execution_identity=execution_identity,
        runtime=runtime,
        evaluated_population_identity=population,
        started_at=claimed.started_at,
        completed_at=_now(),
        artifact_sha256={ARM_RESULT_NAME: artifact_digest},
    )
    if requires_evaluation and lock["evaluated_population_identity"] != population:
        raise M2PersistenceError(
            "The lock's evaluated population identity differs from the result's."
        )
    validate_canonical_run_lock(lock, requires_evaluation=requires_evaluation)
    write_json_atomic(claimed.run_dir / EXPERIMENT_LOCK_NAME, lock)
    validate_canonical_run_lock(
        lock, run_dir=claimed.run_dir, requires_evaluation=requires_evaluation
    )
    staged_path.unlink()
    staging.rmdir()

    status = {
        "experiment_id": claimed.experiment_id,
        "arm": claimed.arm,
        "status": STATUS_COMPLETE,
        "claim_bearing_result_promoted": True,
        "canonical": True,
        "started_at": claimed.started_at,
        "updated_at": _now(),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "artifact_sha256": {ARM_RESULT_NAME: artifact_digest},
        "runtime_identity_checks": runtime.as_dict(),
    }
    write_json_atomic(claimed.run_dir / RUN_STATUS_NAME, status)
    return status


def require_runtime_start(runtime: RuntimeIntegrityRecord, detail: str) -> None:
    """The START enforcement point, before any scientific input is opened."""
    require_runtime_identity(EnforcementPoint.START, record=runtime, detail=detail)
