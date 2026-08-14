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
from collections.abc import Sequence
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
    require_canonical_development_partition,
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

ATTEMPT_FAILURE_RECEIPT_NAME: Final = "M2_ATTEMPT_FAILURE_RECEIPT.json"
FAILURE_REVIEW_SUFFIX: Final = "__failure_review"

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

RECOVERY_LINEAGE_FIELDS: Final = (
    "recovery2_decision_sha256",
    "recovery_from_original_suite_id",
    "recovery1_suite_id",
    "recovery2_suite_id",
    "attempt1_reason_class",
    "recovery1_reason_class",
    "attempt1_scoring_started",
    "attempt1_metrics_computed",
    "attempt1_test_accessed",
    "recovery1_receipt_scoring_started",
    "recovery1_human_forensic_scorer_invocation_observed",
    "recovery1_replay_completed",
    "recovery1_metrics_computed",
    "recovery1_test_accessed",
)


def validate_original_attempt1_failure_lineage(run_root: Path) -> dict[str, Any]:
    """Prove attempt #1 from its ARTIFACTS, not from a directory existing.

    A claim directory shows that something was claimed. It does not show that
    THIS attempt failed, that it failed before any row was scored, that no
    metric was produced, or that the sealed test stayed shut. Every one of those
    is a scientific claim, so each is verified against the frozen preserved
    evidence before a recovery may be authorized.

    Nothing here writes, repairs or normalises anything: if an original artifact
    is absent or mutated, the recovery stops for human review.
    """
    from cardiosentinel.neural.m2_development_run import (
        ORIGINAL_EXCEPTION_SUBSTRING,
        ORIGINAL_EXCEPTION_TYPE,
        ORIGINAL_EXECUTION_GIT_SHA,
        ORIGINAL_FAILED_STAGE,
        ORIGINAL_FAILURE_RECEIPT_FILE_SHA256,
        ORIGINAL_FAILURE_RECEIPT_SHA256,
        ORIGINAL_STATUS_SHA256,
        ORIGINAL_SUITE_ID,
        RECOVERY_DECISION_SHA256,
    )

    root = Path(run_root)

    def refuse(detail: str) -> None:
        raise M2PersistenceError(
            f"The frozen attempt #1 forensic lineage could not be proven: "
            f"{detail} STOP FOR HUMAN REVIEW. The recovery is not authorized, "
            "and no original artifact is repaired, replaced or inferred."
        )

    # 1-3. Both original arm directories and status files, at their frozen bytes.
    for arm in M2_ARMS:
        run_dir = root / arm_experiment_id(ORIGINAL_SUITE_ID, arm)
        if not run_dir.is_dir():
            refuse(f"the original {arm} claim directory {run_dir} is absent.")
        status_path = run_dir / RUN_STATUS_NAME
        if not status_path.is_file():
            refuse(f"the original {arm} {RUN_STATUS_NAME} is absent.")
        observed = sha256_file(status_path)
        if observed != ORIGINAL_STATUS_SHA256[arm]:
            refuse(
                f"the original {arm} {RUN_STATUS_NAME} digests to {observed}, not "
                f"the frozen {ORIGINAL_STATUS_SHA256[arm]}."
            )

        # 4-5. No original arm ever produced claim-bearing science.
        for name in (ARM_RESULT_NAME, EXPERIMENT_LOCK_NAME):
            if (run_dir / name).exists():
                refuse(
                    f"the original {arm} directory contains {name}; attempt #1 is "
                    "recorded as having promoted nothing."
                )

    # 6. Nor did the original suite.
    original_suite = suite_directory(root, ORIGINAL_SUITE_ID) / SUITE_RESULT_NAME
    if original_suite.exists():
        refuse(
            f"the original suite contains {SUITE_RESULT_NAME}; attempt #1 is "
            "recorded as never having completed."
        )

    # 7-9. The additive receipt, at its frozen file digest and self-digest.
    receipt_path = (
        failure_review_directory(root, ORIGINAL_SUITE_ID) / ATTEMPT_FAILURE_RECEIPT_NAME
    )
    if not receipt_path.is_file():
        refuse(f"the additive failure receipt {receipt_path} is absent.")
    file_digest = sha256_file(receipt_path)
    if file_digest != ORIGINAL_FAILURE_RECEIPT_FILE_SHA256:
        refuse(
            f"the failure receipt file digests to {file_digest}, not the frozen "
            f"{ORIGINAL_FAILURE_RECEIPT_FILE_SHA256}."
        )
    receipt = read_json_result(receipt_path)
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    recomputed = canonical_sha256(body)
    if receipt.get("receipt_sha256") != recomputed:
        refuse("the failure receipt's own canonical digest does not validate.")
    if receipt["receipt_sha256"] != ORIGINAL_FAILURE_RECEIPT_SHA256:
        refuse(
            f"the failure receipt digest is {receipt['receipt_sha256']}, not the "
            f"frozen {ORIGINAL_FAILURE_RECEIPT_SHA256}."
        )

    # 10-11. It describes the original suite, and claims no canonical standing.
    if receipt.get("suite_id") != ORIGINAL_SUITE_ID:
        refuse(
            f"the failure receipt names suite {receipt.get('suite_id')!r}, not "
            f"{ORIGINAL_SUITE_ID!r}."
        )
    for flag, expected in (("claim_bearing", False), ("canonical", False)):
        if receipt.get(flag) is not expected:
            refuse(f"the failure receipt records {flag}={receipt.get(flag)!r}.")
    if receipt.get("execution_git_sha") != ORIGINAL_EXECUTION_GIT_SHA:
        refuse(
            f"the failure receipt names execution SHA "
            f"{receipt.get('execution_git_sha')!r}, not "
            f"{ORIGINAL_EXECUTION_GIT_SHA!r}."
        )

    # 12-13. The exact frozen pre-scoring failure, not some other failure.
    if receipt.get("failed_stage") != ORIGINAL_FAILED_STAGE:
        refuse(
            f"the failure receipt records stage {receipt.get('failed_stage')!r}, "
            f"not the frozen {ORIGINAL_FAILED_STAGE!r}."
        )
    if receipt.get("exception_type") != ORIGINAL_EXCEPTION_TYPE:
        refuse(
            f"the failure receipt records exception "
            f"{receipt.get('exception_type')!r}, not the frozen "
            f"{ORIGINAL_EXCEPTION_TYPE!r}."
        )
    if ORIGINAL_EXCEPTION_SUBSTRING not in str(receipt.get("exception_message", "")):
        refuse(
            "the failure receipt's exception message is not the frozen "
            "partition-alignment defect."
        )

    # 14-18. The exposure the recovery lineage will assert downstream.
    for flag, expected in (
        ("validation_opened", True),
        ("scoring_started", False),
        ("metrics_computed", False),
        ("test_accessed", False),
    ):
        if receipt.get(flag) is not expected:
            refuse(
                f"the failure receipt records {flag}={receipt.get(flag)!r}, not "
                f"the frozen {expected!r}."
            )
    if receipt.get("sealed_test_state") != "unopened":
        refuse("the failure receipt does not record the sealed test as unopened.")

    # 19. It binds the preserved status hashes it claims to describe.
    preserved = receipt.get("preserved_status_sha256") or {}
    if preserved != dict(ORIGINAL_STATUS_SHA256):
        refuse(
            "the failure receipt does not bind the frozen preserved status "
            f"digests; it binds {preserved!r}."
        )

    # 20. And the recovery decision that authorised exactly one recovery.
    if receipt.get("recovery_decision_sha256") != RECOVERY_DECISION_SHA256:
        refuse("the failure receipt does not bind the frozen recovery decision digest.")

    return {
        "lineage_class": "m2_attempt1_verified_failure_lineage",
        "original_suite_id": ORIGINAL_SUITE_ID,
        "original_execution_git_sha": ORIGINAL_EXECUTION_GIT_SHA,
        "original_status_sha256": dict(ORIGINAL_STATUS_SHA256),
        "failure_receipt_sha256": ORIGINAL_FAILURE_RECEIPT_SHA256,
        "failure_receipt_file_sha256": ORIGINAL_FAILURE_RECEIPT_FILE_SHA256,
        "recovery_decision_sha256": RECOVERY_DECISION_SHA256,
        "failed_stage": ORIGINAL_FAILED_STAGE,
        "validation_opened": True,
        "scoring_started": False,
        "metrics_computed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "promoted_any_claim_bearing_artifact": False,
        "verified_from_artifacts": True,
    }


def _require_recovery1_receipt_fields(receipt: dict[str, Any], *, refuse=None) -> None:
    """The exposure recovery1's receipt must record, checked by value.

    Separated so it is testable on its own: in the full validator the frozen
    FILE digest already refuses any altered receipt, so these checks would
    otherwise be unreachable.
    """
    if refuse is None:

        def refuse(detail: str) -> None:
            raise M2PersistenceError(
                f"The frozen recovery1 forensic lineage could not be proven: {detail}"
            )

    # The conservative receipt value is preserved as-is and never rewritten.
    if receipt.get("scoring_started") != "indeterminate":
        refuse(
            f"the recovery1 receipt records scoring_started="
            f"{receipt.get('scoring_started')!r}, not the frozen 'indeterminate'."
        )
    for flag, expected in (
        ("validation_opened", True),
        ("replay_completed", False),
        ("post_replay_evaluation_started", False),
        ("metrics_computed_or_completed", False),
        ("test_accessed", False),
    ):
        if receipt.get(flag) is not expected:
            refuse(
                f"the recovery1 receipt records {flag}={receipt.get(flag)!r}, "
                f"not the frozen {expected!r}."
            )
    if receipt.get("metrics_completed_per_arm") != {}:
        refuse("the recovery1 receipt records completed per-arm metrics.")
    if receipt.get("sealed_test_state") != "unopened":
        refuse("the recovery1 receipt does not record the sealed test unopened.")


def validate_recovery1_failure_lineage(run_root: Path) -> dict[str, Any]:
    """Prove RECOVERY1 from its artifacts, exactly as attempt #1 is proven.

    Recovery1 is a second consumed pre-scoring failure with its own frozen
    identity. Recovery2 may not be claimed until BOTH prior lineages verify.
    """
    from cardiosentinel.neural.m2_development_run import (
        RECOVERY1_EXCEPTION_SUBSTRING,
        RECOVERY1_EXCEPTION_TYPE,
        RECOVERY1_EXECUTION_GIT_SHA,
        RECOVERY1_FAILED_STAGE,
        RECOVERY1_FAILURE_RECEIPT_FILE_SHA256,
        RECOVERY1_FAILURE_RECEIPT_SHA256,
        RECOVERY1_STATUS_SHA256,
        RECOVERY1_SUITE_ID,
        RECOVERY2_DECISION_SHA256,
    )

    root = Path(run_root)

    def refuse(detail: str) -> None:
        raise M2PersistenceError(
            f"The frozen recovery1 forensic lineage could not be proven: "
            f"{detail} STOP FOR HUMAN REVIEW. Recovery2 is not authorized, and "
            "no preserved artifact is repaired, replaced or inferred."
        )

    for arm in M2_ARMS:
        run_dir = root / arm_experiment_id(RECOVERY1_SUITE_ID, arm)
        if not run_dir.is_dir():
            refuse(f"the recovery1 {arm} claim directory {run_dir} is absent.")
        status_path = run_dir / RUN_STATUS_NAME
        if not status_path.is_file():
            refuse(f"the recovery1 {arm} {RUN_STATUS_NAME} is absent.")
        observed = sha256_file(status_path)
        if observed != RECOVERY1_STATUS_SHA256[arm]:
            refuse(
                f"the recovery1 {arm} {RUN_STATUS_NAME} digests to {observed}, "
                f"not the frozen {RECOVERY1_STATUS_SHA256[arm]}."
            )
        for name in (ARM_RESULT_NAME, EXPERIMENT_LOCK_NAME):
            if (run_dir / name).exists():
                refuse(
                    f"the recovery1 {arm} directory contains {name}; recovery1 "
                    "is recorded as having promoted nothing."
                )
    if (suite_directory(root, RECOVERY1_SUITE_ID) / SUITE_RESULT_NAME).exists():
        refuse(
            f"the recovery1 suite contains {SUITE_RESULT_NAME}; recovery1 is "
            "recorded as never having completed."
        )

    receipt_path = (
        failure_review_directory(root, RECOVERY1_SUITE_ID)
        / ATTEMPT_FAILURE_RECEIPT_NAME
    )
    if not receipt_path.is_file():
        refuse(f"the recovery1 failure receipt {receipt_path} is absent.")
    file_digest = sha256_file(receipt_path)
    if file_digest != RECOVERY1_FAILURE_RECEIPT_FILE_SHA256:
        refuse(
            f"the recovery1 failure receipt file digests to {file_digest}, not "
            f"the frozen {RECOVERY1_FAILURE_RECEIPT_FILE_SHA256}."
        )
    receipt = read_json_result(receipt_path)
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha256(body):
        refuse("the recovery1 receipt's own canonical digest does not validate.")
    if receipt["receipt_sha256"] != RECOVERY1_FAILURE_RECEIPT_SHA256:
        refuse(
            f"the recovery1 receipt digest is {receipt['receipt_sha256']}, not "
            f"the frozen {RECOVERY1_FAILURE_RECEIPT_SHA256}."
        )
    _require_recovery1_receipt_fields(receipt, refuse=refuse)
    if receipt.get("suite_id") != RECOVERY1_SUITE_ID:
        refuse(f"the recovery1 receipt names suite {receipt.get('suite_id')!r}.")
    if receipt.get("git_sha") != RECOVERY1_EXECUTION_GIT_SHA:
        refuse(f"the recovery1 receipt names execution SHA {receipt.get('git_sha')!r}.")
    for flag, expected in (("claim_bearing", False), ("canonical", False)):
        if receipt.get(flag) is not expected:
            refuse(f"the recovery1 receipt records {flag}={receipt.get(flag)!r}.")
    if receipt.get("failed_stage") != RECOVERY1_FAILED_STAGE:
        refuse(f"the recovery1 receipt records stage {receipt.get('failed_stage')!r}.")
    if receipt.get("exception_type") != RECOVERY1_EXCEPTION_TYPE:
        refuse(
            f"the recovery1 receipt records exception "
            f"{receipt.get('exception_type')!r}."
        )
    if RECOVERY1_EXCEPTION_SUBSTRING not in str(receipt.get("exception_message", "")):
        refuse(
            "the recovery1 receipt's exception message is not the frozen "
            "source-null join-sentinel defect."
        )
    promotion = receipt.get("promotion_state") or {}
    for field in ("arm_result_promoted", "experiment_lock_promoted"):
        values = promotion.get(field) or {}
        if any(values.get(arm) for arm in M2_ARMS):
            refuse(f"the recovery1 receipt records {field} true for some arm.")
    if promotion.get("suite_result_promoted") is not False:
        refuse("the recovery1 receipt records a promoted suite.")
    if receipt.get("preserved_status_sha256"):
        # Recorded at failure time from the STARTED files, before they were
        # rewritten to FAILED; kept as-is and not re-derived here.
        pass

    return {
        "lineage_class": "m2_recovery1_verified_failure_lineage",
        "recovery1_suite_id": RECOVERY1_SUITE_ID,
        "recovery1_execution_git_sha": RECOVERY1_EXECUTION_GIT_SHA,
        "recovery1_status_sha256": dict(RECOVERY1_STATUS_SHA256),
        "recovery1_failure_receipt_sha256": RECOVERY1_FAILURE_RECEIPT_SHA256,
        "recovery1_failure_receipt_file_sha256": (
            RECOVERY1_FAILURE_RECEIPT_FILE_SHA256
        ),
        "recovery2_decision_sha256": RECOVERY2_DECISION_SHA256,
        "recovery1_failed_stage": RECOVERY1_FAILED_STAGE,
        "recovery1_receipt_scoring_started": "indeterminate",
        "recovery1_human_forensic_scorer_invocation_observed": False,
        "recovery1_validation_opened": True,
        "recovery1_replay_completed": False,
        "recovery1_metrics_computed": False,
        "recovery1_test_accessed": False,
        "recovery1_promoted_any_claim_bearing_artifact": False,
        "verified_from_artifacts": True,
    }


def validate_recovery_lineage(payload: dict[str, Any]) -> dict[str, Any]:
    """Every recovery artifact must state what attempt #1 was, by value.

    The recovery never conceals the consumed first attempt: it names it, names
    why it failed, and states that no row was scored, no metric computed and no
    TEST opened before it did.
    """
    from cardiosentinel.neural.m2_development_run import (
        ATTEMPT1_REASON_CLASS,
        CANONICAL_SUITE_ID,
        ORIGINAL_SUITE_ID,
        RECOVERY1_REASON_CLASS,
        RECOVERY1_SUITE_ID,
        RECOVERY2_DECISION_SHA256,
    )

    missing = [field for field in RECOVERY_LINEAGE_FIELDS if field not in payload]
    if missing:
        raise M2PersistenceError(
            f"A recovery artifact must bind its lineage; missing {missing}."
        )
    expectations = {
        "recovery2_decision_sha256": RECOVERY2_DECISION_SHA256,
        "recovery_from_original_suite_id": ORIGINAL_SUITE_ID,
        "recovery1_suite_id": RECOVERY1_SUITE_ID,
        "recovery2_suite_id": CANONICAL_SUITE_ID,
        "attempt1_reason_class": ATTEMPT1_REASON_CLASS,
        "recovery1_reason_class": RECOVERY1_REASON_CLASS,
        "attempt1_scoring_started": False,
        "attempt1_metrics_computed": False,
        "attempt1_test_accessed": False,
        # Two distinct facts, both preserved: the immutable receipt's
        # conservative value, and the human control-flow determination.
        "recovery1_receipt_scoring_started": "indeterminate",
        "recovery1_human_forensic_scorer_invocation_observed": False,
        "recovery1_replay_completed": False,
        "recovery1_metrics_computed": False,
        "recovery1_test_accessed": False,
    }
    for field, expected in expectations.items():
        if payload[field] != expected:
            raise M2PersistenceError(
                f"Recovery lineage {field} is {payload[field]!r}, expected "
                f"{expected!r}."
            )
    return {field: payload[field] for field in RECOVERY_LINEAGE_FIELDS}


POPULATION_IDENTITY_FIELDS: Final = (
    "replay_population_identity",
    "primary_evaluation_population_identity",
    "challenge_evaluation_population_identity",
    "stress_interval_selection_identity",
)
"""The four distinct populations a canonical arm result must bind separately.

One `evaluated_population_identity` used to stand for all of them, which let
the full causal replay population masquerade as a metric denominator. Each
headline section now has to agree with its OWN population.
"""

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
    *POPULATION_IDENTITY_FIELDS,
    "development_source_identity",
    *RECOVERY_LINEAGE_FIELDS,
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
    *POPULATION_IDENTITY_FIELDS,
    "development_source_identity",
    *RECOVERY_LINEAGE_FIELDS,
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
        "test_accessed",
    ):
        if result[flag] is not False:
            raise M2PersistenceError(
                f"A canonical M2 arm result must record {flag}=false."
            )
    # VALIDATION *is* the M2 development evidence partition, so a canonical arm
    # result records that it was read. TEST stays sealed and unopened.
    if result["validation_accessed"] is not True:
        raise M2PersistenceError(
            "Canonical M2 development evidence is computed on VALIDATION and "
            "must record validation_accessed=true."
        )
    if result["memory_selected"] is not None:
        raise M2PersistenceError("A canonical M2 arm result selects no arm.")
    if result["sealed_test_state"] != "unopened":
        raise M2PersistenceError("The B4 sealed test must remain unopened.")
    require_canonical_development_partition(result["partition_accessed"])

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

    replay = validate_replay_population_identity(result["replay_population_identity"])
    primary = validate_primary_population_identity(
        result["primary_evaluation_population_identity"]
    )
    challenge = validate_challenge_population_identity(
        result["challenge_evaluation_population_identity"]
    )
    stress = validate_stress_selection_identity(
        result["stress_interval_selection_identity"]
    )
    if primary == challenge or primary == replay or challenge == replay:
        raise M2PersistenceError(
            "The replay, primary and challenge populations must be distinct "
            "identities; at least two are identical, which means one "
            "denominator is masquerading as another."
        )

    # Every headline section must agree with ITS OWN population. A disagreement
    # means the metric and the declared denominator describe different rows.
    _require_section_population(
        result, "window_evidence", "population_identity", primary
    )
    _require_section_population(
        result, "cold_start_evidence", "population_identity", primary
    )
    _require_section_population(
        result, "false_alarm_evidence", "background_population_identity", primary
    )
    _require_section_population(
        result, "false_alarm_evidence", "challenge_population_identity", challenge
    )
    _require_section_population(
        result, "policy_evidence", "population_identity", replay
    )
    _require_section_population(
        result, "contamination_evidence", "replay_population_identity", replay
    )
    _require_section_population(
        result,
        "contamination_evidence",
        "stress_interval_selection_identity",
        stress,
    )

    # The raw `.stb` provenance must be ONE identity, not several that happen
    # to look alike: the arm result and the stress selection it authorised must
    # name the same verified source.
    validate_recovery_lineage(result)
    source = validate_development_source_identity(result["development_source_identity"])
    if stress.get("development_source_identity") != source:
        raise M2PersistenceError(
            "The stress selection's development_source_identity differs from "
            "the arm result's. The intervals and the verified source must "
            "describe the same files."
        )
    return result


def _require_section_population(
    result: dict[str, Any],
    section: str,
    field: str,
    expected: dict[str, Any],
) -> None:
    """A section's declared population must be exactly the result's own."""
    embedded = result[section].get(field)
    if embedded is None:
        raise M2PersistenceError(
            f"{section} does not declare {field}. Every headline section must "
            "name the population it was computed over, so no section can "
            "silently borrow another's denominator."
        )
    if embedded != expected:
        raise M2PersistenceError(
            f"{section}.{field} differs from the arm result's own identity. The "
            "metric and its declared population must describe the same rows."
        )


def build_canonical_run_lock(
    *,
    experiment_id: str,
    arm: str,
    execution_identity: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
    population_identities: dict[str, Any],
    development_source_identity: dict[str, Any] | None,
    recovery_lineage: dict[str, Any] | None,
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
    partition = require_canonical_development_partition(inputs["partition"])

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
        **{
            field: population_identities.get(field)
            for field in POPULATION_IDENTITY_FIELDS
        },
        # Top-level rather than only nested inside the stress selection: the
        # raw .stb the stress intervals came from is provenance for the whole
        # arm, not a detail of one evidence section.
        "development_source_identity": development_source_identity,
        **dict(recovery_lineage or {}),
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
        "validation_accessed": True,
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
    # `memory_selected` is null by design; the four population identities are
    # governed by `requires_evaluation` below, so a run that produced no
    # label-joined evaluation is not forced to invent one.
    _nullable = {
        "memory_selected",
        "development_source_identity",
        *POPULATION_IDENTITY_FIELDS,
        *RECOVERY_LINEAGE_FIELDS,
    }
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

    require_canonical_development_partition(lock["partition_accessed"])
    if lock["validation_accessed"] is not True:
        raise M2PersistenceError(
            "Canonical M2 development evidence is computed on VALIDATION and "
            "must record validation_accessed=true."
        )
    if lock["test_accessed"] is not False:
        raise M2PersistenceError("A canonical M2 lock must record test_accessed=false.")
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
        validate_replay_population_identity(lock["replay_population_identity"])
        validate_primary_population_identity(
            lock["primary_evaluation_population_identity"]
        )
        validate_challenge_population_identity(
            lock["challenge_evaluation_population_identity"]
        )
        validate_stress_selection_identity(lock["stress_interval_selection_identity"])
        validate_development_source_identity(lock["development_source_identity"])
        validate_recovery_lineage(lock)

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


def _require_identity_payload(identity: Any, name: str) -> dict[str, Any]:
    if not identity or not isinstance(identity, dict):
        raise M2PersistenceError(
            f"A claim-bearing M2 result must bind a real {name}; None, {{}} and "
            "non-object values are refused."
        )
    return dict(identity)


def _require_evaluated_rows(identity: dict[str, Any], name: str) -> None:
    rows = identity.get("evaluated_rows")
    if not isinstance(rows, int) or rows <= 0:
        raise M2PersistenceError(
            f"{name}.evaluated_rows must be a positive integer; received {rows!r}."
        )
    _require_sha256(
        f"{name}.evaluated_ordered_stable_id_sha256",
        identity.get("evaluated_ordered_stable_id_sha256"),
    )
    if not identity.get("identity_key"):
        raise M2PersistenceError(f"{name} names no identity key.")
    if identity.get("positional_join_used") is not False:
        raise M2PersistenceError(f"{name} must record positional_join_used=false.")
    if identity.get("matches_frozen_authority_exactly") is not True:
        raise M2PersistenceError(
            f"{name} must be proven to be EXACTLY the frozen population; a "
            "self-consistent subset is not a headline claim."
        )


def validate_replay_population_identity(
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """The FULL CAUSAL REPLAY population -- never a metric denominator."""
    from cardiosentinel.neural.m2_populations import POPULATION_REPLAY, REPLAY_AUTHORITY

    payload = _require_identity_payload(identity, "replay_population_identity")
    if payload.get("population") != POPULATION_REPLAY:
        raise M2PersistenceError(
            f"replay_population_identity must declare population "
            f"{POPULATION_REPLAY!r}; received {payload.get('population')!r}."
        )
    if payload.get("source") != REPLAY_AUTHORITY:
        raise M2PersistenceError(
            "The replay population must come from the verified full input "
            f"bundle; received {payload.get('source')!r}."
        )
    rows = payload.get("row_count")
    if not isinstance(rows, int) or rows <= 0:
        raise M2PersistenceError(
            f"replay row_count must be a positive integer; received {rows!r}."
        )
    _require_sha256(
        "replay ordered_stable_id_sha256", payload.get("ordered_stable_id_sha256")
    )
    _require_sha256("replay stream_cache_sha256", payload.get("stream_cache_sha256"))
    if payload.get("causal_history_rows_dropped") is not False:
        raise M2PersistenceError(
            "The full replay population must retain every frozen timeline row; "
            "dropping causal history invalidates the memory trajectory."
        )
    return payload


def validate_primary_population_identity(
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """The PRIMARY classification denominator, frozen upstream by P1/M1."""
    from cardiosentinel.neural.m2_populations import (
        POPULATION_PRIMARY,
        PRIMARY_AUTHORITY,
        PRIMARY_VALIDATION_POPULATION,
    )

    payload = _require_identity_payload(
        identity, "primary_evaluation_population_identity"
    )
    if payload.get("population") != POPULATION_PRIMARY:
        raise M2PersistenceError(
            f"primary_evaluation_population_identity must declare "
            f"{POPULATION_PRIMARY!r}; received {payload.get('population')!r}."
        )
    if payload.get("authority") != PRIMARY_AUTHORITY:
        raise M2PersistenceError(
            "The primary metric population must come from the frozen P1 "
            f"validation population; received {payload.get('authority')!r}."
        )
    if payload.get("counts") != dict(PRIMARY_VALIDATION_POPULATION):
        raise M2PersistenceError(
            f"The primary metric population {payload.get('counts')!r} differs "
            f"from the frozen identity {dict(PRIMARY_VALIDATION_POPULATION)!r}."
        )
    if payload.get("membership_derived_from_m2_scores") is not False:
        raise M2PersistenceError(
            "Primary membership is fixed by P1/M1 upstream and may never be "
            "derived from an M2 score."
        )
    if payload.get("binary_labels_present") is not True:
        raise M2PersistenceError(
            "The primary classification denominator requires binary labels."
        )
    _require_evaluated_rows(payload, "primary_evaluation_population_identity")
    return payload


def validate_challenge_population_identity(
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """The CHALLENGE population, from the frozen validation challenge selection."""
    from cardiosentinel.neural.m2_populations import (
        CHALLENGE_AUTHORITY,
        POPULATION_CHALLENGE,
    )
    from cardiosentinel.neural.validation_challenge import (
        CHALLENGE_EXPECTED_COUNTS,
        CHALLENGE_SELECTION_SHA256,
        CHALLENGE_TOTAL_WINDOWS,
    )

    payload = _require_identity_payload(
        identity, "challenge_evaluation_population_identity"
    )
    if payload.get("population") != POPULATION_CHALLENGE:
        raise M2PersistenceError(
            f"challenge_evaluation_population_identity must declare "
            f"{POPULATION_CHALLENGE!r}; received {payload.get('population')!r}."
        )
    if payload.get("authority") != CHALLENGE_AUTHORITY:
        raise M2PersistenceError(
            "The challenge metric population must come from the frozen "
            f"validation challenge selection; received {payload.get('authority')!r}."
        )
    if payload.get("challenge_selection_sha256") != CHALLENGE_SELECTION_SHA256:
        raise M2PersistenceError(
            "The challenge population does not bind the frozen challenge "
            "selection digest."
        )
    if payload.get("row_count") != CHALLENGE_TOTAL_WINDOWS:
        raise M2PersistenceError(
            f"The challenge population holds {payload.get('row_count')!r} "
            f"windows; the frozen selection holds {CHALLENGE_TOTAL_WINDOWS}."
        )
    expected_counts = {k: dict(v) for k, v in CHALLENGE_EXPECTED_COUNTS.items()}
    if payload.get("counts") != expected_counts:
        raise M2PersistenceError(
            "The challenge family counts differ from the frozen identity."
        )
    if payload.get("binary_labels_invented") is not False:
        raise M2PersistenceError(
            "No binary primary label may be invented for a challenge row."
        )
    _require_evaluated_rows(payload, "challenge_evaluation_population_identity")
    return payload


DEVELOPMENT_SOURCE_IDENTITY_CLASS: Final = "m2_v1_development_source_integrity"


def validate_development_source_identity(
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """The proof that the raw `.stb` was the official frozen development source.

    The stress selection reads raw LTSTDB annotations, so the files it read must
    be bound to the official pinned manifest and the frozen feature-corpus
    identity. This validates the receipt the existing repository verifiers
    produced; it invents no second source-identity algorithm.
    """
    payload = _require_identity_payload(identity, "development_source_identity")
    if payload.get("identity_class") != DEVELOPMENT_SOURCE_IDENTITY_CLASS:
        raise M2PersistenceError(
            f"development_source_identity must declare identity_class "
            f"{DEVELOPMENT_SOURCE_IDENTITY_CLASS!r}; received "
            f"{payload.get('identity_class')!r}."
        )
    if payload.get("annotation_set") != "stb":
        raise M2PersistenceError(
            "The stress selection reads the primary `.stb` annotation set; "
            f"received {payload.get('annotation_set')!r}."
        )
    if payload.get("test_partition_hashed") is not False:
        raise M2PersistenceError(
            "Development source verification must never hash a TEST file."
        )
    if payload.get("verified_before_stress_selection") is not True:
        raise M2PersistenceError(
            "The development source must be verified BEFORE any raw annotation "
            "is read for stress selection."
        )
    for name in ("feature_receipt", "source_receipt"):
        receipt = payload.get(name)
        if not isinstance(receipt, dict):
            raise M2PersistenceError(f"{name} is missing from the source identity.")
        if receipt.get("verification_result") != "passed":
            raise M2PersistenceError(
                f"{name}.verification_result is "
                f"{receipt.get('verification_result')!r}, not 'passed'."
            )
    return payload


def validate_stress_selection_identity(
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """The source-defined stress-interval selection and its frozen decision."""
    from cardiosentinel.neural.m2_stress_intervals import (
        DECISION_DOCUMENT,
        DECISION_SHA256,
        EXCLUDED_MARKER_FAMILIES,
        SOURCE_DEFINED_FAMILIES,
    )

    payload = _require_identity_payload(identity, "stress_interval_selection_identity")
    if payload.get("decision_document") != DECISION_DOCUMENT:
        raise M2PersistenceError(
            "The stress selection must bind the frozen eligibility decision "
            f"{DECISION_DOCUMENT!r}."
        )
    if payload.get("decision_sha256") != DECISION_SHA256:
        raise M2PersistenceError(
            "The stress selection does not bind the frozen eligibility decision "
            "digest; the implementation and the decision have drifted apart."
        )
    _require_sha256(
        "stress_interval_selection_sha256",
        payload.get("stress_interval_selection_sha256"),
    )
    if payload.get("source_defined_families") != list(SOURCE_DEFINED_FAMILIES):
        raise M2PersistenceError(
            "The stress selection admits families other than the frozen "
            f"source-defined set {list(SOURCE_DEFINED_FAMILIES)}."
        )
    count = payload.get("eligible_interval_count")
    if not isinstance(count, int) or count < 0:
        raise M2PersistenceError(
            f"eligible_interval_count must be a non-negative integer; received "
            f"{count!r}."
        )
    excluded = payload.get("excluded_marker_families")
    if not isinstance(excluded, dict) or set(excluded) != set(EXCLUDED_MARKER_FAMILIES):
        raise M2PersistenceError(
            "The stress selection must audit every excluded marker family "
            f"{sorted(EXCLUDED_MARKER_FAMILIES)}."
        )
    for family, entry in excluded.items():
        if entry.get("reason") != EXCLUDED_MARKER_FAMILIES[family]:
            raise M2PersistenceError(
                f"Excluded family {family} records a non-canonical reason."
            )
        if entry.get("drift_value_produced") is not False:
            raise M2PersistenceError(
                f"Excluded family {family} must produce no drift value; a zero "
                "is a measurement, and no measurement was possible."
            )
    for flag in (
        "marker_vicinity_reused_as_stress_duration",
        "persistence_duration_invented",
        "merge_gap_applied",
        "selection_influenced_by_m2_outputs",
    ):
        if payload.get(flag) is not False:
            raise M2PersistenceError(f"The stress selection must record {flag}=false.")
    if payload.get("selection_performed_after_label_blind_replay") is not True:
        raise M2PersistenceError(
            "Stress intervals are selected only AFTER the label-blind replay."
        )
    return payload


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


SUITE_CLASS: Final = "m2_v1_two_arm_suite"


def build_suite_body(
    *,
    suite_id: str,
    arm_results: dict[str, dict[str, Any]],
    arm_lock_sha256: dict[str, str],
    population_identities: dict[str, Any],
    development_source_identity: dict[str, Any] | None,
    recovery_lineage: dict[str, Any] | None,
    git_sha: str | None,
) -> dict[str, Any]:
    """The suite aggregation WITHOUT its self-digest.

    Deliberately unsigned: the suite's own PRE_PROMOTION observation does not
    exist yet, and a digest computed before that observation could never cover
    it. `finalize_and_promote_suite_result` embeds the complete runtime block
    and only then signs the payload, so the promoted artifact proves its own
    promotion gate rather than asserting it.
    """
    if set(arm_results) != set(M2_ARMS):
        raise M2PersistenceError(
            f"An M2 suite binds exactly {M2_ARMS}; received {sorted(arm_results)}."
        )
    body: dict[str, Any] = {
        "suite_id": suite_id,
        "suite_class": SUITE_CLASS,
        "arms": list(M2_ARMS),
        "arm_results": arm_results,
        "arm_experiment_ids": {
            arm: arm_experiment_id(suite_id, arm) for arm in M2_ARMS
        },
        "arm_experiment_lock_sha256": dict(arm_lock_sha256),
        "git_sha": git_sha,
        "development_source_identity": development_source_identity,
        **dict(recovery_lineage or {}),
        "memory_selection_performed": False,
        "memory_selected": None,
        "automatic_arm_preference_applied": False,
        "new_scientific_metric_computed": False,
        "human_review_required": True,
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "rollback_evaluated": False,
    }
    for field in POPULATION_IDENTITY_FIELDS:
        body[field] = population_identities.get(field)
    return body


def build_suite_result(
    *,
    suite_id: str,
    arm_results: dict[str, dict[str, Any]],
    arm_lock_sha256: dict[str, str] | None = None,
    population_identities: dict[str, Any] | None = None,
    development_source_identity: dict[str, Any] | None = None,
    recovery_lineage: dict[str, Any] | None = None,
    git_sha: str | None = None,
    runtime_identity_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A two-arm suite that expresses no retention decision.

    An AGGREGATION of two already-frozen arm results. It computes no new
    scientific metric, compares nothing and applies no preference. The digest
    is taken over the body INCLUDING the runtime block, so the suite's own
    promotion evidence is covered by its signature.
    """
    payload = build_suite_body(
        suite_id=suite_id,
        arm_results=arm_results,
        arm_lock_sha256=dict(arm_lock_sha256 or {}),
        population_identities=dict(population_identities or {}),
        development_source_identity=development_source_identity,
        recovery_lineage=recovery_lineage,
        git_sha=git_sha,
    )
    payload["runtime_identity_checks"] = runtime_identity_checks
    payload["m2_suite_sha256"] = canonical_sha256(payload)
    return payload


def _validate_suite_runtime_block(suite: dict[str, Any]) -> None:
    """The suite must carry its own GREEN START and PRE_PROMOTION evidence."""
    checks = suite.get("runtime_identity_checks")
    if not isinstance(checks, dict):
        raise M2PersistenceError(
            "A canonical M2 suite must bind its own runtime_identity_checks; "
            "without them the promoted artifact cannot prove its promotion gate."
        )
    if checks.get("expected_digest") != FROZEN_DEPENDENCY_DIGEST:
        raise M2PersistenceError(
            "The suite runtime block does not expect the frozen scientific identity."
        )
    if checks.get("all_observations_matched") is not True:
        raise M2PersistenceError(
            "Canonical suite evidence requires every runtime observation to match."
        )
    observed = {
        (item.get("enforcement_point"), item.get("detail"))
        for item in checks.get("checks", ())
    }
    points = {point for point, _detail in observed}
    if EnforcementPoint.START.value not in points:
        raise M2PersistenceError(
            "The suite runtime block records no START observation."
        )
    if (EnforcementPoint.PRE_PROMOTION.value, f"promote:{SUITE_RESULT_NAME}") not in (
        observed
    ):
        raise M2PersistenceError(
            f"The suite runtime block records no PRE_PROMOTION observation for "
            f"{SUITE_RESULT_NAME}; an arm's observation is never reused as the "
            "suite's."
        )
    for item in checks.get("checks", ()):
        if item.get("observed_digest") != FROZEN_DEPENDENCY_DIGEST:
            raise M2PersistenceError(
                "A suite runtime observation did not observe the frozen identity."
            )


def validate_suite_result(
    suite: dict[str, Any],
    *,
    run_root: Path | None = None,
    expected_suite_id: str | None = None,
) -> dict[str, Any]:
    """Validate the actual canonical suite contract.

    When `run_root` is supplied the declarations are checked against the ACTUAL
    arm artifacts on disk: a suite is the immutable aggregation of two frozen
    arms, so declaring a digest the files do not have is exactly the failure
    this must catch.
    """
    recorded = suite.get("m2_suite_sha256")
    body = {k: v for k, v in suite.items() if k != "m2_suite_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise M2PersistenceError("M2 suite result failed digest validation.")
    if suite.get("suite_class") != SUITE_CLASS:
        raise M2PersistenceError(
            f"suite_class must be {SUITE_CLASS!r}; received "
            f"{suite.get('suite_class')!r}."
        )
    suite_id = suite.get("suite_id")
    if not suite_id:
        raise M2PersistenceError("A canonical M2 suite must name its suite id.")
    if expected_suite_id is not None and suite_id != expected_suite_id:
        raise M2PersistenceError(
            f"suite_id is {suite_id!r}; the canonical production suite is "
            f"{expected_suite_id!r}."
        )
    if list(suite.get("arms", ())) != list(M2_ARMS):
        raise M2PersistenceError(f"An M2 suite binds exactly {M2_ARMS}, in order.")

    expected_ids = {arm: arm_experiment_id(suite_id, arm) for arm in M2_ARMS}
    if suite.get("arm_experiment_ids") != expected_ids:
        raise M2PersistenceError(
            f"The suite's arm experiment ids must be the deterministic "
            f"{expected_ids}; received {suite.get('arm_experiment_ids')!r}."
        )
    if not isinstance(suite.get("git_sha"), str) or not _GIT_SHA_PATTERN.match(
        suite["git_sha"]
    ):
        raise M2PersistenceError(
            f"suite git_sha is malformed: {suite.get('git_sha')!r}."
        )

    for flag in (
        "memory_selection_performed",
        "automatic_arm_preference_applied",
        "new_scientific_metric_computed",
        "test_accessed",
        "rollback_evaluated",
    ):
        if suite.get(flag) is not False:
            raise M2PersistenceError(f"A canonical M2 suite must record {flag}=false.")
    if suite.get("memory_selected") is not None:
        raise M2PersistenceError("A canonical M2 suite selects no arm.")
    if suite.get("validation_accessed") is not True:
        raise M2PersistenceError(
            "A canonical M2 suite records validation_accessed=true."
        )
    if suite.get("sealed_test_state") != "unopened":
        raise M2PersistenceError("The B4 sealed test must remain unopened.")

    for field in POPULATION_IDENTITY_FIELDS:
        if not suite.get(field):
            raise M2PersistenceError(f"A canonical M2 suite must bind {field}.")
    source_identity = validate_development_source_identity(
        suite.get("development_source_identity")
    )
    validate_recovery_lineage(suite)
    _validate_suite_runtime_block(suite)

    results = suite.get("arm_results") or {}
    locks = suite.get("arm_experiment_lock_sha256") or {}
    if set(results) != set(M2_ARMS) or set(locks) != set(M2_ARMS):
        raise M2PersistenceError(f"An M2 suite binds exactly {M2_ARMS}.")
    for arm in M2_ARMS:
        entry = results[arm] or {}
        if entry.get("experiment_id") != expected_ids[arm]:
            raise M2PersistenceError(
                f"Suite arm {arm} declares experiment_id "
                f"{entry.get('experiment_id')!r}, not {expected_ids[arm]!r}."
            )
        _require_sha256(
            f"suite arm_result_sha256[{arm}]", entry.get("arm_result_sha256")
        )
        _require_sha256(f"suite arm_experiment_lock_sha256[{arm}]", locks.get(arm))

    if run_root is not None:
        _verify_suite_against_arm_artifacts(
            suite, run_root=Path(run_root), source_identity=source_identity
        )
    return suite


def _verify_suite_against_arm_artifacts(
    suite: dict[str, Any], *, run_root: Path, source_identity: dict[str, Any]
) -> None:
    """Prove every suite declaration against the actual promoted arm files."""
    suite_id = suite["suite_id"]
    for arm in M2_ARMS:
        run_dir = Path(run_root) / arm_experiment_id(suite_id, arm)
        result_path = run_dir / ARM_RESULT_NAME
        lock_path = run_dir / EXPERIMENT_LOCK_NAME
        for path in (result_path, lock_path):
            if not path.is_file():
                raise M2PersistenceError(
                    f"Arm {arm} is not COMPLETE ({path.name} is absent); there is "
                    "no canonical suite."
                )
        declared_result = suite["arm_results"][arm]["arm_result_sha256"]
        actual_result = sha256_file(result_path)
        if actual_result != declared_result:
            raise M2PersistenceError(
                f"Arm {arm} result digests to {actual_result}, but the suite "
                f"declares {declared_result}. The suite aggregates the frozen "
                "arms; it never restates them."
            )

        lock = read_json_result(lock_path)
        declared_lock = suite["arm_experiment_lock_sha256"][arm]
        if lock.get("experiment_lock_sha256") != declared_lock:
            raise M2PersistenceError(
                f"Arm {arm} lock digest {lock.get('experiment_lock_sha256')!r} "
                f"differs from the suite's {declared_lock!r}."
            )
        validate_canonical_run_lock(lock, run_dir=run_dir)
        if lock.get("artifact_sha256", {}).get(ARM_RESULT_NAME) != actual_result:
            raise M2PersistenceError(
                f"Arm {arm} lock does not bind the promoted result's digest."
            )
        if lock.get("arm") != arm:
            raise M2PersistenceError(f"Arm {arm} lock records arm {lock.get('arm')!r}.")
        if lock.get("experiment_id") != arm_experiment_id(suite_id, arm):
            raise M2PersistenceError(
                f"Arm {arm} lock records experiment_id {lock.get('experiment_id')!r}."
            )
        if lock.get("git_sha") != suite["git_sha"]:
            raise M2PersistenceError(
                f"Arm {arm} lock binds git_sha {lock.get('git_sha')!r}, but the "
                f"suite binds {suite['git_sha']!r}. Both arms and the suite are "
                "one execution."
            )
        for field in (*POPULATION_IDENTITY_FIELDS, *RECOVERY_LINEAGE_FIELDS):
            if lock.get(field) != suite.get(field):
                raise M2PersistenceError(
                    f"Arm {arm} lock's {field} differs from the suite's."
                )
        if lock.get("development_source_identity") != source_identity:
            raise M2PersistenceError(
                f"Arm {arm} lock's development_source_identity differs from the "
                "suite's."
            )
        if lock.get("test_accessed") is not False:
            raise M2PersistenceError(f"Arm {arm} lock records test_accessed true.")
        if lock.get("sealed_test_state") != "unopened":
            raise M2PersistenceError(f"Arm {arm} lock reports the sealed test opened.")
        if lock.get("memory_selected") is not None:
            raise M2PersistenceError(f"Arm {arm} lock selects an arm.")


# --------------------------------------------------------------------------
# Two-arm suite identity: one suite, two INDEPENDENT canonical attempts
# --------------------------------------------------------------------------


ARM_ID_SEPARATOR: Final = "__"
EVIDENCE_WORKSPACE_SUFFIX: Final = f"{ARM_ID_SEPARATOR}evidence"


def arm_experiment_id(suite_id: str, arm: str) -> str:
    """The deterministic per-arm attempt identity, e.g. `<suite>__M2-G`.

    Each arm needs its OWN immutable claim directory: a shared experiment id
    would make M2-0 claim the directory and M2-G collide with it, so the
    canonical two-arm run could never start. The convention is deterministic --
    never random, never timestamped, and never auto-renamed on collision,
    because any of those would let a consumed attempt be silently re-run.
    """
    if arm not in M2_ARMS:
        raise M2PersistenceError(f"Unknown M2 arm {arm!r}.")
    if ARM_ID_SEPARATOR in str(suite_id):
        raise M2PersistenceError(
            f"A suite id may not contain {ARM_ID_SEPARATOR!r}; it would make the "
            "arm attempt identities ambiguous."
        )
    return f"{suite_id}{ARM_ID_SEPARATOR}{arm}"


def suite_directory(run_root: Path, suite_id: str) -> Path:
    return Path(run_root) / suite_id


def evidence_workspace(run_root: Path, suite_id: str) -> Path:
    """The disk-backed evidence workspace belonging to exactly this suite.

    Derived from the suite attempt rather than caller-selected, so a generic
    root holding a previous attempt's evidence can never be silently reused.
    """
    return Path(run_root) / f"{suite_id}{EVIDENCE_WORKSPACE_SUFFIX}"


def require_unclaimed_suite(run_root: Path, suite_id: str) -> dict[str, Any]:
    """PAIR preflight: prove nothing from this suite attempt already exists.

    Run BEFORE either arm is claimed and therefore before any VALIDATION
    access. A pre-existing arm claim, suite directory, suite result or evidence
    workspace means the attempt is already consumed: it is never deleted,
    reset, renamed, re-rooted, reseeded or automatically retried.
    """
    root = Path(run_root)
    occupied: list[str] = []
    for arm in M2_ARMS:
        arm_dir = root / arm_experiment_id(suite_id, arm)
        if arm_dir.exists():
            occupied.append(str(arm_dir))
        staging = arm_dir.parent / f"{STAGING_PREFIX}{arm_dir.name}"
        if staging.exists():
            occupied.append(str(staging))
    for path in (suite_directory(root, suite_id), evidence_workspace(root, suite_id)):
        if path.exists():
            occupied.append(str(path))
    if occupied:
        raise M2PersistenceError(
            f"Canonical M2 suite {suite_id} is already claimed; these paths "
            f"exist: {sorted(occupied)}. The attempt is consumed. Nothing is "
            "deleted, reset, renamed, re-rooted or reseeded, and no automatic "
            "retry or alternate name is permitted. This requires documented "
            "human review."
        )
    return {
        "suite_id": suite_id,
        "arm_experiment_ids": {
            arm: arm_experiment_id(suite_id, arm) for arm in M2_ARMS
        },
        "existing_arm_claim": False,
        "existing_suite_result": False,
        "existing_evidence_workspace": False,
        "automatic_alternate_name_permitted": False,
    }


def claim_evidence_workspace(run_root: Path, suite_id: str) -> Path:
    """Create this suite's evidence workspace, refusing to reuse another's."""
    workspace = evidence_workspace(run_root, suite_id)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    try:
        workspace.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise M2PersistenceError(
            f"Evidence workspace {workspace} already exists. It may hold another "
            "attempt's evidence; it is never overwritten, cleaned or reused."
        ) from error
    return workspace


def finalize_and_promote_suite_result(
    run_root: Path,
    suite_id: str,
    *,
    suite_body: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
    expected_suite_id: str | None = None,
) -> dict[str, Any]:
    """Sign and atomically promote the one canonical suite result.

    The sequence exists so the promoted artifact can PROVE its own promotion
    gate rather than assert it:

    1. the caller creates the suite record and records START;
    2. every declaration is verified against the ACTUAL arm artifacts;
    3. `M2_SUITE_RESULT.json` takes its OWN PRE_PROMOTION observation -- never
       a reused arm observation;
    4. every suite observation must be GREEN;
    5. the complete runtime block is embedded in the payload;
    6. `m2_suite_sha256` is computed only AFTER that block exists, so the
       signature covers the promotion evidence;
    7. the artifact is promoted atomically and re-validated from its bytes.

    No observation is ever fabricated after hashing. If either arm is not
    COMPLETE there is no canonical suite; if promotion fails, both arm
    artifacts are retained for human review and nothing is re-run
    automatically.
    """
    require_frozen_runtime_record(runtime)
    if "m2_suite_sha256" in suite_body:
        raise M2PersistenceError(
            "The suite body must be unsigned here: a digest taken before the "
            "suite's PRE_PROMOTION observation could never cover it."
        )

    # Verify the aggregation against the real files BEFORE taking the
    # promotion observation, so a mismatched arm never reaches the gate.
    _verify_suite_against_arm_artifacts(
        suite_body,
        run_root=Path(run_root),
        source_identity=validate_development_source_identity(
            suite_body.get("development_source_identity")
        ),
    )

    check = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        expected_digest=runtime.expected_digest,
        detail=f"promote:{SUITE_RESULT_NAME}",
    )
    runtime.record(check)
    if not check.matches:
        raise RuntimeIntegrityError(
            "Runtime identity differed before the suite promotion. The suite was "
            "NOT promoted. Both arm results and locks are retained for human "
            "review, never deleted or blessed, and nothing is retried "
            "automatically."
        )
    if not runtime.all_matched:
        mismatch = runtime.first_mismatch()
        raise RuntimeIntegrityError(
            "Canonical suite evidence requires every runtime observation to "
            f"match; {mismatch.enforcement_point!r} observed "
            f"{mismatch.observed_digest}. The suite was NOT promoted."
        )

    suite = dict(suite_body)
    suite["runtime_identity_checks"] = runtime.as_dict()
    suite["m2_suite_sha256"] = canonical_sha256(suite)
    validate_suite_result(
        suite, run_root=Path(run_root), expected_suite_id=expected_suite_id
    )

    directory = suite_directory(run_root, suite_id)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise M2PersistenceError(
            f"Canonical M2 suite directory {directory} already exists."
        ) from error
    path = directory / SUITE_RESULT_NAME
    write_json_atomic(path, suite)
    validate_suite_result(
        read_json_result(path),
        run_root=Path(run_root),
        expected_suite_id=expected_suite_id,
    )
    return suite


def read_json_result(path: Path) -> dict[str, Any]:
    """Read back a promoted artifact so its persisted bytes are what validate."""
    import json

    return json.loads(Path(path).read_text())


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
    """Mark an attempt FAILED_OR_INTERRUPTED. The claim is never released.

    `claim_bearing_result_promoted` is read from the ACTUAL artifact, not
    assumed false. The finalizer deliberately permits a window where the arm
    result has been promoted and verified but the experiment-lock promotion
    check then fails: on that path the promoted result is preserved, and a
    status file claiming nothing was promoted would misdescribe the filesystem.

    A promoted result without a completed lock is preserved forensic evidence,
    not canonical science -- hence `canonical=false` alongside it. The result is
    never deleted.
    """
    provenance = git_provenance(REPOSITORY_ROOT)
    payload = {
        "experiment_id": claimed.experiment_id,
        "arm": claimed.arm,
        "status": STATUS_FAILED,
        "claim_bearing_result_promoted": (claimed.run_dir / ARM_RESULT_NAME).is_file(),
        "experiment_lock_promoted": (claimed.run_dir / EXPERIMENT_LOCK_NAME).is_file(),
        "promotion_state_source": "filesystem",
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


def failure_review_directory(run_root: Path, suite_id: str) -> Path:
    """Where a suite's ADDITIVE forensic receipt lives.

    Outside every immutable arm directory on purpose: the original claim files
    are historical evidence and are never rewritten to make a failed state look
    cleaner.
    """
    return Path(run_root) / f"{suite_id}{FAILURE_REVIEW_SUFFIX}"


def write_forensic_failure_receipt(
    run_root: Path, suite_id: str, receipt: dict[str, Any]
) -> dict[str, Any]:
    """Write ONE additive non-claim-bearing receipt outside the arm directories.

    The original `M2_RUN_STATUS.json` files are historical evidence. They are
    never rewritten by this function -- a failed attempt is classified beside
    the claim, not by editing it.
    """
    payload = dict(receipt)
    payload["receipt_sha256"] = canonical_sha256(payload)
    directory = failure_review_directory(run_root, suite_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / ATTEMPT_FAILURE_RECEIPT_NAME, payload)
    return payload


def preserved_status_digests(run_root: Path, suite_id: str) -> dict[str, str]:
    """SHA-256 of each preserved claim status file, for forensic binding."""
    digests: dict[str, str] = {}
    for arm in M2_ARMS:
        path = Path(run_root) / arm_experiment_id(suite_id, arm) / RUN_STATUS_NAME
        if path.is_file():
            digests[arm] = sha256_file(path)
    return digests


INDETERMINATE: Final = "indeterminate"
"""Recorded when an abrupt failure leaves a fact genuinely unknowable.

Preferred over a confident `false`: after an uncaught exception, claiming that
scoring never started -- when it may well have -- would understate scientific
exposure in exactly the direction that flatters the run.
"""


def record_attempt_failure(
    run_root: Path,
    suite_id: str,
    *,
    exception: BaseException,
    stage: str,
    claimed_arms: Sequence[str],
    validation_opened: bool,
    exposure: dict[str, Any] | None = None,
    runtime_records: dict[str, RuntimeIntegrityRecord] | None = None,
    promotion_state: dict[str, Any] | None = None,
    decision_sha256: str | None = None,
) -> dict[str, Any]:
    """Deterministic NON-CLAIM-BEARING accounting for a failed attempt.

    M2 development attempt #1 ended with both `M2_RUN_STATUS.json` files still
    reading STARTED, because the exception escaped outside a promotion gate and
    nothing recorded it. Once any arm claim exists, an uncaught canonical-run
    exception must leave deterministic evidence instead.

    **The exposure it records is the REAL execution state**, supplied by the
    caller's tracker. Attempt #1's `scoring_started=false` /
    `metrics_computed=false` are a frozen determination about THAT attempt and
    belong only to its lineage; a future failure after the scorer has been
    invoked must never claim scoring never started. Where a fact cannot be
    proven after an abrupt exception it is recorded as `indeterminate` rather
    than as a confident `false`.

    This deletes nothing, cleans nothing, retries nothing and renames nothing.
    Staged and evidence files are preserved exactly as the failure left them,
    and a partially failed attempt is never made to look COMPLETE.
    """
    provenance = git_provenance(REPOSITORY_ROOT)
    observed = dict(exposure or {})
    arms = sorted(str(arm) for arm in claimed_arms)
    receipt: dict[str, Any] = {
        "artifact_class": "m2_attempt_failure_receipt",
        "claim_bearing": False,
        "scientific_evidence": False,
        "canonical": False,
        "suite_id": suite_id,
        "failed_stage": stage,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "claimed_arms": arms,
        "any_arm_claimed": bool(arms),
        # --- real scientific exposure, not a hard-coded optimism ------------
        "validation_opened": bool(validation_opened),
        "scoring_started": observed.get("scoring_started", INDETERMINATE),
        "replay_completed": observed.get("replay_completed", INDETERMINATE),
        "post_replay_evaluation_started": observed.get(
            "post_replay_evaluation_started", INDETERMINATE
        ),
        "metrics_computed_or_completed": observed.get(
            "metrics_computed_or_completed", INDETERMINATE
        ),
        "metrics_completed_per_arm": dict(
            observed.get("metrics_completed_per_arm", {})
        ),
        "exposure_source": "runtime execution tracker",
        # --- governance --------------------------------------------------
        "memory_selection_performed": False,
        "memory_selected": None,
        "rollback": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "recorded_at": _now(),
        "automatic_retry_performed": False,
        "automatic_cleanup_performed": False,
        "alternate_suite_id_used": False,
        "staged_evidence_preserved": True,
        "human_review_required": True,
        "recovery_decision_sha256": decision_sha256,
        "promotion_state": _reconcile_promotion_state(
            observed_promotion_state(run_root, suite_id, claimed_arms),
            promotion_state,
        ),
    }
    receipt["runtime_identity_checks"] = {
        arm: record.as_dict() for arm, record in sorted((runtime_records or {}).items())
    }
    receipt["preserved_status_sha256"] = preserved_status_digests(run_root, suite_id)
    receipt = write_forensic_failure_receipt(run_root, suite_id, receipt)

    # The established FAILED_OR_INTERRUPTED mechanism, applied to each claim
    for arm in claimed_arms:
        run_dir = Path(run_root) / arm_experiment_id(suite_id, arm)
        if not run_dir.is_dir():
            continue
        status_path = run_dir / RUN_STATUS_NAME
        started_at = None
        if status_path.is_file():
            started_at = read_json_result(status_path).get("started_at")
        promoted = receipt["promotion_state"]["arm_result_promoted"].get(arm, False)
        lock_promoted = receipt["promotion_state"]["experiment_lock_promoted"].get(
            arm, False
        )
        write_json_atomic(
            status_path,
            {
                "experiment_id": arm_experiment_id(suite_id, arm),
                "arm": arm,
                "status": STATUS_FAILED,
                "claim_bearing_result_promoted": bool(promoted),
                "experiment_lock_promoted": bool(lock_promoted),
                "promotion_state_source": "filesystem",
                "canonical": False,
                "reason": f"{stage}: {type(exception).__name__}: {exception}",
                "started_at": started_at or receipt["recorded_at"],
                "updated_at": receipt["recorded_at"],
                "human_review_required": True,
                "repeat_attempt_permitted": False,
                "automatic_retry_performed": False,
            },
        )
    return receipt


def observed_promotion_state(
    run_root: Path, suite_id: str, claimed_arms: Sequence[str]
) -> dict[str, Any]:
    """Promotion state read from the ACTUAL immutable artifacts.

    The filesystem is the forensic authority. The finalizer permits a window in
    which an arm result is promoted and verified but the experiment-lock
    promotion check then fails; a tracker that only records success after the
    whole finalizer returns would report `false` for a file that demonstrably
    exists.
    """
    root = Path(run_root)
    arms = {str(arm) for arm in claimed_arms}
    results: dict[str, bool] = {}
    locks: dict[str, bool] = {}
    for arm in M2_ARMS:
        run_dir = root / arm_experiment_id(suite_id, arm)
        # An unclaimed arm has no directory, so nothing of its own is promoted.
        results[arm] = arm in arms and (run_dir / ARM_RESULT_NAME).is_file()
        locks[arm] = arm in arms and (run_dir / EXPERIMENT_LOCK_NAME).is_file()
    suite = (suite_directory(root, suite_id) / SUITE_RESULT_NAME).is_file()
    return {
        "arm_result_promoted": results,
        "experiment_lock_promoted": locks,
        "suite_result_promoted": suite,
    }


def _require_per_arm_map(value: Any, field: str) -> dict[str, bool]:
    """Forensic promotion evidence is per arm; one boolean names no arm.

    A scalar `true` cannot say WHICH arm promoted, so it is refused rather than
    expanded to both -- silently turning one arm's success into two would be
    exactly the misreport this accounting exists to prevent.
    """
    if not isinstance(value, dict):
        raise M2PersistenceError(
            f"{field} must be a per-arm map; received {value!r}. A scalar cannot "
            "identify which arm promoted."
        )
    unknown = sorted(set(value) - set(M2_ARMS))
    if unknown:
        raise M2PersistenceError(f"{field} names unknown arms {unknown}.")
    return {arm: bool(value.get(arm, False)) for arm in M2_ARMS}


def _reconcile_promotion_state(
    observed: dict[str, Any], tracker: dict[str, Any] | None
) -> dict[str, Any]:
    """Filesystem truth, with the tracker's observation kept alongside it.

    The tracker stays useful as a record of what the run believed, but it can
    never contradict the artifacts: `observed_from_filesystem` is the canonical
    forensic value and `coherent` says whether the two agreed.
    """
    payload: dict[str, Any] = {
        "observed_from_filesystem": observed,
        "authority": "filesystem",
    }
    if tracker is None:
        payload["tracker_observation"] = None
        payload["coherent"] = True
    else:
        normalised = {
            "arm_result_promoted": _require_per_arm_map(
                tracker.get("arm_result_promoted"), "arm_result_promoted"
            ),
            "experiment_lock_promoted": _require_per_arm_map(
                tracker.get("experiment_lock_promoted"), "experiment_lock_promoted"
            ),
            "suite_result_promoted": bool(tracker.get("suite_result_promoted", False)),
        }
        payload["tracker_observation"] = normalised
        payload["coherent"] = normalised == observed
    # The canonical fields ARE the filesystem's.
    payload.update(observed)
    return payload


def audit_forbidden_partitions(execution_identity: dict[str, Any]) -> None:
    """Refuse promotion of anything that touched a forbidden partition."""
    partition = str(execution_identity.get("partition_accessed", "")).lower()
    if partition in FORBIDDEN_PARTITIONS:
        raise M2ExecutionError(
            f"Forbidden partition {partition!r} reached the promotion gate."
        )
    if execution_identity.get("test_accessed") is not False:
        raise M2PersistenceError(
            "A claim-bearing M2 artifact must record test_accessed=false."
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
    # The RESULT owns its population identities; finalization extracts them
    # rather than accepting free-standing arguments that could disagree.
    populations = {field: result.get(field) for field in POPULATION_IDENTITY_FIELDS}
    source_identity = result.get("development_source_identity")

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
        population_identities=populations,
        development_source_identity=source_identity,
        recovery_lineage={
            field: result.get(field) for field in RECOVERY_LINEAGE_FIELDS
        },
        started_at=claimed.started_at,
        completed_at=_now(),
        artifact_sha256={ARM_RESULT_NAME: artifact_digest},
    )
    if requires_evaluation:
        for field in POPULATION_IDENTITY_FIELDS:
            if lock[field] != populations[field]:
                raise M2PersistenceError(
                    f"The lock's {field} differs from the result's."
                )
        if lock["development_source_identity"] != source_identity:
            raise M2PersistenceError(
                "The lock's development_source_identity differs from the result's."
            )
        for field in RECOVERY_LINEAGE_FIELDS:
            if lock[field] != result.get(field):
                raise M2PersistenceError(
                    f"The lock's {field} differs from the result's."
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
