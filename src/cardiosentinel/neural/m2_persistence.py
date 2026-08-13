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
    _nullable = {"memory_selected", *POPULATION_IDENTITY_FIELDS}
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


def build_suite_result(
    *,
    suite_id: str,
    arm_results: dict[str, dict[str, Any]],
    arm_lock_sha256: dict[str, str] | None = None,
    population_identities: dict[str, Any] | None = None,
    development_source_identity: dict[str, Any] | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """A two-arm suite that expresses no retention decision.

    An AGGREGATION of two already-frozen arm results. It computes no new
    scientific metric, compares nothing and applies no preference.
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
        "arm_experiment_ids": {
            arm: arm_experiment_id(suite_id, arm) for arm in M2_ARMS
        },
        "arm_experiment_lock_sha256": dict(arm_lock_sha256 or {}),
        "git_sha": git_sha,
        "development_source_identity": development_source_identity,
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
        payload[field] = (population_identities or {}).get(field)
    payload["m2_suite_sha256"] = canonical_sha256(payload)
    return payload


def validate_suite_result(suite: dict[str, Any]) -> dict[str, Any]:
    """Re-verify a suite's self-digest and its no-selection invariants."""
    recorded = suite.get("m2_suite_sha256")
    body = {k: v for k, v in suite.items() if k != "m2_suite_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise M2PersistenceError("M2 suite result failed digest validation.")
    if list(suite.get("arms", ())) != list(M2_ARMS):
        raise M2PersistenceError(f"An M2 suite binds exactly {M2_ARMS}.")
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
    if not suite.get("development_source_identity"):
        raise M2PersistenceError(
            "A canonical M2 suite must bind the development source-integrity "
            "identity that the raw .stb stress selection was proven against."
        )
    return suite


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
    suite: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
    arm_run_dirs: dict[str, Path],
) -> dict[str, Any]:
    """Validate and atomically promote the one canonical suite result.

    `M2_SUITE_RESULT.json` is claim-bearing in its own right, so it takes its
    OWN PRE_PROMOTION observation -- never a reused arm observation. If either
    arm is not COMPLETE there is no canonical suite, and if suite promotion
    fails the two arm artifacts are retained for human review and nothing is
    re-run automatically.
    """
    require_frozen_runtime_record(runtime)
    validate_suite_result(suite)

    for arm, run_dir in sorted(arm_run_dirs.items()):
        for name in (ARM_RESULT_NAME, EXPERIMENT_LOCK_NAME):
            if not (Path(run_dir) / name).is_file():
                raise M2PersistenceError(
                    f"Arm {arm} is not COMPLETE ({name} is absent); there is no "
                    "canonical suite."
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
    validate_suite_result(read_json_result(path))
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
