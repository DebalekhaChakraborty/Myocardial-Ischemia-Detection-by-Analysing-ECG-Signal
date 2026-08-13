"""Claim-bearing M2 result/run-lock persistence with staged-then-promote safety.

A partial or interrupted run must never be able to look complete. Canonical
artifacts are staged first and promoted only after every gate passes, in this
order:

1. computation finished successfully and the result is staged, not canonical;
2. the forbidden-partition and artifact audits are GREEN;
3. the COMPLETION runtime check is taken, after the last scientific computation;
4. a final PRE_PROMOTION check is taken immediately before the promotion
   itself -- the frozen design requires one before EVERY claim-bearing
   promotion, including the arm claim directory, and that count is never
   reduced to simplify ordering;
5. the canonical provenance block is finalized INCLUDING start, every promotion
   check and completion;
6. the artifact hash is computed over exactly the bytes that get promoted;
7. the finalized artifact is atomically promoted and the run marked COMPLETE.

The COMPLETION observation is deliberately taken BEFORE the canonical artifact
is written, so a promoted result carries a genuinely observed end digest rather
than a fabricated one. A COMPLETION mismatch INVALIDATES canonical standing:
nothing is promoted, the run can never end COMPLETE/canonical, and the staged
result is retained as non-claim-bearing forensic material rather than deleted
or silently rewritten.

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
from cardiosentinel.neural.m2_scorer import RETAINED_M1L_LOCK_SHA256
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
    "evaluated_population_identity",
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
    evaluated_population_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every identity a canonical M2 run must carry.

    `evaluated_population_identity` is the exact evaluated row set, bound
    through the existing frozen ordered-stable-ID digest -- not merely a row
    count, and deliberately not a second dataset identity mechanism. It comes
    from `M2EvaluationBundle.population_identity()`. It is `None` only for a
    run that produced no label-joined evaluation.
    """
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
        "evaluated_population_identity": evaluated_population_identity,
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


CLAIM_DIRECTORY_PROMOTION_DETAIL: Final = "arm_claim_directory"


def claim_run_directory(
    run_root: Path,
    experiment_id: str,
    arm: str,
    *,
    runtime: RuntimeIntegrityRecord,
) -> M2RunDirectory:
    """Atomically claim the one canonical attempt. The directory IS the claim.

    Creating an arm claim directory is itself a claim-bearing promotion under
    the frozen sentinel design, so it carries its own PRE_PROMOTION runtime
    check. That check is taken BEFORE `mkdir`: on mismatch no canonical claim
    directory comes into existence at all. The START check is not accepted as a
    substitute -- the environment can move between the two.

    There is no force, overwrite, retry, reseed or delete path. If the
    directory already exists in any state the attempt is consumed and this
    refuses, exactly as the M1 convention does.
    """
    if arm not in M2_ARMS:
        raise M2PersistenceError(f"Unknown M2 arm {arm!r}.")

    start = runtime.digest_at(EnforcementPoint.START)
    if start is None:
        raise M2PersistenceError(
            "A canonical M2 claim requires a successful START runtime check "
            "before any scientific input is opened; none is recorded."
        )
    if not all(
        check.matches
        for check in runtime.checks
        if check.enforcement_point == EnforcementPoint.START.value
    ):
        raise M2PersistenceError(
            "The recorded START runtime check did not match the frozen "
            "identity; no canonical claim may be created."
        )

    # Claim-boundary PRE_PROMOTION check. Raises before mkdir on mismatch.
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


def validate_complete_runtime_identity(
    runtime: RuntimeIntegrityRecord,
) -> dict[str, Any]:
    """Require a COMPLETE, GREEN runtime block -- not merely a present field.

    A key whose value is `None` does not satisfy a provenance contract. Every
    required observation must exist AND match, so it is impossible to promote
    canonical COMPLETE evidence with a missing or mismatched digest at any
    enforcement point.
    """
    if runtime.expected_digest != FROZEN_DEPENDENCY_DIGEST:
        raise M2PersistenceError(
            f"The runtime record expects {runtime.expected_digest!r}, not the "
            f"frozen scientific identity {FROZEN_DEPENDENCY_DIGEST!r}."
        )
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


def validate_evaluated_population_identity(
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Require a real, non-empty evaluated-population identity.

    Reuses the existing frozen ordered-stable-ID digest; no second population
    identity algorithm is introduced.
    """
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
    digest = identity.get("evaluated_ordered_stable_id_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise M2PersistenceError(
            f"evaluated_ordered_stable_id_sha256 is malformed: {digest!r}."
        )
    try:
        int(digest, 16)
    except ValueError as error:
        raise M2PersistenceError(
            f"evaluated_ordered_stable_id_sha256 is not hexadecimal: {digest!r}."
        ) from error
    if not identity.get("identity_key"):
        raise M2PersistenceError("evaluated population identity names no identity key.")
    if identity.get("positional_join_used") is not False:
        raise M2PersistenceError(
            "A claim-bearing evaluated population must record "
            "positional_join_used=false."
        )
    return dict(identity)


def validate_claim_bearing_arm_result(
    result: dict[str, Any],
    *,
    execution_identity: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
    requires_evaluation: bool,
) -> dict[str, Any]:
    """The single final gate an M2 arm result must pass before promotion.

    It verifies identities, thresholds, partition authorization, runtime
    completeness and the absence of any retention selection. It computes and
    tunes no scientific metric, and it never selects an arm.
    """
    from cardiosentinel.neural.m2_scorer import (
        M1L_CLASSIFICATION_THRESHOLD,
        NORMAL_EVIDENCE_THRESHOLD,
    )

    if result.get("arm") not in M2_ARMS:
        raise M2PersistenceError(
            f"Unknown or missing arm in result: {result.get('arm')!r}."
        )
    if result.get("scientific_computation_completed") is not True:
        raise M2PersistenceError(
            "A claim-bearing result must record scientific_computation_completed=true."
        )
    if result.get("m2_protocol_sha256") != GATE.M2_PROTOCOL_SHA256:
        raise M2PersistenceError("The result does not bind the frozen M2 protocol.")
    if result.get("m2_gate_receipt_sha256") != GATE.M2_GATE_RECEIPT_SHA256:
        raise M2PersistenceError("The result does not bind the canonical gate receipt.")

    scorer = execution_identity["scorer_identity"]
    if scorer["retained_lock_sha256"] != RETAINED_M1L_LOCK_SHA256:
        raise M2PersistenceError("The result does not bind the retained M1L lock.")
    if scorer["classification_threshold"] != M1L_CLASSIFICATION_THRESHOLD:
        raise M2PersistenceError("The result does not bind the frozen M1L threshold.")
    if scorer["memory_admission_threshold"] != NORMAL_EVIDENCE_THRESHOLD:
        raise M2PersistenceError("The result does not bind the frozen M2 margin.")
    if scorer["classification_threshold_used_for_memory_admission"] is not False:
        raise M2PersistenceError(
            "The classification threshold must never gate memory admission."
        )

    audit_forbidden_partitions(execution_identity)
    validate_complete_runtime_identity(runtime)

    if requires_evaluation:
        validate_evaluated_population_identity(
            result.get("evaluated_population_identity")
        )
    if result.get("memory_selection_performed") not in (None, False):
        raise M2PersistenceError("A canonical M2 run performs no arm selection.")
    if result.get("memory_selected") not in (None,):
        raise M2PersistenceError("A canonical M2 run selects no arm automatically.")
    if result.get("rollback") not in (None, False):
        raise M2PersistenceError("Rollback is excluded from the claim-bearing core.")
    return result


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


def finalize_and_promote_arm_result(
    claimed: M2RunDirectory,
    *,
    result: dict[str, Any],
    execution_identity: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
    evaluated_population_identity: dict[str, Any] | None = None,
    requires_evaluation: bool = True,
) -> dict[str, Any]:
    """Stage, gate, finalize, then atomically promote one arm's result.

    The ordering matters and is the frozen sentinel design's, not a
    convenience: the COMPLETION observation is taken BEFORE the canonical
    provenance block is constructed, so the promoted artifact carries a
    genuinely observed end digest rather than a fabricated one.

    A. computation has finished (asserted by the caller's result flag);
    B. the result is staged, not yet canonical;
    C. forbidden-partition and artifact validation succeed;
    D. the COMPLETION runtime check is taken after the last computation;
    E. a final PRE_PROMOTION check is taken immediately before the promotion
       itself -- the design requires one before EVERY claim-bearing promotion,
       and that count is not reduced to simplify ordering;
    F. the canonical provenance block is finalized INCLUDING start, every
       promotion check and completion;
    G. the artifact hash is computed over the finalized bytes actually promoted;
    H. the finalized artifact is atomically promoted;
    I. the run is marked COMPLETE.

    Any runtime mismatch at C-E refuses canonical promotion outright. A
    COMPLETION mismatch INVALIDATES canonical standing: the run can never end
    COMPLETE/canonical, nothing is promoted to the canonical location, and the
    staged result is retained as non-claim-bearing forensic material rather
    than deleted or silently rewritten.
    """
    staging = claimed.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    staged_path = staging / ARM_RESULT_NAME

    if result.get("arm") != claimed.arm:
        raise M2PersistenceError("The staged result's arm disagrees with the claim.")
    staged = dict(result)
    if evaluated_population_identity is not None:
        staged["evaluated_population_identity"] = dict(evaluated_population_identity)
    write_json_atomic(staged_path, staged)

    # C. partition/authorization audit before anything else is considered.
    audit_forbidden_partitions(execution_identity)

    # D. COMPLETION, taken after the last scientific computation.
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

    # E. final PRE_PROMOTION, immediately before the promotion itself.
    pre_promotion = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        expected_digest=runtime.expected_digest,
        detail=f"promote:{ARM_RESULT_NAME}",
    )
    runtime.record(pre_promotion)
    if not pre_promotion.matches:
        _retain_forensic_failure(
            claimed,
            pre_promotion,
            "runtime identity differed immediately before promotion",
        )
        raise RuntimeIntegrityError(
            "Runtime identity differed at PRE_PROMOTION; the claim-bearing "
            "result was NOT promoted, the environment was NOT repaired and the "
            "attempt is consumed."
        )

    # F. finalize the canonical provenance with every observation present.
    finalized = dict(staged)
    finalized["runtime_identity_checks"] = validate_complete_runtime_identity(runtime)
    finalized["runtime_dependency_digest_start"] = runtime.digest_at(
        EnforcementPoint.START
    )
    finalized["runtime_dependency_digest_pre_promotion"] = runtime.digest_at(
        EnforcementPoint.PRE_PROMOTION
    )
    finalized["runtime_dependency_digest_end"] = runtime.digest_at(
        EnforcementPoint.COMPLETION
    )
    finalized["completed_at"] = _now()
    validate_claim_bearing_arm_result(
        finalized,
        execution_identity=execution_identity,
        runtime=runtime,
        requires_evaluation=requires_evaluation,
    )

    # G-H. hash exactly the bytes promoted, then promote atomically.
    write_json_atomic(staged_path, finalized)
    artifact_digest = sha256_file(staged_path)
    canonical_path = claimed.run_dir / ARM_RESULT_NAME
    write_json_atomic(canonical_path, finalized)
    if sha256_file(canonical_path) != artifact_digest:
        raise M2PersistenceError(
            "The promoted canonical artifact does not match the hashed bytes."
        )
    staged_path.unlink()
    staging.rmdir()

    # I. COMPLETE.
    status = {
        "experiment_id": claimed.experiment_id,
        "arm": claimed.arm,
        "status": STATUS_COMPLETE,
        "claim_bearing_result_promoted": True,
        "canonical": True,
        "started_at": claimed.started_at,
        "updated_at": _now(),
        "artifact_sha256": {ARM_RESULT_NAME: artifact_digest},
        "runtime_identity_checks": runtime.as_dict(),
    }
    write_json_atomic(claimed.run_dir / RUN_STATUS_NAME, status)
    return status


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


def require_runtime_start(runtime: RuntimeIntegrityRecord, detail: str) -> None:
    """The START enforcement point, before any scientific input is opened."""
    require_runtime_identity(EnforcementPoint.START, record=runtime, detail=detail)
