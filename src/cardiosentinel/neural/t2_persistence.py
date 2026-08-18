"""Claim-bearing T2 persistence: one canonical TRAIN-only attempt, one lock.

This follows the M2/U1 convention rather than inventing a second provenance
system. `require_frozen_runtime_record` is imported from `m2_persistence`
unchanged, so a T2 claim rests on exactly the same frozen-digest invariant a
canonical U1 claim did, and the runtime-integrity sentinel is the existing one.

**The directory is the claim.** `t2-v1-training` is created with
`exist_ok=False`; if it exists in ANY state the attempt is consumed and the run
STOPS. Nothing is deleted, reset, renamed, re-rooted, reseeded or retried, and
there is no alternate suite name to fall back on.

**Outer VALIDATION is not merely unimplemented here -- it is refused.** The
activation constant below is the only switch, it is `False`, and there is no
environment variable, flag or argument that can move it. A future activation
change set flips it after the TRAIN-only artifacts are human-reviewed.
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
from cardiosentinel.neural.t2_protocol import (
    T2_ARM_GRU,
    T2_ARM_S4D,
    T2_EXPECTED_PARAMETER_COUNTS,
    T2_INTERNAL_SPLIT_SHA256,
    T2_PROTOCOL_SHA256,
    T2_SPLIT_SHA256,
)

# ---------------------------------------------------------------------------
# The frozen execution specification
# ---------------------------------------------------------------------------
T2_EXECUTION_SPEC_NAME: Final = "T2_CANONICAL_TRAINING_EXECUTION_SPEC_V1"
T2_EXECUTION_SPEC_PATH: Final = (
    REPOSITORY_ROOT / "docs" / f"{T2_EXECUTION_SPEC_NAME}.md"
)
T2_EXECUTION_SPEC_SHA256: Final = (
    "af6ebf1a6314edb86cce7aa88a6260dd1bd155fd0aebe472d3745b6c823b8054"
)

# ---------------------------------------------------------------------------
# The one canonical TRAIN-only claim
# ---------------------------------------------------------------------------
T2_EXPERIMENT_IDENTITY: Final = "T2_temporal_v1"
T2_TRAINING_ATTEMPT_ID: Final = "t2-v1-training"
T2_RUN_ROOT: Final = Path("cardiosentinel-runs/phase8-t2-development-v1")

STATUS_STARTED: Final = "STARTED"
STATUS_COMPLETE: Final = "COMPLETE"
STATUS_FAILED: Final = "FAILED_OR_INTERRUPTED"

RUN_STATUS_NAME: Final = "T2_TRAINING_STATUS.json"
PREFLIGHT_NAME: Final = "T2_TRAINING_PREFLIGHT.json"
INTERNAL_SPLIT_NAME: Final = "T2_INTERNAL_SPLIT.json"
POPULATION_NAME: Final = "T2_TRAINING_POPULATION.json"
GRU_RESULT_NAME: Final = "T2_GRU_TRAINING_RESULT.json"
S4D_RESULT_NAME: Final = "T2_S4D_TRAINING_RESULT.json"
RESULT_NAME: Final = "T2_TRAINING_RESULT.json"
EXPERIMENT_LOCK_NAME: Final = "T2_TRAINING_EXPERIMENT_LOCK.json"
FAILURE_RECEIPT_NAME: Final = "T2_TRAINING_FAILURE_RECEIPT.json"

CHECKPOINT_NAME: Final = {
    T2_ARM_GRU: "T2_GRU_BEST_CHECKPOINT.pt",
    T2_ARM_S4D: "T2_S4D_BEST_CHECKPOINT.pt",
}
CHECKPOINT_LOCK_NAME: Final = {
    T2_ARM_GRU: "T2_GRU_CHECKPOINT_LOCK.json",
    T2_ARM_S4D: "T2_S4D_CHECKPOINT_LOCK.json",
}
ARM_RESULT_NAME: Final = {T2_ARM_GRU: GRU_RESULT_NAME, T2_ARM_S4D: S4D_RESULT_NAME}

COMPONENT_ARTIFACTS: Final = (
    PREFLIGHT_NAME,
    INTERNAL_SPLIT_NAME,
    POPULATION_NAME,
    GRU_RESULT_NAME,
    S4D_RESULT_NAME,
)

STAGING_PREFIX: Final = ".staging-"
ID_SEPARATOR: Final = "__"
REVIEW_SUFFIX: Final = f"{ID_SEPARATOR}review"

RESULT_CLASS: Final = "t2_v1_canonical_training_result"
LOCK_CLASS: Final = "t2_v1_canonical_training_run_lock"

ARM_SELECTION_PENDING: Final = "pending_one_shot_outer_validation"

# The single activation switch. There is deliberately no setter, no environment
# variable and no CLI flag: flipping it is a reviewed change set.
T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED: Final = False

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")


class T2PersistenceError(RuntimeError):
    """Raised when a claim-bearing T2 artifact cannot be persisted safely."""


class T2ActivationError(RuntimeError):
    """Raised when outer-VALIDATION execution is attempted while unauthorized."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _require_sha256(label: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.match(value):
        raise T2PersistenceError(f"{label} is not a SHA-256 digest: {value!r}.")
    return value


def validate_t2_execution_spec(path: Path = T2_EXECUTION_SPEC_PATH) -> str:
    """Verify the frozen execution specification byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise T2PersistenceError(f"T2 execution spec is missing at {document}.")
    digest = sha256_file(document)
    if digest != T2_EXECUTION_SPEC_SHA256:
        raise T2PersistenceError(
            f"T2 execution spec digest {digest} differs from the frozen "
            f"{T2_EXECUTION_SPEC_SHA256}. The specification is immutable."
        )
    return digest


# ---------------------------------------------------------------------------
# Outer-VALIDATION activation firewall
# ---------------------------------------------------------------------------


def require_outer_validation_authorized() -> None:
    """Refuse before any VALIDATION per-row artifact can be opened.

    This is called at the top of every outer-VALIDATION entry point, before path
    resolution, before the representation memmap and before any label read.
    """
    if T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is not True:
        raise T2ActivationError(
            "T2 outer VALIDATION execution is not authorized by the frozen "
            "activation state. No VALIDATION representation array, label or "
            "timeline evidence is opened. A separate reviewed activation change "
            "set authorizes the one-shot run after the TRAIN-only artifacts are "
            "reviewed; there is no flag, argument or environment variable that "
            "bypasses this."
        )


# ---------------------------------------------------------------------------
# Deterministic paths -- no timestamp, no uuid, no random suffix
# ---------------------------------------------------------------------------


def t2_run_directory(run_root: Path, attempt_id: str) -> Path:
    if ID_SEPARATOR in str(attempt_id):
        raise T2PersistenceError(f"A T2 attempt id may not contain {ID_SEPARATOR!r}.")
    return Path(run_root) / str(attempt_id)


def t2_review_directory(run_root: Path, attempt_id: str) -> Path:
    """Where additive, non-claim-bearing receipts live, OUTSIDE the claim."""
    return Path(run_root) / f"{attempt_id}{REVIEW_SUFFIX}"


def require_unclaimed_t2_attempt(run_root: Path, attempt_id: str) -> dict[str, Any]:
    """Prove nothing from this attempt exists. Run BEFORE any timeline access."""
    run_dir = t2_run_directory(run_root, attempt_id)
    occupied = [
        str(path)
        for path in (
            run_dir,
            run_dir.parent / f"{STAGING_PREFIX}{run_dir.name}",
            t2_review_directory(run_root, attempt_id),
        )
        if path.exists()
    ]
    if occupied:
        raise T2PersistenceError(
            f"Canonical T2 attempt {attempt_id} is already claimed; these paths "
            f"exist: {sorted(occupied)}. The attempt is consumed. Nothing is "
            "deleted, reset, renamed, re-rooted or reseeded, no automatic retry "
            "or alternate name is permitted, and this requires documented human "
            "review."
        )
    return {
        "attempt_id": str(attempt_id),
        "existing_run_directory": False,
        "existing_review_directory": False,
        "automatic_alternate_name_permitted": False,
        "automatic_retry_permitted": False,
    }


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class T2RunDirectory:
    """A claimed canonical T2 attempt and the artifacts promoted into it."""

    run_dir: Path
    attempt_id: str
    started_at: str
    promoted: dict[str, str] = field(default_factory=dict)

    @property
    def staging_dir(self) -> Path:
        return self.run_dir.parent / f"{STAGING_PREFIX}{self.run_dir.name}"


def claim_t2_run_directory(
    run_root: Path, attempt_id: str, *, runtime: RuntimeIntegrityRecord
) -> T2RunDirectory:
    """Atomically claim the one canonical T2 attempt. The directory IS the claim."""
    require_frozen_runtime_record(runtime)
    require_runtime_identity(
        EnforcementPoint.PRE_PROMOTION, record=runtime, detail="t2_claim_directory"
    )
    run_dir = t2_run_directory(run_root, attempt_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise T2PersistenceError(
            f"Canonical T2 attempt {attempt_id} is already claimed at {run_dir}. "
            "Automatic rerun, retry, selective rerun and fresh-seed restart are "
            "prohibited and require documented human review."
        ) from error
    claimed = T2RunDirectory(
        run_dir=run_dir, attempt_id=str(attempt_id), started_at=_now()
    )
    write_json_atomic(
        run_dir / RUN_STATUS_NAME,
        {
            "attempt_id": claimed.attempt_id,
            "experiment_identity": T2_EXPERIMENT_IDENTITY,
            "status": STATUS_STARTED,
            "claim_bearing_result_promoted": False,
            "started_at": claimed.started_at,
            "updated_at": claimed.started_at,
            "arm_selection_status": ARM_SELECTION_PENDING,
            "outer_validation_accessed": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        },
    )
    return claimed


def promote_component(
    claimed: T2RunDirectory,
    name: str,
    payload: dict[str, Any],
    *,
    runtime: RuntimeIntegrityRecord,
) -> str:
    """Promote one component artifact under its own PRE_PROMOTION observation."""
    if name not in COMPONENT_ARTIFACTS:
        raise T2PersistenceError(f"{name!r} is not a T2 component artifact.")
    if name in claimed.promoted:
        raise T2PersistenceError(f"{name} was already promoted; it is immutable.")
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


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def promote_checkpoint(
    claimed: T2RunDirectory,
    arm: str,
    state_dict: dict[str, Any],
    *,
    identity: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
) -> dict[str, Any]:
    """Persist one arm's retained checkpoint and its immutable identity lock."""
    import torch

    if arm not in CHECKPOINT_NAME:
        raise T2PersistenceError(f"{arm!r} is not a frozen T2 candidate.")
    require_frozen_runtime_record(runtime)
    check = observe_runtime_identity(
        EnforcementPoint.PRE_PROMOTION,
        expected_digest=runtime.expected_digest,
        detail=f"promote:{CHECKPOINT_NAME[arm]}",
    )
    runtime.record(check)
    if not check.matches:
        raise RuntimeIntegrityError(
            f"Runtime identity differed before promoting the {arm} checkpoint; "
            "it was NOT promoted and nothing is retried."
        )
    path = claimed.run_dir / CHECKPOINT_NAME[arm]
    if path.exists():
        raise T2PersistenceError(f"{path.name} already exists; it is immutable.")
    torch.save(state_dict, path)
    digest = sha256_file(path)
    lock = {
        "lock_class": "t2_checkpoint_lock",
        "architecture": arm,
        "checkpoint_file": CHECKPOINT_NAME[arm],
        "checkpoint_sha256": digest,
        "trainable_parameters": T2_EXPECTED_PARAMETER_COUNTS[arm],
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": T2_EXECUTION_SPEC_SHA256,
        "internal_split_sha256": T2_INTERNAL_SPLIT_SHA256,
        "arm_selection_status": ARM_SELECTION_PENDING,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        **dict(identity),
    }
    lock["checkpoint_lock_sha256"] = canonical_sha256(lock)
    write_json_atomic(claimed.run_dir / CHECKPOINT_LOCK_NAME[arm], lock)
    return lock


def load_checkpoint(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load a checkpoint by the safest supported route, digest checked first.

    `weights_only=True` refuses to deserialize arbitrary executable Python, and
    the digest is verified before the file is opened as a model, so a mutated
    checkpoint never reaches `torch.load`.
    """
    import torch

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise T2PersistenceError(f"No T2 checkpoint at {checkpoint}.")
    observed = sha256_file(checkpoint)
    if observed != expected_sha256:
        raise T2PersistenceError(
            f"Checkpoint {checkpoint.name} digests to {observed}, but its lock "
            f"binds {expected_sha256}. A mutated checkpoint is refused."
        )
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Runtime and lock
# ---------------------------------------------------------------------------


def runtime_provenance() -> dict[str, Any]:
    """Interpreter, dependency and device facts, without importing Torch twice."""
    import torch

    environment = dependency_environment()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = None
    if device_type == "cuda":  # pragma: no cover - no CUDA in this runtime
        device_name = torch.cuda.get_device_name(0)
    return {
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "package_count": int(environment["installed_package_count"]),
        "dependency_digest": str(environment["installed_packages_sha256"]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_type": device_type,
        "device_name": device_name,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }


def require_single_runtime(first: dict[str, Any], second: dict[str, Any]) -> None:
    """Both arms must share one device and runtime. Otherwise STOP."""
    for field_ in (
        "device_type",
        "device_name",
        "torch_version",
        "cuda_version",
        "dependency_digest",
        "deterministic_algorithms",
    ):
        if first.get(field_) != second.get(field_):
            raise T2PersistenceError(
                f"The two T2 arms observed different {field_}: "
                f"{first.get(field_)!r} vs {second.get(field_)!r}. A mixed "
                "runtime comparison is not admissible; execution STOPS."
            )


REQUIRED_LOCK_FIELDS: Final = (
    "git_sha",
    "git_dirty",
    "interpreter",
    "package_count",
    "dependency_digest",
    "t2_protocol_sha256",
    "t2_execution_spec_sha256",
    "split_sha256",
    "internal_split_sha256",
    "train_timeline_identity",
    "checkpoint_sha256",
    "artifact_sha256",
)


def build_t2_run_lock(
    *,
    attempt_id: str,
    runtime: RuntimeIntegrityRecord,
    provenance: dict[str, Any],
    started_at: str,
    completed_at: str,
    artifact_sha256: dict[str, str],
) -> dict[str, Any]:
    """Construct the one canonical T2 training lock. The only identity assembly."""
    git = git_provenance(REPOSITORY_ROOT)
    environment = runtime_provenance()
    lock: dict[str, Any] = {
        "lock_class": LOCK_CLASS,
        "attempt_id": str(attempt_id),
        "experiment_identity": T2_EXPERIMENT_IDENTITY,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        **environment,
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": T2_EXECUTION_SPEC_SHA256,
        "split_sha256": T2_SPLIT_SHA256,
        "internal_split_sha256": T2_INTERNAL_SPLIT_SHA256,
        "train_timeline_identity": provenance["train_timeline_identity"],
        "fit_subjects": provenance["fit_subjects"],
        "internal_dev_subjects": provenance["internal_dev_subjects"],
        "positive_class_weight": provenance["positive_class_weight"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "internal_dev_thresholds": provenance["internal_dev_thresholds"],
        "trainable_parameters": dict(T2_EXPECTED_PARAMETER_COUNTS),
        "arm_selection_status": ARM_SELECTION_PENDING,
        "arm_selected": None,
        "outer_validation_accessed": False,
        "outer_validation_execution_authorized": (
            T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED
        ),
        "runtime_dependency_digest_start": runtime.digest_at(EnforcementPoint.START),
        "runtime_dependency_digest_pre_promotion": runtime.digest_at(
            EnforcementPoint.PRE_PROMOTION
        ),
        "runtime_dependency_digest_end": runtime.digest_at(EnforcementPoint.COMPLETION),
        "runtime_identity_checks": runtime.as_dict(),
        "partition_accessed": "train",
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "started_at": started_at,
        "completed_at": completed_at,
        "artifact_sha256": dict(artifact_sha256),
        "automatic_retry_performed": False,
        "repeat_attempt_permitted": False,
    }
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    return lock


def validate_t2_run_lock(
    lock: dict[str, Any], *, run_dir: Path | None = None
) -> dict[str, Any]:
    """Validate the ACTUAL values of a T2 lock, not merely the presence of keys."""
    recorded = lock.get("experiment_lock_sha256")
    body = {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise T2PersistenceError("The T2 experiment lock failed digest validation.")
    if lock.get("lock_class") != LOCK_CLASS:
        raise T2PersistenceError(f"Unknown T2 lock class {lock.get('lock_class')!r}.")
    missing = [name for name in REQUIRED_LOCK_FIELDS if name not in lock]
    if missing:
        raise T2PersistenceError(f"The canonical T2 lock is missing {missing}.")
    if not isinstance(lock["git_sha"], str) or not _GIT_SHA_PATTERN.match(
        lock["git_sha"]
    ):
        raise T2PersistenceError(f"git_sha is malformed: {lock['git_sha']!r}.")
    if lock["git_dirty"] is not False:
        raise T2PersistenceError(
            "Canonical T2 evidence requires a clean Git checkout, matching the "
            "existing P1/M1/M2/U1 convention."
        )
    for field_, expected in (
        ("t2_protocol_sha256", T2_PROTOCOL_SHA256),
        ("t2_execution_spec_sha256", T2_EXECUTION_SPEC_SHA256),
        ("split_sha256", T2_SPLIT_SHA256),
        ("internal_split_sha256", T2_INTERNAL_SPLIT_SHA256),
    ):
        _require_sha256(field_, lock[field_])
        if lock[field_] != expected:
            raise T2PersistenceError(
                f"{field_} is {lock[field_]!r}, expected the frozen {expected!r}."
            )
    for flag in (
        "validation_accessed",
        "test_accessed",
        "outer_validation_accessed",
        "outer_validation_execution_authorized",
        "automatic_retry_performed",
        "repeat_attempt_permitted",
    ):
        if lock[flag] is not False:
            raise T2PersistenceError(f"A canonical T2 lock must record {flag}=false.")
    if lock["sealed_test_state"] != "unopened":
        raise T2PersistenceError("The B4 sealed test must remain unopened.")
    if lock["arm_selection_status"] != ARM_SELECTION_PENDING:
        raise T2PersistenceError(
            "TRAIN-only execution selects no arm; the status must remain "
            f"{ARM_SELECTION_PENDING!r}."
        )
    if lock["arm_selected"] is not None:
        raise T2PersistenceError(
            "A winner cannot exist before the one-shot outer VALIDATION."
        )
    for label in ("start", "pre_promotion", "end"):
        digest = lock[f"runtime_dependency_digest_{label}"]
        _require_sha256(f"runtime_dependency_digest_{label}", digest)
        if digest != FROZEN_DEPENDENCY_DIGEST:
            raise T2PersistenceError(
                f"runtime_dependency_digest_{label} is not the frozen identity."
            )
    if lock["runtime_identity_checks"].get("all_observations_matched") is not True:
        raise T2PersistenceError(
            "Canonical T2 evidence requires every runtime observation to match."
        )
    artifacts = dict(lock["artifact_sha256"])
    if not artifacts:
        raise T2PersistenceError("A canonical T2 lock binds no artifact hash.")
    for name, digest in artifacts.items():
        _require_sha256(f"artifact_sha256[{name}]", digest)
        if run_dir is not None:
            path = Path(run_dir) / name
            if not path.is_file() or sha256_file(path) != digest:
                raise T2PersistenceError(
                    f"T2 artifact {name} does not match its lock digest."
                )
    for arm, digest in dict(lock["checkpoint_sha256"]).items():
        _require_sha256(f"checkpoint_sha256[{arm}]", digest)
        if run_dir is not None:
            path = Path(run_dir) / CHECKPOINT_NAME[arm]
            if not path.is_file() or sha256_file(path) != digest:
                raise T2PersistenceError(
                    f"The {arm} checkpoint does not match its lock digest."
                )
    return lock


def validate_t2_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    """A complete T2 training result in its own right, before anything is hashed."""
    if result.get("artifact_class") != RESULT_CLASS:
        raise T2PersistenceError(
            f"Unknown T2 result class {result.get('artifact_class')!r}."
        )
    missing = [
        name
        for name in (
            "attempt_id",
            "component_sha256",
            "checkpoint_sha256",
            "internal_dev_thresholds",
            "arm_selection_status",
            "outer_validation_accessed",
            "test_accessed",
        )
        if name not in result
    ]
    if missing:
        raise T2PersistenceError(f"The T2 result is missing {missing}.")
    absent = [
        name for name in COMPONENT_ARTIFACTS if name not in result["component_sha256"]
    ]
    if absent:
        raise T2PersistenceError(
            f"The T2 result does not bind every component artifact: {absent}."
        )
    if result["arm_selection_status"] != ARM_SELECTION_PENDING:
        raise T2PersistenceError(
            "TRAIN-only execution declares no winner; the status must remain "
            f"{ARM_SELECTION_PENDING!r}."
        )
    if result.get("arm_selected") is not None:
        raise T2PersistenceError("A T2 training result may not select an arm.")
    if result.get("outer_validation_accessed") is not False:
        raise T2PersistenceError("A T2 training result records VALIDATION access.")
    if result.get("test_accessed") is not False:
        raise T2PersistenceError("A T2 result must record test_accessed=false.")
    if result.get("sealed_test_state") != "unopened":
        raise T2PersistenceError("The B4 sealed test must remain unopened.")
    return result


def validate_canonical_t2_attempt(run_root: Path, attempt_id: str) -> dict[str, Any]:
    """The one complete canonical verification of a promoted T2 attempt."""
    run_dir = t2_run_directory(run_root, attempt_id)
    result_path = run_dir / RESULT_NAME
    lock_path = run_dir / EXPERIMENT_LOCK_NAME
    for path in (result_path, lock_path):
        if not path.is_file():
            raise T2PersistenceError(f"No canonical T2 artifact at {path}.")
    result = json.loads(result_path.read_text())
    validate_t2_result_payload(result)
    for name, digest in result["component_sha256"].items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise T2PersistenceError(
                f"T2 component {name} does not match the digest the result binds."
            )
    lock = validate_t2_run_lock(json.loads(lock_path.read_text()), run_dir=run_dir)
    if dict(result["checkpoint_sha256"]) != dict(lock["checkpoint_sha256"]):
        raise T2PersistenceError(
            "The result and the lock disagree about the retained checkpoints."
        )
    return {
        "verification_class": "t2_canonical_attempt_verification",
        "attempt_id": str(attempt_id),
        "result_sha256": sha256_file(result_path),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "component_sha256": dict(result["component_sha256"]),
        "checkpoint_sha256": dict(lock["checkpoint_sha256"]),
        "arm_selection_status": ARM_SELECTION_PENDING,
        "verified": True,
    }


def record_t2_attempt_failure(
    run_root: Path,
    claimed: T2RunDirectory,
    *,
    exception: BaseException,
    stage: str,
    arm: str | None,
    exposure: dict[str, Any],
    runtime: RuntimeIntegrityRecord | None = None,
) -> dict[str, Any]:
    """One additive forensic receipt for a post-claim failure. No retry."""
    git = git_provenance(REPOSITORY_ROOT)
    promoted = {
        name: sha256_file(claimed.run_dir / name)
        for name in (*COMPONENT_ARTIFACTS, RESULT_NAME, EXPERIMENT_LOCK_NAME)
        if (claimed.run_dir / name).is_file()
    }
    receipt = {
        "receipt_class": "t2_training_failure_receipt",
        "claim_bearing": False,
        "attempt_id": claimed.attempt_id,
        "failed_stage": str(stage),
        "arm": arm,
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
        "arm_selection_status": ARM_SELECTION_PENDING,
        "outer_validation_accessed": False,
        "automatic_retry_performed": False,
        "repeat_attempt_permitted": False,
        "attempt_consumed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "human_review_required": True,
        "recorded_at": _now(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    directory = t2_review_directory(run_root, claimed.attempt_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / FAILURE_RECEIPT_NAME, receipt)
    write_json_atomic(
        claimed.run_dir / RUN_STATUS_NAME,
        {
            "attempt_id": claimed.attempt_id,
            "status": STATUS_FAILED,
            "claim_bearing_result_promoted": (claimed.run_dir / RESULT_NAME).is_file(),
            "promotion_state_source": "filesystem",
            "canonical": False,
            "failed_stage": str(stage),
            "failure_receipt_sha256": receipt["receipt_sha256"],
            "started_at": claimed.started_at,
            "updated_at": _now(),
            "human_review_required": True,
            "repeat_attempt_permitted": False,
            "automatic_retry_performed": False,
            "outer_validation_accessed": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        },
    )
    return receipt


def finalize_and_promote_t2_result(
    claimed: T2RunDirectory,
    *,
    result: dict[str, Any],
    provenance: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
) -> dict[str, Any]:
    """Validate, gate, promote and lock the one canonical T2 training result."""
    require_frozen_runtime_record(runtime)
    validate_t2_result_payload(result)
    for name in COMPONENT_ARTIFACTS:
        if claimed.promoted.get(name) != result["component_sha256"].get(name):
            raise T2PersistenceError(
                f"The result's digest for {name} differs from the promoted bytes."
            )
    completion = observe_runtime_identity(
        EnforcementPoint.COMPLETION,
        expected_digest=runtime.expected_digest,
        detail=RESULT_NAME,
    )
    runtime.record(completion)
    if not completion.matches:
        raise RuntimeIntegrityError(
            "Runtime identity differed at COMPLETION. Canonical promotion is "
            "INVALIDATED: the T2 result was NOT promoted, the environment was "
            "NOT repaired and already-promoted components are retained as "
            "non-claim-bearing forensic material."
        )
    result_path = claimed.run_dir / RESULT_NAME
    write_json_atomic(result_path, result)
    result_digest = sha256_file(result_path)
    lock = build_t2_run_lock(
        attempt_id=claimed.attempt_id,
        runtime=runtime,
        provenance=provenance,
        started_at=claimed.started_at,
        completed_at=_now(),
        artifact_sha256={**claimed.promoted, RESULT_NAME: result_digest},
    )
    validate_t2_run_lock(lock)
    write_json_atomic(claimed.run_dir / EXPERIMENT_LOCK_NAME, lock)
    validate_t2_run_lock(lock, run_dir=claimed.run_dir)
    status = {
        "attempt_id": claimed.attempt_id,
        "experiment_identity": T2_EXPERIMENT_IDENTITY,
        "status": STATUS_COMPLETE,
        "claim_bearing_result_promoted": True,
        "canonical": True,
        "started_at": claimed.started_at,
        "updated_at": _now(),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "artifact_sha256": dict(lock["artifact_sha256"]),
        "checkpoint_sha256": dict(lock["checkpoint_sha256"]),
        "arm_selection_status": ARM_SELECTION_PENDING,
        "human_review_required": True,
        "outer_validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "runtime_identity_checks": runtime.as_dict(),
    }
    write_json_atomic(claimed.run_dir / RUN_STATUS_NAME, status)
    return {"result": result, "lock": lock, "status": status}
