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
from collections.abc import Sequence
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
    T2_ARMS,
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
T2_OUTER_VALIDATION_ATTEMPT_ID: Final = "t2-v1-outer-validation"

# Repository-anchored, not cwd-relative. Git provenance is already evaluated
# against REPOSITORY_ROOT, so a one-shot scientific claim must have exactly one
# physical location too: a cwd-relative path would let the same attempt be
# claimed twice from two shells, and would make "is this attempt consumed?"
# depend on where the human happened to be standing.
T2_RUN_ROOT: Final = (
    REPOSITORY_ROOT / "cardiosentinel-runs" / "phase8-t2-development-v1"
)

# Both attempts are siblings under the one absolute run root.
T2_ATTEMPT_IDS: Final = (T2_TRAINING_ATTEMPT_ID, T2_OUTER_VALIDATION_ATTEMPT_ID)

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

# Runtime-sentinel stage labels. The frozen sentinel design has exactly three
# enforcement points, so the finer training stages ride in the observation's
# `detail` field rather than inventing a fourth point in a frozen enum.
STAGE_TRAINING_START: Final = "t2_training_start"
STAGE_CLAIM: Final = "t2_claim_directory"


def stage_pre_model_construction(arm: str) -> str:
    return f"pre_model_construction:{arm}"


def stage_pre_checkpoint_promotion(arm: str) -> str:
    return f"pre_checkpoint_promotion:{arm}"


def required_runtime_stage_order(arms: Sequence[str]) -> tuple[str, ...]:
    """The frozen enforcement choreography, as `detail` labels in order."""
    stages: list[str] = [STAGE_TRAINING_START]
    for arm in arms:
        stages.append(stage_pre_model_construction(arm))
        stages.append(stage_pre_checkpoint_promotion(arm))
    stages.append(RESULT_NAME)
    return tuple(stages)


STAGING_PREFIX: Final = ".staging-"
ID_SEPARATOR: Final = "__"
REVIEW_SUFFIX: Final = f"{ID_SEPARATOR}review"

RESULT_CLASS: Final = "t2_v1_canonical_training_result"
LOCK_CLASS: Final = "t2_v1_canonical_training_run_lock"
CHECKPOINT_LOCK_CLASS: Final = "t2_checkpoint_lock"

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
        EnforcementPoint.PRE_PROMOTION, record=runtime, detail=STAGE_CLAIM
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


def observe_t2_runtime_stage(
    runtime: RuntimeIntegrityRecord,
    *,
    point: EnforcementPoint,
    detail: str,
) -> None:
    """Observe, record and refuse to continue past a mismatch at one stage.

    This is how the canonical body actually visits its enforcement points, so
    the persisted `runtime_identity_checks` sequence is evidence that they were
    visited rather than a claim that they could have been.

    START is the one point that cannot require a prior START, so it is checked
    against the frozen identity directly; every later point additionally
    requires that a successful START is already on the record.
    """
    if EnforcementPoint(point) is EnforcementPoint.START:
        if runtime.expected_digest != FROZEN_DEPENDENCY_DIGEST:
            raise T2PersistenceError(
                "A canonical T2 attempt requires the frozen scientific identity "
                f"{FROZEN_DEPENDENCY_DIGEST!r}; this record expects "
                f"{runtime.expected_digest!r}."
            )
    else:
        require_frozen_runtime_record(runtime)
    check = observe_runtime_identity(
        point, expected_digest=runtime.expected_digest, detail=detail
    )
    runtime.record(check)
    if not check.matches:
        raise RuntimeIntegrityError(
            f"Runtime identity differed at {detail}. Execution STOPS: the "
            "environment is NOT repaired, nothing is installed and nothing is "
            "retried."
        )


def observed_runtime_stages(runtime: RuntimeIntegrityRecord) -> tuple[str, ...]:
    """The `detail` labels of every recorded observation, in order."""
    return tuple(str(check.detail) for check in runtime.checks)


def require_runtime_stage_order(
    runtime: RuntimeIntegrityRecord, arms: Sequence[str]
) -> tuple[str, ...]:
    """Prove the frozen enforcement stages were visited, in the frozen order."""
    required = required_runtime_stage_order(arms)
    observed = observed_runtime_stages(runtime)
    filtered = tuple(label for label in observed if label in set(required))
    if filtered != required:
        raise T2PersistenceError(
            f"The runtime enforcement choreography was {list(filtered)}, not the "
            f"frozen {list(required)}."
        )
    return required


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
    observe_t2_runtime_stage(
        runtime,
        point=EnforcementPoint.PRE_PROMOTION,
        detail=stage_pre_checkpoint_promotion(arm),
    )
    path = claimed.run_dir / CHECKPOINT_NAME[arm]
    if path.exists():
        raise T2PersistenceError(f"{path.name} already exists; it is immutable.")
    torch.save(state_dict, path)
    digest = sha256_file(path)
    # The caller's identity block is laid down FIRST so the authoritative
    # fields below overwrite it. Spreading it last would let a caller supply
    # its own `checkpoint_sha256` or `architecture` and have the lock attest
    # to something other than the bytes just written.
    lock = {
        **dict(identity),
        "lock_class": CHECKPOINT_LOCK_CLASS,
        "attempt_id": claimed.attempt_id,
        "experiment_identity": T2_EXPERIMENT_IDENTITY,
        "architecture": arm,
        "checkpoint_file": CHECKPOINT_NAME[arm],
        "checkpoint_sha256": digest,
        "trainable_parameters": T2_EXPECTED_PARAMETER_COUNTS[arm],
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": T2_EXECUTION_SPEC_SHA256,
        "internal_split_sha256": T2_INTERNAL_SPLIT_SHA256,
        "arm_selection_status": ARM_SELECTION_PENDING,
        "arm_selected": None,
        "outer_validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    lock["checkpoint_lock_sha256"] = canonical_sha256(lock)
    lock_path = claimed.run_dir / CHECKPOINT_LOCK_NAME[arm]
    if lock_path.exists():
        raise T2PersistenceError(f"{lock_path.name} already exists; it is immutable.")
    write_json_atomic(lock_path, lock)
    validate_checkpoint_lock(lock, arm, run_dir=claimed.run_dir)
    return lock


def validate_checkpoint_lock(
    lock: dict[str, Any], arm: str, *, run_dir: Path | None = None
) -> dict[str, Any]:
    """Verify one checkpoint lock completely, against its own bytes and the file.

    Refused: a mutated checkpoint; a mutated lock; a lock whose self-digest no
    longer covers its body; a lock pointing at another arm's checkpoint file; a
    drifted architecture, parameter count, protocol digest, execution-spec
    digest or internal-split digest.
    """
    if arm not in CHECKPOINT_NAME:
        raise T2PersistenceError(f"{arm!r} is not a frozen T2 candidate.")
    recorded = lock.get("checkpoint_lock_sha256")
    body = {k: v for k, v in lock.items() if k != "checkpoint_lock_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise T2PersistenceError(
            f"The {arm} checkpoint lock failed its own digest validation."
        )
    if lock.get("lock_class") != CHECKPOINT_LOCK_CLASS:
        raise T2PersistenceError(
            f"Unknown checkpoint lock class {lock.get('lock_class')!r}."
        )
    for field_, expected in (
        ("architecture", arm),
        ("checkpoint_file", CHECKPOINT_NAME[arm]),
        ("trainable_parameters", T2_EXPECTED_PARAMETER_COUNTS[arm]),
        ("t2_protocol_sha256", T2_PROTOCOL_SHA256),
        ("t2_execution_spec_sha256", T2_EXECUTION_SPEC_SHA256),
        ("internal_split_sha256", T2_INTERNAL_SPLIT_SHA256),
        ("arm_selection_status", ARM_SELECTION_PENDING),
    ):
        if lock.get(field_) != expected:
            raise T2PersistenceError(
                f"The {arm} checkpoint lock binds {field_}={lock.get(field_)!r}, "
                f"expected {expected!r}."
            )
    if lock.get("arm_selected") is not None:
        raise T2PersistenceError(
            "A TRAIN-only checkpoint lock cannot name a selected arm."
        )
    for flag in ("outer_validation_accessed", "test_accessed"):
        if lock.get(flag) is not False:
            raise T2PersistenceError(f"A T2 checkpoint lock must record {flag}=false.")
    if lock.get("sealed_test_state") != "unopened":
        raise T2PersistenceError("The B4 sealed test must remain unopened.")
    digest = _require_sha256("checkpoint_sha256", lock.get("checkpoint_sha256"))
    if run_dir is not None:
        checkpoint = Path(run_dir) / CHECKPOINT_NAME[arm]
        if not checkpoint.is_file():
            raise T2PersistenceError(f"No {arm} checkpoint at {checkpoint}.")
        observed = sha256_file(checkpoint)
        if observed != digest:
            raise T2PersistenceError(
                f"The {arm} checkpoint digests to {observed}, but its lock binds "
                f"{digest}. A mutated checkpoint is refused."
            )
    return lock


def read_checkpoint_lock(run_dir: Path, arm: str) -> dict[str, Any]:
    """Read and completely verify one persisted checkpoint lock."""
    if arm not in CHECKPOINT_LOCK_NAME:
        raise T2PersistenceError(f"{arm!r} is not a frozen T2 candidate.")
    path = Path(run_dir) / CHECKPOINT_LOCK_NAME[arm]
    if not path.is_file():
        raise T2PersistenceError(f"No {arm} checkpoint lock at {path}.")
    return validate_checkpoint_lock(
        json.loads(path.read_text()), arm, run_dir=Path(run_dir)
    )


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


def canonical_execution_device() -> Any:
    """The one host-supported device. No flag, no override, no fallback.

    `cuda:0` when CUDA is available, `cpu` otherwise. This is selected exactly
    once per canonical attempt and everything scientific then actually runs on
    it -- parameters, inputs, masks, targets, carried states, internal-dev
    scoring and the reloaded-checkpoint threshold pass.
    """
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def runtime_provenance(device: Any | None = None) -> dict[str, Any]:
    """Interpreter, dependency and device facts, without importing Torch twice.

    `device_type` describes where the science will actually run, not merely
    what the host could offer. Reporting `cuda` because CUDA happens to be
    installed, while every tensor stays on the CPU, would be false provenance
    on any CUDA-capable host; `require_execution_device_agreement` refuses that
    by comparing this record against the model's real parameter device.
    """
    import torch

    environment = dependency_environment()
    selected = (
        torch.device(device) if device is not None else (canonical_execution_device())
    )
    device_type = selected.type
    device_index = selected.index
    device_name = None
    if device_type == "cuda":  # pragma: no cover - no CUDA in this runtime
        device_name = torch.cuda.get_device_name(device_index or 0)
    return {
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "package_count": int(environment["installed_package_count"]),
        "dependency_digest": str(environment["installed_packages_sha256"]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "declared_execution_device": str(selected),
        "device_type": device_type,
        "device_index": device_index,
        "device_name": device_name,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_override_permitted": False,
        "silent_cpu_fallback_permitted": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }


def model_parameter_device(model: Any) -> str:
    """Where the model's parameters actually live. One device or STOP."""
    devices = {str(parameter.device) for parameter in model.parameters()}
    if len(devices) != 1:
        raise T2PersistenceError(
            f"A T2 candidate's parameters span {sorted(devices)}. A split model "
            "is not the specified model; execution STOPS."
        )
    return devices.pop()


def require_execution_device_agreement(
    provenance: dict[str, Any], observed_device: str
) -> dict[str, Any]:
    """The declared device and the observed parameter device must agree.

    This is the check that makes `device_type: cuda` mean something. A record
    claiming CUDA while the parameters sit on the CPU is refused rather than
    promoted, so no artifact can attest to a device the computation never
    touched.
    """
    declared = str(provenance.get("declared_execution_device"))
    observed = str(observed_device)
    if _same_device(declared, observed):
        return {
            "declared_execution_device": declared,
            "model_parameter_device": observed,
            "execution_device_agrees": True,
        }
    raise T2PersistenceError(
        f"The runtime record declares execution on {declared!r}, but the model's "
        f"parameters are on {observed!r}. Provenance may not claim a device the "
        "computation did not run on; execution STOPS and nothing is promoted."
    )


def _same_device(first: str, second: str) -> bool:
    """`cuda:0` and `cuda` name the same device; `cuda:0` and `cpu` do not."""
    import torch

    left = torch.device(first)
    right = torch.device(second)
    if left.type != right.type:
        return False
    return (left.index or 0) == (right.index or 0)


def require_deterministic_execution(device: Any) -> dict[str, Any]:
    """Determinism is required on the selected device, never traded away.

    If the selected device cannot satisfy a required deterministic operation,
    execution STOPS FOR HUMAN REVIEW. It does not quietly fall back to the CPU:
    a silent fallback would produce evidence whose device provenance is a
    guess, and it would change what was computed without saying so.
    """
    import torch

    selected = torch.device(device)
    if not torch.are_deterministic_algorithms_enabled():
        raise T2PersistenceError(
            f"Deterministic algorithms are not enabled for execution on "
            f"{selected}. This is a STOP condition: canonical T2 evidence is "
            "never produced with determinism silently disabled, nothing is "
            "installed and there is no CPU fallback."
        )
    return {
        "execution_device": str(selected),
        "deterministic_algorithms": True,
        "silent_cpu_fallback_performed": False,
    }


def _require_sha256_like_git(value: Any) -> str:
    if not isinstance(value, str) or not _GIT_SHA_PATTERN.match(value):
        raise T2PersistenceError(f"Not a Git commit identity: {value!r}.")
    return value


def require_authorized_git_identity(authorized_git_sha: str) -> dict[str, Any]:
    """Re-read HEAD and prove it is still the commit the human authorized.

    A canonical attempt can run for hours. Preflight proved the checkout was
    clean and at the authorized commit; nothing stopped HEAD moving afterwards,
    and a result written at commit A beside a lock written at commit B would be
    two independently well-formed artifacts describing different code. This is
    called once, immediately before the claim-bearing promotion, and its result
    is what both artifacts then carry.

    A drift STOPS the attempt. The attempt is consumed, the normal additive
    failure receipt is written, and nothing is retried.
    """
    expected = _require_sha256_like_git(authorized_git_sha)
    git = git_provenance(REPOSITORY_ROOT)
    if git["git_dirty"] is not False:
        raise T2PersistenceError(
            "The working tree became dirty during the canonical attempt. "
            "Canonical T2 evidence requires a clean checkout at promotion as "
            "well as at preflight; the attempt is consumed and nothing is "
            "promoted, repaired or retried."
        )
    if git["git_sha"] != expected:
        raise T2PersistenceError(
            f"HEAD moved during the canonical attempt: it is now "
            f"{git['git_sha']}, but the attempt was authorized for {expected}. "
            "A result and a lock written at different commits would describe "
            "different code; the attempt is consumed and nothing is promoted, "
            "repaired or retried."
        )
    return {
        "authorized_git_sha": expected,
        "git_sha": str(git["git_sha"]),
        "git_dirty": False,
        "git_identity_reverified_before_promotion": True,
    }


def build_execution_device_proof(
    per_arm_proof: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """One top-level device proof, derived only after both arms have passed.

    The top-level claim is not a fresh reading: it is the agreement of the two
    arms' own observed parameter devices. If they disagree, or if either
    disagrees with the device it declared, there is no single device the
    canonical comparison ran on and execution STOPS.
    """
    missing = [arm for arm in T2_ARMS if arm not in per_arm_proof]
    if missing:
        raise T2PersistenceError(
            f"A top-level execution-device proof needs both frozen arms; "
            f"missing {missing}."
        )
    declared = {str(per_arm_proof[arm]["declared_execution_device"]) for arm in T2_ARMS}
    observed = {str(per_arm_proof[arm]["model_parameter_device"]) for arm in T2_ARMS}
    for arm in T2_ARMS:
        if per_arm_proof[arm].get("execution_device_agrees") is not True:
            raise T2PersistenceError(
                f"The {arm} execution-device proof does not agree; no top-level "
                "proof can be derived from it."
            )
    first_declared = sorted(declared)[0]
    first_observed = sorted(observed)[0]
    if len(observed) != 1 or not all(
        _same_device(first_observed, other) for other in observed
    ):
        raise T2PersistenceError(
            f"The two T2 arms executed on different devices: {sorted(observed)}. "
            "A mixed-device scientific comparison is not admissible."
        )
    if not all(_same_device(first_declared, other) for other in declared):
        raise T2PersistenceError(
            f"The two T2 arms declared different devices: {sorted(declared)}."
        )
    if not _same_device(first_declared, first_observed):
        raise T2PersistenceError(
            f"The arms declared {first_declared} but executed on {first_observed}."
        )
    return {
        "declared_execution_device": first_declared,
        "model_parameter_device": first_observed,
        "execution_device_agrees": True,
        "derived_from": "both_arm_observed_parameter_devices",
    }


def require_execution_device_cross_binding(
    top_level: dict[str, Any], parts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Every artifact that names a device must name the same one.

    The top-level lock, the top-level result, both arm results and both
    checkpoint locks each carry a device proof. A rewritten top-level proof
    with a repaired self-digest is still refused, because the arm artifacts
    disagree with it.
    """
    declared = str(top_level["declared_execution_device"])
    observed = str(top_level["model_parameter_device"])
    require_execution_device_agreement(top_level, observed)
    for name, proof in parts.items():
        if proof.get("execution_device_agrees") is not True:
            raise T2PersistenceError(f"{name} records a disagreeing device proof.")
        if not _same_device(str(proof["declared_execution_device"]), declared):
            raise T2PersistenceError(
                f"{name} declares {proof['declared_execution_device']!r} but the "
                f"canonical attempt declares {declared!r}."
            )
        if not _same_device(str(proof["model_parameter_device"]), observed):
            raise T2PersistenceError(
                f"{name} executed on {proof['model_parameter_device']!r} but the "
                f"canonical attempt records {observed!r}."
            )
    return {
        "declared_execution_device": declared,
        "model_parameter_device": observed,
        "execution_device_agrees": True,
        "cross_bound_artifacts": sorted(parts),
    }


def require_single_runtime(first: dict[str, Any], second: dict[str, Any]) -> None:
    """Both arms must share one device and runtime. Otherwise STOP."""
    for field_ in (
        "declared_execution_device",
        "device_type",
        "device_index",
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
    "authorized_git_sha",
    "git_sha",
    "git_dirty",
    "execution_device_proof",
    "per_arm_execution_device_proof",
    "interpreter",
    "package_count",
    "dependency_digest",
    "t2_protocol_sha256",
    "t2_execution_spec_sha256",
    "split_sha256",
    "internal_split_sha256",
    "train_timeline_identity",
    "target_authority_identity",
    "checkpoint_sha256",
    "checkpoint_lock_sha256",
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
    """Construct the one canonical T2 training lock. The only identity assembly.

    Two identities are supplied rather than re-observed here.

    **Git.** `require_authorized_git_identity` has already re-read HEAD and
    proved it is still the commit the human authorized at preflight. Re-reading
    it independently a third time would let the lock attest to a commit the
    result never named, so the proven pair travels in through `provenance`.

    **Device.** The runtime record is the one both arms actually executed on,
    complete with the observed `model_parameter_device`. A fresh
    `runtime_provenance()` here would carry the declared device and no proof
    that anything ran there, which is exactly the gap this closes.
    """
    authorized = _require_sha256_like_git(provenance["authorized_git_sha"])
    git = dict(provenance["git_identity"])
    environment = dict(provenance["runtime"])
    lock: dict[str, Any] = {
        "lock_class": LOCK_CLASS,
        "attempt_id": str(attempt_id),
        "experiment_identity": T2_EXPERIMENT_IDENTITY,
        "authorized_git_sha": authorized,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        **environment,
        "execution_device_proof": dict(provenance["execution_device_proof"]),
        "per_arm_execution_device_proof": dict(
            provenance["per_arm_execution_device_proof"]
        ),
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": T2_EXECUTION_SPEC_SHA256,
        "split_sha256": T2_SPLIT_SHA256,
        "internal_split_sha256": T2_INTERNAL_SPLIT_SHA256,
        "train_timeline_identity": provenance["train_timeline_identity"],
        "target_authority_identity": provenance["target_authority_identity"],
        "fit_subjects": provenance["fit_subjects"],
        "internal_dev_subjects": provenance["internal_dev_subjects"],
        "positive_class_weight": provenance["positive_class_weight"],
        "fit_positive_count": provenance["fit_positive_count"],
        "fit_negative_count": provenance["fit_negative_count"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_lock_sha256": provenance["checkpoint_lock_sha256"],
        "checkpoint_lock_self_sha256": provenance["checkpoint_lock_self_sha256"],
        "internal_dev_thresholds": provenance["internal_dev_thresholds"],
        # Derived from the record itself, after COMPLETION has been observed,
        # so the persisted sequence is what actually happened rather than what
        # the caller believed had happened when it assembled its provenance.
        "runtime_enforcement_stages": list(observed_runtime_stages(runtime)),
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
    # The commit the lock was written at must be the commit the human
    # authorized. Two independently well-formed artifacts at different commits
    # describe different code.
    _require_sha256_like_git(lock["authorized_git_sha"])
    if lock["git_sha"] != lock["authorized_git_sha"]:
        raise T2PersistenceError(
            f"The canonical T2 lock was written at {lock['git_sha']} but the "
            f"attempt was authorized for {lock['authorized_git_sha']}."
        )
    require_execution_device_cross_binding(
        dict(lock["execution_device_proof"]),
        {
            f"{arm} arm device proof": dict(proof)
            for arm, proof in dict(lock["per_arm_execution_device_proof"]).items()
        },
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
    checkpoints = dict(lock["checkpoint_sha256"])
    checkpoint_locks = dict(lock["checkpoint_lock_sha256"])
    if sorted(checkpoints) != sorted(T2_ARMS) or sorted(checkpoint_locks) != sorted(
        T2_ARMS
    ):
        raise T2PersistenceError(
            "A canonical T2 lock binds one checkpoint and one checkpoint lock "
            f"per frozen arm; got {sorted(checkpoints)} / "
            f"{sorted(checkpoint_locks)}."
        )
    for arm, digest in checkpoints.items():
        _require_sha256(f"checkpoint_sha256[{arm}]", digest)
        if run_dir is not None:
            path = Path(run_dir) / CHECKPOINT_NAME[arm]
            if not path.is_file() or sha256_file(path) != digest:
                raise T2PersistenceError(
                    f"The {arm} checkpoint does not match its lock digest."
                )
    # The checkpoint LOCK is a claim-bearing component in its own right: its
    # file bytes, its self-digest and the checkpoint it names are all bound
    # here, so no caller has to remember a separate validator.
    for arm, digest in checkpoint_locks.items():
        _require_sha256(f"checkpoint_lock_sha256[{arm}]", digest)
        if run_dir is None:
            continue
        path = Path(run_dir) / CHECKPOINT_LOCK_NAME[arm]
        if not path.is_file() or sha256_file(path) != digest:
            raise T2PersistenceError(
                f"The {arm} checkpoint lock file does not match the digest the "
                "experiment lock binds."
            )
        checkpoint_lock = read_checkpoint_lock(Path(run_dir), arm)
        if checkpoint_lock["checkpoint_sha256"] != checkpoints[arm]:
            raise T2PersistenceError(
                f"The {arm} checkpoint lock names checkpoint "
                f"{checkpoint_lock['checkpoint_sha256']}, but the experiment "
                f"lock binds {checkpoints[arm]}."
            )
        expected_self = dict(lock["checkpoint_lock_self_sha256"]).get(arm)
        if checkpoint_lock["checkpoint_lock_sha256"] != expected_self:
            raise T2PersistenceError(
                f"The {arm} checkpoint lock self-digest is "
                f"{checkpoint_lock['checkpoint_lock_sha256']}, not the bound "
                f"{expected_self}."
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
            "authorized_git_sha",
            "git_sha",
            "execution_device_proof",
            "component_sha256",
            "checkpoint_sha256",
            "checkpoint_lock_sha256",
            "internal_dev_thresholds",
            "target_authority_identity",
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
    _require_sha256_like_git(result["authorized_git_sha"])
    if result["git_sha"] != result["authorized_git_sha"]:
        raise T2PersistenceError(
            f"The canonical T2 result was written at {result['git_sha']} but the "
            f"attempt was authorized for {result['authorized_git_sha']}."
        )
    if result["execution_device_proof"].get("execution_device_agrees") is not True:
        raise T2PersistenceError(
            "A canonical T2 result must carry a passing execution-device proof."
        )
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
    # `validate_t2_run_lock` with a run_dir re-reads and completely verifies
    # both checkpoints AND both checkpoint locks, so a caller of the canonical
    # validator never has to remember a second one.
    lock = validate_t2_run_lock(json.loads(lock_path.read_text()), run_dir=run_dir)
    if dict(result["checkpoint_sha256"]) != dict(lock["checkpoint_sha256"]):
        raise T2PersistenceError(
            "The result and the lock disagree about the retained checkpoints."
        )
    if dict(result["checkpoint_lock_sha256"]) != dict(lock["checkpoint_lock_sha256"]):
        raise T2PersistenceError(
            "The result and the lock disagree about the checkpoint lock files."
        )
    checkpoint_locks = {arm: read_checkpoint_lock(run_dir, arm) for arm in T2_ARMS}
    for arm, checkpoint_lock in checkpoint_locks.items():
        if checkpoint_lock["attempt_id"] != str(attempt_id):
            raise T2PersistenceError(
                f"The {arm} checkpoint lock was claimed by attempt "
                f"{checkpoint_lock['attempt_id']!r}, not {attempt_id!r}."
            )

    # ONE authorized commit across every artifact that names one.
    authorized = _require_sha256_like_git(result["authorized_git_sha"])
    for label, observed in (
        ("result.git_sha", result["git_sha"]),
        ("lock.authorized_git_sha", lock["authorized_git_sha"]),
        ("lock.git_sha", lock["git_sha"]),
    ):
        if observed != authorized:
            raise T2PersistenceError(
                f"{label} is {observed!r}, but the attempt was authorized for "
                f"{authorized!r}. Every canonical artifact names one commit."
            )

    # ONE execution device across every artifact that names one: the top-level
    # result, the top-level lock, both arm results and both checkpoint locks.
    parts: dict[str, dict[str, Any]] = {}
    for arm in T2_ARMS:
        arm_result = json.loads((run_dir / ARM_RESULT_NAME[arm]).read_text())
        parts[f"{arm} arm result"] = dict(arm_result["execution_device_proof"])
        parts[f"{arm} checkpoint lock"] = dict(
            checkpoint_locks[arm]["execution_device_proof"]
        )
    parts["experiment lock"] = dict(lock["execution_device_proof"])
    device = require_execution_device_cross_binding(
        dict(result["execution_device_proof"]), parts
    )
    return {
        "verification_class": "t2_canonical_attempt_verification",
        "attempt_id": str(attempt_id),
        "result_sha256": sha256_file(result_path),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "component_sha256": dict(result["component_sha256"]),
        "checkpoint_sha256": dict(lock["checkpoint_sha256"]),
        "checkpoint_lock_sha256": dict(lock["checkpoint_lock_sha256"]),
        "checkpoint_locks_verified": True,
        "authorized_git_sha": authorized,
        "git_identity_verified": True,
        "execution_device_proof": device,
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
    # BEFORE anything claim-bearing is written. A drift here consumes the
    # attempt through the caller's failure receipt rather than promoting a
    # result and a lock that name different commits.
    git_identity = require_authorized_git_identity(provenance["authorized_git_sha"])
    provenance = {**provenance, "git_identity": git_identity}
    if result["authorized_git_sha"] != git_identity["authorized_git_sha"]:
        raise T2PersistenceError(
            "The canonical result names a different authorized commit than the "
            "provenance the lock will be built from."
        )
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
    require_runtime_stage_order(runtime, T2_ARMS)
    result_path = claimed.run_dir / RESULT_NAME
    write_json_atomic(result_path, result)
    result_digest = sha256_file(result_path)
    lock = build_t2_run_lock(
        attempt_id=claimed.attempt_id,
        runtime=runtime,
        provenance=provenance,
        started_at=claimed.started_at,
        completed_at=_now(),
        artifact_sha256={
            **claimed.promoted,
            **{
                CHECKPOINT_LOCK_NAME[arm]: digest
                for arm, digest in dict(provenance["checkpoint_lock_sha256"]).items()
            },
            RESULT_NAME: result_digest,
        },
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


# ---------------------------------------------------------------------------
# The one canonical outer-VALIDATION claim
#
# Written now, while the activation state is False, so a future activation
# change set flips a switch rather than inventing persistence, failure
# semantics or a validator once the TRAIN numbers are known.
# ---------------------------------------------------------------------------

OUTER_STATUS_NAME: Final = "T2_OUTER_VALIDATION_STATUS.json"
OUTER_RESULT_NAME: Final = "T2_OUTER_VALIDATION_RESULT.json"
OUTER_LOCK_NAME: Final = "T2_OUTER_VALIDATION_EXPERIMENT_LOCK.json"
OUTER_FAILURE_RECEIPT_NAME: Final = "T2_OUTER_VALIDATION_FAILURE_RECEIPT.json"
OUTER_EVIDENCE_DIRNAME: Final = "row_evidence"

OUTER_RESULT_CLASS: Final = "t2_v1_outer_validation_result"
OUTER_LOCK_CLASS: Final = "t2_v1_canonical_outer_validation_run_lock"

STAGE_OUTER_START: Final = "t2_outer_validation_start"
STAGE_OUTER_CLAIM: Final = "t2_outer_claim_directory"


def stage_pre_checkpoint_load(arm: str) -> str:
    return f"pre_checkpoint_load:{arm}"


def required_outer_runtime_stage_order(arms: Sequence[str]) -> tuple[str, ...]:
    """The frozen outer enforcement choreography, as `detail` labels in order."""
    stages: list[str] = [STAGE_OUTER_START]
    stages.extend(stage_pre_checkpoint_load(arm) for arm in arms)
    stages.append(OUTER_EVIDENCE_DIRNAME)
    stages.append(OUTER_RESULT_NAME)
    return tuple(stages)


def require_outer_runtime_stage_order(
    runtime: RuntimeIntegrityRecord, arms: Sequence[str]
) -> tuple[str, ...]:
    """Prove the frozen outer stages were visited, in the frozen order."""
    required = required_outer_runtime_stage_order(arms)
    observed = observed_runtime_stages(runtime)
    filtered = tuple(label for label in observed if label in set(required))
    if filtered != required:
        raise T2PersistenceError(
            f"The outer runtime enforcement choreography was {list(filtered)}, "
            f"not the frozen {list(required)}."
        )
    return required


def require_unclaimed_outer_attempt(
    run_root: Path = T2_RUN_ROOT, attempt_id: str = T2_OUTER_VALIDATION_ATTEMPT_ID
) -> dict[str, Any]:
    """Prove the one outer attempt is unconsumed. The directory IS the claim."""
    return require_unclaimed_t2_attempt(run_root, attempt_id)


def claim_t2_outer_directory(
    run_root: Path = T2_RUN_ROOT,
    attempt_id: str = T2_OUTER_VALIDATION_ATTEMPT_ID,
    *,
    runtime: RuntimeIntegrityRecord,
) -> T2RunDirectory:
    """Atomically claim the one canonical outer attempt.

    A sibling of `t2-v1-training` under the same absolute run root. No
    timestamp, no UUID, no random suffix, no automatic `recovery1`, no retry
    name: once this directory exists the outer attempt is consumed.
    """
    require_frozen_runtime_record(runtime)
    require_runtime_identity(
        EnforcementPoint.PRE_PROMOTION, record=runtime, detail=STAGE_OUTER_CLAIM
    )
    run_dir = t2_run_directory(run_root, attempt_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise T2PersistenceError(
            f"Canonical T2 outer attempt {attempt_id} is already claimed at "
            f"{run_dir}. There is exactly one outer VALIDATION attempt; "
            "automatic rerun, retry, selective arm rerun and fresh-seed restart "
            "are prohibited and require documented human review."
        ) from error
    claimed = T2RunDirectory(
        run_dir=run_dir, attempt_id=str(attempt_id), started_at=_now()
    )
    write_json_atomic(
        run_dir / OUTER_STATUS_NAME,
        {
            "attempt_id": claimed.attempt_id,
            "experiment_identity": T2_EXPERIMENT_IDENTITY,
            "status": STATUS_STARTED,
            "claim_bearing_result_promoted": False,
            "started_at": claimed.started_at,
            "updated_at": claimed.started_at,
            "validation_accessed": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        },
    )
    return claimed


REQUIRED_OUTER_RESULT_FIELDS: Final = (
    "artifact_class",
    "attempt_id",
    "git_sha",
    "git_dirty",
    "t2_protocol_sha256",
    "t2_execution_spec_sha256",
    "training_attempt_id",
    "training_result_sha256",
    "training_experiment_lock_sha256",
    "checkpoint_sha256",
    "checkpoint_lock_sha256",
    "checkpoint_lock_self_sha256",
    "internal_dev_thresholds",
    "validation_timeline_identity",
    "target_authority_identity",
    "row_accounting",
    "primary_population_identity",
    "challenge_population_identity",
    "unavailable_row_census",
    "per_arm_evidence",
    "row_evidence_store",
    "subject_bootstrap",
    "temporal_descriptors",
    "selection_decision",
    "selected_arm",
    "runtime",
    "latency_used_in_selection",
    "challenge_used_in_selection",
    "automatic_retry_performed",
    "validation_accessed",
    "test_accessed",
    "sealed_test_state",
)


def validate_t2_outer_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    """A complete outer result in its own right, before anything is hashed."""
    if result.get("artifact_class") != OUTER_RESULT_CLASS:
        raise T2PersistenceError(
            f"Unknown T2 outer result class {result.get('artifact_class')!r}."
        )
    missing = [name for name in REQUIRED_OUTER_RESULT_FIELDS if name not in result]
    if missing:
        raise T2PersistenceError(f"The T2 outer result is missing {missing}.")
    for arm in T2_ARMS:
        if arm not in result["per_arm_evidence"]:
            raise T2PersistenceError(f"No outer evidence for {arm}.")
        for block in ("checkpoint_sha256", "checkpoint_lock_sha256"):
            if arm not in result[block]:
                raise T2PersistenceError(
                    f"The outer result does not bind {block}[{arm}]."
                )
    if result["selected_arm"] not in T2_ARMS:
        raise T2PersistenceError(
            f"The outer result selects {result['selected_arm']!r}, which is not a "
            f"frozen candidate."
        )
    if result.get("git_dirty") is not False:
        raise T2PersistenceError(
            "Canonical outer evidence requires a clean Git checkout."
        )
    for flag in (
        "latency_used_in_selection",
        "challenge_used_in_selection",
        "automatic_retry_performed",
        "test_accessed",
    ):
        if result.get(flag) is not False:
            raise T2PersistenceError(
                f"A canonical outer result must record {flag}=false."
            )
    if result.get("validation_accessed") is not True:
        raise T2PersistenceError(
            "An outer-VALIDATION result that did not access VALIDATION is not an "
            "outer-VALIDATION result."
        )
    if result.get("sealed_test_state") != "unopened":
        raise T2PersistenceError("The B4 sealed test must remain unopened.")
    for field_, expected in (
        ("t2_protocol_sha256", T2_PROTOCOL_SHA256),
        ("t2_execution_spec_sha256", T2_EXECUTION_SPEC_SHA256),
    ):
        _require_sha256(field_, result[field_])
        if result[field_] != expected:
            raise T2PersistenceError(
                f"{field_} is {result[field_]!r}, expected the frozen {expected!r}."
            )
    return result


def build_t2_outer_lock(
    *,
    attempt_id: str,
    runtime: RuntimeIntegrityRecord,
    provenance: dict[str, Any],
    started_at: str,
    completed_at: str,
    artifact_sha256: dict[str, str],
) -> dict[str, Any]:
    """Construct the one canonical outer lock. The only identity assembly."""
    git = git_provenance(REPOSITORY_ROOT)
    lock: dict[str, Any] = {
        "lock_class": OUTER_LOCK_CLASS,
        "attempt_id": str(attempt_id),
        "experiment_identity": T2_EXPERIMENT_IDENTITY,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "authorized_git_sha": provenance["authorized_git_sha"],
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": T2_EXECUTION_SPEC_SHA256,
        "training_attempt_id": provenance["training_attempt_id"],
        "training_result_sha256": provenance["training_result_sha256"],
        "training_experiment_lock_sha256": (
            provenance["training_experiment_lock_sha256"]
        ),
        "checkpoint_sha256": dict(provenance["checkpoint_sha256"]),
        "checkpoint_lock_sha256": dict(provenance["checkpoint_lock_sha256"]),
        "checkpoint_lock_self_sha256": dict(provenance["checkpoint_lock_self_sha256"]),
        "internal_dev_thresholds": provenance["internal_dev_thresholds"],
        "validation_timeline_identity": provenance["validation_timeline_identity"],
        "target_authority_identity": provenance["target_authority_identity"],
        "row_accounting": provenance["row_accounting"],
        "row_evidence_store_sha256": provenance["row_evidence_store_sha256"],
        "selected_arm": provenance["selected_arm"],
        "selection_decision": provenance["selection_decision"],
        "runtime": provenance["runtime"],
        "runtime_identity_checks": runtime.as_dict(),
        "runtime_enforcement_stages": list(observed_runtime_stages(runtime)),
        "runtime_dependency_digest_start": runtime.digest_at(EnforcementPoint.START),
        "runtime_dependency_digest_end": runtime.digest_at(EnforcementPoint.COMPLETION),
        "latency_used_in_selection": False,
        "challenge_used_in_selection": False,
        "automatic_retry_performed": False,
        "repeat_attempt_permitted": False,
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "started_at": started_at,
        "completed_at": completed_at,
        "artifact_sha256": dict(artifact_sha256),
    }
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    return lock


def validate_t2_outer_lock(
    lock: dict[str, Any], *, run_dir: Path | None = None
) -> dict[str, Any]:
    """Validate the ACTUAL values of an outer lock, not merely its keys."""
    recorded = lock.get("experiment_lock_sha256")
    body = {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise T2PersistenceError(
            "The T2 outer experiment lock failed digest validation."
        )
    if lock.get("lock_class") != OUTER_LOCK_CLASS:
        raise T2PersistenceError(
            f"Unknown outer lock class {lock.get('lock_class')!r}."
        )
    if lock["git_dirty"] is not False:
        raise T2PersistenceError(
            "Canonical outer evidence requires a clean Git checkout."
        )
    if not _GIT_SHA_PATTERN.match(str(lock["authorized_git_sha"])):
        raise T2PersistenceError(
            f"authorized_git_sha is malformed: {lock['authorized_git_sha']!r}."
        )
    if lock["git_sha"] != lock["authorized_git_sha"]:
        raise T2PersistenceError(
            f"The outer attempt ran at {lock['git_sha']} but was authorized for "
            f"{lock['authorized_git_sha']}."
        )
    for field_, expected in (
        ("t2_protocol_sha256", T2_PROTOCOL_SHA256),
        ("t2_execution_spec_sha256", T2_EXECUTION_SPEC_SHA256),
    ):
        if lock[field_] != expected:
            raise T2PersistenceError(
                f"{field_} is {lock[field_]!r}, expected the frozen {expected!r}."
            )
    for flag in (
        "latency_used_in_selection",
        "challenge_used_in_selection",
        "automatic_retry_performed",
        "repeat_attempt_permitted",
        "test_accessed",
    ):
        if lock[flag] is not False:
            raise T2PersistenceError(
                f"A canonical outer lock must record {flag}=false."
            )
    if lock["sealed_test_state"] != "unopened":
        raise T2PersistenceError("The B4 sealed test must remain unopened.")
    if lock["selected_arm"] not in T2_ARMS:
        raise T2PersistenceError(
            f"The outer lock selects {lock['selected_arm']!r}, not a frozen arm."
        )
    for label in ("start", "end"):
        digest = lock[f"runtime_dependency_digest_{label}"]
        _require_sha256(f"runtime_dependency_digest_{label}", digest)
        if digest != FROZEN_DEPENDENCY_DIGEST:
            raise T2PersistenceError(
                f"runtime_dependency_digest_{label} is not the frozen identity."
            )
    if lock["runtime_identity_checks"].get("all_observations_matched") is not True:
        raise T2PersistenceError(
            "Canonical outer evidence requires every runtime observation to match."
        )
    require_execution_device_agreement(
        lock["runtime"], lock["runtime"].get("model_parameter_device", "")
    )
    artifacts = dict(lock["artifact_sha256"])
    if not artifacts:
        raise T2PersistenceError("A canonical outer lock binds no artifact hash.")
    for name, digest in artifacts.items():
        _require_sha256(f"artifact_sha256[{name}]", digest)
        if run_dir is not None:
            path = Path(run_dir) / name
            if not path.is_file() or sha256_file(path) != digest:
                raise T2PersistenceError(
                    f"Outer artifact {name} does not match its lock digest."
                )
    return lock


def finalize_and_promote_t2_outer_result(
    claimed: T2RunDirectory,
    *,
    result: dict[str, Any],
    provenance: dict[str, Any],
    runtime: RuntimeIntegrityRecord,
) -> dict[str, Any]:
    """Validate, gate, promote and lock the one canonical outer result."""
    require_frozen_runtime_record(runtime)
    validate_t2_outer_result_payload(result)
    completion = observe_runtime_identity(
        EnforcementPoint.COMPLETION,
        expected_digest=runtime.expected_digest,
        detail=OUTER_RESULT_NAME,
    )
    runtime.record(completion)
    if not completion.matches:
        raise RuntimeIntegrityError(
            "Runtime identity differed at COMPLETION. Canonical outer promotion "
            "is INVALIDATED: the result was NOT promoted, the environment was "
            "NOT repaired and already-promoted evidence is retained as "
            "non-claim-bearing forensic material."
        )
    require_outer_runtime_stage_order(runtime, T2_ARMS)
    result_path = claimed.run_dir / OUTER_RESULT_NAME
    write_json_atomic(result_path, result)
    lock = build_t2_outer_lock(
        attempt_id=claimed.attempt_id,
        runtime=runtime,
        provenance=provenance,
        started_at=claimed.started_at,
        completed_at=_now(),
        artifact_sha256={
            **claimed.promoted,
            OUTER_RESULT_NAME: sha256_file(result_path),
        },
    )
    validate_t2_outer_lock(lock)
    write_json_atomic(claimed.run_dir / OUTER_LOCK_NAME, lock)
    validate_t2_outer_lock(lock, run_dir=claimed.run_dir)
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
        "selected_arm": lock["selected_arm"],
        "human_review_required": True,
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "runtime_identity_checks": runtime.as_dict(),
    }
    write_json_atomic(claimed.run_dir / OUTER_STATUS_NAME, status)
    return {"result": result, "lock": lock, "status": status}


def record_t2_outer_attempt_failure(
    run_root: Path,
    claimed: T2RunDirectory,
    *,
    exception: BaseException,
    stage: str,
    arm: str | None,
    exposure: dict[str, Any],
    runtime: RuntimeIntegrityRecord | None = None,
) -> dict[str, Any]:
    """One additive forensic receipt for a post-claim outer failure. No retry."""
    git = git_provenance(REPOSITORY_ROOT)
    # Start from what the attempt recorded as promoted -- which already
    # includes the row-evidence manifest once it is written -- and then re-read
    # the filesystem for anything claim-bearing that landed after that mapping
    # was populated. Scanning only the result and the lock would silently omit
    # row evidence that was promoted before the failure, and the receipt is the
    # only record a consumed attempt leaves.
    promoted: dict[str, str] = {}
    for name, digest in dict(claimed.promoted).items():
        path = claimed.run_dir / name
        promoted[name] = sha256_file(path) if path.is_file() else str(digest)
    from cardiosentinel.neural.t2_outer_evidence import T2_OUTER_STORE_MANIFEST_NAME

    for name in (
        OUTER_RESULT_NAME,
        OUTER_LOCK_NAME,
        f"{OUTER_EVIDENCE_DIRNAME}/{T2_OUTER_STORE_MANIFEST_NAME}",
    ):
        path = claimed.run_dir / name
        if path.is_file():
            promoted[name] = sha256_file(path)
    receipt = {
        "receipt_class": "t2_outer_validation_failure_receipt",
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
        "promotion_state_source": "claim_record_and_filesystem",
        "promoted_artifacts": promoted,
        "row_evidence_manifest_sha256": promoted.get(
            f"{OUTER_EVIDENCE_DIRNAME}/{T2_OUTER_STORE_MANIFEST_NAME}"
        ),
        "runtime_identity_checks": runtime.as_dict() if runtime is not None else None,
        "automatic_retry_performed": False,
        "repeat_attempt_permitted": False,
        "selective_arm_rerun_permitted": False,
        "alternate_attempt_name_permitted": False,
        "attempt_consumed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "human_review_required": True,
        "recorded_at": _now(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    directory = t2_review_directory(run_root, claimed.attempt_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / OUTER_FAILURE_RECEIPT_NAME, receipt)
    write_json_atomic(
        claimed.run_dir / OUTER_STATUS_NAME,
        {
            "attempt_id": claimed.attempt_id,
            "status": STATUS_FAILED,
            "claim_bearing_result_promoted": (
                claimed.run_dir / OUTER_RESULT_NAME
            ).is_file(),
            "promotion_state_source": "filesystem",
            "canonical": False,
            "failed_stage": str(stage),
            "arm": arm,
            "failure_receipt_sha256": receipt["receipt_sha256"],
            "started_at": claimed.started_at,
            "updated_at": _now(),
            "human_review_required": True,
            "repeat_attempt_permitted": False,
            "automatic_retry_performed": False,
            "selective_arm_rerun_permitted": False,
            "validation_accessed": bool(exposure.get("validation_accessed")),
            "test_accessed": False,
            "sealed_test_state": "unopened",
        },
    )
    return receipt


def validate_canonical_t2_outer_validation_attempt(
    run_root: Path = T2_RUN_ROOT,
    attempt_id: str = T2_OUTER_VALIDATION_ATTEMPT_ID,
) -> dict[str, Any]:
    """The one complete canonical verification of a promoted outer attempt.

    Verifies actual bytes, not merely JSON values: the outer result, the outer
    lock, the per-row evidence manifest and every per-row array, the referenced
    TRAIN attempt in full, both checkpoints and both checkpoint locks. No caller
    has to remember a second validator.
    """
    from cardiosentinel.neural.t2_outer_evidence import (
        T2_OUTER_STORE_MANIFEST_NAME,
        validate_t2_outer_evidence_store,
    )

    run_dir = t2_run_directory(run_root, attempt_id)
    result_path = run_dir / OUTER_RESULT_NAME
    lock_path = run_dir / OUTER_LOCK_NAME
    for path in (result_path, lock_path):
        if not path.is_file():
            raise T2PersistenceError(f"No canonical outer artifact at {path}.")
    result = json.loads(result_path.read_text())
    validate_t2_outer_result_payload(result)
    lock = validate_t2_outer_lock(json.loads(lock_path.read_text()), run_dir=run_dir)

    # The referenced TRAIN attempt is re-verified in full, including both
    # checkpoints and both checkpoint locks: an outer result bound to a mutated
    # training attempt describes a model that no longer exists.
    training = validate_canonical_t2_attempt(run_root, result["training_attempt_id"])
    if training["result_sha256"] != result["training_result_sha256"]:
        raise T2PersistenceError(
            "The outer result binds a TRAIN result digest that no longer matches."
        )
    if training["experiment_lock_sha256"] != result["training_experiment_lock_sha256"]:
        raise T2PersistenceError(
            "The outer result binds a TRAIN lock digest that no longer matches."
        )
    for block in ("checkpoint_sha256", "checkpoint_lock_sha256"):
        if dict(result[block]) != dict(training[block]):
            raise T2PersistenceError(
                f"The outer result and the TRAIN attempt disagree about {block}."
            )
        if dict(lock[block]) != dict(result[block]):
            raise T2PersistenceError(
                f"The outer result and the outer lock disagree about {block}."
            )

    evidence_root = run_dir / OUTER_EVIDENCE_DIRNAME
    manifest_path = evidence_root / T2_OUTER_STORE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise T2PersistenceError(f"No per-row outer evidence store at {manifest_path}.")
    manifest = json.loads(manifest_path.read_text())
    try:
        validate_t2_outer_evidence_store(manifest, root=evidence_root)
    except Exception as error:
        # One validator, one exception class for its callers. The store's own
        # error text is preserved so the cause is not lost.
        raise T2PersistenceError(
            f"The per-row outer evidence store failed verification: {error}"
        ) from error
    if manifest["content_sha256"] != result["row_evidence_store"]["content_sha256"]:
        raise T2PersistenceError(
            "The outer result binds a row-evidence manifest digest that no longer "
            "matches the persisted store."
        )
    if manifest["content_sha256"] != lock["row_evidence_store_sha256"]:
        raise T2PersistenceError(
            "The outer lock binds a row-evidence manifest digest that no longer "
            "matches the persisted store."
        )
    return {
        "verification_class": "t2_canonical_outer_validation_verification",
        "attempt_id": str(attempt_id),
        "result_sha256": sha256_file(result_path),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "row_evidence_store_sha256": manifest["content_sha256"],
        "training_attempt_verification": training,
        "checkpoint_sha256": dict(result["checkpoint_sha256"]),
        "checkpoint_lock_sha256": dict(result["checkpoint_lock_sha256"]),
        "selected_arm": result["selected_arm"],
        "row_accounting": dict(result["row_accounting"]),
        "verified": True,
    }
