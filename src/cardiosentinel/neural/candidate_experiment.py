"""Canonical B4-B and B4-C train/validation runners with no sealed-test route.

Each candidate is an independently canonical one-shot experiment frozen by
`docs/B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.md`. A completed B4-B run does not
block the one canonical B4-C run, and vice versa.

The historical B4-A runner in `experiment.py` is deliberately not made generic.
This module reuses its reviewed helpers unchanged so B4-A scientific behaviour
cannot shift, and adds only what the candidates require: exact-environment
governance and numerical-integrity abort handling.

Nothing here can select a threshold, search a hyperparameter, or reach the test
partition.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cardiosentinel.baseline.cache import (
    read_json,
    require_nonversioned_path,
    write_json_atomic,
)
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.neural.candidates import (
    B4B_EXPERIMENT_ID,
    B4B_FP32_PARAMETER_BYTES,
    B4B_TRAINABLE_PARAMETERS,
    B4C_EXPERIMENT_ID,
    B4C_FP32_PARAMETER_BYTES,
    B4C_TRAINABLE_PARAMETERS,
    B4CSSMCNN,
    MODEL_DIM,
    SSM_STATE_DIM,
    TOKENS,
    B4BTransformerCNN,
    DiagonalGatedSSMBlock,
    b4b_model_identity,
    b4c_model_identity,
)
from cardiosentinel.neural.determinism import initialize_determinism
from cardiosentinel.neural.experiment import (
    CHECKPOINT_RULE,
    EPOCH_HISTORY_NAME,
    EXPERIMENT_LOCK_NAME,
    LOCK_STATUS,
    PRIOR_RUN_ARTIFACTS,
    PROGRAM_IDENTITY,
    RUN_MANIFEST_NAME,
    RUN_STATUS_NAME,
    SELECTED_MODEL_NAME,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_RUNNING,
    THRESHOLD_RULE,
    TRAINING_CHECKPOINT_NAME,
    VALIDATION_METRICS_NAME,
    VALIDATION_PREDICTIONS_NAME,
    VALIDATION_THRESHOLD_NAME,
    _epoch_payload,
    _index_summary,
    _require_cache_identity,
    _require_frozen_counts,
    _require_output_disk,
    _save_state_dict_atomic,
    _validation_evidence,
    _write_status,
    _write_validation_predictions,
    input_contract,
    training_configuration,
)
from cardiosentinel.neural.integrity import (
    canonical_sha256,
    validate_development_feature_integrity,
    validate_development_source_integrity,
)
from cardiosentinel.neural.metadata import B4MetadataIndex
from cardiosentinel.neural.protocol import (
    B4_SPLIT_SHA256,
    DATASET,
    DATASET_VERSION,
    FEATURE_CORPUS_SHA256,
    REPOSITORY_ROOT,
    SEED,
    validate_frozen_protocol,
)
from cardiosentinel.neural.provenance import runtime_environment
from cardiosentinel.neural.training import (
    CompletedEpoch,
    build_training_loader,
    build_validation_loader,
    run_frozen_training,
    validation_f1_threshold,
    validation_scores,
)
from cardiosentinel.neural.waveform_cache import (
    B4CachedWaveformDataset,
    build_development_indexes,
    validate_waveform_cache,
)

ARCHITECTURE_PROTOCOL_PATH = (
    REPOSITORY_ROOT / "docs" / "B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.md"
)
ARCHITECTURE_PROTOCOL_SHA256 = (
    "986bc166f7f4a787423e1ac33cad65342ae7a700f85bfd8bb9d0291f64d2a0dc"
)

RUN_COLLECTION = "phase3b2-architecture-v1"
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "cardiosentinel-runs" / RUN_COLLECTION
DEFAULT_COMMAND = "cardiosentinel b4 candidate run-train-validation"
PREFLIGHT_COMMAND = "cardiosentinel b4 candidate run-preflight"

# The canonical candidate runs must execute in the exact B4-A scientific
# software environment. This is execution governance, not an architecture rule.
B4A_DEPENDENCY_DIGEST = (
    "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
)
REQUIRED_ENVIRONMENT = {
    "python_version": "3.12.6",
    "torch_version": "2.13.0+cpu",
    "numpy_version": "2.3.2",
}
REQUIRED_KEY_DEPENDENCIES = {
    "numpy": "2.3.2",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "torch": "2.13.0+cpu",
    "wfdb": "4.3.1",
}

CANDIDATE_SELECTORS = {"b4b": B4B_EXPERIMENT_ID, "b4c": B4C_EXPERIMENT_ID}
CANDIDATE_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    B4B_EXPERIMENT_ID: {
        "architecture": "B4BTransformerCNN",
        "factory": B4BTransformerCNN,
        "identity": b4b_model_identity,
        "trainable_parameter_count": B4B_TRAINABLE_PARAMETERS,
        "fp32_parameter_payload_bytes": B4B_FP32_PARAMETER_BYTES,
    },
    B4C_EXPERIMENT_ID: {
        "architecture": "B4CSSMCNN",
        "factory": B4CSSMCNN,
        "identity": b4c_model_identity,
        "trainable_parameter_count": B4C_TRAINABLE_PARAMETERS,
        "fp32_parameter_payload_bytes": B4C_FP32_PARAMETER_BYTES,
    },
}


class CandidateNumericalIntegrityError(RuntimeError):
    """Raised when an impossible numerical state consumes a canonical attempt."""


def resolve_candidate_selector(selector: str) -> str:
    """Map the only two permitted selectors onto canonical experiment IDs."""
    if selector not in CANDIDATE_SELECTORS:
        raise ValueError(
            f"Unknown B4 candidate selector {selector!r}; permitted values are "
            f"{sorted(CANDIDATE_SELECTORS)}."
        )
    return CANDIDATE_SELECTORS[selector]


def resolve_candidate_run_dir(run_root: Path, experiment_id: str) -> Path:
    """Resolve one candidate's independent canonical directory outside Git."""
    if experiment_id not in CANDIDATE_SPECIFICATIONS:
        raise ValueError(f"Unknown B4 candidate experiment {experiment_id!r}.")
    root = require_nonversioned_path(run_root, "B4 candidate run root")
    return root / experiment_id


def validate_architecture_protocol(
    path: Path = ARCHITECTURE_PROTOCOL_PATH,
) -> str:
    """Fail if the architecture protocol bytes differ from the frozen digest."""
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ARCHITECTURE_PROTOCOL_SHA256:
        raise ValueError(
            "B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.md differs from its frozen "
            "SHA-256."
        )
    return digest


def require_exact_scientific_environment(environment: dict[str, Any]) -> str:
    """Refuse a canonical run outside the exact B4-A scientific environment."""
    for key, expected in REQUIRED_ENVIRONMENT.items():
        if environment.get(key) != expected:
            raise ValueError(
                f"Candidate runs require {key} == {expected}; observed "
                f"{environment.get(key)!r}. Refusing the scientific run."
            )
    dependencies = environment.get("dependencies", {})
    observed = dependencies.get("key_dependencies", {})
    for package, expected in REQUIRED_KEY_DEPENDENCIES.items():
        if observed.get(package) != expected:
            raise ValueError(
                f"Candidate runs require {package} == {expected}; observed "
                f"{observed.get(package)!r}. Refusing the scientific run."
            )
    digest = dependencies.get("installed_packages_sha256")
    if digest != B4A_DEPENDENCY_DIGEST:
        raise ValueError(
            "Candidate runs require the exact B4-A dependency snapshot "
            f"{B4A_DEPENDENCY_DIGEST}; observed {digest!r}. Refusing the "
            "scientific run. Do not change packages to satisfy this check."
        )
    return digest


def expected_candidate_identity(experiment_id: str) -> dict[str, Any]:
    """Report the expected identity from constants, constructing no model."""
    specification = CANDIDATE_SPECIFICATIONS.get(experiment_id)
    if specification is None:
        raise ValueError(f"Unknown B4 candidate experiment {experiment_id!r}.")
    return {
        "identity_source": "frozen_protocol_constants",
        "verified_against_constructed_model": False,
        "architecture": specification["architecture"],
        "experiment_id": experiment_id,
        "trainable_parameter_count": specification["trainable_parameter_count"],
        "fp32_parameter_payload_bytes": specification[
            "fp32_parameter_payload_bytes"
        ],
        "tokens": TOKENS,
        "model_dim": MODEL_DIM,
        "output": "single_raw_logit",
    }


@dataclass(frozen=True, slots=True)
class CandidateExecutionRequest:
    """Structured, resolved invocation arguments bound into run provenance."""

    experiment_id: str
    command: str
    source: Path
    feature_root: Path
    cache_root: Path
    run_root: Path
    workers: int = 0
    save_validation_predictions: bool | None = None

    def payload(self, resolved_device: str) -> dict[str, Any]:
        paths = {
            name: str(Path(getattr(self, name)).expanduser().resolve())
            for name in ("source", "feature_root", "cache_root", "run_root")
        }
        selector = next(
            key
            for key, value in CANDIDATE_SELECTORS.items()
            if value == self.experiment_id
        )
        rendered = [
            PROGRAM_IDENTITY, *self.command.split()[1:],
            "--candidate", selector,
        ]
        for name in ("source", "feature_root", "cache_root", "run_root"):
            rendered += [f"--{name.replace('_', '-')}", paths[name]]
        rendered += ["--workers", str(self.workers)]
        return {
            "experiment_id": self.experiment_id,
            "candidate": selector,
            "program": PROGRAM_IDENTITY,
            "command": self.command,
            **paths,
            "requested_device": "cpu",
            "resolved_device": resolved_device,
            "workers": self.workers,
            "require_clean": True,
            "save_validation_predictions": self.save_validation_predictions,
            "shell_command": " ".join(rendered),
        }


@dataclass(frozen=True, slots=True)
class PreparedCandidateExperiment:
    run_dir: Path
    experiment_id: str
    indexes: dict[str, B4MetadataIndex]
    cache: Any
    device: str
    report: dict[str, Any]


def _describe_prior_candidate_run(run_dir: Path) -> str:
    """Summarize an existing claim without letting corruption mask the refusal."""
    if not run_dir.is_dir():
        return "path_exists_but_is_not_a_directory"
    status_path = run_dir / RUN_STATUS_NAME
    if status_path.is_file():
        try:
            return f"status={read_json(status_path).get('status')}"
        except (OSError, ValueError):
            return "status=unreadable_or_corrupt"
    try:
        if not any(run_dir.iterdir()):
            return "empty_directory"
    except OSError:
        return "unreadable_directory"
    artifacts = sorted(
        name for name in PRIOR_RUN_ARTIFACTS if (run_dir / name).exists()
    )
    return f"partial_without_status, artifacts={artifacts}"


def _require_no_prior_candidate_run(run_dir: Path, experiment_id: str) -> None:
    """Refuse if the canonical candidate directory exists in any state.

    The directory itself is the claim, so its mere existence means the one
    canonical attempt for this experiment has been consumed. This covers an
    empty, partial, corrupt, RUNNING, FAILED_OR_INTERRUPTED or COMPLETE
    directory alike. Nothing is ever deleted or reset here.

    This is an early, friendly check only. Exclusivity is guaranteed by the
    atomic claim in `claim_candidate_run_directory`, not by this check.
    """
    if not run_dir.exists():
        return
    raise ValueError(
        f"Canonical candidate experiment {experiment_id} has already been claimed "
        f"at {run_dir} ({_describe_prior_candidate_run(run_dir)}). Automatic "
        "rerun, restart or fresh-seed retry is prohibited and requires "
        "documented human review."
    )


def claim_candidate_run_directory(run_dir: Path, experiment_id: str) -> Path:
    """Atomically claim the one canonical attempt for this candidate.

    The candidate run directory *is* the claim. `mkdir(exist_ok=False)` is atomic
    on POSIX, so exactly one process can ever create it. A check-then-create
    sequence could not provide that guarantee: two processes could both observe
    an absent directory and both proceed to train.

    Once created, the attempt is consumed. The directory is never removed on
    failure, so a crash after this point still blocks any automatic rerun.
    """
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ValueError(
            f"Canonical candidate experiment {experiment_id} has already been "
            "claimed. Automatic rerun, restart or fresh-seed retry is prohibited "
            "and requires documented human review."
        ) from error
    return run_dir


def _require_finite_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise CandidateNumericalIntegrityError(
                f"Non-finite value in parameter {name} after an optimizer step."
            )


def _require_finite_state_space(model: torch.nn.Module) -> None:
    """Reject a non-finite derived SSM transition or input gain; never repair."""
    for index, block in enumerate(model.modules()):
        if not isinstance(block, DiagonalGatedSSMBlock):
            continue
        with torch.no_grad():
            transition, input_gain, output = block.discrete_state_space()
        for label, tensor in (
            ("transition", transition),
            ("input_gain", input_gain),
            ("output", output),
        ):
            if not torch.isfinite(tensor).all():
                raise CandidateNumericalIntegrityError(
                    f"Non-finite derived SSM {label} in block {index}."
                )


def require_numerical_integrity(model: torch.nn.Module, epoch: CompletedEpoch) -> None:
    """Abort for human review on an impossible numerical state.

    Nothing is clamped, repaired, restarted, re-typed, or substituted. This is
    integrity failure handling, not a training rule.
    """
    if not np.isfinite(epoch.mean_training_loss):
        raise CandidateNumericalIntegrityError(
            f"Non-finite mean training loss at epoch {epoch.epoch}."
        )
    if not np.isfinite(epoch.validation_auprc):
        raise CandidateNumericalIntegrityError(
            f"Non-finite validation score at epoch {epoch.epoch}."
        )
    _require_finite_parameters(model)
    _require_finite_state_space(model)


def prepare_candidate_experiment(
    execution: CandidateExecutionRequest,
) -> PreparedCandidateExperiment:
    """Validate every frozen identity and abort before any model is initialized.

    No candidate model is constructed here: the expected identity comes from
    committed constants so nothing exists before determinism is established.
    """
    experiment_id = execution.experiment_id
    if experiment_id not in CANDIDATE_SPECIFICATIONS:
        raise ValueError(f"Unknown B4 candidate experiment {experiment_id!r}.")
    if execution.workers < 0:
        raise ValueError("Candidate worker count cannot be negative.")

    protocol_sha256 = validate_frozen_protocol()
    architecture_sha256 = validate_architecture_protocol()
    provenance = git_provenance(REPOSITORY_ROOT)
    if provenance["git_dirty"]:
        raise ValueError("A canonical candidate run requires a clean Git checkout.")

    run_dir = resolve_candidate_run_dir(execution.run_root, experiment_id)
    _require_no_prior_candidate_run(run_dir, experiment_id)
    resources = _require_output_disk(run_dir.parent)

    feature_receipt = validate_development_feature_integrity(execution.feature_root)
    source_receipt = validate_development_source_integrity(
        execution.source, feature_receipt
    )
    indexes = build_development_indexes(execution.feature_root)
    _require_frozen_counts(indexes)
    cache = validate_waveform_cache(execution.cache_root, indexes)
    _require_cache_identity(cache.manifest, feature_receipt, source_receipt)

    determinism = initialize_determinism(requested_device="cpu")
    environment = runtime_environment(determinism.device, execution.workers)
    dependency_digest = require_exact_scientific_environment(environment)
    if environment.get("amp_enabled") is not False:
        raise ValueError("Candidate runs forbid automatic mixed precision.")
    if not determinism.deterministic_algorithms:
        raise ValueError("Candidate runs require deterministic PyTorch algorithms.")

    report = {
        "status": "ready_for_canonical_development_run",
        "experiment_id": experiment_id,
        "candidate_architecture": CANDIDATE_SPECIFICATIONS[experiment_id][
            "architecture"
        ],
        "run_dir": str(run_dir),
        "protocol_sha256": protocol_sha256,
        "architecture_protocol_sha256": architecture_sha256,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "training_selection_sha256": indexes["train"].selection_sha256,
        "development_feature_integrity_sha256": feature_receipt[
            "development_feature_integrity_sha256"
        ],
        "development_source_integrity_sha256": source_receipt[
            "development_source_integrity_sha256"
        ],
        "waveform_cache_sha256": cache.manifest["waveform_cache_sha256"],
        "cache_complete": cache.manifest["cache_complete"],
        "equivalence_audit": cache.manifest["equivalence_audit"],
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "input_contract": input_contract(),
        "model": expected_candidate_identity(experiment_id),
        "training_configuration": training_configuration(),
        "seed": SEED,
        "device": determinism.device,
        "determinism": {
            "deterministic_algorithms": determinism.deterministic_algorithms,
            "cudnn_benchmark": determinism.cudnn_benchmark,
            "cudnn_deterministic": determinism.cudnn_deterministic,
            "cuda_workspace_config": determinism.cuda_workspace_config,
        },
        "environment": environment,
        "environment_dependency_digest": dependency_digest,
        "execution": execution.payload(determinism.device),
        "git": provenance,
        "partitions": {
            partition: _index_summary(indexes[partition])
            for partition in sorted(indexes)
        },
        "test_partition_access": None,
        "resources": resources,
    }
    return PreparedCandidateExperiment(
        run_dir=run_dir,
        experiment_id=experiment_id,
        indexes=indexes,
        cache=cache,
        device=determinism.device,
        report=report,
    )


def candidate_scientific_preflight(
    selector: str,
    source: Path,
    feature_root: Path,
    cache_root: Path,
    run_root: Path = DEFAULT_RUN_ROOT,
    *,
    workers: int = 0,
) -> dict[str, Any]:
    """Report canonical-run readiness without initializing or training a model."""
    experiment_id = resolve_candidate_selector(selector)
    prepared = prepare_candidate_experiment(
        CandidateExecutionRequest(
            experiment_id=experiment_id,
            command=PREFLIGHT_COMMAND,
            source=source,
            feature_root=feature_root,
            cache_root=cache_root,
            run_root=run_root,
            workers=workers,
            save_validation_predictions=None,
        )
    )
    return prepared.report


def build_candidate_lock(
    prepared: PreparedCandidateExperiment,
    *,
    command: str,
    epoch_history: tuple[dict[str, Any], ...],
    selected_epoch: int,
    selected_validation_auprc: float,
    threshold: float,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    training_checkpoint_sha256: str,
    validation_evidence_sha256: str,
    validation_predictions_sha256: str | None,
    duration_seconds: float,
) -> dict[str, Any]:
    """Assemble the immutable candidate lock and bind its own SHA-256."""
    report = prepared.report
    payload: dict[str, Any] = {
        "experiment_id": prepared.experiment_id,
        "candidate_architecture": report["candidate_architecture"],
        "status": LOCK_STATUS,
        "git_sha": report["git"]["git_sha"],
        "git_dirty": report["git"]["git_dirty"],
        "protocol_sha256": report["protocol_sha256"],
        "architecture_protocol_sha256": report["architecture_protocol_sha256"],
        "split_sha256": report["split_sha256"],
        "feature_corpus_sha256": report["feature_corpus_sha256"],
        "training_selection_sha256": report["training_selection_sha256"],
        "development_feature_integrity_sha256": report[
            "development_feature_integrity_sha256"
        ],
        "development_source_integrity_sha256": report[
            "development_source_integrity_sha256"
        ],
        "waveform_cache_sha256": report["waveform_cache_sha256"],
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "input_contract": report["input_contract"],
        "model": report["model"],
        "trainable_parameter_count": report["model"]["trainable_parameter_count"],
        "fp32_parameter_payload_bytes": report["model"][
            "fp32_parameter_payload_bytes"
        ],
        "training_configuration": report["training_configuration"],
        "seed": SEED,
        "environment": report["environment"],
        "environment_dependency_digest": report["environment_dependency_digest"],
        "device": report["device"],
        "determinism": report["determinism"],
        "training_rows": report["partitions"]["train"],
        "validation_rows": report["partitions"]["validation"],
        "epoch_history_digest": canonical_sha256(list(epoch_history)),
        "completed_epochs": len(epoch_history),
        "selected_epoch": selected_epoch,
        "selected_validation_auprc": selected_validation_auprc,
        "validation_threshold": threshold,
        "threshold_selection_rule": THRESHOLD_RULE,
        "checkpoint_selection_rule": CHECKPOINT_RULE,
        "locked_inference_model": SELECTED_MODEL_NAME,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "training_checkpoint": TRAINING_CHECKPOINT_NAME,
        "training_checkpoint_sha256": training_checkpoint_sha256,
        "validation_evidence_sha256": validation_evidence_sha256,
        "validation_predictions_sha256": validation_predictions_sha256,
        "command": command,
        "execution": report["execution"],
        "total_duration_seconds": duration_seconds,
        "test": None,
    }
    specification = CANDIDATE_SPECIFICATIONS[prepared.experiment_id]
    if payload["git_dirty"]:
        raise ValueError("A candidate lock requires a clean Git checkout.")
    if payload["architecture_protocol_sha256"] != ARCHITECTURE_PROTOCOL_SHA256:
        raise ValueError("A candidate lock requires the frozen architecture protocol.")
    if payload["trainable_parameter_count"] != specification[
        "trainable_parameter_count"
    ]:
        raise ValueError("A candidate lock has the wrong parameter count.")
    if not payload["model"]["verified_against_constructed_model"]:
        raise ValueError("A candidate lock requires a verified constructed model.")
    if payload["test"] is not None:
        raise ValueError("A candidate lock must record test as null.")
    payload["experiment_lock_sha256"] = canonical_sha256(payload)
    return payload


def validate_candidate_lock(run_dir: Path) -> dict[str, Any]:
    """Re-derive the canonical lock digest and confirm the bound checkpoint."""
    lock_path = run_dir / EXPERIMENT_LOCK_NAME
    if not lock_path.is_file():
        raise ValueError("The candidate run has no EXPERIMENT_LOCK.json.")
    lock = read_json(lock_path)
    recorded = lock.pop("experiment_lock_sha256", None)
    if recorded is None or recorded != canonical_sha256(lock):
        raise ValueError("Candidate experiment lock hash validation failed.")
    lock["experiment_lock_sha256"] = recorded
    if "test" not in lock or lock["test"] is not None:
        raise ValueError("Candidate experiment lock must record test as null.")
    if lock.get("experiment_id") not in CANDIDATE_SPECIFICATIONS:
        raise ValueError("Candidate experiment lock has an unknown experiment.")
    model_path = run_dir / str(lock["locked_inference_model"])
    if not model_path.is_file() or sha256_file(model_path) != lock["checkpoint_sha256"]:
        raise ValueError("Candidate locked inference model failed hash validation.")
    return lock


def run_candidate_train_validation(
    selector: str,
    source: Path,
    feature_root: Path,
    cache_root: Path,
    run_root: Path = DEFAULT_RUN_ROOT,
    *,
    command: str = DEFAULT_COMMAND,
    workers: int = 0,
    save_validation_predictions: bool = True,
) -> dict[str, Any]:
    """Run one candidate's canonical train/validation experiment and lock it.

    This command has no test-partition route and no configuration override. A
    clean Git checkout and the exact B4-A scientific environment are structurally
    mandatory: there is no parameter that can relax either.
    """
    started = time.monotonic()
    experiment_id = resolve_candidate_selector(selector)
    execution = CandidateExecutionRequest(
        experiment_id=experiment_id,
        command=command,
        source=source,
        feature_root=feature_root,
        cache_root=cache_root,
        run_root=run_root,
        workers=workers,
        save_validation_predictions=save_validation_predictions,
    )
    prepared = prepare_candidate_experiment(execution)
    run_dir = prepared.run_dir
    # Atomically consume the one canonical attempt for this candidate. This is
    # deliberately outside the try below: a refused claim belongs to another
    # process, so this one must never write status into that directory.
    claim_candidate_run_directory(run_dir, prepared.experiment_id)
    _write_status(
        run_dir, STATUS_RUNNING, experiment_id=experiment_id, command=command
    )
    write_json_atomic(
        run_dir / RUN_MANIFEST_NAME,
        {**prepared.report, "command": command, "status": STATUS_RUNNING},
    )

    try:
        specification = CANDIDATE_SPECIFICATIONS[experiment_id]
        train_dataset = B4CachedWaveformDataset(
            prepared.cache, prepared.indexes["train"]
        )
        validation_dataset = B4CachedWaveformDataset(
            prepared.cache, prepared.indexes["validation"]
        )
        training_loader = build_training_loader(train_dataset, workers=workers)
        validation_loader = build_validation_loader(
            validation_dataset, workers=workers
        )

        # Re-establish every frozen seed immediately before construction so the
        # canonical model is initialized exactly once from the protocol seed.
        # Nothing may consume RNG between this call and the constructor.
        initialize_determinism(requested_device=prepared.device)
        model = specification["factory"]()
        prepared.report["model"] = specification["identity"](model)
        device = torch.device(prepared.device)
        model = model.to(device)

        history: list[dict[str, Any]] = []

        def persist_epoch(epoch: CompletedEpoch) -> None:
            require_numerical_integrity(model, epoch)
            history.append(_epoch_payload(epoch))
            write_json_atomic(
                run_dir / EPOCH_HISTORY_NAME,
                {"experiment_id": experiment_id, "epochs": history},
            )

        result = run_frozen_training(
            model,
            training_loader,
            validation_loader,
            device,
            run_dir / TRAINING_CHECKPOINT_NAME,
            epoch_callback=persist_epoch,
        )

        labels, scores = validation_scores(model, validation_loader, device)
        if not np.isfinite(scores).all():
            raise CandidateNumericalIntegrityError(
                "Non-finite validation score from the selected checkpoint."
            )
        threshold = validation_f1_threshold(labels, scores)
        if threshold != result.validation_threshold:
            raise ValueError(
                "Candidate threshold re-derivation disagrees with the frozen run."
            )
        evidence = _validation_evidence(
            prepared.indexes["validation"], labels, scores, threshold
        )
        evidence_digest = canonical_sha256(evidence)
        write_json_atomic(run_dir / VALIDATION_METRICS_NAME, evidence)
        write_json_atomic(
            run_dir / VALIDATION_THRESHOLD_NAME,
            {
                "experiment_id": experiment_id,
                "threshold": threshold,
                "threshold_rule": THRESHOLD_RULE,
                "selected_from": "validation",
                "test_informed": False,
            },
        )

        predictions_sha256 = None
        if save_validation_predictions:
            predictions_path = run_dir / VALIDATION_PREDICTIONS_NAME
            _write_validation_predictions(
                predictions_path, prepared.indexes["validation"], labels, scores
            )
            predictions_sha256 = sha256_file(predictions_path)

        selected_path = run_dir / SELECTED_MODEL_NAME
        _save_state_dict_atomic(selected_path, model.state_dict())
        lock = build_candidate_lock(
            prepared,
            command=command,
            epoch_history=tuple(history),
            selected_epoch=result.selected_checkpoint_epoch,
            selected_validation_auprc=result.selected_validation_auprc,
            threshold=threshold,
            checkpoint_sha256=sha256_file(selected_path),
            checkpoint_bytes=selected_path.stat().st_size,
            training_checkpoint_sha256=sha256_file(
                run_dir / TRAINING_CHECKPOINT_NAME
            ),
            validation_evidence_sha256=evidence_digest,
            validation_predictions_sha256=predictions_sha256,
            duration_seconds=time.monotonic() - started,
        )
        write_json_atomic(run_dir / EXPERIMENT_LOCK_NAME, lock)

        model.eval()
        model.requires_grad_(False)
        _write_status(
            run_dir,
            STATUS_COMPLETE,
            experiment_id=experiment_id,
            command=command,
            selected_epoch=result.selected_checkpoint_epoch,
            experiment_lock_sha256=lock["experiment_lock_sha256"],
            model_locked=True,
        )
        write_json_atomic(
            run_dir / RUN_MANIFEST_NAME,
            {**prepared.report, "command": command, "status": STATUS_COMPLETE},
        )
        return {
            "status": STATUS_COMPLETE,
            "experiment_id": experiment_id,
            "run_dir": str(run_dir),
            "completed_epochs": len(history),
            "selected_epoch": result.selected_checkpoint_epoch,
            "selected_validation_auprc": result.selected_validation_auprc,
            "validation_threshold": threshold,
            "validation_evidence": evidence,
            "experiment_lock_sha256": lock["experiment_lock_sha256"],
            "checkpoint_sha256": lock["checkpoint_sha256"],
            "test": None,
        }
    except BaseException as error:
        _write_status(
            run_dir,
            STATUS_FAILED,
            experiment_id=experiment_id,
            command=command,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(limit=20),
            human_review_required=True,
            automatic_restart_performed=False,
            numerical_integrity_failure=isinstance(
                error, CandidateNumericalIntegrityError
            ),
        )
        raise


__all__ = [
    "ARCHITECTURE_PROTOCOL_SHA256",
    "B4A_DEPENDENCY_DIGEST",
    "CANDIDATE_SELECTORS",
    "CANDIDATE_SPECIFICATIONS",
    "DEFAULT_RUN_ROOT",
    "MODEL_DIM",
    "SSM_STATE_DIM",
    "CandidateExecutionRequest",
    "CandidateNumericalIntegrityError",
    "PreparedCandidateExperiment",
    "build_candidate_lock",
    "candidate_scientific_preflight",
    "claim_candidate_run_directory",
    "expected_candidate_identity",
    "prepare_candidate_experiment",
    "require_exact_scientific_environment",
    "require_numerical_integrity",
    "resolve_candidate_run_dir",
    "resolve_candidate_selector",
    "run_candidate_train_validation",
    "validate_architecture_protocol",
    "validate_candidate_lock",
]
