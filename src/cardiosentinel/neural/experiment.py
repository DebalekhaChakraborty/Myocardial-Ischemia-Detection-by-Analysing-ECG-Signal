"""Canonical B4 train/validation experiment runner with no sealed-test route.

This module drives the single prospective B4 experiment frozen by
`docs/B4_PROTOCOL_V1.md`. It owns provenance, crash-safety, and lock creation
only; every scientific decision remains in `cardiosentinel.neural.training`.

There is exactly one canonical B4 run. This module never enumerates, loads, or
hashes a sealed-test row, and exposes no argument that could select one.
"""

from __future__ import annotations

import os
import shutil
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
from cardiosentinel.baseline.metrics import binary_metrics, subject_macro_metrics
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.neural.determinism import initialize_determinism
from cardiosentinel.neural.integrity import (
    DEVELOPMENT_PARTITIONS,
    canonical_sha256,
    validate_development_feature_integrity,
    validate_development_source_integrity,
)
from cardiosentinel.neural.metadata import B4MetadataIndex
from cardiosentinel.neural.model import (
    B4CompactCNN,
    fp32_parameter_payload_bytes,
    local_receptive_field_samples,
    trainable_parameter_count,
)
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    B4_SPLIT_SHA256,
    DATASET,
    DATASET_VERSION,
    EXPECTED_COUNTS,
    FEATURE_CORPUS_SHA256,
    FP32_PARAMETER_BYTES,
    LOCAL_RECEPTIVE_FIELD_SAMPLES,
    REPOSITORY_ROOT,
    SAMPLING_FREQUENCY_HZ,
    SEED,
    TEMPORAL_LENGTHS,
    TRAINABLE_PARAMETER_COUNT,
    WINDOW_SAMPLES,
    validate_frozen_protocol,
)
from cardiosentinel.neural.provenance import runtime_environment
from cardiosentinel.neural.training import (
    BATCH_SIZE,
    EARLY_STOPPING_DELTA,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    CompletedEpoch,
    build_training_loader,
    build_validation_loader,
    run_frozen_training,
    validation_f1_threshold,
    validation_scores,
)
from cardiosentinel.neural.waveform_cache import (
    TRAINING_SELECTION_SHA256,
    B4CachedWaveformDataset,
    build_development_indexes,
    validate_waveform_cache,
)

EXPERIMENT_ID = "B4_raw_compact_cnn_v1"
RUN_COLLECTION = "phase3b2-b4-v1"
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "cardiosentinel-runs" / RUN_COLLECTION
DEFAULT_COMMAND = "cardiosentinel b4 run-train-validation"
PREFLIGHT_COMMAND = "cardiosentinel b4 run-preflight"
PROGRAM_IDENTITY = "python -m cardiosentinel"

RUN_STATUS_NAME = "RUN_STATUS.json"
RUN_MANIFEST_NAME = "RUN_MANIFEST.json"
EPOCH_HISTORY_NAME = "EPOCH_HISTORY.json"
VALIDATION_METRICS_NAME = "VALIDATION_METRICS.json"
VALIDATION_THRESHOLD_NAME = "VALIDATION_THRESHOLD.json"
EXPERIMENT_LOCK_NAME = "EXPERIMENT_LOCK.json"
SELECTED_MODEL_NAME = "model_selected.pt"
TRAINING_CHECKPOINT_NAME = "training_checkpoint.pt"
VALIDATION_PREDICTIONS_NAME = "validation_predictions.npz"

# Any of these proves a canonical run already began; none may be overwritten.
PRIOR_RUN_ARTIFACTS = (
    RUN_STATUS_NAME,
    RUN_MANIFEST_NAME,
    EPOCH_HISTORY_NAME,
    EXPERIMENT_LOCK_NAME,
    SELECTED_MODEL_NAME,
    TRAINING_CHECKPOINT_NAME,
)

STATUS_RUNNING = "RUNNING"
STATUS_COMPLETE = "COMPLETE"
STATUS_FAILED = "FAILED_OR_INTERRUPTED"

LOCK_STATUS = "locked_for_one_shot_test"
THRESHOLD_RULE = (
    "maximum validation F1 over exact observed validation scores; "
    "the highest threshold wins an exact tie"
)
CHECKPOINT_RULE = (
    "maximum full primary validation AUPRC; the earliest epoch wins an exact tie"
)
REQUIRED_FREE_BYTES = 2 * 1024**3


def input_contract() -> dict[str, Any]:
    """Return the frozen predictive input contract recorded in every artifact."""
    return {
        "channels": 1,
        "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "samples": WINDOW_SAMPLES,
        "physical_unit": "mV",
        "dtype": "float32",
        "processing_profile": "raw",
        "batch_shape": f"[B, 1, {WINDOW_SAMPLES}]",
        "handcrafted_features_used": False,
    }


def _require_consistent_frozen_constants() -> None:
    """Check frozen model arithmetic without constructing any PyTorch module."""
    if FP32_PARAMETER_BYTES != TRAINABLE_PARAMETER_COUNT * 4:
        raise ValueError("Frozen B4 parameter payload contradicts its parameter count.")
    if local_receptive_field_samples() != LOCAL_RECEPTIVE_FIELD_SAMPLES:
        raise ValueError("B4 receptive field differs from the frozen protocol.")
    if TEMPORAL_LENGTHS != (2500, 1250, 625, 313, 157, 79):
        raise ValueError("B4 temporal lengths differ from the frozen protocol.")


def frozen_model_identity() -> dict[str, Any]:
    """Report the expected model identity from committed protocol constants.

    Scientific preflight uses this so it never instantiates a PyTorch module
    before frozen seeds and deterministic settings are established. Actual
    implementation drift is caught by `model_identity` on the canonical model.
    """
    _require_consistent_frozen_constants()
    return {
        "identity_source": "frozen_protocol_constants",
        "verified_against_constructed_model": False,
        "architecture": "B4CompactCNN",
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "fp32_parameter_payload_bytes": FP32_PARAMETER_BYTES,
        "local_receptive_field_samples": LOCAL_RECEPTIVE_FIELD_SAMPLES,
        "temporal_lengths": list(TEMPORAL_LENGTHS),
        "output": "single_raw_logit",
    }


def model_identity(model: torch.nn.Module) -> dict[str, Any]:
    """Describe the constructed model and fail if it drifts from the protocol."""
    _require_consistent_frozen_constants()
    parameters = trainable_parameter_count(model)
    payload_bytes = fp32_parameter_payload_bytes(model)
    if parameters != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("B4 model parameter count differs from the frozen protocol.")
    if payload_bytes != FP32_PARAMETER_BYTES:
        raise ValueError("B4 model payload differs from the frozen protocol.")
    return {
        "identity_source": "constructed_model",
        "verified_against_constructed_model": True,
        "architecture": "B4CompactCNN",
        "trainable_parameter_count": parameters,
        "fp32_parameter_payload_bytes": payload_bytes,
        "local_receptive_field_samples": local_receptive_field_samples(),
        "temporal_lengths": list(TEMPORAL_LENGTHS),
        "output": "single_raw_logit",
    }


def training_configuration() -> dict[str, Any]:
    """Return the frozen training configuration; this module never varies it."""
    return {
        "seed": SEED,
        "loss": "BCEWithLogitsLoss(reduction=mean)",
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "batch_size": BATCH_SIZE,
        "drop_last": False,
        "max_epochs": MAX_EPOCHS,
        "scheduler": None,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_delta": EARLY_STOPPING_DELTA,
        "checkpoint_criterion": CHECKPOINT_RULE,
        "mixed_precision": False,
        "class_weighting": None,
        "augmentation": None,
        "hyperparameter_search": None,
        "restart_selection": None,
    }


@dataclass(frozen=True, slots=True)
class B4ExecutionRequest:
    """Structured, resolved invocation arguments bound into run provenance.

    Scientific provenance reads these typed fields; the rendered shell string is
    a human convenience and is never the authoritative record.
    """

    command: str
    source: Path
    feature_root: Path
    cache_root: Path
    run_root: Path
    requested_device: str | None = None
    workers: int = 0
    require_clean: bool = True
    save_validation_predictions: bool | None = None

    def payload(self, resolved_device: str) -> dict[str, Any]:
        """Render every resolved argument actually used by this invocation."""
        paths = {
            name: str(Path(getattr(self, name)).expanduser().resolve())
            for name in ("source", "feature_root", "cache_root", "run_root")
        }
        rendered = [PROGRAM_IDENTITY, *self.command.split()[1:]]
        for name in ("source", "feature_root", "cache_root", "run_root"):
            rendered += [f"--{name.replace('_', '-')}", paths[name]]
        if self.requested_device is not None:
            rendered += ["--device", self.requested_device]
        rendered += ["--workers", str(self.workers)]
        if self.save_validation_predictions is False:
            rendered.append("--no-validation-predictions")
        return {
            "experiment_id": EXPERIMENT_ID,
            "program": PROGRAM_IDENTITY,
            "command": self.command,
            **paths,
            "requested_device": self.requested_device,
            "resolved_device": resolved_device,
            "workers": self.workers,
            "require_clean": self.require_clean,
            "save_validation_predictions": self.save_validation_predictions,
            "shell_command": " ".join(rendered),
        }


@dataclass(frozen=True, slots=True)
class PreparedB4Experiment:
    """Validated objects plus the JSON-serializable scientific preflight report."""

    run_dir: Path
    indexes: dict[str, B4MetadataIndex]
    cache: Any
    device: str
    report: dict[str, Any]


def _index_summary(index: B4MetadataIndex) -> dict[str, Any]:
    return {
        "partition": index.partition,
        "positive": index.positive_count,
        "negative": index.negative_count,
        "total": index.total_count,
        "subjects": index.subject_count,
        "selection_sha256": index.selection_sha256,
    }


def resolve_run_dir(run_root: Path) -> Path:
    """Resolve the single canonical experiment directory outside Git tracking."""
    root = require_nonversioned_path(run_root, "B4 experiment run root")
    return root / EXPERIMENT_ID


def _require_no_prior_experiment(run_dir: Path) -> None:
    existing = sorted(
        name for name in PRIOR_RUN_ARTIFACTS if (run_dir / name).exists()
    )
    if not existing:
        return
    status = None
    status_path = run_dir / RUN_STATUS_NAME
    if status_path.is_file():
        status = read_json(status_path).get("status")
    raise ValueError(
        f"Canonical B4 experiment {EXPERIMENT_ID} already has evidence in "
        f"{run_dir} (status={status}, artifacts={existing}). There is exactly one "
        "canonical B4 run: a rerun, restart, or fresh-seed retry is not automatic "
        "and requires documented human review."
    )


def _require_output_disk(run_root: Path) -> dict[str, int]:
    probe = run_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < REQUIRED_FREE_BYTES:
        raise ValueError(
            f"B4 run requires {REQUIRED_FREE_BYTES} free output bytes, but only "
            f"{free} are available."
        )
    return {"available_bytes": free, "required_bytes": REQUIRED_FREE_BYTES}


def _require_frozen_counts(indexes: dict[str, B4MetadataIndex]) -> None:
    if set(indexes) != DEVELOPMENT_PARTITIONS:
        raise ValueError("B4 run requires train and validation indexes only.")
    for partition in sorted(DEVELOPMENT_PARTITIONS):
        expected = EXPECTED_COUNTS[partition]
        index = indexes[partition]
        if index.total_count != expected["total"]:
            raise ValueError(f"B4 {partition} row count differs from frozen counts.")
        if index.positive_count != expected["positive"]:
            raise ValueError(f"B4 {partition} positives differ from frozen counts.")
        if index.negative_count != expected["negative"]:
            raise ValueError(f"B4 {partition} negatives differ from frozen counts.")
    if indexes["train"].selection_sha256 != TRAINING_SELECTION_SHA256:
        raise ValueError("Frozen B4 training selection SHA-256 differs.")


def _require_cache_identity(
    manifest: dict[str, Any],
    feature_receipt: dict[str, Any],
    source_receipt: dict[str, Any],
) -> None:
    """Bind the cache to freshly recomputed development-integrity receipts."""
    expected = {
        "protocol_sha256": B4_PROTOCOL_SHA256,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "training_selection_sha256": TRAINING_SELECTION_SHA256,
        "development_feature_integrity_sha256": feature_receipt[
            "development_feature_integrity_sha256"
        ],
        "development_source_integrity_sha256": source_receipt[
            "development_source_integrity_sha256"
        ],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"B4 waveform cache identity differs: {key}.")
    audit = manifest.get("equivalence_audit", {})
    if audit.get("exact_mismatches") != 0 or audit.get("result") != "passed":
        raise ValueError("B4 waveform cache lacks a zero-mismatch equivalence audit.")
    if manifest.get("cache_complete") is not True:
        raise ValueError("B4 waveform cache is not complete.")


def prepare_b4_experiment(
    execution: B4ExecutionRequest,
) -> PreparedB4Experiment:
    """Validate every frozen identity and abort before any model is initialized.

    This function never instantiates `B4CompactCNN`. It reports the expected
    model identity from committed protocol constants so that no PyTorch module
    exists before frozen seeds and deterministic settings are established.
    """
    workers = execution.workers
    if workers < 0:
        raise ValueError("B4 worker count cannot be negative.")
    protocol_sha256 = validate_frozen_protocol()
    provenance = git_provenance(REPOSITORY_ROOT)
    if execution.require_clean and provenance["git_dirty"]:
        raise ValueError("The canonical B4 run requires a clean Git checkout.")

    run_dir = resolve_run_dir(execution.run_root)
    _require_no_prior_experiment(run_dir)
    resources = _require_output_disk(run_dir.parent)

    feature_receipt = validate_development_feature_integrity(execution.feature_root)
    source_receipt = validate_development_source_integrity(
        execution.source, feature_receipt
    )
    indexes = build_development_indexes(execution.feature_root)
    _require_frozen_counts(indexes)
    cache = validate_waveform_cache(execution.cache_root, indexes)
    _require_cache_identity(cache.manifest, feature_receipt, source_receipt)

    # No model is constructed here. Determinism is established first, and the
    # canonical model is built only inside run_b4_train_validation.
    determinism = initialize_determinism(
        requested_device=execution.requested_device
    )
    model_config = frozen_model_identity()
    environment = runtime_environment(determinism.device, workers)
    if environment.get("amp_enabled") is not False:
        raise ValueError("B4 forbids automatic mixed precision.")
    if not determinism.deterministic_algorithms:
        raise ValueError("B4 requires deterministic PyTorch algorithms.")

    report = {
        "status": "ready_for_canonical_development_run",
        "experiment_id": EXPERIMENT_ID,
        "run_dir": str(run_dir),
        "protocol_sha256": protocol_sha256,
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
        "model": model_config,
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
        "execution": execution.payload(determinism.device),
        "git": provenance,
        "partitions": {
            partition: _index_summary(indexes[partition])
            for partition in sorted(DEVELOPMENT_PARTITIONS)
        },
        "test_partition_access": None,
        "resources": resources,
    }
    return PreparedB4Experiment(
        run_dir=run_dir,
        indexes=indexes,
        cache=cache,
        device=determinism.device,
        report=report,
    )


def b4_scientific_preflight(
    source: Path,
    feature_root: Path,
    cache_root: Path,
    run_root: Path = DEFAULT_RUN_ROOT,
    *,
    requested_device: str | None = None,
    require_clean: bool = True,
    workers: int = 0,
) -> dict[str, Any]:
    """Report canonical-run readiness without initializing or training a model."""
    prepared = prepare_b4_experiment(
        B4ExecutionRequest(
            command=PREFLIGHT_COMMAND,
            source=source,
            feature_root=feature_root,
            cache_root=cache_root,
            run_root=run_root,
            requested_device=requested_device,
            workers=workers,
            require_clean=require_clean,
            save_validation_predictions=None,
        )
    )
    return prepared.report


def _write_status(run_dir: Path, status: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    write_json_atomic(run_dir / RUN_STATUS_NAME, payload)
    return payload


def _save_state_dict_atomic(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _epoch_payload(epoch: CompletedEpoch) -> dict[str, Any]:
    return {
        "epoch": epoch.epoch,
        "mean_training_loss": epoch.mean_training_loss,
        "validation_auprc": epoch.validation_auprc,
        "checkpoint_saved": epoch.checkpoint_saved,
        "early_stopping_patience": epoch.early_stopping_patience,
    }


def _validation_subjects(index: B4MetadataIndex, count: int) -> np.ndarray:
    if index.total_count != count:
        raise ValueError("B4 validation scores are not aligned with metadata rows.")
    return np.asarray(
        [reference.subject_id for reference in index.references], dtype=np.str_
    )


def _validation_evidence(
    index: B4MetadataIndex,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    integer_labels = np.asarray(labels, dtype=np.int64)
    float_scores = np.asarray(scores, dtype=np.float64)
    subjects = _validation_subjects(index, int(integer_labels.size))
    pooled = binary_metrics(integer_labels, float_scores, threshold)
    macro = subject_macro_metrics(integer_labels, float_scores, subjects, threshold)
    return {
        "partition": "validation",
        "evidence_class": "development_validation_result",
        "sampled": False,
        "row_count": int(integer_labels.size),
        "positive_count": int(pooled["positive_count"]),
        "negative_count": int(pooled["negative_count"]),
        "positive_prevalence": pooled["positive_prevalence"],
        "subject_count": int(np.unique(subjects).size),
        "threshold": threshold,
        "threshold_rule": THRESHOLD_RULE,
        "pooled": pooled,
        "subject_macro": macro,
        "score_semantics": (
            "uncalibrated sigmoid model score; not calibrated probability"
        ),
    }


def _write_validation_predictions(
    path: Path,
    index: B4MetadataIndex,
    labels: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Persist non-predictive identity plus label and score; never a waveform."""
    references = index.references
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            stable_id=np.asarray(
                [item.stable_id for item in references], dtype=np.str_
            ),
            subject_id=np.asarray(
                [item.subject_id for item in references], dtype=np.str_
            ),
            record_id=np.asarray(
                [item.record_id for item in references], dtype=np.str_
            ),
            channel_index=np.asarray(
                [item.channel_index for item in references], dtype=np.int64
            ),
            target_family=np.asarray(
                [item.target_family for item in references], dtype=np.str_
            ),
            context_flags=np.asarray(
                ["|".join(item.context_flags) for item in references], dtype=np.str_
            ),
            label=np.asarray(labels, dtype=np.int64),
            score=np.asarray(scores, dtype=np.float64),
        )
    os.replace(temporary, path)


def build_experiment_lock(
    prepared: PreparedB4Experiment,
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
    """Assemble the immutable lock and bind its own canonical SHA-256."""
    report = prepared.report
    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": LOCK_STATUS,
        "git_sha": report["git"]["git_sha"],
        "git_dirty": report["git"]["git_dirty"],
        "protocol_sha256": report["protocol_sha256"],
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
        "training_configuration": report["training_configuration"],
        "seed": SEED,
        "environment": report["environment"],
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
    if payload["git_dirty"]:
        raise ValueError("The B4 experiment lock requires a clean Git checkout.")
    if payload["execution"]["require_clean"] is not True:
        raise ValueError("The canonical B4 run must require a clean checkout.")
    if payload["execution"]["command"] != payload["command"]:
        raise ValueError("B4 execution provenance disagrees with the run command.")
    if not payload["environment"]["dependencies"]["installed_packages"]:
        raise ValueError("The B4 experiment lock requires a dependency snapshot.")
    if not payload["model"]["verified_against_constructed_model"]:
        raise ValueError("The B4 lock requires a verified constructed-model identity.")
    if payload["trainable_parameter_count"] != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("The B4 experiment lock has the wrong parameter count.")
    if payload["test"] is not None:
        raise ValueError("The B4 experiment lock must record test as null.")
    payload["experiment_lock_sha256"] = canonical_sha256(payload)
    return payload


def validate_experiment_lock(run_dir: Path) -> dict[str, Any]:
    """Re-derive the canonical lock digest and confirm the bound checkpoint."""
    lock_path = run_dir / EXPERIMENT_LOCK_NAME
    if not lock_path.is_file():
        raise ValueError("The canonical B4 run has no EXPERIMENT_LOCK.json.")
    lock = read_json(lock_path)
    recorded = lock.pop("experiment_lock_sha256", None)
    if recorded is None or recorded != canonical_sha256(lock):
        raise ValueError("B4 experiment lock hash validation failed.")
    lock["experiment_lock_sha256"] = recorded
    # A subscript, never a call argument, so the module keeps a zero-exception
    # firewall against passing a test partition name into any function.
    if "test" not in lock or lock["test"] is not None:
        raise ValueError("B4 experiment lock must record test as null.")
    model_path = run_dir / str(lock["locked_inference_model"])
    if not model_path.is_file() or sha256_file(model_path) != lock["checkpoint_sha256"]:
        raise ValueError("B4 locked inference model failed hash validation.")
    return lock


def run_b4_train_validation(
    source: Path,
    feature_root: Path,
    cache_root: Path,
    run_root: Path = DEFAULT_RUN_ROOT,
    *,
    command: str = DEFAULT_COMMAND,
    requested_device: str | None = None,
    workers: int = 0,
    save_validation_predictions: bool = True,
) -> dict[str, Any]:
    """Run the one canonical B4 train/validation experiment and lock its result.

    This command has no test-partition route: it builds development indexes
    only, reads the validated train/validation waveform cache, and writes a lock
    whose `test` field is always null. A clean Git checkout is structurally
    mandatory: there is deliberately no parameter that can relax it.
    """
    started = time.monotonic()
    execution = B4ExecutionRequest(
        command=command,
        source=source,
        feature_root=feature_root,
        cache_root=cache_root,
        run_root=run_root,
        requested_device=requested_device,
        workers=workers,
        require_clean=True,
        save_validation_predictions=save_validation_predictions,
    )
    prepared = prepare_b4_experiment(execution)
    run_dir = prepared.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_status(run_dir, STATUS_RUNNING, command=command)
    write_json_atomic(
        run_dir / RUN_MANIFEST_NAME,
        {**prepared.report, "command": command, "status": STATUS_RUNNING},
    )

    try:
        train_dataset = B4CachedWaveformDataset(
            prepared.cache, prepared.indexes["train"]
        )
        validation_dataset = B4CachedWaveformDataset(
            prepared.cache, prepared.indexes["validation"]
        )
        training_loader = build_training_loader(train_dataset, workers=workers)
        validation_loader = build_validation_loader(validation_dataset, workers=workers)

        # Re-establish every frozen seed immediately before construction so the
        # canonical model is initialized exactly once from the protocol seed,
        # unaffected by any earlier RNG use. Nothing may consume RNG between
        # this call and the constructor on the next line.
        initialize_determinism(requested_device=prepared.device)
        model = B4CompactCNN()
        # Validate the real constructed model so implementation drift fails
        # before a single training batch is processed.
        prepared.report["model"] = model_identity(model)
        device = torch.device(prepared.device)
        model = model.to(device)

        history: list[dict[str, Any]] = []

        def persist_epoch(epoch: CompletedEpoch) -> None:
            history.append(_epoch_payload(epoch))
            write_json_atomic(
                run_dir / EPOCH_HISTORY_NAME,
                {"experiment_id": EXPERIMENT_ID, "epochs": history},
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
        threshold = validation_f1_threshold(labels, scores)
        if threshold != result.validation_threshold:
            raise ValueError(
                "B4 threshold re-derivation disagrees with the frozen training run."
            )
        evidence = _validation_evidence(
            prepared.indexes["validation"], labels, scores, threshold
        )
        evidence_digest = canonical_sha256(evidence)
        write_json_atomic(run_dir / VALIDATION_METRICS_NAME, evidence)
        write_json_atomic(
            run_dir / VALIDATION_THRESHOLD_NAME,
            {
                "experiment_id": EXPERIMENT_ID,
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
        lock = build_experiment_lock(
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

        # The experiment is locked: refuse further gradient or weight mutation.
        model.eval()
        model.requires_grad_(False)

        _write_status(
            run_dir,
            STATUS_COMPLETE,
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
            "experiment_id": EXPERIMENT_ID,
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
            command=command,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(limit=20),
            human_review_required=True,
            automatic_restart_performed=False,
        )
        raise
