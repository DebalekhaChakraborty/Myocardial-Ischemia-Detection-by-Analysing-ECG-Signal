"""P1 physiology-fusion development machinery.

Implements the frozen procedure in `docs/P1_PHYSIOLOGY_FUSION_PROTOCOL_V1.md`:
the B4-B frozen-embedding cache contract, the train-only physiology transform,
the matched P1-A / P1-B heads, and the canonical run claim.

B4-B is a fixed representation extractor here. It is never fine-tuned, its
weights are never written, and there is no route to the sealed test: `test` is
refused on every public path.

Nothing in this module computes a scientific result on its own; the canonical
P1 run is a separately authorised step.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor, nn

from cardiosentinel.baseline.cache import require_nonversioned_path, write_json_atomic
from cardiosentinel.features.schema import COMBINED_V1, MORPHOLOGY_V1
from cardiosentinel.neural.candidates import B4BTransformerCNN
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    REPOSITORY_ROOT,
    protocol_sha256,
    require_development_partition,
)

P1_PROTOCOL_NAME: Final = "P1_PHYSIOLOGY_FUSION_PROTOCOL_V1"
P1_PROTOCOL_PATH: Final = (
    REPOSITORY_ROOT / "docs" / "P1_PHYSIOLOGY_FUSION_PROTOCOL_V1.md"
)
P1_PROTOCOL_SHA256: Final = (
    "66e91c6cda73ac66c7dfddb2cf25a601af383ed8a84ba9f24dfed82519d8f256"
)

MORPHOLOGY_SCHEMA_SHA256: Final = (
    "13f60be400b5b957c1eb592bbafd8206d4d2855c1aa657a058671fb8d7cab434"
)
PHYSIOLOGY_FEATURE_NAMES: Final = tuple(MORPHOLOGY_V1.names)
PHYSIOLOGY_DIM: Final = len(PHYSIOLOGY_FEATURE_NAMES)
EMBEDDING_DIM: Final = 128
EMBEDDING_TAP: Final = "B4BTransformerCNN.encode:pooled_post_final_norm"

# Engineering/scientific proxy groupings, NOT validated clinical measurements.
PHYSIOLOGY_FEATURE_GROUPS: Final = {
    "detection_quality": (
        "detected_r_peak_count",
        "usable_beat_count",
        "morphology_valid",
    ),
    "rr_rhythm": (
        "rr_mean_ms",
        "rr_median_ms",
        "rr_std_ms",
        "rr_cv",
        "estimated_hr_bpm",
    ),
    "qrs_amplitude": (
        "pre_r_baseline_median_mv",
        "qrs_proxy_peak_to_peak_mv",
    ),
    "st_t_proxy": (
        "post_r_80ms_delta_mv",
        "post_r_120ms_delta_mv",
        "post_r_160ms_delta_mv",
        "post_r_200ms_delta_mv",
        "post_r_80_160_slope_mv_per_s",
        "post_r_80_200_area_mv_s",
    ),
    "beat_template": (
        "beat_template_correlation_median",
        "beat_template_variability",
    ),
}
VALIDITY_FEATURE: Final = "morphology_valid"

B4B_EXPERIMENT_ID: Final = "B4B_cnn_transformer_v1"
B4B_CHECKPOINT_SHA256: Final = (
    "b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9"
)
B4B_EXPERIMENT_LOCK_SHA256: Final = (
    "58e44a09ce3ebffecfcd49d957acfa368fc03b534fdcd990aedb9b6b0e9bda7b"
)

P1A_EXPERIMENT_ID: Final = "P1A_neural_head_v1"
P1B_EXPERIMENT_ID: Final = "P1B_phys_fusion_v1"
P1_RUN_COLLECTION: Final = "phase4-p1-physiology-v1"

HEAD_HIDDEN_DIM: Final = 64
HEAD_DROPOUT: Final = 0.10
P1_SEED: Final = 2026
P1_BATCH_SIZE: Final = 256
P1_MAX_EPOCHS: Final = 30
P1_LEARNING_RATE: Final = 1e-3
P1_WEIGHT_DECAY: Final = 1e-4
P1_EARLY_STOPPING_PATIENCE: Final = 4
P1_EARLY_STOPPING_DELTA: Final = 1e-6

PERMITTED_PARTITIONS: Final = ("train", "validation")
FORBIDDEN_PARTITIONS: Final = frozenset({"test"})

RUN_STATUS_NAME: Final = "RUN_STATUS.json"
ATTEMPT_STATUS_STARTED: Final = "STARTED"
ATTEMPT_STATUS_COMPLETE: Final = "COMPLETE"
ATTEMPT_STATUS_FAILED: Final = "FAILED_OR_INTERRUPTED"

_MORPHOLOGY_COLUMNS: Final = tuple(
    list(COMBINED_V1.names).index(name) for name in PHYSIOLOGY_FEATURE_NAMES
)


class PhysiologyFusionError(RuntimeError):
    """Raised when a P1 development step cannot proceed with full integrity."""


def validate_p1_protocol(path: Path = P1_PROTOCOL_PATH) -> str:
    """Fail if the frozen P1 procedure bytes have changed."""
    digest = protocol_sha256(path)
    if digest != P1_PROTOCOL_SHA256:
        raise PhysiologyFusionError(
            "P1_PHYSIOLOGY_FUSION_PROTOCOL_V1.md differs from its frozen SHA-256."
        )
    return digest


def require_p1_partition(partition: str) -> str:
    """Accept train/validation only; the sealed test has no P1 route."""
    if partition in FORBIDDEN_PARTITIONS:
        raise PhysiologyFusionError(
            f"P1 development must never access the {partition!r} partition."
        )
    if partition not in PERMITTED_PARTITIONS:
        raise PhysiologyFusionError(f"Unsupported P1 partition {partition!r}.")
    return require_development_partition(partition)


def validate_physiology_schema(names: tuple[str, ...] | list[str]) -> None:
    """Refuse a renamed, reordered or resized physiology vector."""
    if tuple(names) != PHYSIOLOGY_FEATURE_NAMES:
        raise PhysiologyFusionError(
            "Physiology feature names/order do not match the frozen "
            "morphology_v1 schema."
        )
    observed = MORPHOLOGY_V1.as_dict()["feature_schema_sha256"]
    if observed != MORPHOLOGY_SCHEMA_SHA256:
        raise PhysiologyFusionError(
            "morphology_v1 schema SHA-256 differs from the frozen identity."
        )


def physiology_feature_groups() -> dict[str, tuple[str, ...]]:
    """Return the frozen groups after checking they tile the schema exactly."""
    assigned = [name for group in PHYSIOLOGY_FEATURE_GROUPS.values() for name in group]
    if sorted(assigned) != sorted(PHYSIOLOGY_FEATURE_NAMES):
        raise PhysiologyFusionError(
            "Physiology feature groups do not exactly cover the frozen schema."
        )
    if len(assigned) != len(set(assigned)):
        raise PhysiologyFusionError("Physiology feature groups overlap.")
    return dict(PHYSIOLOGY_FEATURE_GROUPS)


def morphology_columns() -> tuple[int, ...]:
    """Column indices of morphology_v1 inside the combined feature matrix."""
    return _MORPHOLOGY_COLUMNS


@dataclass(frozen=True, slots=True)
class PhysiologyTransform:
    """A train-fitted imputation + standardisation transform.

    `fit` is train-only by construction. `transform` may be applied to train,
    validation and challenge rows; it can never be applied to test because
    `require_p1_partition` refuses that partition.
    """

    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    zero_variance_features: tuple[str, ...]
    imputed_counts: dict[str, int]
    fitted_rows: int
    schema_sha256: str
    training_selection_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "transform": "train_median_impute_then_standardize",
            "feature_names": list(self.feature_names),
            "feature_order_is_frozen": True,
            "physiology_dim": len(self.feature_names),
            "medians": list(self.medians),
            "means": list(self.means),
            "scales": list(self.scales),
            "zero_variance_features": list(self.zero_variance_features),
            "zero_variance_policy": "constant_zero_column_after_centering",
            "missingness_policy": (
                "non-finite values replaced by the per-feature TRAIN median; "
                "morphology_valid retained and never imputed; no row dropped"
            ),
            "validity_feature": VALIDITY_FEATURE,
            "imputed_counts": dict(self.imputed_counts),
            "fitted_rows": self.fitted_rows,
            "fitted_on_partition": "train",
            "schema_sha256": self.schema_sha256,
            "training_selection_sha256": self.training_selection_sha256,
        }
        payload["transform_sha256"] = canonical_sha256(payload)
        return payload

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the frozen train-derived transform. Never refits."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise PhysiologyFusionError(
                f"Physiology matrix must be [N, {len(self.feature_names)}]."
            )
        medians = np.asarray(self.medians, dtype=np.float64)
        output = np.where(np.isfinite(matrix), matrix, medians)
        validity = self.feature_names.index(VALIDITY_FEATURE)
        # morphology_valid is a reliability signal, never an imputed quantity.
        output[:, validity] = matrix[:, validity]
        if not np.all(np.isfinite(output[:, validity])):
            raise PhysiologyFusionError(
                f"{VALIDITY_FEATURE} must be finite in every row."
            )
        output = (output - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.all(np.isfinite(output)):
            raise PhysiologyFusionError(
                "Physiology transform produced a non-finite value."
            )
        return output.astype(np.float32)


def fit_physiology_transform(
    values: np.ndarray,
    *,
    partition: str = "train",
    feature_names: tuple[str, ...] | list[str] = PHYSIOLOGY_FEATURE_NAMES,
    training_selection_sha256: str | None = None,
) -> PhysiologyTransform:
    """Fit imputation and standardisation statistics on TRAIN rows only."""
    if require_p1_partition(partition) != "train":
        raise PhysiologyFusionError(
            "Physiology transform statistics may only be fitted on train."
        )
    validate_physiology_schema(feature_names)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != PHYSIOLOGY_DIM:
        raise PhysiologyFusionError(
            f"Train matrix must be [N, {PHYSIOLOGY_DIM}]."
        )
    if matrix.shape[0] == 0:
        raise PhysiologyFusionError(
            "Cannot fit the physiology transform on no rows."
        )

    medians: list[float] = []
    imputed: dict[str, int] = {}
    for index, name in enumerate(PHYSIOLOGY_FEATURE_NAMES):
        column = matrix[:, index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            raise PhysiologyFusionError(
                f"Feature {name!r} has zero finite train support; refusing to "
                "invent a fill value."
            )
        medians.append(float(np.median(finite)))
        imputed[name] = int(np.sum(~np.isfinite(column)))

    filled = np.where(np.isfinite(matrix), matrix, np.asarray(medians))
    means = filled.mean(axis=0)
    stds = filled.std(axis=0)
    zero_variance = tuple(
        name
        for index, name in enumerate(PHYSIOLOGY_FEATURE_NAMES)
        if stds[index] == 0.0
    )
    scales = np.where(stds == 0.0, 1.0, stds)
    return PhysiologyTransform(
        feature_names=tuple(PHYSIOLOGY_FEATURE_NAMES),
        medians=tuple(medians),
        means=tuple(float(v) for v in means),
        scales=tuple(float(v) for v in scales),
        zero_variance_features=zero_variance,
        imputed_counts=imputed,
        fitted_rows=int(matrix.shape[0]),
        schema_sha256=MORPHOLOGY_SCHEMA_SHA256,
        training_selection_sha256=training_selection_sha256,
    )


class P1FusionHead(nn.Module):
    """The matched P1 head. P1-A and P1-B differ only in input width."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise PhysiologyFusionError("P1 head input dimension must be positive.")
        self.input_dim = input_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, HEAD_HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN_DIM, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise PhysiologyFusionError(
                f"P1 head expects [B, {self.input_dim}]."
            )
        return self.head(features).squeeze(-1)


def build_p1_head(experiment_id: str) -> P1FusionHead:
    """Construct the frozen head for one P1 arm."""
    if experiment_id == P1A_EXPERIMENT_ID:
        return P1FusionHead(EMBEDDING_DIM)
    if experiment_id == P1B_EXPERIMENT_ID:
        return P1FusionHead(EMBEDDING_DIM + PHYSIOLOGY_DIM)
    raise PhysiologyFusionError(f"Unknown P1 experiment {experiment_id!r}.")


def p1_training_configuration() -> dict[str, Any]:
    """The single frozen training contract shared by both P1 arms."""
    return {
        "seed": P1_SEED,
        "loss": "BCEWithLogitsLoss(reduction=mean)",
        "optimizer": "AdamW",
        "learning_rate": P1_LEARNING_RATE,
        "weight_decay": P1_WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "batch_size": P1_BATCH_SIZE,
        "drop_last": False,
        "max_epochs": P1_MAX_EPOCHS,
        "scheduler": None,
        "augmentation": None,
        "class_weighting": None,
        "mixed_precision": False,
        "early_stopping_patience": P1_EARLY_STOPPING_PATIENCE,
        "early_stopping_delta": P1_EARLY_STOPPING_DELTA,
        "checkpoint_criterion": (
            "maximum full primary validation AUPRC; the earliest epoch wins an "
            "exact tie"
        ),
        "encoder": "frozen B4-B; not fine-tuned",
    }


def p1_head_identity(head: P1FusionHead) -> dict[str, Any]:
    """Describe a constructed P1 head."""
    parameters = sum(p.numel() for p in head.parameters() if p.requires_grad)
    return {
        "input_dim": head.input_dim,
        "hidden_dim": HEAD_HIDDEN_DIM,
        "activation": "SiLU",
        "dropout": HEAD_DROPOUT,
        "output": "single_raw_logit",
        "trainable_parameter_count": parameters,
        "fp32_parameter_payload_bytes": parameters * 4,
    }


def embedding_cache_contract(
    *,
    partition: str,
    stable_ids: tuple[str, ...] | list[str],
    embeddings: np.ndarray,
    split_sha256: str,
    feature_corpus_sha256: str,
    git_sha: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Build the provenance record binding one frozen B4-B embedding cache.

    The cache is development-only: `require_p1_partition` refuses `test`, so no
    test embedding cache can be described by this contract.
    """
    evaluated = require_p1_partition(partition)
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
        raise PhysiologyFusionError(
            f"Embedding cache must be [N, {EMBEDDING_DIM}]."
        )
    if matrix.shape[0] != len(stable_ids):
        raise PhysiologyFusionError("Embedding rows and stable IDs are misaligned.")
    if len(set(stable_ids)) != len(stable_ids):
        raise PhysiologyFusionError("Embedding cache has duplicate stable IDs.")
    if not np.all(np.isfinite(matrix)):
        raise PhysiologyFusionError("Embedding cache contains a non-finite value.")
    payload = {
        "cache_kind": "b4b_frozen_embedding_cache_v1",
        "partition": evaluated,
        "encoder_experiment_id": B4B_EXPERIMENT_ID,
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_fine_tuned": False,
        "embedding_tap": EMBEDDING_TAP,
        "embedding_dim": EMBEDDING_DIM,
        "dtype": str(matrix.dtype),
        "rows": int(matrix.shape[0]),
        "stable_id_sha256": canonical_sha256(sorted(stable_ids)),
        "split_sha256": split_sha256,
        "feature_corpus_sha256": feature_corpus_sha256,
        "git_sha": git_sha,
        "environment": environment,
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "test_accessed": False,
    }
    payload["cache_sha256"] = canonical_sha256(payload)
    return payload


def _model_state_digest(model: nn.Module) -> str:
    with torch.no_grad():
        items = [
            (name, np.asarray(tensor.detach().cpu().numpy()).tobytes().hex())
            for name, tensor in sorted(model.state_dict().items())
        ]
    return canonical_sha256(items)


def extract_frozen_embeddings(
    encoder: B4BTransformerCNN, waveforms: Tensor
) -> tuple[Tensor, dict[str, Any]]:
    """Extract pooled B4-B embeddings without touching the encoder's weights."""
    encoder.eval()
    encoder.requires_grad_(False)
    before = _model_state_digest(encoder)
    with torch.no_grad():
        embeddings = encoder.encode(waveforms)
    after = _model_state_digest(encoder)
    if before != after:
        raise PhysiologyFusionError(
            "The frozen B4-B encoder state changed during embedding extraction."
        )
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise PhysiologyFusionError("Unexpected B4-B embedding dimension.")
    receipt = {
        "encoder_state_sha256_before": before,
        "encoder_state_sha256_after": after,
        "encoder_state_unchanged": True,
        "encoder_fine_tuned": False,
        "gradients_enabled": False,
        "inference_mode": "torch.no_grad + eval + requires_grad_(False)",
        "embedding_tap": EMBEDDING_TAP,
        "embedding_dim": EMBEDDING_DIM,
    }
    return embeddings, receipt


def resolve_p1_run_dir(run_root: Path, experiment_id: str) -> Path:
    """Resolve one canonical P1 run directory under a non-versioned root."""
    if experiment_id not in (P1A_EXPERIMENT_ID, P1B_EXPERIMENT_ID):
        raise PhysiologyFusionError(f"Unknown P1 experiment {experiment_id!r}.")
    root = require_nonversioned_path(Path(run_root), "P1 development evidence")
    return root / experiment_id


def claim_p1_run_directory(run_dir: Path, experiment_id: str) -> Path:
    """Atomically claim the one canonical attempt for a P1 arm.

    The directory itself is the claim. There is no force, overwrite, retry or
    delete path, and the claim is never released on failure.
    """
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise PhysiologyFusionError(
            f"Canonical P1 experiment {experiment_id} has already been claimed "
            f"at {run_dir}. Automatic rerun, retry and fresh-seed restart are "
            "prohibited and require documented human review."
        ) from error
    descriptor = os.open(run_dir.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return run_dir


def write_p1_status(
    run_dir: Path, status: str, *, experiment_id: str, **fields: Any
) -> dict[str, Any]:
    """Write the P1 heartbeat for the calling experiment.

    `experiment_id` is required rather than defaulted, for the same reason the
    B4 runner now requires it: a default silently mislabels one experiment's
    status with another's identity.
    """
    payload = {
        "experiment_id": experiment_id,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    write_json_atomic(Path(run_dir) / RUN_STATUS_NAME, payload)
    return payload


def record_p1_failure(
    run_dir: Path, experiment_id: str, error: BaseException
) -> dict[str, Any]:
    """Record a post-claim failure. The claim is never released."""
    return write_p1_status(
        run_dir,
        ATTEMPT_STATUS_FAILED,
        experiment_id=experiment_id,
        error_type=type(error).__name__,
        error=str(error),
        traceback=traceback.format_exc(limit=20),
        human_review_required=True,
        repeat_attempt_permitted=False,
        automatic_retry_performed=False,
    )


def p1_run_manifest(
    experiment_id: str,
    *,
    head: P1FusionHead,
    transform: PhysiologyTransform | None,
    git_sha: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the provenance manifest for one P1 arm."""
    if experiment_id not in (P1A_EXPERIMENT_ID, P1B_EXPERIMENT_ID):
        raise PhysiologyFusionError(f"Unknown P1 experiment {experiment_id!r}.")
    uses_physiology = experiment_id == P1B_EXPERIMENT_ID
    if uses_physiology and transform is None:
        raise PhysiologyFusionError("P1-B requires a fitted physiology transform.")
    if not uses_physiology and transform is not None:
        raise PhysiologyFusionError(
            "P1-A is the neural-only control and must not receive physiology."
        )
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "arm": (
            "neural_plus_physiology" if uses_physiology else "neural_only_control"
        ),
        "uses_physiology": uses_physiology,
        "encoder_experiment_id": B4B_EXPERIMENT_ID,
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_fine_tuned": False,
        "embedding_tap": EMBEDDING_TAP,
        "embedding_dim": EMBEDDING_DIM,
        "physiology_dim": PHYSIOLOGY_DIM if uses_physiology else 0,
        "physiology_schema_sha256": (
            MORPHOLOGY_SCHEMA_SHA256 if uses_physiology else None
        ),
        "physiology_transform": transform.as_dict() if transform else None,
        "physiology_feature_groups": {
            k: list(v) for k, v in physiology_feature_groups().items()
        },
        "head": p1_head_identity(head),
        "training_configuration": p1_training_configuration(),
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "git_sha": git_sha,
        "environment": environment,
        "partitions_permitted": list(PERMITTED_PARTITIONS),
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "architecture_selection_performed": False,
    }
    payload["p1_manifest_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_TAP",
    "MORPHOLOGY_SCHEMA_SHA256",
    "P1A_EXPERIMENT_ID",
    "P1B_EXPERIMENT_ID",
    "P1_PROTOCOL_SHA256",
    "PHYSIOLOGY_DIM",
    "PHYSIOLOGY_FEATURE_GROUPS",
    "PHYSIOLOGY_FEATURE_NAMES",
    "P1FusionHead",
    "PhysiologyFusionError",
    "PhysiologyTransform",
    "build_p1_head",
    "claim_p1_run_directory",
    "embedding_cache_contract",
    "extract_frozen_embeddings",
    "fit_physiology_transform",
    "morphology_columns",
    "p1_head_identity",
    "p1_run_manifest",
    "p1_training_configuration",
    "physiology_feature_groups",
    "record_p1_failure",
    "require_p1_partition",
    "resolve_p1_run_dir",
    "validate_p1_protocol",
    "validate_physiology_schema",
    "write_p1_status",
]
