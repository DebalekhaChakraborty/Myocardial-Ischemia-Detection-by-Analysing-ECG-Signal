"""Canonical P1 Stage-1 execution path: matched P1-A vs P1-B ablation.

This is the official scientific route promised by
`docs/P1_PHYSIOLOGY_FUSION_PROTOCOL_V1.md`. It owns embedding-cache
materialization and validation, deterministic head training over precomputed
frozen embeddings, validation evidence, development challenge evidence, the
immutable run lock, and the Stage-1 suite that requires *both* arms.

The B4-B encoder is frozen throughout: it is loaded from `model_selected.pt`
only, never fine-tuned, never placed in an optimizer, and its state digest is
checked before and after every extraction.

Every official entry point takes an explicit partition and calls
`require_p1_partition`, so `test` cannot enter any scientific route. Low-level
numeric helpers operate on unlabelled arrays and deliberately do not claim to
provide that firewall themselves.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from cardiosentinel.baseline.cache import read_json, write_json_atomic
from cardiosentinel.baseline.metrics import (
    binary_metrics,
    challenge_metrics,
    subject_macro_metrics,
)
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.evaluation.metrics import select_validation_f1_threshold
from cardiosentinel.evaluation.protocol import challenge_evidence_policy
from cardiosentinel.neural.candidate_experiment import (
    require_exact_scientific_environment,
)
from cardiosentinel.neural.data import B4WaveformDataset
from cardiosentinel.neural.determinism import initialize_determinism
from cardiosentinel.neural.integrity import (
    canonical_sha256,
    validate_development_feature_integrity,
    validate_development_source_integrity,
)
from cardiosentinel.neural.physiology_fusion import (
    B4B_CHECKPOINT_SHA256,
    B4B_EXPERIMENT_LOCK_SHA256,
    EMBEDDING_DIM,
    EMBEDDING_TAP,
    MORPHOLOGY_SCHEMA_SHA256,
    P1_BATCH_SIZE,
    P1_EARLY_STOPPING_PATIENCE,
    P1_LEARNING_RATE,
    P1_MAX_EPOCHS,
    P1_PROTOCOL_SHA256,
    P1_SEED,
    P1_WEIGHT_DECAY,
    P1A_EXPERIMENT_ID,
    P1B_EXPERIMENT_ID,
    PHYSIOLOGY_DIM,
    PHYSIOLOGY_FEATURE_NAMES,
    PhysiologyTransform,
    build_p1_head,
    claim_p1_run_directory,
    extract_frozen_embeddings,
    fit_physiology_transform,
    morphology_columns,
    p1_head_identity,
    p1_training_configuration,
    record_p1_failure,
    require_p1_partition,
    resolve_p1_run_dir,
    validate_p1_protocol,
    validate_physiology_schema,
    write_p1_status,
)
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    B4_SPLIT_SHA256,
    FEATURE_CORPUS_SHA256,
    REPOSITORY_ROOT,
)
from cardiosentinel.neural.provenance import runtime_environment
from cardiosentinel.neural.resource_benchmark import (
    load_locked_model,
    validate_locked_model,
)
from cardiosentinel.neural.training import CheckpointTracker
from cardiosentinel.neural.validation_challenge import (
    CHALLENGE_EXPECTED_COUNTS,
    CHALLENGE_TOTAL_WINDOWS,
    build_validation_challenge_index,
)
from cardiosentinel.neural.waveform_cache import (
    B4CachedWaveformDataset,
    build_development_indexes,
    validate_waveform_cache,
)

CACHE_MANIFEST_NAME: Final = "P1_EMBEDDING_CACHE_MANIFEST.json"
CACHE_ARRAY_NAME: Final = "p1_embeddings.npz"
CACHE_CLAIM_NAME: Final = "P1_EMBEDDING_CACHE_CLAIM.json"

RUN_MANIFEST_NAME: Final = "RUN_MANIFEST.json"
EPOCH_HISTORY_NAME: Final = "EPOCH_HISTORY.json"
PHYSIOLOGY_TRANSFORM_NAME: Final = "PHYSIOLOGY_TRANSFORM.json"
VALIDATION_METRICS_NAME: Final = "VALIDATION_METRICS.json"
VALIDATION_THRESHOLD_NAME: Final = "VALIDATION_THRESHOLD.json"
VALIDATION_PREDICTIONS_NAME: Final = "VALIDATION_PREDICTIONS.npz"
CHALLENGE_METRICS_NAME: Final = "CHALLENGE_METRICS.json"
SELECTED_MODEL_NAME: Final = "model_selected.pt"
TRAINING_CHECKPOINT_NAME: Final = "training_checkpoint.pt"
EXPERIMENT_LOCK_NAME: Final = "EXPERIMENT_LOCK.json"
STAGE1_RESULTS_NAME: Final = "P1_STAGE1_RESULTS.json"

STATUS_RUNNING: Final = "RUNNING"
STATUS_COMPLETE: Final = "COMPLETE"
LOCK_STATUS: Final = "locked_p1_development_result"

P1_ARM_ORDER: Final = (P1A_EXPERIMENT_ID, P1B_EXPERIMENT_ID)
TRAINING_SELECTION_SHA256: Final = (
    "318da148da5d638af44e73c06c00cc4df2815017d4ce8bb1a1b864e53eda8009"
)
EXPECTED_POPULATIONS: Final = {
    "train": {
        "total": 374_452,
        "positive": 93_613,
        "negative": 280_839,
        "subjects": 56,
    },
    "validation": {
        "total": 473_897,
        "positive": 21_628,
        "negative": 452_269,
        "subjects": 12,
    },
}
CHALLENGE_SELECTION_SHA256: Final = (
    "49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a"
)
CHALLENGE_NAMES: Final = ("rate_related", "axis_shift", "conduction_change")
FROZEN_DEPENDENCY_DIGEST: Final = (
    "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
)

THRESHOLD_RULE: Final = (
    "maximum validation F1 over exact observed validation scores; "
    "the highest threshold wins an exact tie"
)
CHECKPOINT_RULE: Final = (
    "maximum full primary validation AUPRC; the earliest epoch wins an exact tie"
)
SCORE_SEMANTICS: Final = (
    "uncalibrated sigmoid model score; not calibrated probability"
)


class P1ExecutionError(RuntimeError):
    """Raised when a canonical P1 step cannot proceed with full integrity."""


# --------------------------------------------------------------------------
# Provenance digests
# --------------------------------------------------------------------------


def ordered_stable_id_digest(stable_ids) -> str:
    """Order-SENSITIVE digest of the embedding row identities.

    A sorted digest would not detect a row-order change, which would silently
    misalign embeddings against labels, physiology and subjects.
    """
    identifiers = [str(value) for value in stable_ids]
    if len(set(identifiers)) != len(identifiers):
        raise P1ExecutionError("Embedding stable IDs contain duplicates.")
    return canonical_sha256({"order": "row_order", "stable_ids": identifiers})


def embedding_content_digest(matrix: np.ndarray) -> str:
    """Digest the exact embedding content: shape, dtype and contiguous bytes."""
    array = np.ascontiguousarray(matrix)
    hasher = hashlib.sha256()
    hasher.update(repr((array.shape, str(array.dtype))).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def require_p1_runtime() -> tuple[dict[str, Any], str]:
    """Verify the CURRENT runtime against the frozen scientific environment."""
    determinism = initialize_determinism(requested_device="cpu")
    environment = runtime_environment(determinism.device, 0)
    dependency_digest = require_exact_scientific_environment(environment)
    if environment.get("amp_enabled") is not False:
        raise P1ExecutionError("P1 forbids automatic mixed precision.")
    return environment, dependency_digest


def require_clean_checkout() -> dict[str, Any]:
    """Refuse a canonical P1 claim from a dirty checkout."""
    provenance = git_provenance(REPOSITORY_ROOT)
    if provenance["git_dirty"]:
        raise P1ExecutionError("Canonical P1 evidence requires a clean Git checkout.")
    return provenance


# --------------------------------------------------------------------------
# Embedding cache
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class P1EmbeddingCache:
    """An ordered frozen-B4-B embedding cache for one development partition."""

    partition: str
    stable_ids: tuple[str, ...]
    embeddings: np.ndarray
    labels: np.ndarray
    subject_ids: tuple[str, ...]
    manifest: dict[str, Any]


def _population_of(labels: np.ndarray, subject_ids) -> dict[str, int]:
    return {
        "total": int(labels.shape[0]),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "subjects": int(len(set(subject_ids))),
    }


def build_embedding_cache_manifest(
    *,
    partition: str,
    stable_ids,
    embeddings: np.ndarray,
    labels: np.ndarray,
    subject_ids,
    git_sha: str,
    git_dirty: bool,
    dependency_digest: str,
    encoder_receipt: dict[str, Any],
    require_expected_population: bool = True,
) -> dict[str, Any]:
    """Bind ordered identity, content and every frozen identity for one cache."""
    evaluated = require_p1_partition(partition)
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
        raise P1ExecutionError(f"Embeddings must be [N, {EMBEDDING_DIM}].")
    if matrix.dtype != np.float32:
        raise P1ExecutionError("Embeddings must be float32.")
    if not np.all(np.isfinite(matrix)):
        raise P1ExecutionError("Embedding cache contains a non-finite value.")
    if not (len(stable_ids) == matrix.shape[0] == labels.shape[0] == len(subject_ids)):
        raise P1ExecutionError("Embedding cache columns are misaligned.")
    if not np.all(np.isin(labels, (0, 1))):
        raise P1ExecutionError("Embedding cache labels must be 0 or 1.")
    if git_dirty:
        raise P1ExecutionError("Embedding cache requires a clean Git checkout.")

    population = _population_of(labels, subject_ids)
    if require_expected_population and population != EXPECTED_POPULATIONS[evaluated]:
        raise P1ExecutionError(
            f"{evaluated} population {population} differs from the frozen "
            f"identity {EXPECTED_POPULATIONS[evaluated]}."
        )
    payload = {
        "cache_kind": "b4b_frozen_embedding_cache_v1",
        "partition": evaluated,
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_fine_tuned": False,
        "encoder_receipt": encoder_receipt,
        "embedding_tap": EMBEDDING_TAP,
        "embedding_dim": EMBEDDING_DIM,
        "dtype": str(matrix.dtype),
        "rows": int(matrix.shape[0]),
        "population": population,
        "ordered_stable_id_sha256": ordered_stable_id_digest(stable_ids),
        "sorted_stable_id_sha256": canonical_sha256(sorted(str(v) for v in stable_ids)),
        "embedding_content_sha256": embedding_content_digest(matrix),
        "label_content_sha256": embedding_content_digest(
            np.asarray(labels, dtype=np.int64)
        ),
        # Subject IDs drive subject-macro metrics, so their exact ordered
        # content is bound too: a reassigned subject must not validate.
        "ordered_subject_id_sha256": canonical_sha256(
            {"order": "row_order", "subject_ids": [str(v) for v in subject_ids]}
        ),
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "training_selection_sha256": (
            TRAINING_SELECTION_SHA256 if evaluated == "train" else None
        ),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "environment_dependency_digest": dependency_digest,
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "test_accessed": False,
    }
    payload["cache_sha256"] = canonical_sha256(payload)
    return payload


def load_official_b4b_encoder(b4b_run_dir: Path) -> nn.Module:
    """Load the canonical B4-B encoder, proving its identity before use.

    The official path must never receive an arbitrary preconstructed module and
    then stamp it with the frozen checkpoint SHA.
    """
    lock = validate_locked_model(Path(b4b_run_dir), official_model="B4-B")
    if lock["experiment_lock_sha256"] != B4B_EXPERIMENT_LOCK_SHA256:
        raise P1ExecutionError("B4-B experiment lock SHA-256 is not the selected one.")
    if lock["checkpoint_sha256"] != B4B_CHECKPOINT_SHA256:
        raise P1ExecutionError("B4-B checkpoint SHA-256 is not the selected one.")
    if lock.get("test") is not None:
        raise P1ExecutionError("The B4-B lock must record test as null.")
    encoder = load_locked_model(Path(b4b_run_dir), lock)
    if type(encoder).__name__ != "B4BTransformerCNN":
        raise P1ExecutionError("The locked B4-B model is not a B4BTransformerCNN.")
    if not hasattr(encoder, "encode"):
        raise P1ExecutionError("The locked B4-B model exposes no encode() tap.")
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def _claim_cache_directory(directory: Path, partition: str) -> None:
    """Refuse any pre-existing canonical cache directory, partial or complete."""
    if directory.exists():
        raise P1ExecutionError(
            f"A canonical P1 embedding cache directory already exists at "
            f"{directory}. It is never overwritten, even if partial: an "
            "incomplete cache requires human review or explicit read-only "
            "validation."
        )
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        directory / CACHE_CLAIM_NAME,
        {
            "claim": "p1_embedding_cache",
            "partition": partition,
            "claim_status": "STARTED",
            "overwrite_permitted": False,
        },
    )


def materialize_p1_embedding_cache(
    b4b_run_dir: Path,
    waveform_batches,
    *,
    partition: str,
    stable_ids,
    labels: np.ndarray,
    subject_ids,
    cache_root: Path,
    require_expected_population: bool = True,
) -> P1EmbeddingCache:
    """Materialize one development embedding cache with the locked B4-B model.

    `waveform_batches` yields `[B, 1, 2500]` float32 tensors in the exact frozen
    row order. The encoder is loaded from the canonical B4-B run and proven
    before use; it is never fine-tuned and never enters an optimizer. Any
    pre-existing cache directory, complete or partial, is refused.
    """
    evaluated = require_p1_partition(partition)
    validate_p1_protocol()
    environment, dependency_digest = require_p1_runtime()
    provenance = require_clean_checkout()
    encoder = load_official_b4b_encoder(b4b_run_dir)

    directory = Path(cache_root) / evaluated
    manifest_path = directory / CACHE_MANIFEST_NAME
    _claim_cache_directory(directory, evaluated)

    chunks: list[np.ndarray] = []
    receipt: dict[str, Any] = {}
    for batch in waveform_batches:
        embeddings, receipt = extract_frozen_embeddings(encoder, batch)
        chunks.append(embeddings.to(torch.float32).numpy())
    if not chunks:
        raise P1ExecutionError("No waveform batches were provided.")
    matrix = np.concatenate(chunks, axis=0).astype(np.float32)

    manifest = build_embedding_cache_manifest(
        partition=evaluated,
        stable_ids=stable_ids,
        embeddings=matrix,
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=subject_ids,
        git_sha=provenance["git_sha"],
        git_dirty=provenance["git_dirty"],
        dependency_digest=dependency_digest,
        encoder_receipt=receipt,
        require_expected_population=require_expected_population,
    )
    manifest["environment"] = environment

    array_path = directory / CACHE_ARRAY_NAME
    temporary = directory / f".{CACHE_ARRAY_NAME}.tmp.npz"
    np.savez_compressed(
        temporary,
        stable_id=np.asarray([str(v) for v in stable_ids], dtype=np.str_),
        subject_id=np.asarray([str(v) for v in subject_ids], dtype=np.str_),
        label=np.asarray(labels, dtype=np.int64),
        embedding=matrix,
    )
    temporary.replace(array_path)
    manifest["artifact"] = CACHE_ARRAY_NAME
    manifest["artifact_sha256"] = sha256_file(array_path)
    manifest["cache_sha256"] = canonical_sha256(
        {k: v for k, v in manifest.items() if k != "cache_sha256"}
    )
    write_json_atomic(manifest_path, manifest)
    return P1EmbeddingCache(
        partition=evaluated,
        stable_ids=tuple(str(v) for v in stable_ids),
        embeddings=matrix,
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=tuple(str(v) for v in subject_ids),
        manifest=manifest,
    )


def load_p1_embedding_cache(cache_root: Path, partition: str) -> P1EmbeddingCache:
    """Load and fully re-verify a materialized development embedding cache."""
    evaluated = require_p1_partition(partition)
    directory = Path(cache_root) / evaluated
    manifest_path = directory / CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise P1ExecutionError(f"No P1 embedding cache manifest at {manifest_path}.")
    manifest = read_json(manifest_path)

    recorded = manifest.get("cache_sha256")
    body = {k: v for k, v in manifest.items() if k != "cache_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise P1ExecutionError("P1 embedding cache manifest failed digest validation.")
    if manifest["partition"] != evaluated:
        raise P1ExecutionError("P1 embedding cache manifest partition mismatch.")
    for field, expected in (
        ("encoder_checkpoint_sha256", B4B_CHECKPOINT_SHA256),
        ("encoder_experiment_lock_sha256", B4B_EXPERIMENT_LOCK_SHA256),
        ("split_sha256", B4_SPLIT_SHA256),
        ("feature_corpus_sha256", FEATURE_CORPUS_SHA256),
        ("embedding_tap", EMBEDDING_TAP),
        ("p1_protocol_sha256", P1_PROTOCOL_SHA256),
    ):
        if manifest.get(field) != expected:
            raise P1ExecutionError(
                f"P1 embedding cache binds {field}={manifest.get(field)!r}, "
                f"expected {expected!r}."
            )
    if manifest.get("git_dirty") is not False:
        raise P1ExecutionError("P1 embedding cache was built from a dirty checkout.")
    if manifest.get("population") != EXPECTED_POPULATIONS[evaluated]:
        raise P1ExecutionError(
            f"P1 embedding cache population {manifest.get('population')} differs "
            f"from the frozen identity {EXPECTED_POPULATIONS[evaluated]}."
        )
    if evaluated == "train" and (
        manifest.get("training_selection_sha256") != TRAINING_SELECTION_SHA256
    ):
        raise P1ExecutionError(
            "P1 train embedding cache does not bind the frozen training selection."
        )
    if manifest.get("environment_dependency_digest") != FROZEN_DEPENDENCY_DIGEST:
        raise P1ExecutionError(
            "P1 embedding cache does not bind the frozen dependency digest."
        )

    array_path = directory / str(manifest["artifact"])
    if sha256_file(array_path) != manifest["artifact_sha256"]:
        raise P1ExecutionError("P1 embedding cache artifact SHA-256 does not match.")
    with np.load(array_path, allow_pickle=False) as archive:
        stable_ids = tuple(archive["stable_id"].tolist())
        subject_ids = tuple(archive["subject_id"].tolist())
        labels = np.asarray(archive["label"], dtype=np.int64)
        embeddings = np.asarray(archive["embedding"], dtype=np.float32)

    if ordered_stable_id_digest(stable_ids) != manifest["ordered_stable_id_sha256"]:
        raise P1ExecutionError(
            "P1 embedding cache row order does not match its ordered identity."
        )
    if embedding_content_digest(embeddings) != manifest["embedding_content_sha256"]:
        raise P1ExecutionError("P1 embedding content does not match its digest.")
    if embedding_content_digest(labels) != manifest["label_content_sha256"]:
        raise P1ExecutionError("P1 embedding cache labels do not match their digest.")
    subject_digest = canonical_sha256(
        {"order": "row_order", "subject_ids": [str(v) for v in subject_ids]}
    )
    if subject_digest != manifest["ordered_subject_id_sha256"]:
        raise P1ExecutionError(
            "P1 embedding cache subject assignment does not match its digest."
        )
    if _population_of(labels, subject_ids) != manifest["population"]:
        raise P1ExecutionError("P1 embedding cache population does not match.")
    return P1EmbeddingCache(
        partition=evaluated,
        stable_ids=stable_ids,
        embeddings=embeddings,
        labels=labels,
        subject_ids=subject_ids,
        manifest=manifest,
    )


@dataclass(frozen=True, slots=True)
class P1PhysiologyBundle:
    """Transformed physiology bound to an exact ordered stable-ID sequence.

    A bare array cannot detect a row permutation; this carries the ordered IDs
    and content digest so misalignment against the embedding cache is refused.
    """

    partition: str
    stable_ids: tuple[str, ...]
    values: np.ndarray
    schema_sha256: str
    transform_sha256: str

    @property
    def ordered_stable_id_sha256(self) -> str:
        return ordered_stable_id_digest(self.stable_ids)

    @property
    def content_sha256(self) -> str:
        return embedding_content_digest(self.values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "rows": int(self.values.shape[0]),
            "physiology_dim": int(self.values.shape[1]),
            "ordered_stable_id_sha256": self.ordered_stable_id_sha256,
            "content_sha256": self.content_sha256,
            "schema_sha256": self.schema_sha256,
            "transform_sha256": self.transform_sha256,
        }


def build_physiology_bundle(
    *,
    partition: str,
    stable_ids,
    raw_by_stable_id: dict[str, np.ndarray],
    transform: PhysiologyTransform,
) -> P1PhysiologyBundle:
    """Join frozen morphology_v1 by stable ID, in the cache's exact row order."""
    evaluated = require_p1_partition(partition)
    identifiers = [str(v) for v in stable_ids]
    missing = [key for key in identifiers if key not in raw_by_stable_id]
    if missing:
        raise P1ExecutionError(
            f"{len(missing)} rows have no morphology_v1 record; the first is "
            f"{missing[0]!r}."
        )
    raw = np.stack([np.asarray(raw_by_stable_id[key], dtype=np.float64)
                    for key in identifiers])
    if raw.shape[1] != PHYSIOLOGY_DIM:
        raise P1ExecutionError(f"Physiology rows must be [N, {PHYSIOLOGY_DIM}].")
    return P1PhysiologyBundle(
        partition=evaluated,
        stable_ids=tuple(identifiers),
        values=transform.transform(raw),
        schema_sha256=MORPHOLOGY_SCHEMA_SHA256,
        transform_sha256=transform.as_dict()["transform_sha256"],
    )


def require_aligned_physiology(
    bundle: P1PhysiologyBundle, stable_ids, partition: str
) -> np.ndarray:
    """Refuse physiology whose ordered identity differs from the cache's."""
    if bundle.partition != partition:
        raise P1ExecutionError(
            f"Physiology bundle is for {bundle.partition!r}, not {partition!r}."
        )
    if bundle.ordered_stable_id_sha256 != ordered_stable_id_digest(stable_ids):
        raise P1ExecutionError(
            "Physiology rows are not in the embedding cache's exact row order."
        )
    return bundle.values


# --------------------------------------------------------------------------
# Deterministic training
# --------------------------------------------------------------------------


def build_deterministic_p1_head(experiment_id: str):
    """Construct a P1 head under a deterministic reseed.

    Standard PyTorch `nn.Linear` initialization is used; determinism comes from
    reseeding immediately before construction so nothing may consume RNG in
    between.
    """
    torch.manual_seed(P1_SEED)
    return build_p1_head(experiment_id)


def p1_epoch_order(rows: int, epoch: int) -> np.ndarray:
    """Frozen deterministic train shuffle: seed 2026 offset by the epoch."""
    generator = torch.Generator().manual_seed(P1_SEED + epoch)
    return torch.randperm(rows, generator=generator).numpy()


def _features_for(
    arm: str, embeddings: np.ndarray, physiology: np.ndarray | None
) -> np.ndarray:
    if arm == P1A_EXPERIMENT_ID:
        if physiology is not None:
            raise P1ExecutionError(
                "P1-A is the neural-only control and must not receive physiology."
            )
        return embeddings
    if physiology is None:
        raise P1ExecutionError("P1-B requires transformed physiology features.")
    if physiology.shape != (embeddings.shape[0], PHYSIOLOGY_DIM):
        raise P1ExecutionError(f"Physiology must be [N, {PHYSIOLOGY_DIM}].")
    return np.concatenate([embeddings, physiology], axis=1).astype(np.float32)


def _require_finite_state(
    head: nn.Module, loss: float, auprc: float, epoch: int
) -> None:
    if not np.isfinite(loss):
        raise P1ExecutionError(f"Non-finite mean training loss at epoch {epoch}.")
    if not np.isfinite(auprc):
        raise P1ExecutionError(f"Non-finite validation AUPRC at epoch {epoch}.")
    for name, parameter in head.named_parameters():
        if not torch.isfinite(parameter).all():
            raise P1ExecutionError(f"Non-finite P1 parameter {name} at epoch {epoch}.")


def _scores(head: nn.Module, features: np.ndarray) -> np.ndarray:
    head.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], P1_BATCH_SIZE):
            chunk = torch.from_numpy(features[start : start + P1_BATCH_SIZE])
            outputs.append(torch.sigmoid(head(chunk)).to(torch.float64).numpy())
    scores = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise P1ExecutionError("P1 head produced a non-finite prediction.")
    return scores


def train_p1_arm(
    experiment_id: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    max_epochs: int = P1_MAX_EPOCHS,
) -> dict[str, Any]:
    """Train one P1 arm on precomputed frozen embeddings.

    The B4-B encoder does not execute here and is not in the optimizer: only the
    small head is trained.
    """
    head = build_deterministic_p1_head(experiment_id)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=P1_LEARNING_RATE,
        weight_decay=P1_WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    features = torch.from_numpy(np.ascontiguousarray(train_features))
    targets = torch.from_numpy(np.asarray(train_labels, dtype=np.float32))

    history: list[dict[str, Any]] = []
    # Reuse the reviewed B4 tracker: checkpoint saving is a strict numerical
    # maximum, while patience resets only on an improvement beyond the delta.
    # A 5e-7 gain therefore becomes the selected checkpoint without resetting
    # patience. Conflating the two was a real defect in the previous head.
    tracker = CheckpointTracker()
    best_state: dict[str, Any] | None = None

    for epoch in range(1, max_epochs + 1):
        head.train()
        order = p1_epoch_order(features.shape[0], epoch)
        total = 0.0
        batches = 0
        for start in range(0, len(order), P1_BATCH_SIZE):
            index = order[start : start + P1_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(head(features[index]), targets[index])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        mean_loss = total / max(batches, 1)

        scores = _scores(head, validation_features)
        auprc = float(binary_metrics(validation_labels, scores, 0.5)["auprc"])
        _require_finite_state(head, mean_loss, auprc, epoch)

        decision = tracker.update(epoch, auprc)
        if decision.save_checkpoint:
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "mean_training_loss": mean_loss,
                "validation_auprc": auprc,
                "checkpoint_saved": decision.save_checkpoint,
                "early_stopping_patience": decision.patience,
            }
        )
        if decision.stop_training:
            break

    if best_state is None:
        raise P1ExecutionError("P1 training selected no checkpoint.")
    head.load_state_dict(best_state)
    return {
        "head": head,
        "epoch_history": tuple(history),
        "selected_epoch": tracker.best_epoch,
        "selected_validation_auprc": tracker.best_auprc,
        "completed_epochs": len(history),
        "stop_reason": (
            "early_stopping"
            if tracker.patience >= P1_EARLY_STOPPING_PATIENCE
            else "max_epochs"
        ),
    }


def select_p1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Maximum-F1 validation threshold via the reviewed exact sweep.

    This delegates to `select_validation_f1_threshold`, the repository's exact
    O(N log N) cumulative implementation. A second P1-specific implementation is
    deliberately not maintained: the previous per-candidate loop recomputed full
    metrics (including AUPRC/AUROC) for every unique score, which is
    quadratic-ish and unacceptable at 473,897 validation rows.

    Semantics are unchanged: validation only, maximum F1 over exact observed
    scores, highest threshold winning an exact tie.
    """
    return select_validation_f1_threshold(
        np.asarray(labels, dtype=np.int64).tolist(),
        np.asarray(scores, dtype=np.float64).tolist(),
        partition="validation",
    )


def p1_validation_evidence(
    labels: np.ndarray, scores: np.ndarray, subject_ids, threshold: float
) -> dict[str, Any]:
    """Pooled and subject-macro validation evidence from reviewed metrics."""
    subjects = np.asarray([str(v) for v in subject_ids], dtype=np.str_)
    return {
        "evidence_class": "p1_development_validation_result",
        "partition": "validation",
        "score_semantics": SCORE_SEMANTICS,
        "threshold": threshold,
        "threshold_rule": THRESHOLD_RULE,
        "window_count": int(labels.shape[0]),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "subject_count": int(len(set(subjects.tolist()))),
        "pooled": binary_metrics(labels, scores, threshold),
        "subject_macro": subject_macro_metrics(labels, scores, subjects, threshold),
    }


def p1_challenge_evidence(
    target_families, scores: np.ndarray, subject_ids, threshold: float
) -> dict[str, Any]:
    """Development challenge evidence, reusing the frozen production metric."""
    families = np.asarray([str(v) for v in target_families], dtype=np.str_)
    subjects = np.asarray([str(v) for v in subject_ids], dtype=np.str_)
    frozen = challenge_metrics(families, scores, subjects, threshold)
    evidence: dict[str, Any] = {
        "partition": "validation",
        "challenge_selection_sha256": CHALLENGE_SELECTION_SHA256,
        "threshold": threshold,
        "threshold_source": "locked_p1_validation_threshold",
    }
    for name in CHALLENGE_NAMES:
        policy = challenge_evidence_policy(name)
        measured = frozen[name]
        evidence[name] = {
            "target_family": policy.target_family,
            "evidence_status": policy.evidence_level,
            "is_headline_metric": policy.is_headline_metric,
            "challenge_window_count": measured["challenge_window_count"],
            "false_positive_count": measured["false_positive_count"],
            "false_positive_fraction": measured["false_positive_fraction"],
            "supporting_subject_count": measured["contributing_subject_count"],
            "bootstrap_permitted": measured["bootstrap_permitted"],
            "frozen_metric": measured,
        }
    return evidence


@dataclass(frozen=True, slots=True)
class P1ChallengeSet:
    """The frozen validation challenge rows prepared for P1 scoring."""

    stable_ids: tuple[str, ...]
    target_families: tuple[str, ...]
    subject_ids: tuple[str, ...]
    embeddings: np.ndarray
    physiology: P1PhysiologyBundle | None
    selection_sha256: str
    counts: dict[str, dict[str, int]]


def prepare_p1_challenge_set(
    feature_root: Path,
    *,
    embeddings_by_stable_id: dict[str, np.ndarray],
    raw_physiology_by_stable_id: dict[str, np.ndarray],
    transform: PhysiologyTransform,
) -> P1ChallengeSet:
    """Rebuild and verify the frozen validation challenge population for P1.

    The identity is rebuilt through the reviewed B4 challenge index rather than
    stamped, so an arbitrary row set cannot masquerade as the frozen selection.
    """
    require_p1_partition("validation")
    index = build_validation_challenge_index(Path(feature_root))
    if index.selection_sha256 != CHALLENGE_SELECTION_SHA256:
        raise P1ExecutionError(
            "Rebuilt challenge selection digest differs from the frozen identity."
        )
    if index.counts != {
        family: dict(counts) for family, counts in CHALLENGE_EXPECTED_COUNTS.items()
    }:
        raise P1ExecutionError(
            f"Challenge population {index.counts} differs from the frozen identity."
        )
    if len(index.references) != CHALLENGE_TOTAL_WINDOWS:
        raise P1ExecutionError(
            f"Expected {CHALLENGE_TOTAL_WINDOWS} challenge windows, "
            f"observed {len(index.references)}."
        )
    stable_ids = tuple(item.stable_id for item in index.references)
    missing = [key for key in stable_ids if key not in embeddings_by_stable_id]
    if missing:
        raise P1ExecutionError(
            f"{len(missing)} challenge rows have no frozen B4-B embedding."
        )
    embeddings = np.stack(
        [np.asarray(embeddings_by_stable_id[key], dtype=np.float32)
         for key in stable_ids]
    )
    physiology = build_physiology_bundle(
        partition="validation",
        stable_ids=stable_ids,
        raw_by_stable_id=raw_physiology_by_stable_id,
        transform=transform,
    )
    return P1ChallengeSet(
        stable_ids=stable_ids,
        target_families=tuple(item.target_family for item in index.references),
        subject_ids=tuple(item.subject_id for item in index.references),
        embeddings=embeddings,
        physiology=physiology,
        selection_sha256=index.selection_sha256,
        counts=index.counts,
    )


# --------------------------------------------------------------------------
# Immutable lock
# --------------------------------------------------------------------------


def build_p1_lock(
    experiment_id: str,
    *,
    head: nn.Module,
    result: dict[str, Any],
    threshold: float,
    validation_evidence: dict[str, Any],
    challenge_evidence: dict[str, Any] | None,
    transform: PhysiologyTransform | None,
    train_cache: dict[str, Any],
    validation_cache: dict[str, Any],
    artifact_hashes: dict[str, Any],
    provenance: dict[str, Any],
    environment: dict[str, Any],
    dependency_digest: str,
) -> dict[str, Any]:
    """Assemble the immutable P1 lock and bind its own canonical digest."""
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": LOCK_STATUS,
        "arm": (
            "neural_plus_physiology"
            if experiment_id == P1B_EXPERIMENT_ID
            else "neural_only_control"
        ),
        "head": p1_head_identity(head),
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_fine_tuned": False,
        "embedding_tap": EMBEDDING_TAP,
        "embedding_dim": EMBEDDING_DIM,
        "train_embedding_cache_sha256": train_cache["cache_sha256"],
        "validation_embedding_cache_sha256": validation_cache["cache_sha256"],
        "train_ordered_stable_id_sha256": train_cache["ordered_stable_id_sha256"],
        "validation_ordered_stable_id_sha256": validation_cache[
            "ordered_stable_id_sha256"
        ],
        "physiology_schema_sha256": (
            MORPHOLOGY_SCHEMA_SHA256 if transform else None
        ),
        "physiology_transform_sha256": (
            transform.as_dict()["transform_sha256"] if transform else None
        ),
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "environment_dependency_digest": dependency_digest,
        "environment": environment,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "training_selection_sha256": TRAINING_SELECTION_SHA256,
        "training_rows": train_cache["population"],
        "validation_rows": validation_cache["population"],
        "training_configuration": p1_training_configuration(),
        "checkpoint_selection_rule": CHECKPOINT_RULE,
        "threshold_selection_rule": THRESHOLD_RULE,
        "selected_epoch": result["selected_epoch"],
        "selected_validation_auprc": result["selected_validation_auprc"],
        "completed_epochs": result["completed_epochs"],
        "stop_reason": result["stop_reason"],
        "validation_threshold": threshold,
        "epoch_history_digest": canonical_sha256(list(result["epoch_history"])),
        "validation_evidence_sha256": canonical_sha256(validation_evidence),
        "challenge_evidence_sha256": (
            canonical_sha256(challenge_evidence) if challenge_evidence else None
        ),
        "challenge_selection_sha256": (
            CHALLENGE_SELECTION_SHA256 if challenge_evidence else None
        ),
        "score_semantics": SCORE_SEMANTICS,
        "test": None,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        **artifact_hashes,
    }
    payload["experiment_lock_sha256"] = canonical_sha256(payload)
    return payload


def validate_p1_lock(run_dir: Path) -> dict[str, Any]:
    """Re-derive a P1 lock digest and confirm its bound artifacts."""
    lock_path = Path(run_dir) / EXPERIMENT_LOCK_NAME
    if not lock_path.is_file():
        raise P1ExecutionError(f"No {EXPERIMENT_LOCK_NAME} in {run_dir}.")
    lock = read_json(lock_path)
    recorded = lock.pop("experiment_lock_sha256", None)
    if recorded is None or recorded != canonical_sha256(lock):
        raise P1ExecutionError("P1 experiment lock hash validation failed.")
    lock["experiment_lock_sha256"] = recorded
    bound = lock.get("artifact_sha256", {})
    if not bound:
        raise P1ExecutionError("P1 lock binds no claim-bearing artifact hashes.")
    for name, expected in bound.items():
        path = Path(run_dir) / name
        if not path.is_file():
            raise P1ExecutionError(f"P1 locked artifact {name} is absent.")
        if sha256_file(path) != expected:
            raise P1ExecutionError(f"P1 locked artifact {name} failed hash validation.")
    if "test" not in lock or lock["test"] is not None:
        raise P1ExecutionError("P1 experiment lock must record test as null.")
    if lock.get("experiment_id") not in P1_ARM_ORDER:
        raise P1ExecutionError("P1 experiment lock has an unknown experiment.")
    if lock.get("encoder_fine_tuned") is not False:
        raise P1ExecutionError("P1 lock claims the frozen encoder was fine-tuned.")
    checkpoint = Path(run_dir) / SELECTED_MODEL_NAME
    if not checkpoint.is_file() or sha256_file(checkpoint) != lock["checkpoint_sha256"]:
        raise P1ExecutionError("P1 locked head failed hash validation.")
    return lock


# --------------------------------------------------------------------------
# Canonical arm and Stage-1 suite
# --------------------------------------------------------------------------


def run_p1_arm(
    experiment_id: str,
    *,
    run_root: Path,
    train_cache: P1EmbeddingCache,
    validation_cache: P1EmbeddingCache,
    transform: PhysiologyTransform | None,
    challenge: P1ChallengeSet,
    train_physiology: P1PhysiologyBundle | None = None,
    validation_physiology: P1PhysiologyBundle | None = None,
    command: str = "cardiosentinel p1 run-stage1",
) -> dict[str, Any]:
    """Execute and lock one canonical P1 arm. The claim is never released."""
    if experiment_id not in P1_ARM_ORDER:
        raise P1ExecutionError(f"Unknown P1 experiment {experiment_id!r}.")
    validate_p1_protocol()
    environment, dependency_digest = require_p1_runtime()
    provenance = require_clean_checkout()
    if train_cache.partition != "train" or validation_cache.partition != "validation":
        raise P1ExecutionError("P1 requires a train cache and a validation cache.")

    run_dir = resolve_p1_run_dir(Path(run_root), experiment_id)
    claim_p1_run_directory(run_dir, experiment_id)
    write_p1_status(
        run_dir, STATUS_RUNNING, experiment_id=experiment_id, command=command
    )
    started = time.monotonic()
    try:
        uses_physiology = experiment_id == P1B_EXPERIMENT_ID
        train_values = (
            require_aligned_physiology(
                train_physiology, train_cache.stable_ids, "train"
            )
            if uses_physiology
            else None
        )
        validation_values = (
            require_aligned_physiology(
                validation_physiology, validation_cache.stable_ids, "validation"
            )
            if uses_physiology
            else None
        )
        train_features = _features_for(
            experiment_id, train_cache.embeddings, train_values
        )
        validation_features = _features_for(
            experiment_id, validation_cache.embeddings, validation_values
        )
        result = train_p1_arm(
            experiment_id,
            train_features,
            train_cache.labels,
            validation_features,
            validation_cache.labels,
        )
        head = result["head"]
        scores = _scores(head, validation_features)
        threshold = select_p1_threshold(validation_cache.labels, scores)
        evidence = p1_validation_evidence(
            validation_cache.labels, scores, validation_cache.subject_ids, threshold
        )

        if challenge.selection_sha256 != CHALLENGE_SELECTION_SHA256:
            raise P1ExecutionError("P1 challenge set is not the frozen selection.")
        challenge_values = (
            require_aligned_physiology(
                challenge.physiology, challenge.stable_ids, "validation"
            )
            if uses_physiology
            else None
        )
        challenge_features = _features_for(
            experiment_id, challenge.embeddings, challenge_values
        )
        challenge_evidence = p1_challenge_evidence(
            challenge.target_families,
            _scores(head, challenge_features),
            challenge.subject_ids,
            threshold,
        )
        challenge_evidence["challenge_population"] = challenge.counts

        write_json_atomic(
            run_dir / EPOCH_HISTORY_NAME,
            {"experiment_id": experiment_id, "epochs": list(result["epoch_history"])},
        )
        write_json_atomic(run_dir / VALIDATION_METRICS_NAME, evidence)
        write_json_atomic(
            run_dir / VALIDATION_THRESHOLD_NAME,
            {
                "experiment_id": experiment_id,
                "selected_from": "validation",
                "test_informed": False,
                "threshold": threshold,
                "threshold_rule": THRESHOLD_RULE,
            },
        )
        write_json_atomic(
            run_dir / PHYSIOLOGY_TRANSFORM_NAME,
            transform.as_dict() if transform else {"physiology_transform": None},
        )
        write_json_atomic(run_dir / CHALLENGE_METRICS_NAME, challenge_evidence)
        np.savez_compressed(
            run_dir / VALIDATION_PREDICTIONS_NAME,
            stable_id=np.asarray(validation_cache.stable_ids, dtype=np.str_),
            subject_id=np.asarray(validation_cache.subject_ids, dtype=np.str_),
            label=validation_cache.labels,
            score=scores,
        )
        torch.save(head.state_dict(), run_dir / SELECTED_MODEL_NAME)
        torch.save(head.state_dict(), run_dir / TRAINING_CHECKPOINT_NAME)

        write_json_atomic(
            run_dir / RUN_MANIFEST_NAME,
            {
                "experiment_id": experiment_id,
                "command": command,
                "status": STATUS_COMPLETE,
            },
        )
        # Every claim-bearing result file is bound, so tampering any of them is
        # refused by the lock validator.
        claim_bearing = (
            EPOCH_HISTORY_NAME,
            PHYSIOLOGY_TRANSFORM_NAME,
            VALIDATION_METRICS_NAME,
            VALIDATION_THRESHOLD_NAME,
            VALIDATION_PREDICTIONS_NAME,
            CHALLENGE_METRICS_NAME,
            SELECTED_MODEL_NAME,
            TRAINING_CHECKPOINT_NAME,
            RUN_MANIFEST_NAME,
        )
        artifacts = {
            "locked_inference_model": SELECTED_MODEL_NAME,
            "checkpoint_sha256": sha256_file(run_dir / SELECTED_MODEL_NAME),
            "checkpoint_bytes": (run_dir / SELECTED_MODEL_NAME).stat().st_size,
            "artifact_sha256": {
                name: sha256_file(run_dir / name) for name in claim_bearing
            },
        }
        lock = build_p1_lock(
            experiment_id,
            head=head,
            result=result,
            threshold=threshold,
            validation_evidence=evidence,
            challenge_evidence=challenge_evidence,
            transform=transform,
            train_cache=train_cache.manifest,
            validation_cache=validation_cache.manifest,
            artifact_hashes=artifacts,
            provenance=provenance,
            environment=environment,
            dependency_digest=dependency_digest,
        )
        write_json_atomic(run_dir / EXPERIMENT_LOCK_NAME, lock)
        write_p1_status(
            run_dir,
            STATUS_COMPLETE,
            experiment_id=experiment_id,
            command=command,
            selected_epoch=result["selected_epoch"],
            experiment_lock_sha256=lock["experiment_lock_sha256"],
        )
        return {
            "experiment_id": experiment_id,
            "status": STATUS_COMPLETE,
            "run_dir": str(run_dir),
            "selected_epoch": result["selected_epoch"],
            "selected_validation_auprc": result["selected_validation_auprc"],
            "validation_threshold": threshold,
            "experiment_lock_sha256": lock["experiment_lock_sha256"],
            "duration_seconds": time.monotonic() - started,
            "test": None,
        }
    except BaseException as error:
        record_p1_failure(run_dir, experiment_id, error)
        raise


def run_p1_stage1_suite(
    *,
    run_root: Path,
    train_cache: P1EmbeddingCache,
    validation_cache: P1EmbeddingCache,
    transform: PhysiologyTransform,
    train_physiology: P1PhysiologyBundle,
    validation_physiology: P1PhysiologyBundle,
    challenge: P1ChallengeSet,
    command: str = "cardiosentinel p1 run-stage1",
) -> dict[str, Any]:
    """Run the official Stage P1-1 ablation. BOTH arms are mandatory.

    There is deliberately no route that runs one arm alone and calls Stage P1-1
    complete, and no selective retry: a claimed arm is immutable.
    """
    validate_p1_protocol()
    environment, dependency_digest = require_p1_runtime()
    provenance = require_clean_checkout()
    started = time.monotonic()

    if challenge.selection_sha256 != CHALLENGE_SELECTION_SHA256:
        raise P1ExecutionError(
            "Stage P1-1 requires the frozen validation challenge selection."
        )
    results: dict[str, Any] = {}
    for arm in P1_ARM_ORDER:
        uses_physiology = arm == P1B_EXPERIMENT_ID
        results[arm] = run_p1_arm(
            arm,
            run_root=run_root,
            train_cache=train_cache,
            validation_cache=validation_cache,
            transform=transform if uses_physiology else None,
            challenge=challenge,
            train_physiology=train_physiology if uses_physiology else None,
            validation_physiology=(
                validation_physiology if uses_physiology else None
            ),
            command=command,
        )

    suite = {
        "suite": "P1_stage1_physiology_ablation_v1",
        "stage": "P1-1",
        "command": command,
        "arm_order": list(P1_ARM_ORDER),
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_fine_tuned": False,
        "physiology_transform_sha256": transform.as_dict()["transform_sha256"],
        "challenge_selection_sha256": challenge.selection_sha256,
        "challenge_population": challenge.counts,
        "train_embedding_cache_sha256": train_cache.manifest["cache_sha256"],
        "validation_embedding_cache_sha256": validation_cache.manifest["cache_sha256"],
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "environment_dependency_digest": dependency_digest,
        "runtime_environment": environment,
        "arm_results": results,
        "physiology_retained": None,
        "retention_decision_performed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "suite_duration_seconds": time.monotonic() - started,
    }
    suite["p1_stage1_suite_sha256"] = canonical_sha256(suite)
    write_json_atomic(Path(run_root) / STAGE1_RESULTS_NAME, suite)
    return suite


def prepare_p1_challenge_embeddings(
    b4b_run_dir: Path,
    feature_root: Path,
    source: Path,
    *,
    batch_size: int = P1_BATCH_SIZE,
) -> dict[str, Any]:
    """Score the frozen validation challenge rows with the locked B4-B encoder.

    The challenge rows are a SEPARATE validation population from the primary
    cache: the primary cache holds only `ischemic_positive` +
    `background_negative`, so challenge embeddings can never be looked up there.
    They are produced here through the same validated raw physical-mV path the
    reviewed B4 challenge evaluator uses.
    """
    require_p1_partition("validation")
    index = build_validation_challenge_index(Path(feature_root))
    if index.selection_sha256 != CHALLENGE_SELECTION_SHA256:
        raise P1ExecutionError(
            "Rebuilt challenge selection digest differs from the frozen identity."
        )
    encoder = load_official_b4b_encoder(Path(b4b_run_dir))
    reader = B4WaveformDataset(index.references, Path(source))
    chunks: list[np.ndarray] = []
    receipt: dict[str, Any] = {}
    for start in range(0, len(index.references), batch_size):
        batch = torch.stack(
            [
                reader.read_waveform(reference)
                for reference in index.references[start : start + batch_size]
            ]
        )
        embeddings, receipt = extract_frozen_embeddings(encoder, batch)
        chunks.append(embeddings.to(torch.float32).numpy())
    matrix = np.concatenate(chunks, axis=0).astype(np.float32)
    stable_ids = tuple(item.stable_id for item in index.references)
    return {
        "index": index,
        "stable_ids": stable_ids,
        "embeddings": matrix,
        "encoder_receipt": receipt,
        "ordered_stable_id_sha256": ordered_stable_id_digest(stable_ids),
        "embedding_content_sha256": embedding_content_digest(matrix),
        "waveform_reads": reader.stats.source_reads,
    }


def prepare_p1_embedding_caches(
    *,
    cache_root: Path,
    feature_root: Path,
    source: Path,
    b4b_run_dir: Path,
    waveform_cache_root: Path,
) -> dict[str, P1EmbeddingCache]:
    """Load, or canonically materialize, both primary embedding caches.

    Valid existing caches are loaded and verified, never regenerated. A partial
    cache directory stops for human review. Nothing here requires the caller to
    assemble waveform batches, stable IDs, labels or subjects by hand.
    """
    caches: dict[str, P1EmbeddingCache] = {}
    indexes = build_development_indexes(Path(feature_root))
    validated = validate_waveform_cache(Path(waveform_cache_root), indexes)
    for partition in ("train", "validation"):
        directory = Path(cache_root) / partition
        if (directory / CACHE_MANIFEST_NAME).is_file():
            caches[partition] = load_p1_embedding_cache(Path(cache_root), partition)
            continue
        if directory.exists():
            raise P1ExecutionError(
                f"A partial P1 embedding cache exists at {directory}; human "
                "review is required before Stage P1-1 may proceed."
            )
        index = indexes[partition]
        dataset = B4CachedWaveformDataset(validated, index)
        references = index.references

        def batches(dataset=dataset, total=len(references)):
            for start in range(0, total, P1_BATCH_SIZE):
                yield torch.stack(
                    [
                        dataset[row].waveform
                        for row in range(start, min(start + P1_BATCH_SIZE, total))
                    ]
                )

        caches[partition] = materialize_p1_embedding_cache(
            Path(b4b_run_dir),
            batches(),
            partition=partition,
            stable_ids=[item.stable_id for item in references],
            labels=np.asarray(
                [int(item.binary_label) for item in references], dtype=np.int64
            ),
            subject_ids=[item.subject_id for item in references],
            cache_root=Path(cache_root),
        )
    return caches


def execute_p1_stage1(
    *,
    run_root: Path,
    cache_root: Path,
    feature_root: Path,
    source: Path,
    b4b_run_dir: Path,
    waveform_cache_root: Path,
    command: str = "cardiosentinel p1 run-stage1",
) -> dict[str, Any]:
    """Assemble every canonical input and run Stage P1-1.

    This is the single official orchestration the CLI invokes: it validates the
    caches, fits the train-only physiology transform, builds the stable-ID bound
    physiology, rebuilds the frozen challenge population, and runs both arms.
    No manual Python assembly step exists for the scientific run.
    """
    validate_p1_protocol()
    require_p1_runtime()
    require_clean_checkout()
    load_official_b4b_encoder(Path(b4b_run_dir))
    feature_receipt = validate_development_feature_integrity(Path(feature_root))
    source_receipt = validate_development_source_integrity(
        Path(source), feature_receipt
    )

    caches = prepare_p1_embedding_caches(
        cache_root=Path(cache_root),
        feature_root=Path(feature_root),
        source=Path(source),
        b4b_run_dir=Path(b4b_run_dir),
        waveform_cache_root=Path(waveform_cache_root),
    )
    train_cache, validation_cache = caches["train"], caches["validation"]
    raw_train = read_frozen_physiology(Path(feature_root), "train")
    raw_validation = read_frozen_physiology(Path(feature_root), "validation")

    transform = fit_physiology_transform(
        np.stack([raw_train[key] for key in train_cache.stable_ids]),
        partition="train",
        training_selection_sha256=TRAINING_SELECTION_SHA256,
    )
    train_physiology = build_physiology_bundle(
        partition="train",
        stable_ids=train_cache.stable_ids,
        raw_by_stable_id=raw_train,
        transform=transform,
    )
    validation_physiology = build_physiology_bundle(
        partition="validation",
        stable_ids=validation_cache.stable_ids,
        raw_by_stable_id=raw_validation,
        transform=transform,
    )
    # Challenge embeddings come from the DEDICATED validation-challenge path,
    # never from the primary validation cache: the two populations are disjoint.
    challenge_embeddings = prepare_p1_challenge_embeddings(
        Path(b4b_run_dir), Path(feature_root), Path(source)
    )
    challenge = prepare_p1_challenge_set(
        Path(feature_root),
        embeddings_by_stable_id=dict(
            zip(challenge_embeddings["stable_ids"], challenge_embeddings["embeddings"])
        ),
        raw_physiology_by_stable_id=raw_validation,
        transform=transform,
    )
    suite = run_p1_stage1_suite(
        run_root=Path(run_root),
        train_cache=train_cache,
        validation_cache=validation_cache,
        transform=transform,
        train_physiology=train_physiology,
        validation_physiology=validation_physiology,
        challenge=challenge,
        command=command,
    )
    suite["challenge_embedding_provenance"] = {
        "ordered_stable_id_sha256": challenge_embeddings["ordered_stable_id_sha256"],
        "embedding_content_sha256": challenge_embeddings["embedding_content_sha256"],
        "encoder_receipt": challenge_embeddings["encoder_receipt"],
        "waveform_reads": challenge_embeddings["waveform_reads"],
        "source": "dedicated_validation_challenge_locked_encoder_path",
    }
    suite["development_feature_integrity_sha256"] = feature_receipt[
        "development_feature_integrity_sha256"
    ]
    suite["development_source_integrity_sha256"] = source_receipt[
        "development_source_integrity_sha256"
    ]
    return suite


def read_frozen_physiology(
    feature_root: Path, partition: str
) -> dict[str, np.ndarray]:
    """Read frozen morphology_v1 rows by stable ID for one development partition."""
    evaluated = require_p1_partition(partition)
    columns = morphology_columns()
    values: dict[str, np.ndarray] = {}
    for path in sorted((Path(feature_root) / evaluated).glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            identifiers = archive["stable_ids"]
            features = archive["features"][:, columns]
            for index, key in enumerate(identifiers.tolist()):
                values[str(key)] = np.asarray(features[index], dtype=np.float64)
    if not values:
        raise P1ExecutionError(
            f"No frozen morphology_v1 rows found for {evaluated}."
        )
    return values


def p1_preflight(
    run_root: Path,
    cache_root: Path,
    *,
    b4b_run_dir: Path | None = None,
    feature_root: Path | None = None,
    source: Path | None = None,
) -> dict[str, Any]:
    """Read-only Stage P1-1 readiness gate. Creates nothing.

    This is a real gate: it reports `ready_for_canonical_p1_stage1` only when
    both canonical embedding caches exist AND fully validate. Absent caches are
    reported as requiring materialization, never as directly runnable.
    """
    validate_p1_protocol()
    validate_physiology_schema(PHYSIOLOGY_FEATURE_NAMES)
    environment, dependency_digest = require_p1_runtime()
    provenance = require_clean_checkout()

    claimed = {arm: (Path(run_root) / arm).exists() for arm in P1_ARM_ORDER}
    caches: dict[str, Any] = {}
    for partition in ("train", "validation"):
        present = (Path(cache_root) / partition / CACHE_MANIFEST_NAME).exists()
        entry: dict[str, Any] = {"present": present, "validated": False}
        if present:
            try:
                cache = load_p1_embedding_cache(Path(cache_root), partition)
                entry["validated"] = True
                entry["population"] = cache.manifest["population"]
                entry["cache_sha256"] = cache.manifest["cache_sha256"]
            except Exception as error:  # surfaced, never silently ignored
                entry["error"] = f"{type(error).__name__}: {error}"
        caches[partition] = entry

    encoder_state: dict[str, Any] = {"validated": False}
    if b4b_run_dir is not None:
        try:
            lock = validate_locked_model(Path(b4b_run_dir), official_model="B4-B")
            encoder_state = {
                "validated": True,
                "experiment_lock_sha256": lock["experiment_lock_sha256"],
                "checkpoint_sha256": lock["checkpoint_sha256"],
                "test": lock["test"],
                "matches_selected_encoder": (
                    lock["experiment_lock_sha256"] == B4B_EXPERIMENT_LOCK_SHA256
                    and lock["checkpoint_sha256"] == B4B_CHECKPOINT_SHA256
                    and lock["test"] is None
                ),
            }
        except Exception as error:
            encoder_state = {
                "validated": False,
                "error": f"{type(error).__name__}: {error}",
            }

    challenge_state: dict[str, Any] = {"validated": False}
    if feature_root is not None:
        try:
            index = build_validation_challenge_index(Path(feature_root))
            challenge_state = {
                "validated": index.selection_sha256 == CHALLENGE_SELECTION_SHA256,
                "selection_sha256": index.selection_sha256,
                "counts": index.counts,
            }
        except Exception as error:
            challenge_state = {
                "validated": False,
                "error": f"{type(error).__name__}: {error}",
            }

    integrity: dict[str, Any] = {"validated": False}
    if feature_root is not None and source is not None:
        try:
            feature_receipt = validate_development_feature_integrity(Path(feature_root))
            source_receipt = validate_development_source_integrity(
                Path(source), feature_receipt
            )
            integrity = {
                "validated": True,
                "development_feature_integrity_sha256": feature_receipt[
                    "development_feature_integrity_sha256"
                ],
                "development_source_integrity_sha256": source_receipt[
                    "development_source_integrity_sha256"
                ],
            }
        except Exception as error:
            integrity = {
                "validated": False,
                "error": f"{type(error).__name__}: {error}",
            }

    test_artifacts = sorted(
        path.name
        for path in Path(REPOSITORY_ROOT).glob("cardiosentinel-runs/**/TEST_*")
    )
    caches_ready = all(entry["validated"] for entry in caches.values())
    encoder_ready = bool(encoder_state.get("matches_selected_encoder"))
    challenge_ready = bool(challenge_state.get("validated"))

    if any(claimed.values()):
        status = "attempt_already_claimed"
    elif test_artifacts:
        status = "test_artifact_present_human_review_required"
    elif not caches_ready:
        status = "embedding_cache_materialization_required"
    elif b4b_run_dir is not None and not encoder_ready:
        status = "selected_encoder_not_verified"
    elif feature_root is not None and not challenge_ready:
        status = "challenge_population_not_verified"
    elif source is not None and not integrity.get("validated"):
        status = "development_integrity_not_verified"
    else:
        status = "ready_for_canonical_p1_stage1"

    report = {
        "selected_encoder": encoder_state,
        "challenge_population": challenge_state,
        "development_integrity": integrity,
        "test_artifacts_present": test_artifacts,
        "embedding_caches_ready": caches_ready,
        "command": "cardiosentinel p1 preflight",
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "physiology_schema_sha256": MORPHOLOGY_SCHEMA_SHA256,
        "expected_populations": EXPECTED_POPULATIONS,
        "challenge_selection_sha256": CHALLENGE_SELECTION_SHA256,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "environment_dependency_digest": dependency_digest,
        "runtime_environment": environment,
        "arm_order": list(P1_ARM_ORDER),
        "canonical_arm_claimed": claimed,
        "embedding_cache": caches,
        "partitions_permitted": ["train", "validation"],
        "test_partition_access": None,
        "models_constructed": 0,
        "artifacts_created": 0,
        "status": status,
    }
    report["preflight_sha256"] = canonical_sha256(report)
    return report


__all__ = [
    "CACHE_MANIFEST_NAME",
    "P1ChallengeSet",
    "P1PhysiologyBundle",
    "build_physiology_bundle",
    "load_official_b4b_encoder",
    "execute_p1_stage1",
    "prepare_p1_challenge_embeddings",
    "prepare_p1_challenge_set",
    "prepare_p1_embedding_caches",
    "read_frozen_physiology",
    "require_aligned_physiology",
    "CHALLENGE_SELECTION_SHA256",
    "EXPECTED_POPULATIONS",
    "P1EmbeddingCache",
    "P1ExecutionError",
    "build_deterministic_p1_head",
    "build_embedding_cache_manifest",
    "build_p1_lock",
    "embedding_content_digest",
    "load_p1_embedding_cache",
    "materialize_p1_embedding_cache",
    "ordered_stable_id_digest",
    "p1_challenge_evidence",
    "p1_epoch_order",
    "p1_preflight",
    "p1_validation_evidence",
    "require_clean_checkout",
    "require_p1_runtime",
    "run_p1_arm",
    "run_p1_stage1_suite",
    "select_p1_threshold",
    "train_p1_arm",
    "validate_p1_lock",
]
