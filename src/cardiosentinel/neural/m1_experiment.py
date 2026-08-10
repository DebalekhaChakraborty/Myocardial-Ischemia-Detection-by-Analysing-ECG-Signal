"""Canonical M1 Stage-1 execution path: M1S / M1L / M1D against frozen P1-B.

This is the official scientific route promised by
`docs/M1_DUAL_MEMORY_PROTOCOL_V1.md`. It owns full-stream representation
assembly, the train-only distance standardizer, the immutable stream memory
cache, deterministic head training over precomputed features, validation and
development challenge evidence, the immutable arm locks, and the Stage-1 suite
that requires all three arms.

Everything upstream is frozen: the B4-B encoder is loaded from
`model_selected.pt` and never fine-tuned, and the P1 physiology transform is
read from the canonical P1-B run and never refitted.

The memory trajectory is built over the FULL development stream. Labels select
supervised training membership, primary validation membership and challenge
reporting strata only — they never decide whether a window exists in history.

> This update policy is intentionally NOT contamination-safe. M2 is required
> before any safe-adaptation or deployment-safe personalization claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from cardiosentinel.baseline.cache import read_json, write_json_atomic
from cardiosentinel.baseline.metrics import binary_metrics
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.integrity import (
    canonical_sha256,
    validate_development_feature_integrity,
    validate_development_source_integrity,
)
from cardiosentinel.neural.metadata import B4WindowReference, load_b4_references
from cardiosentinel.neural.p1_experiment import (
    CHALLENGE_METRICS_NAME,
    CHALLENGE_SELECTION_SHA256,
    EPOCH_HISTORY_NAME,
    EXPECTED_POPULATIONS,
    EXPERIMENT_LOCK_NAME,
    FROZEN_DEPENDENCY_DIGEST,
    LOCK_STATUS,
    PHYSIOLOGY_TRANSFORM_NAME,
    SCORE_SEMANTICS,
    SELECTED_MODEL_NAME,
    THRESHOLD_RULE,
    VALIDATION_METRICS_NAME,
    VALIDATION_THRESHOLD_NAME,
    build_physiology_bundle,
    embedding_content_digest,
    load_official_b4b_encoder,
    load_p1_embedding_cache,
    ordered_stable_id_digest,
    p1_challenge_evidence,
    p1_epoch_order,
    p1_validation_evidence,
    read_frozen_physiology,
    require_clean_checkout,
    require_p1_runtime,
    select_p1_threshold,
)
from cardiosentinel.neural.patient_memory import (
    ALPHA_LONG,
    ALPHA_SHORT,
    ATTEMPT_STATUS_COMPLETE,
    ATTEMPT_STATUS_STARTED,
    COLD_START_BINS,
    CONTAMINATION_SAFE,
    GLOBAL_CONTROL_EXPERIMENT_ID,
    M1_ARM_FEATURES,
    M1_EXPERIMENT_IDS,
    M1_PROTOCOL_SHA256,
    P1_RETENTION_DECISION_SHA256,
    REPRESENTATION_DIM,
    STANDARDIZER_NAME,
    STREAM_CACHE_ARRAY_NAME,
    STREAM_CACHE_CLAIM_NAME,
    STREAM_CACHE_MANIFEST_NAME,
    UPDATE_POLICY,
    M1DistanceStandardizer,
    M1MemoryError,
    M1StreamMemory,
    build_causal_streams,
    build_deterministic_m1_head,
    claim_m1_run_directory,
    fit_distance_standardizer,
    generate_stream_memory,
    m1_alpha_identity,
    m1_arm_features,
    m1_boundary_statement,
    m1_head_identity,
    m1_training_configuration,
    ordered_chronology_digest,
    record_m1_failure,
    require_m1_experiment,
    resolve_m1_run_dir,
    select_rows,
    validate_m1_protocol,
    write_m1_status,
)
from cardiosentinel.neural.physiology_fusion import (
    B4B_CHECKPOINT_SHA256,
    B4B_EXPERIMENT_LOCK_SHA256,
    EMBEDDING_DIM,
    EMBEDDING_TAP,
    MORPHOLOGY_SCHEMA_SHA256,
    P1_BATCH_SIZE,
    P1_EARLY_STOPPING_PATIENCE,
    P1_MAX_EPOCHS,
    P1_PROTOCOL_SHA256,
    P1B_EXPERIMENT_ID,
    PHYSIOLOGY_DIM,
    PhysiologyTransform,
    extract_frozen_embeddings,
    require_p1_partition,
)
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    B4_SPLIT_SHA256,
    FEATURE_CORPUS_SHA256,
)
from cardiosentinel.neural.resource_benchmark import validate_locked_model
from cardiosentinel.neural.training import CheckpointTracker
from cardiosentinel.neural.validation_challenge import build_validation_challenge_index

M1_STAGE1_RESULT_NAME: Final = "M1_STAGE1_RESULTS.json"
RUN_MANIFEST_NAME: Final = "RUN_MANIFEST.json"
MEMORY_FEATURE_NAME: Final = "MEMORY_FEATURES.json"
M1_ARM_ORDER: Final = M1_EXPERIMENT_IDS
PRIMARY_AUDIT_ROWS: Final = 64

M1_SUITE_STATUS: Final = "locked_m1_development_result"
M1_LOCK_STATUS: Final = LOCK_STATUS.replace("p1", "m1")


# --------------------------------------------------------------------------
# Full-stream representation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M1StreamRepresentation:
    """Fused 146-d representations for every causal window of one partition."""

    partition: str
    stable_ids: tuple[str, ...]
    matrix: np.ndarray
    streams: dict[tuple[str, int], tuple[B4WindowReference, ...]]
    reused_primary_rows: int
    newly_extracted_rows: int
    primary_audit: dict[str, Any]

    def by_stable_id(self) -> dict[str, np.ndarray]:
        return {key: self.matrix[i] for i, key in enumerate(self.stable_ids)}


def _verified_p1b_artifact(p1b_run_dir: Path, name: str) -> dict[str, Any]:
    """Read one P1-B artifact and prove it against the lock's own digest."""
    lock = read_json(Path(p1b_run_dir) / EXPERIMENT_LOCK_NAME)
    expected = dict(lock.get("artifact_sha256") or {}).get(name)
    path = Path(p1b_run_dir) / name
    if expected is None or not path.is_file():
        raise M1MemoryError(f"The P1-B lock does not bind an artifact named {name}.")
    if sha256_file(path) != expected:
        raise M1MemoryError(f"P1-B artifact {name} does not match its lock digest.")
    return read_json(path)


def load_frozen_physiology_transform(p1b_run_dir: Path) -> PhysiologyTransform:
    """Read the frozen P1 physiology transform. M1 never refits physiology.

    The transform lives in its own artifact; the lock binds only its digest, so
    both the artifact hash and the reconstructed canonical digest are checked
    before the transform is allowed to touch an M1 representation.
    """
    lock = read_json(Path(p1b_run_dir) / EXPERIMENT_LOCK_NAME)
    payload = _verified_p1b_artifact(p1b_run_dir, PHYSIOLOGY_TRANSFORM_NAME)
    transform = PhysiologyTransform(
        feature_names=tuple(payload["feature_names"]),
        medians=tuple(float(v) for v in payload["medians"]),
        means=tuple(float(v) for v in payload["means"]),
        scales=tuple(float(v) for v in payload["scales"]),
        zero_variance_features=tuple(payload["zero_variance_features"]),
        imputed_counts=dict(payload["imputed_counts"]),
        fitted_rows=int(payload["fitted_rows"]),
        schema_sha256=str(payload["schema_sha256"]),
        training_selection_sha256=payload.get("training_selection_sha256"),
    )
    recomputed = transform.as_dict()["transform_sha256"]
    if recomputed != payload.get("transform_sha256"):
        raise M1MemoryError(
            "The reconstructed physiology transform does not reproduce its "
            "frozen digest."
        )
    if recomputed != lock.get("physiology_transform_sha256"):
        raise M1MemoryError(
            "The P1-B physiology transform differs from the digest recorded in "
            "its immutable lock."
        )
    return transform


def load_frozen_control_evidence(p1b_run_dir: Path) -> dict[str, Any]:
    """Read the frozen P1-B control evidence. M1 never retrains the control."""
    return {
        "validation_evidence": _verified_p1b_artifact(
            p1b_run_dir, VALIDATION_METRICS_NAME
        ),
        "challenge_evidence": _verified_p1b_artifact(
            p1b_run_dir, CHALLENGE_METRICS_NAME
        ),
    }


def _fuse(embedding: np.ndarray, physiology: np.ndarray) -> np.ndarray:
    fused = np.concatenate([embedding, physiology], axis=1).astype(np.float32)
    if fused.shape[1] != REPRESENTATION_DIM:
        raise M1MemoryError(f"Fused representation must be {REPRESENTATION_DIM}-d.")
    if not np.all(np.isfinite(fused)):
        raise M1MemoryError(
            "A non-finite fused representation was produced despite the frozen "
            "P1 transformation. M1 refuses rather than skipping the window."
        )
    return fused


def prepare_stream_representations(
    partition: str,
    *,
    cache_root: Path,
    feature_root: Path,
    b4b_run_dir: Path,
    p1b_run_dir: Path,
    waveform_batches_for=None,
) -> M1StreamRepresentation:
    """Assemble fused 146-d representations for the full development stream.

    Primary rows reuse the canonical frozen P1 embedding cache verbatim. Any
    additional full-stream row is extracted with the same locked B4-B encoder.
    The overlap is proven rather than assumed: the extra set is required to be
    disjoint from the cache, and a deterministic audit subset of primary rows is
    re-extracted and required to match the cache bit-for-bit.
    """
    evaluated = require_p1_partition(partition)
    transform = load_frozen_physiology_transform(p1b_run_dir)
    cache = load_p1_embedding_cache(Path(cache_root), evaluated)
    references = load_b4_references(Path(feature_root), evaluated, primary_only=False)
    streams = build_causal_streams(references)

    raw_physiology = read_frozen_physiology(Path(feature_root), evaluated)
    primary_embeddings = {
        key: cache.embeddings[i] for i, key in enumerate(cache.stable_ids)
    }
    ordered = [
        reference for key in sorted(streams) for reference in streams[key]
    ]
    extra = [item for item in ordered if item.stable_id not in primary_embeddings]

    encoder = None
    extracted: dict[str, np.ndarray] = {}
    if extra:
        if waveform_batches_for is None:
            raise M1MemoryError(
                f"{len(extra)} full-stream rows are outside the frozen P1 "
                "embedding cache and require waveform access."
            )
        encoder = load_official_b4b_encoder(Path(b4b_run_dir))
        for identifiers, batch in waveform_batches_for(
            evaluated, tuple(item.stable_id for item in extra)
        ):
            embeddings, _ = extract_frozen_embeddings(encoder, batch)
            block = embeddings.to(torch.float32).numpy()
            for offset, key in enumerate(identifiers):
                if str(key) in primary_embeddings:
                    raise M1MemoryError(
                        f"Row {key} was re-extracted although it is already in "
                        "the frozen P1 cache; the overlap must be reused, not "
                        "duplicated."
                    )
                extracted[str(key)] = block[offset]

    audit: dict[str, Any] = {
        "primary_rows_reused": len(primary_embeddings),
        "rows_newly_extracted": len(extracted),
        "extra_disjoint_from_primary_cache": True,
        "re_extracted_primary_rows": 0,
        "re_extracted_primary_bitwise_identical": None,
    }
    if encoder is not None and waveform_batches_for is not None:
        sample = [
            key
            for index, key in enumerate(cache.stable_ids)
            if index % max(len(cache.stable_ids) // PRIMARY_AUDIT_ROWS, 1) == 0
        ][:PRIMARY_AUDIT_ROWS]
        identical = True
        checked = 0
        for identifiers, batch in waveform_batches_for(evaluated, tuple(sample)):
            embeddings, _ = extract_frozen_embeddings(encoder, batch)
            block = embeddings.to(torch.float32).numpy()
            for offset, key in enumerate(identifiers):
                checked += 1
                if not np.array_equal(block[offset], primary_embeddings[str(key)]):
                    identical = False
        audit["re_extracted_primary_rows"] = checked
        audit["re_extracted_primary_bitwise_identical"] = identical
        if checked and not identical:
            raise M1MemoryError(
                "Re-extracted primary embeddings differ from the frozen P1 "
                "cache; the overlap identity is not exact."
            )

    stable_ids = tuple(item.stable_id for item in ordered)
    absent = [
        key
        for key in stable_ids
        if key not in primary_embeddings and key not in extracted
    ]
    if absent:
        raise M1MemoryError(
            f"{len(absent)} full-stream rows have no frozen B4-B embedding; the "
            f"first is {absent[0]!r}."
        )
    embeddings = np.stack(
        [
            np.asarray(
                primary_embeddings[key] if key in primary_embeddings
                else extracted[key],
                dtype=np.float32,
            )
            for key in stable_ids
        ]
    )
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise M1MemoryError(f"Full-stream embeddings must be {EMBEDDING_DIM}-d.")
    bundle = build_physiology_bundle(
        partition=evaluated,
        stable_ids=stable_ids,
        raw_by_stable_id=raw_physiology,
        transform=transform,
    )
    return M1StreamRepresentation(
        partition=evaluated,
        stable_ids=stable_ids,
        matrix=_fuse(embeddings, bundle.values),
        streams=streams,
        reused_primary_rows=len(primary_embeddings),
        newly_extracted_rows=len(extracted),
        primary_audit=audit,
    )


# --------------------------------------------------------------------------
# Distance standardizer artifact
# --------------------------------------------------------------------------


def build_distance_standardizer(
    representation: M1StreamRepresentation,
    *,
    primary_train_stable_ids: Sequence[str],
) -> M1DistanceStandardizer:
    """Fit the frozen distance space on primary TRAIN rows only."""
    if representation.partition != "train":
        raise M1MemoryError("The M1 distance standardizer is fitted on train only.")
    expected = EXPECTED_POPULATIONS["train"]["total"]
    if len(primary_train_stable_ids) != expected:
        raise M1MemoryError(
            f"The standardizer must be fitted on the frozen {expected} primary "
            f"TRAIN rows, received {len(primary_train_stable_ids)}."
        )
    lookup = representation.by_stable_id()
    matrix = np.stack(
        [np.asarray(lookup[key], dtype=np.float64) for key in primary_train_stable_ids]
    )
    return fit_distance_standardizer(
        matrix,
        partition="train",
        input_identities={
            "m1_protocol_sha256": M1_PROTOCOL_SHA256,
            "p1_protocol_sha256": P1_PROTOCOL_SHA256,
            "p1b_experiment_lock_sha256": None,
            "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
            "ordered_stable_id_sha256": ordered_stable_id_digest(
                primary_train_stable_ids
            ),
            "representation_content_sha256": embedding_content_digest(matrix),
        },
    )


# --------------------------------------------------------------------------
# Immutable stream cache
# --------------------------------------------------------------------------


def build_stream_cache_manifest(
    memory: M1StreamMemory,
    representation: M1StreamRepresentation,
    *,
    standardizer_sha256: str,
    p1_stage1_suite_sha256: str,
    p1b_lock_sha256: str,
    physiology_transform_sha256: str,
    embedding_cache_sha256: str,
    git_sha: str,
    git_dirty: bool,
    dependency_digest: str,
) -> dict[str, Any]:
    """Bind every identity the stream cache depends on."""
    if memory.stable_ids != representation.stable_ids:
        raise M1MemoryError("Stream memory and representation rows are misaligned.")
    lookup = representation.by_stable_id()
    matrix = np.stack([lookup[key] for key in memory.stable_ids])
    manifest: dict[str, Any] = {
        "artifact_class": "m1_full_stream_memory_cache",
        "partition": memory.partition,
        "m1_protocol_sha256": M1_PROTOCOL_SHA256,
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "p1_retention_decision_sha256": P1_RETENTION_DECISION_SHA256,
        "p1_stage1_suite_sha256": p1_stage1_suite_sha256,
        "p1b_experiment_lock_sha256": p1b_lock_sha256,
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "embedding_tap": EMBEDDING_TAP,
        "physiology_transform_sha256": physiology_transform_sha256,
        "physiology_schema_sha256": MORPHOLOGY_SCHEMA_SHA256,
        "p1_embedding_cache_sha256": embedding_cache_sha256,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "distance_standardizer_sha256": standardizer_sha256,
        "representation_dim": REPRESENTATION_DIM,
        "full_stream_row_count": int(len(memory.stable_ids)),
        "stream_count": int(len(memory.streams)),
        "record_ids": sorted({key[0] for key in memory.streams}),
        "channel_indices": sorted({int(key[1]) for key in memory.streams}),
        "ordered_stable_id_sha256": ordered_stable_id_digest(memory.stable_ids),
        "ordered_chronology_sha256": memory.chronology_sha256,
        "representation_content_sha256": embedding_content_digest(matrix),
        "d_short_content_sha256": embedding_content_digest(memory.d_short),
        "d_long_content_sha256": embedding_content_digest(memory.d_long),
        "history_count_sha256": embedding_content_digest(
            np.stack([memory.past_observed_count, memory.past_update_count], axis=1)
        ),
        "primary_rows_reused": representation.reused_primary_rows,
        "rows_newly_extracted": representation.newly_extracted_rows,
        "primary_overlap_audit": dict(representation.primary_audit),
        "label_independent_history": True,
        "update_policy": UPDATE_POLICY,
        "contamination_safe": CONTAMINATION_SAFE,
        "alpha_short": ALPHA_SHORT,
        "alpha_long": ALPHA_LONG,
        "memory_features": list(("d_short", "d_long")),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "environment_dependency_digest": dependency_digest,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    return manifest


def _claim_stream_cache(directory: Path, partition: str) -> None:
    """Claim the stream-cache directory. A partial cache stops for review."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise M1MemoryError(
            f"An M1 stream cache already exists at {directory}. A complete "
            "cache must be reused and a partial cache requires documented "
            "human review; it is never overwritten or automatically repaired."
        ) from error
    write_json_atomic(
        directory / STREAM_CACHE_CLAIM_NAME,
        {
            "partition": partition,
            "claim": "m1_full_stream_memory_cache",
            "m1_protocol_sha256": M1_PROTOCOL_SHA256,
            "overwrite_permitted": False,
            "automatic_repair_permitted": False,
        },
    )


def materialize_stream_cache(
    memory: M1StreamMemory,
    representation: M1StreamRepresentation,
    *,
    cache_root: Path,
    manifest_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the immutable full-stream memory cache for one partition."""
    directory = Path(cache_root) / memory.partition
    _claim_stream_cache(directory, memory.partition)
    lookup = representation.by_stable_id()
    matrix = np.stack([lookup[key] for key in memory.stable_ids])
    manifest = build_stream_cache_manifest(
        memory, representation, **dict(manifest_fields)
    )
    array_path = directory / STREAM_CACHE_ARRAY_NAME
    temporary = directory / f".{STREAM_CACHE_ARRAY_NAME}.tmp.npz"
    np.savez_compressed(
        temporary,
        stable_id=np.asarray(memory.stable_ids, dtype=np.str_),
        record_id=np.asarray(memory.record_ids, dtype=np.str_),
        channel_index=memory.channel_indices,
        start_sample=memory.start_samples,
        representation=matrix,
        d_short=memory.d_short,
        d_long=memory.d_long,
        past_observed_count=memory.past_observed_count,
        past_update_count=memory.past_update_count,
        prototype_disagreement=memory.prototype_disagreement,
        recording_age_seconds=memory.recording_age_seconds,
        cold_start_bin=np.asarray(memory.cold_start_bins, dtype=np.str_),
    )
    temporary.replace(array_path)
    manifest = dict(manifest)
    manifest.pop("stream_cache_sha256", None)
    manifest["artifact"] = STREAM_CACHE_ARRAY_NAME
    manifest["artifact_sha256"] = sha256_file(array_path)
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    write_json_atomic(directory / STREAM_CACHE_MANIFEST_NAME, manifest)
    return manifest


def load_stream_cache(cache_root: Path, partition: str) -> tuple[
    M1StreamMemory, np.ndarray, dict[str, Any]
]:
    """Load and fully re-verify a materialized M1 stream cache."""
    evaluated = require_p1_partition(partition)
    directory = Path(cache_root) / evaluated
    manifest_path = directory / STREAM_CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        if directory.exists():
            raise M1MemoryError(
                f"A partial M1 stream cache exists at {directory} without a "
                "manifest. This requires documented human review and is never "
                "overwritten or automatically repaired."
            )
        raise M1MemoryError(f"No M1 stream cache manifest at {manifest_path}.")
    manifest = read_json(manifest_path)
    recorded = manifest.get("stream_cache_sha256")
    body = {k: v for k, v in manifest.items() if k != "stream_cache_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise M1MemoryError("M1 stream cache manifest failed digest validation.")
    for field, expected in (
        ("m1_protocol_sha256", M1_PROTOCOL_SHA256),
        ("p1_protocol_sha256", P1_PROTOCOL_SHA256),
        ("p1_retention_decision_sha256", P1_RETENTION_DECISION_SHA256),
        ("encoder_checkpoint_sha256", B4B_CHECKPOINT_SHA256),
        ("encoder_experiment_lock_sha256", B4B_EXPERIMENT_LOCK_SHA256),
        ("split_sha256", B4_SPLIT_SHA256),
        ("feature_corpus_sha256", FEATURE_CORPUS_SHA256),
        ("update_policy", UPDATE_POLICY),
        ("alpha_short", ALPHA_SHORT),
        ("alpha_long", ALPHA_LONG),
        ("partition", evaluated),
    ):
        if manifest.get(field) != expected:
            raise M1MemoryError(
                f"M1 stream cache binds {field}={manifest.get(field)!r}, "
                f"expected {expected!r}."
            )
    if manifest.get("test_accessed") is not False:
        raise M1MemoryError("M1 stream cache does not record test_accessed=false.")
    if manifest.get("git_dirty") is not False:
        raise M1MemoryError("M1 stream cache was built from a dirty checkout.")
    if manifest.get("environment_dependency_digest") != FROZEN_DEPENDENCY_DIGEST:
        raise M1MemoryError(
            "M1 stream cache does not bind the frozen dependency digest."
        )

    array_path = directory / str(manifest["artifact"])
    if sha256_file(array_path) != manifest["artifact_sha256"]:
        raise M1MemoryError("M1 stream cache artifact SHA-256 does not match.")
    with np.load(array_path, allow_pickle=False) as archive:
        memory = M1StreamMemory(
            partition=evaluated,
            stable_ids=tuple(archive["stable_id"].tolist()),
            record_ids=tuple(archive["record_id"].tolist()),
            channel_indices=np.asarray(archive["channel_index"], dtype=np.int64),
            start_samples=np.asarray(archive["start_sample"], dtype=np.int64),
            d_short=np.asarray(archive["d_short"], dtype=np.float64),
            d_long=np.asarray(archive["d_long"], dtype=np.float64),
            past_observed_count=np.asarray(
                archive["past_observed_count"], dtype=np.int64
            ),
            past_update_count=np.asarray(
                archive["past_update_count"], dtype=np.int64
            ),
            prototype_disagreement=np.asarray(
                archive["prototype_disagreement"], dtype=np.float64
            ),
            recording_age_seconds=np.asarray(
                archive["recording_age_seconds"], dtype=np.float64
            ),
            cold_start_bins=tuple(archive["cold_start_bin"].tolist()),
            streams=tuple(
                sorted(
                    {
                        (str(record), int(channel))
                        for record, channel in zip(
                            archive["record_id"].tolist(),
                            archive["channel_index"].tolist(),
                        )
                    }
                )
            ),
            chronology_sha256=str(manifest["ordered_chronology_sha256"]),
        )
        matrix = np.asarray(archive["representation"], dtype=np.float32)

    if ordered_stable_id_digest(memory.stable_ids) != manifest[
        "ordered_stable_id_sha256"
    ]:
        raise M1MemoryError("M1 stream cache row order does not match its identity.")
    for field, values in (
        ("representation_content_sha256", matrix),
        ("d_short_content_sha256", memory.d_short),
        ("d_long_content_sha256", memory.d_long),
    ):
        if embedding_content_digest(values) != manifest[field]:
            raise M1MemoryError(f"M1 stream cache {field} does not match.")
    if int(manifest["full_stream_row_count"]) != len(memory.stable_ids):
        raise M1MemoryError("M1 stream cache row count does not match.")
    return memory, matrix, manifest


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _scores(head: nn.Module, features: np.ndarray) -> np.ndarray:
    head.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], P1_BATCH_SIZE):
            chunk = torch.from_numpy(features[start : start + P1_BATCH_SIZE])
            outputs.append(torch.sigmoid(head(chunk)).to(torch.float64).numpy())
    scores = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise M1MemoryError("An M1 head produced a non-finite prediction.")
    return scores


def train_m1_arm(
    experiment_id: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    max_epochs: int = P1_MAX_EPOCHS,
) -> dict[str, Any]:
    """Train one M1 arm under the P1 training contract, reused unchanged.

    Only the small head trains. The B4-B encoder does not execute here, the
    physiology transform is frozen, and the memory cache is precomputed.
    """
    require_m1_experiment(experiment_id)
    head = build_deterministic_m1_head(experiment_id)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=m1_training_configuration()["learning_rate"],
        weight_decay=m1_training_configuration()["weight_decay"],
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
    # The reviewed B4/P1 tracker: checkpoint saving is a strict numerical
    # maximum while patience resets only beyond the delta. The two remain
    # separate quantities.
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
        if not np.isfinite(mean_loss) or not np.isfinite(auprc):
            raise M1MemoryError(f"Non-finite M1 training state at epoch {epoch}.")
        for name, parameter in head.named_parameters():
            if not torch.isfinite(parameter).all():
                raise M1MemoryError(
                    f"Non-finite M1 parameter {name} at epoch {epoch}."
                )

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
        raise M1MemoryError("M1 training selected no checkpoint.")
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


# --------------------------------------------------------------------------
# Cold-start and personalization evidence
# --------------------------------------------------------------------------


def cold_start_evidence(
    memory: M1StreamMemory,
    rows: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Supporting evidence stratified by the frozen recording-age bins."""
    bins = np.asarray(memory.cold_start_bins, dtype=np.str_)[rows]
    evidence: dict[str, Any] = {}
    for name, _low, _high in COLD_START_BINS:
        mask = bins == name
        subset = int(np.sum(mask))
        entry: dict[str, Any] = {
            "window_count": subset,
            "evidence_status": "supporting",
        }
        if subset:
            entry["metrics"] = binary_metrics(labels[mask], scores[mask], threshold)
        evidence[name] = entry
    return evidence


def memory_descriptives(memory: M1StreamMemory, rows: np.ndarray) -> dict[str, Any]:
    """Descriptive memory statistics. Disagreement is never a model input."""
    return {
        "window_count": int(rows.shape[0]),
        "finite_representation_rate": 1.0,
        "past_observed_count_mean": float(
            np.mean(memory.past_observed_count[rows])
        ),
        "past_update_count_mean": float(np.mean(memory.past_update_count[rows])),
        "past_counts_agree": bool(
            np.array_equal(
                memory.past_observed_count[rows], memory.past_update_count[rows]
            )
        ),
        "d_short_mean": float(np.mean(memory.d_short[rows])),
        "d_long_mean": float(np.mean(memory.d_long[rows])),
        "prototype_disagreement_mean": float(
            np.mean(memory.prototype_disagreement[rows])
        ),
        "prototype_disagreement_is_model_input": False,
    }


# --------------------------------------------------------------------------
# Immutable lock
# --------------------------------------------------------------------------


def build_m1_lock(
    experiment_id: str,
    *,
    head: nn.Module,
    result: dict[str, Any],
    threshold: float,
    validation_evidence: dict[str, Any],
    challenge_evidence: dict[str, Any],
    cold_start: dict[str, Any],
    descriptives: dict[str, Any],
    train_cache: Mapping[str, Any],
    validation_cache: Mapping[str, Any],
    standardizer: Mapping[str, Any],
    artifact_hashes: Mapping[str, Any],
    provenance: Mapping[str, Any],
    environment: Mapping[str, Any],
    dependency_digest: str,
    p1_stage1_suite_sha256: str,
    p1b_lock_sha256: str,
) -> dict[str, Any]:
    """Assemble the immutable lock for one M1 arm."""
    require_m1_experiment(experiment_id)
    lock: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": M1_LOCK_STATUS,
        "phase": "phase5_m1_dual_memory",
        "m1_protocol_sha256": M1_PROTOCOL_SHA256,
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "p1_retention_decision_sha256": P1_RETENTION_DECISION_SHA256,
        "p1_stage1_suite_sha256": p1_stage1_suite_sha256,
        "global_control_experiment_id": GLOBAL_CONTROL_EXPERIMENT_ID,
        "global_control_lock_sha256": p1b_lock_sha256,
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "embedding_tap": EMBEDDING_TAP,
        "representation_dim": REPRESENTATION_DIM,
        "memory_features": list(M1_ARM_FEATURES[experiment_id]),
        "head": m1_head_identity(experiment_id, head),
        "training": m1_training_configuration(),
        "memory": m1_alpha_identity(),
        "boundary": m1_boundary_statement(),
        "selected_epoch": result["selected_epoch"],
        "selected_validation_auprc": result["selected_validation_auprc"],
        "completed_epochs": result["completed_epochs"],
        "stop_reason": result["stop_reason"],
        "threshold": threshold,
        "threshold_rule": THRESHOLD_RULE,
        "score_semantics": SCORE_SEMANTICS,
        "validation_evidence": validation_evidence,
        "challenge_evidence": challenge_evidence,
        "cold_start_evidence": cold_start,
        "memory_descriptives": descriptives,
        "train_stream_cache": dict(train_cache),
        "validation_stream_cache": dict(validation_cache),
        "distance_standardizer": dict(standardizer),
        "artifact_sha256": dict(artifact_hashes),
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "environment": dict(environment),
        "environment_dependency_digest": dependency_digest,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "test_metrics": None,
        "repeat_attempt_permitted": False,
        "automatic_retry_performed": False,
    }
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    return lock


def validate_m1_lock(run_dir: Path) -> dict[str, Any]:
    """Re-verify one immutable M1 arm lock."""
    path = Path(run_dir) / EXPERIMENT_LOCK_NAME
    if not path.is_file():
        raise M1MemoryError(f"No M1 experiment lock at {path}.")
    lock = read_json(path)
    recorded = lock.get("experiment_lock_sha256")
    body = {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise M1MemoryError("M1 experiment lock failed digest validation.")
    require_m1_experiment(str(lock.get("experiment_id")))
    if lock.get("m1_protocol_sha256") != M1_PROTOCOL_SHA256:
        raise M1MemoryError("M1 lock does not bind the frozen M1 protocol.")
    if lock.get("test_accessed") is not False or lock.get("test_metrics") is not None:
        raise M1MemoryError("M1 lock records test access.")
    for name, digest in dict(lock.get("artifact_sha256") or {}).items():
        artifact = Path(run_dir) / name
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise M1MemoryError(f"M1 artifact {name} does not match its lock digest.")
    return lock


# --------------------------------------------------------------------------
# Stage-1 suite
# --------------------------------------------------------------------------


def _arm_result_summary(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": lock["experiment_id"],
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "memory_features": list(lock["memory_features"]),
        "head": lock["head"],
        "threshold": lock["threshold"],
        "validation_evidence": lock["validation_evidence"],
        "challenge_evidence": lock["challenge_evidence"],
        "cold_start_evidence": lock["cold_start_evidence"],
        "memory_descriptives": lock["memory_descriptives"],
    }


def build_m1_stage1_result(
    arm_locks: Mapping[str, Mapping[str, Any]],
    *,
    control: Mapping[str, Any],
    stream_caches: Mapping[str, Any],
    standardizer: Mapping[str, Any],
    provenance: Mapping[str, Any],
    environment: Mapping[str, Any],
    dependency_digest: str,
) -> dict[str, Any]:
    """Assemble the combined M1 Stage-1 result. All three arms are required."""
    missing = [arm for arm in M1_ARM_ORDER if arm not in arm_locks]
    if missing:
        raise M1MemoryError(f"M1 Stage-1 requires all three arms; missing {missing}.")
    payload: dict[str, Any] = {
        "result_class": M1_SUITE_STATUS,
        "phase": "phase5_m1_dual_memory",
        "m1_protocol_sha256": M1_PROTOCOL_SHA256,
        "p1_retention_decision_sha256": P1_RETENTION_DECISION_SHA256,
        "arm_order": list(M1_ARM_ORDER),
        "arm_results": {
            arm: _arm_result_summary(arm_locks[arm]) for arm in M1_ARM_ORDER
        },
        "global_control": dict(control),
        "stream_caches": dict(stream_caches),
        "distance_standardizer": dict(standardizer),
        "memory": m1_alpha_identity(),
        "boundary": m1_boundary_statement(),
        "primary_metric": "pooled_validation_auprc",
        "comparison_rule": (
            "bounded Pareto judgement across the frozen P1-B global control, "
            "M1S, M1L and M1D; no weighted score is computed and dual memory "
            "is not automatically selected"
        ),
        "weighted_score_used": False,
        "memory_selection_performed": False,
        "memory_selected": None,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "environment": dict(environment),
        "environment_dependency_digest": dependency_digest,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "repeat_attempt_permitted": False,
    }
    payload["m1_stage1_suite_sha256"] = canonical_sha256(payload)
    return payload


def validate_m1_stage1_results(run_root: Path) -> dict[str, Any]:
    """Re-verify the combined M1 Stage-1 result and all three arm locks."""
    path = Path(run_root) / M1_STAGE1_RESULT_NAME
    if not path.is_file():
        raise M1MemoryError(f"No M1 Stage-1 result at {path}.")
    payload = read_json(path)
    recorded = payload.get("m1_stage1_suite_sha256")
    body = {k: v for k, v in payload.items() if k != "m1_stage1_suite_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise M1MemoryError("M1 Stage-1 result failed digest validation.")
    if payload.get("m1_protocol_sha256") != M1_PROTOCOL_SHA256:
        raise M1MemoryError("M1 Stage-1 result does not bind the frozen protocol.")
    if payload.get("test_accessed") is not False:
        raise M1MemoryError("M1 Stage-1 result records test access.")
    for arm in M1_ARM_ORDER:
        lock = validate_m1_lock(Path(run_root) / arm)
        recorded_arm = payload["arm_results"][arm]["experiment_lock_sha256"]
        if lock["experiment_lock_sha256"] != recorded_arm:
            raise M1MemoryError(f"M1 arm {arm} lock differs from the suite record.")
    return payload


# --------------------------------------------------------------------------
# Read-only preflight
# --------------------------------------------------------------------------


def m1_preflight(
    run_root: Path,
    stream_cache_root: Path,
    *,
    p1_run_root: Path | None = None,
    b4b_run_dir: Path | None = None,
    feature_root: Path | None = None,
    source: Path | None = None,
) -> dict[str, Any]:
    """Read-only Stage M1-1 readiness gate. Creates zero models and artifacts.

    An absent stream cache is reported as `stream_cache_materialization_required`,
    which is the expected healthy initial state and not a failure.
    """
    validate_m1_protocol()
    environment, dependency_digest = require_p1_runtime()
    provenance = require_clean_checkout()
    alphas = m1_alpha_identity()

    claimed = {arm: (Path(run_root) / arm).exists() for arm in M1_ARM_ORDER}
    caches: dict[str, Any] = {}
    for partition in ("train", "validation"):
        directory = Path(stream_cache_root) / partition
        entry: dict[str, Any] = {
            "present": (directory / STREAM_CACHE_MANIFEST_NAME).exists(),
            "directory_exists": directory.exists(),
            "validated": False,
        }
        if entry["present"]:
            try:
                _, _, manifest = load_stream_cache(Path(stream_cache_root), partition)
                entry["validated"] = True
                entry["stream_cache_sha256"] = manifest["stream_cache_sha256"]
                entry["full_stream_row_count"] = manifest["full_stream_row_count"]
                entry["stream_count"] = manifest["stream_count"]
            except Exception as error:  # surfaced, never silently ignored
                entry["error"] = f"{type(error).__name__}: {error}"
        elif entry["directory_exists"]:
            entry["error"] = (
                "A partial stream cache directory exists without a manifest; "
                "human review is required and it is never overwritten."
            )
        caches[partition] = entry

    p1_state: dict[str, Any] = {"validated": False}
    if p1_run_root is not None:
        try:
            from cardiosentinel.neural.p1_experiment import validate_p1_stage1_results

            suite = validate_p1_stage1_results(Path(p1_run_root))
            transform = load_frozen_physiology_transform(
                Path(p1_run_root) / P1B_EXPERIMENT_ID
            )
            p1_state = {
                "validated": True,
                "p1_stage1_suite_sha256": suite["p1_stage1_suite_sha256"],
                "p1b_experiment_lock_sha256": suite["arm_results"][
                    P1B_EXPERIMENT_ID
                ]["experiment_lock_sha256"],
                "physiology_transform_sha256": transform.as_dict()[
                    "transform_sha256"
                ],
                "physiology_dim": PHYSIOLOGY_DIM,
                "physiology_refitted": False,
                "test_accessed": suite["test_accessed"],
            }
        except Exception as error:
            p1_state = {"validated": False, "error": f"{type(error).__name__}: {error}"}

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

    chronology: dict[str, Any] = {"validated": False}
    challenge_state: dict[str, Any] = {"validated": False}
    integrity: dict[str, Any] = {"validated": False}
    if feature_root is not None:
        try:
            streams = {
                partition: build_causal_streams(
                    load_b4_references(
                        Path(feature_root), partition, primary_only=False
                    )
                )
                for partition in ("train", "validation")
            }
            chronology = {
                "validated": True,
                "causal_order_field": "window_start_samples",
                "stream_key": ["record_id", "channel_index"],
                "memory_resets_at_recording_channel_boundary": True,
                "partitions": {
                    partition: {
                        "stream_count": len(value),
                        "full_stream_row_count": sum(
                            len(rows) for rows in value.values()
                        ),
                        "record_count": len({key[0] for key in value}),
                        "channel_indices": sorted({int(key[1]) for key in value}),
                        "ordered_chronology_sha256": ordered_chronology_digest(value),
                    }
                    for partition, value in streams.items()
                },
            }
        except Exception as error:
            chronology = {
                "validated": False,
                "error": f"{type(error).__name__}: {error}",
            }
        try:
            index = build_validation_challenge_index(Path(feature_root))
            challenge_state = {
                "validated": index.selection_sha256 == CHALLENGE_SELECTION_SHA256,
                "selection_sha256": index.selection_sha256,
                "counts": index.counts,
                "scored_at_causal_stream_position": True,
            }
        except Exception as error:
            challenge_state = {
                "validated": False,
                "error": f"{type(error).__name__}: {error}",
            }
        if source is not None:
            try:
                feature_receipt = validate_development_feature_integrity(
                    Path(feature_root)
                )
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

    caches_ready = all(entry.get("validated") for entry in caches.values())
    gates_ready = all(
        state.get("validated")
        for state in (p1_state, encoder_state, chronology, challenge_state, integrity)
    )
    if any(claimed.values()):
        status = "m1_arm_already_claimed"
    elif not gates_ready:
        status = "upstream_gates_incomplete"
    elif not caches_ready:
        status = "stream_cache_materialization_required"
    else:
        status = "ready_for_canonical_m1_stage1"

    return {
        "preflight_class": "m1_stage1_readiness",
        "status": status,
        "healthy_initial_status": "stream_cache_materialization_required",
        "ready_for_canonical_m1_stage1": status == "ready_for_canonical_m1_stage1",
        "m1_protocol_sha256": M1_PROTOCOL_SHA256,
        "p1_retention_decision_sha256": P1_RETENTION_DECISION_SHA256,
        "representation_dim": REPRESENTATION_DIM,
        "memory": alphas,
        "boundary": m1_boundary_statement(),
        "arm_claims": claimed,
        "stream_caches": caches,
        "p1_evidence": p1_state,
        "encoder": encoder_state,
        "chronology": chronology,
        "validation_challenge": challenge_state,
        "development_integrity": integrity,
        "test_artifacts_present": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "models_created": 0,
        "artifacts_created": 0,
        "read_only": True,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "environment": environment,
        "environment_dependency_digest": dependency_digest,
    }


# --------------------------------------------------------------------------
# Canonical Stage-1 route
# --------------------------------------------------------------------------


def run_m1_arm(
    experiment_id: str,
    *,
    run_root: Path,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_subject_ids: Sequence[str],
    challenge_features: np.ndarray,
    challenge_families: Sequence[str],
    challenge_subject_ids: Sequence[str],
    validation_memory: M1StreamMemory,
    validation_rows: np.ndarray,
    train_cache: Mapping[str, Any],
    validation_cache: Mapping[str, Any],
    standardizer: Mapping[str, Any],
    provenance: Mapping[str, Any],
    environment: Mapping[str, Any],
    dependency_digest: str,
    p1_stage1_suite_sha256: str,
    p1b_lock_sha256: str,
    max_epochs: int = P1_MAX_EPOCHS,
) -> dict[str, Any]:
    """Claim, run and freeze exactly one M1 arm.

    Provenance is mandatory rather than appended afterwards, so a result can
    never be digested and written before its provenance is bound.
    """
    require_m1_experiment(experiment_id)
    run_dir = claim_m1_run_directory(
        resolve_m1_run_dir(Path(run_root), experiment_id), experiment_id
    )
    try:
        write_m1_status(
            run_dir,
            ATTEMPT_STATUS_STARTED,
            experiment_id=experiment_id,
            m1_protocol_sha256=M1_PROTOCOL_SHA256,
            repeat_attempt_permitted=False,
        )
        result = train_m1_arm(
            experiment_id,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            max_epochs=max_epochs,
        )
        head = result["head"]
        scores = _scores(head, validation_features)
        threshold = select_p1_threshold(validation_labels, scores)
        validation_evidence = p1_validation_evidence(
            validation_labels, scores, validation_subject_ids, threshold
        )
        challenge_scores = _scores(head, challenge_features)
        challenge = p1_challenge_evidence(
            challenge_families, challenge_scores, challenge_subject_ids, threshold
        )
        cold_start = cold_start_evidence(
            validation_memory, validation_rows, validation_labels, scores, threshold
        )
        descriptives = memory_descriptives(validation_memory, validation_rows)

        torch.save(head.state_dict(), run_dir / SELECTED_MODEL_NAME)
        write_json_atomic(
            run_dir / EPOCH_HISTORY_NAME,
            {"experiment_id": experiment_id, "epochs": list(result["epoch_history"])},
        )
        write_json_atomic(run_dir / VALIDATION_METRICS_NAME, validation_evidence)
        write_json_atomic(
            run_dir / VALIDATION_THRESHOLD_NAME,
            {
                "experiment_id": experiment_id,
                "threshold": threshold,
                "threshold_rule": THRESHOLD_RULE,
                "score_semantics": SCORE_SEMANTICS,
            },
        )
        write_json_atomic(run_dir / CHALLENGE_METRICS_NAME, challenge)
        write_json_atomic(
            run_dir / MEMORY_FEATURE_NAME,
            {
                "experiment_id": experiment_id,
                "memory_features": list(M1_ARM_FEATURES[experiment_id]),
                "memory": m1_alpha_identity(),
                "cold_start_evidence": cold_start,
                "descriptives": descriptives,
            },
        )
        artifact_hashes = {
            name: sha256_file(run_dir / name)
            for name in (
                SELECTED_MODEL_NAME,
                EPOCH_HISTORY_NAME,
                VALIDATION_METRICS_NAME,
                VALIDATION_THRESHOLD_NAME,
                CHALLENGE_METRICS_NAME,
                MEMORY_FEATURE_NAME,
            )
        }
        lock = build_m1_lock(
            experiment_id,
            head=head,
            result=result,
            threshold=threshold,
            validation_evidence=validation_evidence,
            challenge_evidence=challenge,
            cold_start=cold_start,
            descriptives=descriptives,
            train_cache=train_cache,
            validation_cache=validation_cache,
            standardizer=standardizer,
            artifact_hashes=artifact_hashes,
            provenance=provenance,
            environment=environment,
            dependency_digest=dependency_digest,
            p1_stage1_suite_sha256=p1_stage1_suite_sha256,
            p1b_lock_sha256=p1b_lock_sha256,
        )
        write_json_atomic(run_dir / EXPERIMENT_LOCK_NAME, lock)
        write_m1_status(
            run_dir,
            ATTEMPT_STATUS_COMPLETE,
            experiment_id=experiment_id,
            experiment_lock_sha256=lock["experiment_lock_sha256"],
            repeat_attempt_permitted=False,
        )
        return lock
    except BaseException as error:
        record_m1_failure(run_dir, experiment_id, error)
        raise


def execute_m1_stage1(
    *,
    run_root: Path,
    stream_cache_root: Path,
    cache_root: Path,
    feature_root: Path,
    source: Path,
    b4b_run_dir: Path,
    p1_run_root: Path,
    waveform_batches_for=None,
    max_epochs: int = P1_MAX_EPOCHS,
) -> dict[str, Any]:
    """The one canonical M1 Stage-1 route.

    Order is fixed: validate every gate, construct or reuse the immutable
    full-stream memory cache, validate it, then claim/run/freeze M1S, M1L and
    M1D, then write the combined result. There is no single-arm public route,
    no force, no retry, no alternate seed and no selective rerun.
    """
    from cardiosentinel.neural.p1_experiment import validate_p1_stage1_results

    validate_m1_protocol()
    environment, dependency_digest = require_p1_runtime()
    provenance = require_clean_checkout()
    m1_alpha_identity()

    suite = validate_p1_stage1_results(Path(p1_run_root))
    p1b_lock_sha256 = suite["arm_results"][P1B_EXPERIMENT_ID][
        "experiment_lock_sha256"
    ]
    encoder_lock = validate_locked_model(Path(b4b_run_dir), official_model="B4-B")
    if encoder_lock["test"] is not None:
        raise M1MemoryError("The selected encoder lock records test evidence.")
    feature_receipt = validate_development_feature_integrity(Path(feature_root))
    validate_development_source_integrity(Path(source), feature_receipt)
    transform_sha256 = load_frozen_physiology_transform(
        Path(p1_run_root) / P1B_EXPERIMENT_ID
    ).as_dict()["transform_sha256"]
    # The control's evidence is read from its own immutable lock: the suite
    # summary carries only identities, and M1 must never retrain the control.
    control_dir = Path(p1_run_root) / P1B_EXPERIMENT_ID
    control_lock = read_json(control_dir / EXPERIMENT_LOCK_NAME)
    if control_lock.get("experiment_lock_sha256") != p1b_lock_sha256:
        raise M1MemoryError(
            "The P1-B control lock differs from the digest recorded by the "
            "frozen P1 Stage-1 suite."
        )
    control_evidence = load_frozen_control_evidence(control_dir)

    # --- full-stream memory caches -------------------------------------
    memories: dict[str, M1StreamMemory] = {}
    matrices: dict[str, np.ndarray] = {}
    manifests: dict[str, dict[str, Any]] = {}
    standardizer_payload: dict[str, Any] | None = None
    standardizer: M1DistanceStandardizer | None = None

    for partition in ("train", "validation"):
        directory = Path(stream_cache_root) / partition
        if (directory / STREAM_CACHE_MANIFEST_NAME).is_file():
            memory, matrix, manifest = load_stream_cache(
                Path(stream_cache_root), partition
            )
            memories[partition] = memory
            matrices[partition] = matrix
            manifests[partition] = manifest
            continue

        representation = prepare_stream_representations(
            partition,
            cache_root=Path(cache_root),
            feature_root=Path(feature_root),
            b4b_run_dir=Path(b4b_run_dir),
            p1b_run_dir=Path(p1_run_root) / P1B_EXPERIMENT_ID,
            waveform_batches_for=waveform_batches_for,
        )
        cache = load_p1_embedding_cache(Path(cache_root), partition)
        if standardizer is None:
            if partition != "train":
                raise M1MemoryError(
                    "The distance standardizer must be fitted from the train "
                    "stream before any validation memory is generated."
                )
            standardizer = build_distance_standardizer(
                representation, primary_train_stable_ids=cache.stable_ids
            )
            standardizer_payload = standardizer.as_dict()
            write_json_atomic(
                Path(stream_cache_root) / STANDARDIZER_NAME, standardizer_payload
            )
        memory = generate_stream_memory(
            representation.streams,
            partition=partition,
            representations=representation.by_stable_id(),
            standardizer=standardizer,
        )
        manifest = materialize_stream_cache(
            memory,
            representation,
            cache_root=Path(stream_cache_root),
            manifest_fields={
                "standardizer_sha256": standardizer_payload["standardizer_sha256"],
                "p1_stage1_suite_sha256": suite["p1_stage1_suite_sha256"],
                "p1b_lock_sha256": p1b_lock_sha256,
                "physiology_transform_sha256": transform_sha256,
                "embedding_cache_sha256": cache.manifest["cache_sha256"],
                "git_sha": provenance["git_sha"],
                "git_dirty": provenance["git_dirty"],
                "dependency_digest": dependency_digest,
            },
        )
        memory, matrix, manifest = load_stream_cache(
            Path(stream_cache_root), partition
        )
        memories[partition] = memory
        matrices[partition] = matrix
        manifests[partition] = manifest

    if standardizer_payload is None:
        standardizer_payload = read_json(Path(stream_cache_root) / STANDARDIZER_NAME)

    # --- supervised membership -----------------------------------------
    train_cache = load_p1_embedding_cache(Path(cache_root), "train")
    validation_cache = load_p1_embedding_cache(Path(cache_root), "validation")
    train_rows = select_rows(memories["train"], train_cache.stable_ids)
    validation_rows = select_rows(
        memories["validation"], validation_cache.stable_ids
    )
    challenge = build_validation_challenge_index(Path(feature_root))
    if challenge.selection_sha256 != CHALLENGE_SELECTION_SHA256:
        raise M1MemoryError("The rebuilt challenge selection is not the frozen one.")
    challenge_ids = tuple(item.stable_id for item in challenge.references)
    if set(challenge_ids) & set(validation_cache.stable_ids):
        raise M1MemoryError(
            "The primary validation and challenge populations must stay "
            "disjoint; a shared row indicates a lookup built from the wrong "
            "population."
        )
    challenge_rows = select_rows(memories["validation"], challenge_ids)

    locks: dict[str, dict[str, Any]] = {}
    for experiment_id in M1_ARM_ORDER:
        locks[experiment_id] = run_m1_arm(
            experiment_id,
            run_root=Path(run_root),
            train_features=m1_arm_features(
                experiment_id,
                matrices["train"][train_rows],
                memories["train"].memory_matrix(experiment_id)[train_rows],
            ),
            train_labels=train_cache.labels,
            validation_features=m1_arm_features(
                experiment_id,
                matrices["validation"][validation_rows],
                memories["validation"].memory_matrix(experiment_id)[validation_rows],
            ),
            validation_labels=validation_cache.labels,
            validation_subject_ids=validation_cache.subject_ids,
            challenge_features=m1_arm_features(
                experiment_id,
                matrices["validation"][challenge_rows],
                memories["validation"].memory_matrix(experiment_id)[challenge_rows],
            ),
            challenge_families=tuple(
                item.target_family for item in challenge.references
            ),
            challenge_subject_ids=tuple(
                item.subject_id for item in challenge.references
            ),
            validation_memory=memories["validation"],
            validation_rows=validation_rows,
            train_cache={
                "stream_cache_sha256": manifests["train"]["stream_cache_sha256"],
                "full_stream_row_count": manifests["train"]["full_stream_row_count"],
                "supervised_rows": int(train_rows.shape[0]),
            },
            validation_cache={
                "stream_cache_sha256": manifests["validation"][
                    "stream_cache_sha256"
                ],
                "full_stream_row_count": manifests["validation"][
                    "full_stream_row_count"
                ],
                "primary_rows": int(validation_rows.shape[0]),
                "challenge_rows": int(challenge_rows.shape[0]),
            },
            standardizer={
                "standardizer_sha256": standardizer_payload["standardizer_sha256"],
                "fitted_rows": standardizer_payload["fitted_rows"],
                "fitted_on_partition": standardizer_payload["fitted_on_partition"],
            },
            provenance=provenance,
            environment=environment,
            dependency_digest=dependency_digest,
            p1_stage1_suite_sha256=suite["p1_stage1_suite_sha256"],
            p1b_lock_sha256=p1b_lock_sha256,
            max_epochs=max_epochs,
        )

    payload = build_m1_stage1_result(
        locks,
        control={
            "experiment_id": GLOBAL_CONTROL_EXPERIMENT_ID,
            "experiment_lock_sha256": p1b_lock_sha256,
            "p1_stage1_suite_sha256": suite["p1_stage1_suite_sha256"],
            "retrained_by_m1": False,
            "validation_evidence": control_evidence["validation_evidence"],
            "challenge_evidence": control_evidence["challenge_evidence"],
        },
        stream_caches={
            partition: {
                "stream_cache_sha256": manifest["stream_cache_sha256"],
                "full_stream_row_count": manifest["full_stream_row_count"],
                "stream_count": manifest["stream_count"],
                "ordered_chronology_sha256": manifest["ordered_chronology_sha256"],
            }
            for partition, manifest in manifests.items()
        },
        standardizer=standardizer_payload,
        provenance=provenance,
        environment=environment,
        dependency_digest=dependency_digest,
    )
    write_json_atomic(Path(run_root) / M1_STAGE1_RESULT_NAME, payload)
    return payload
