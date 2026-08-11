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
from cardiosentinel.neural.data import B4WaveformDataset
from cardiosentinel.neural.integrity import (
    canonical_sha256,
    validate_development_feature_integrity,
    validate_development_source_integrity,
)
from cardiosentinel.neural.m1_store import (
    CHANNEL_INDEX_FILE,
    COLD_START_BIN_FILE,
    D_LONG_FILE,
    D_SHORT_FILE,
    DEFAULT_CHUNK_ROWS,
    DISAGREEMENT_FILE,
    M1_STREAM_CACHE_SCHEMA,
    PAST_OBSERVED_FILE,
    PAST_UPDATE_FILE,
    RECORD_ID_FILE,
    RECORDING_AGE_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
    M1RowStore,
    M1StoreSpec,
    StreamingContentDigest,
    locate_rows,
    streaming_ordered_stable_id_digest,
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
    DualTimescaleMemory,
    M1DistanceStandardizer,
    M1MemoryError,
    M1StreamMemory,
    StreamKey,
    build_causal_streams,
    build_deterministic_m1_head,
    claim_m1_run_directory,
    cold_start_bin,
    fit_distance_standardizer,
    m1_alpha_identity,
    m1_arm_features,
    m1_boundary_statement,
    m1_head_identity,
    m1_training_configuration,
    ordered_chronology_digest,
    record_m1_failure,
    require_m1_experiment,
    resolve_m1_run_dir,
    stream_key,
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
    morphology_columns,
    require_p1_partition,
)
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    B4_SPLIT_SHA256,
    FEATURE_CORPUS_SHA256,
    REPOSITORY_ROOT,
    SAMPLING_FREQUENCY_HZ,
)
from cardiosentinel.neural.resource_benchmark import validate_locked_model
from cardiosentinel.neural.training import CheckpointTracker
from cardiosentinel.neural.validation_challenge import (
    _model_state_digest,
    build_validation_challenge_index,
)

M1_STAGE1_RESULT_NAME: Final = "M1_STAGE1_RESULTS.json"
RUN_MANIFEST_NAME: Final = "RUN_MANIFEST.json"
MEMORY_FEATURE_NAME: Final = "MEMORY_FEATURES.json"
M1_ARM_ORDER: Final = M1_EXPERIMENT_IDS
PRIMARY_AUDIT_ROWS: Final = 64
# Re-extracting a scattered audit sample groups rows into different batches than
# the original cache build, and float32 GEMM reassociation makes that grouping
# visible in the last bits. Bitwise equality is therefore recorded but not
# required; the admission bound is orders of magnitude below any difference a
# wrong encoder, wrong cache or wrong waveform would produce.
PRIMARY_AUDIT_TOLERANCE: Final = 1e-5

# --------------------------------------------------------------------------
# Exact frozen upstream identities
#
# The M1 protocol binds specific historical evidence, not "whatever valid
# artifact happens to be supplied". Recording a digest is not enforcement, so
# every one of these is checked before any M1 representation is built.
# --------------------------------------------------------------------------
FROZEN_P1_STAGE1_SUITE_SHA256: Final = (
    "cc354ef64415d9c0dafcffdc0fdfa2446cd81a7d0c30fa9c58b0095cbc0be772"
)
FROZEN_P1B_LOCK_SHA256: Final = (
    "796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0"
)
FROZEN_PHYSIOLOGY_TRANSFORM_SHA256: Final = (
    "cc6bd3a353f0ac6cad342114ed96e135cbf3c61e2946f847d5b95358b6bd51a9"
)
FROZEN_P1_EMBEDDING_CACHE_SHA256: Final = {
    "train": "0a5f021b89597d245a2afdc51fe1a65ba5cd6a090beba429f38bbccff8c372dd",
    "validation": "c533db3acfdfa1057c2ac9d8e77d011d3ac5f87fc7a872399227f94f526db0c3",
}
FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256: Final = (
    "8a7977dc4f0ac7308fa0a5ad439bb5961f806f049ddcb27fd6de461a05d690fd"
)
FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256: Final = (
    "a56e283631c9db51762118d6574ae1171840836dd4ff1d33d952fb51442571c1"
)

# Attempt 1 of the canonical Stage-1 run was consumed without producing any
# scientific artifact. Every future preflight reports this so no report can
# imply that no execution has ever occurred. It does NOT authorize a
# replacement run: human governance stays external.
ATTEMPT1_FAILURE_DOCUMENT: Final = "docs/M1_STAGE1_ATTEMPT1_FAILURE.md"
PRIOR_AUTHORIZED_INVOCATION_COUNT: Final = 1

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


WAVEFORM_BATCH_SIZE: Final = 256


def canonical_waveform_batches(
    references: Sequence[B4WindowReference],
    source: Path,
    *,
    batch_size: int = WAVEFORM_BATCH_SIZE,
):
    """The production full-stream waveform iterator.

    This is the canonical route, not an injected callback: `m1 run-stage1` must
    work from `--source` alone. It reuses the reviewed `B4WaveformDataset`
    contract, so every window is read as validated physical mV at the exact
    record/channel/start/end of its own reference. No metadata reaches the
    tensor.
    """
    if not references:
        return
    dataset = B4WaveformDataset(tuple(references), Path(source))
    for start in range(0, len(references), batch_size):
        block = tuple(references[start : start + batch_size])
        waveforms = torch.stack([dataset.read_waveform(item) for item in block])
        yield tuple(item.stable_id for item in block), waveforms, dataset.stats


def _extract_through_encoder(
    encoder,
    references: Sequence[B4WindowReference],
    batches,
    *,
    permitted: set[str],
    forbidden: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run the locked encoder over a batch stream and prove what it consumed.

    Every requested identifier must appear exactly once and nothing else may
    appear, so a silently truncated or duplicated read cannot pass as complete.
    """
    requested = [item.stable_id for item in references]
    extracted: dict[str, np.ndarray] = {}
    receipt: dict[str, Any] = {}
    reads = 0
    before = _model_state_digest(encoder)
    for identifiers, waveforms, stats in batches:
        embeddings, receipt = extract_frozen_embeddings(encoder, waveforms)
        block = embeddings.to(torch.float32).numpy()
        reads = int(getattr(stats, "source_reads", reads))
        for offset, key in enumerate(identifiers):
            identifier = str(key)
            if identifier not in permitted:
                raise M1MemoryError(
                    f"The waveform iterator produced unexpected row {identifier}."
                )
            if identifier in extracted:
                raise M1MemoryError(
                    f"Row {identifier} was produced more than once."
                )
            if forbidden is not None and identifier in forbidden:
                raise M1MemoryError(
                    f"Row {identifier} was re-extracted although it is already "
                    "in the frozen P1 cache; the overlap must be reused, not "
                    "duplicated."
                )
            extracted[identifier] = block[offset]
    missing = [key for key in requested if key not in extracted]
    if missing:
        raise M1MemoryError(
            f"{len(missing)} requested rows were never produced by the waveform "
            f"iterator; the first is {missing[0]!r}."
        )
    after = _model_state_digest(encoder)
    if before != after:
        raise M1MemoryError(
            "The locked B4-B encoder state changed during full-stream extraction."
        )
    return extracted, {
        "encoder_state_sha256_before": before,
        "encoder_state_sha256_after": after,
        "encoder_state_unchanged": True,
        "encoder_fine_tuned": False,
        "waveform_source_reads": reads,
        "rows_extracted": len(extracted),
        **{k: v for k, v in receipt.items() if k in ("embedding_tap", "embedding_dim")},
    }


def prepare_stream_representations(
    partition: str,
    *,
    cache_root: Path,
    feature_root: Path,
    source: Path,
    b4b_run_dir: Path,
    p1b_run_dir: Path,
    _waveform_batches_for=None,
) -> M1StreamRepresentation:
    """Assemble fused 146-d representations for the full development stream.

    Primary rows reuse the canonical frozen P1 embedding cache verbatim. Every
    additional full-stream row is read through the canonical waveform path and
    embedded with the same locked B4-B encoder. The overlap is proven rather
    than assumed: extra rows must be disjoint from the cache, and a
    deterministic audit subset of primary rows is re-extracted through the same
    path and required to match the cache bit-for-bit.

    `_waveform_batches_for` is a private seam for synthetic tests. The canonical
    production route does not use it and works from `source` alone.
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
    ordered = [reference for key in sorted(streams) for reference in streams[key]]
    extra = [item for item in ordered if item.stable_id not in primary_embeddings]
    by_id = {item.stable_id: item for item in ordered}

    def batches_for(rows: Sequence[B4WindowReference]):
        if _waveform_batches_for is not None:
            return _waveform_batches_for(evaluated, tuple(r.stable_id for r in rows))
        return canonical_waveform_batches(rows, Path(source))

    audit: dict[str, Any] = {
        "primary_rows_reused": len(primary_embeddings),
        "rows_newly_extracted": 0,
        "extra_disjoint_from_primary_cache": True,
        "re_extracted_primary_rows": 0,
        "re_extracted_primary_bitwise_identical": None,
        "re_extracted_primary_max_abs_deviation": None,
        "primary_audit_tolerance": PRIMARY_AUDIT_TOLERANCE,
    }
    extracted: dict[str, np.ndarray] = {}
    extraction_receipt: dict[str, Any] = {"rows_extracted": 0}
    audit_receipt: dict[str, Any] = {}

    if extra:
        encoder = load_official_b4b_encoder(Path(b4b_run_dir))
        extracted, extraction_receipt = _extract_through_encoder(
            encoder,
            extra,
            batches_for(extra),
            permitted={item.stable_id for item in extra},
            forbidden=primary_embeddings,
        )
        audit["rows_newly_extracted"] = len(extracted)

        # Deliberate audit sample: the ONLY primary rows re-generated.
        step = max(len(cache.stable_ids) // PRIMARY_AUDIT_ROWS, 1)
        sample = [
            by_id[key]
            for index, key in enumerate(cache.stable_ids)
            if index % step == 0 and key in by_id
        ][:PRIMARY_AUDIT_ROWS]
        if sample:
            replayed, audit_receipt = _extract_through_encoder(
                encoder,
                sample,
                batches_for(sample),
                permitted={item.stable_id for item in sample},
            )
            identical = all(
                np.array_equal(values, primary_embeddings[key])
                for key, values in replayed.items()
            )
            deviation = max(
                float(
                    np.max(
                        np.abs(
                            values.astype(np.float64)
                            - primary_embeddings[key].astype(np.float64)
                        )
                    )
                )
                for key, values in replayed.items()
            )
            audit["re_extracted_primary_rows"] = len(replayed)
            audit["re_extracted_primary_bitwise_identical"] = identical
            audit["re_extracted_primary_max_abs_deviation"] = deviation
            audit["primary_audit_tolerance"] = PRIMARY_AUDIT_TOLERANCE
            if deviation > PRIMARY_AUDIT_TOLERANCE:
                raise M1MemoryError(
                    "Re-extracted primary embeddings differ from the frozen P1 "
                    f"cache by {deviation}, beyond the batch-grouping tolerance "
                    f"{PRIMARY_AUDIT_TOLERANCE}. The overlap identity is not "
                    "exact: the encoder, cache or waveform source differs."
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
                primary_embeddings[key]
                if key in primary_embeddings
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
    extra_ids = tuple(item.stable_id for item in extra)
    audit["extra_ordered_stable_id_sha256"] = (
        ordered_stable_id_digest(extra_ids) if extra_ids else None
    )
    audit["extra_embedding_content_sha256"] = (
        embedding_content_digest(np.stack([extracted[key] for key in extra_ids]))
        if extra_ids
        else None
    )
    audit["waveform_source_reads"] = int(
        extraction_receipt.get("waveform_source_reads", 0)
    ) + int(audit_receipt.get("waveform_source_reads", 0))
    audit["extraction_receipt"] = extraction_receipt
    audit["primary_audit_receipt"] = audit_receipt
    return M1StreamRepresentation(
        partition=evaluated,
        stable_ids=stable_ids,
        matrix=_fuse(embeddings, bundle.values),
        streams=streams,
        reused_primary_rows=len(primary_embeddings),
        newly_extracted_rows=len(extracted),
        primary_audit=audit,
    )


def _require_exact(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise M1MemoryError(
            f"{name} is {observed!r} but the frozen M1 protocol binds "
            f"{expected!r}. M1 consumes exactly the historical evidence it "
            "names; a different valid artifact is not a substitute."
        )


def require_frozen_upstream_identities(
    *,
    p1_suite: Mapping[str, Any],
    p1b_lock: Mapping[str, Any],
    physiology_transform_sha256: str,
    embedding_caches: Mapping[str, Mapping[str, Any]],
    encoder_lock: Mapping[str, Any],
    feature_receipt: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    challenge_selection_sha256: str,
) -> dict[str, Any]:
    """Enforce every exact upstream digest the M1 protocol binds."""
    _require_exact(
        "P1 Stage-1 suite",
        p1_suite.get("p1_stage1_suite_sha256"),
        FROZEN_P1_STAGE1_SUITE_SHA256,
    )
    _require_exact(
        "P1-B experiment lock",
        p1b_lock.get("experiment_lock_sha256"),
        FROZEN_P1B_LOCK_SHA256,
    )
    _require_exact(
        "P1 physiology transform",
        physiology_transform_sha256,
        FROZEN_PHYSIOLOGY_TRANSFORM_SHA256,
    )
    for partition, expected in FROZEN_P1_EMBEDDING_CACHE_SHA256.items():
        _require_exact(
            f"P1 {partition} embedding cache",
            dict(embedding_caches.get(partition) or {}).get("cache_sha256"),
            expected,
        )
    _require_exact(
        "B4-B checkpoint",
        encoder_lock.get("checkpoint_sha256"),
        B4B_CHECKPOINT_SHA256,
    )
    _require_exact(
        "B4-B experiment lock",
        encoder_lock.get("experiment_lock_sha256"),
        B4B_EXPERIMENT_LOCK_SHA256,
    )
    if encoder_lock.get("test") is not None:
        raise M1MemoryError("The selected encoder lock records test evidence.")
    _require_exact(
        "development feature integrity",
        feature_receipt.get("development_feature_integrity_sha256"),
        FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256,
    )
    _require_exact(
        "development source integrity",
        source_receipt.get("development_source_integrity_sha256"),
        FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256,
    )
    _require_exact(
        "challenge selection",
        challenge_selection_sha256,
        CHALLENGE_SELECTION_SHA256,
    )
    _require_exact(
        "P1 retention decision",
        sha256_file(
            REPOSITORY_ROOT / "docs" / "P1_PHYSIOLOGY_RETENTION_DECISION_V1.md"
        ),
        P1_RETENTION_DECISION_SHA256,
    )
    if p1_suite.get("test_accessed") is not False:
        raise M1MemoryError("The frozen P1 Stage-1 suite records test access.")
    return {
        "p1_stage1_suite_sha256": FROZEN_P1_STAGE1_SUITE_SHA256,
        "p1b_experiment_lock_sha256": FROZEN_P1B_LOCK_SHA256,
        "physiology_transform_sha256": FROZEN_PHYSIOLOGY_TRANSFORM_SHA256,
        "p1_train_embedding_cache_sha256": FROZEN_P1_EMBEDDING_CACHE_SHA256["train"],
        "p1_validation_embedding_cache_sha256": FROZEN_P1_EMBEDDING_CACHE_SHA256[
            "validation"
        ],
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "development_feature_integrity_sha256": (
            FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256
        ),
        "development_source_integrity_sha256": (
            FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256
        ),
        "challenge_selection_sha256": CHALLENGE_SELECTION_SHA256,
        "p1_retention_decision_sha256": P1_RETENTION_DECISION_SHA256,
        "m1_protocol_sha256": M1_PROTOCOL_SHA256,
        "all_frozen_identities_enforced": True,
    }


def build_distance_standardizer(
    representation: M1StreamRepresentation,
    *,
    primary_train_stable_ids: Sequence[str],
    upstream_identities: Mapping[str, Any],
) -> M1DistanceStandardizer:
    """Fit the frozen distance space on primary TRAIN rows only."""
    if representation.partition != "train":
        raise M1MemoryError("The M1 distance standardizer is fitted on train only.")
    required = (
        "p1_stage1_suite_sha256",
        "p1b_experiment_lock_sha256",
        "physiology_transform_sha256",
        "p1_train_embedding_cache_sha256",
        "encoder_checkpoint_sha256",
    )
    upstream = dict(upstream_identities)
    missing = [key for key in required if not upstream.get(key)]
    if missing:
        raise M1MemoryError(
            f"The distance standardizer must bind exact upstream identities; "
            f"{missing} are absent or null."
        )
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
            "p1_stage1_suite_sha256": upstream["p1_stage1_suite_sha256"],
            "p1b_experiment_lock_sha256": upstream["p1b_experiment_lock_sha256"],
            "physiology_transform_sha256": upstream["physiology_transform_sha256"],
            "p1_train_embedding_cache_sha256": upstream[
                "p1_train_embedding_cache_sha256"
            ],
            "encoder_checkpoint_sha256": upstream["encoder_checkpoint_sha256"],
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
    feature_integrity_sha256: str,
    source_integrity_sha256: str,
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
        "development_feature_integrity_sha256": feature_integrity_sha256,
        "development_source_integrity_sha256": source_integrity_sha256,
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
    # A self-consistent but WRONG alternate cache must not be accepted: the
    # manifest's own digest only proves internal consistency, so every exact
    # frozen upstream identity is rechecked here as well.
    for field, expected in (
        ("m1_protocol_sha256", M1_PROTOCOL_SHA256),
        ("p1_protocol_sha256", P1_PROTOCOL_SHA256),
        ("p1_retention_decision_sha256", P1_RETENTION_DECISION_SHA256),
        ("p1_stage1_suite_sha256", FROZEN_P1_STAGE1_SUITE_SHA256),
        ("p1b_experiment_lock_sha256", FROZEN_P1B_LOCK_SHA256),
        ("physiology_transform_sha256", FROZEN_PHYSIOLOGY_TRANSFORM_SHA256),
        (
            "p1_embedding_cache_sha256",
            FROZEN_P1_EMBEDDING_CACHE_SHA256[evaluated],
        ),
        (
            "development_feature_integrity_sha256",
            FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256,
        ),
        (
            "development_source_integrity_sha256",
            FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256,
        ),
        ("encoder_checkpoint_sha256", B4B_CHECKPOINT_SHA256),
        ("encoder_experiment_lock_sha256", B4B_EXPERIMENT_LOCK_SHA256),
        ("split_sha256", B4_SPLIT_SHA256),
        ("feature_corpus_sha256", FEATURE_CORPUS_SHA256),
        ("physiology_schema_sha256", MORPHOLOGY_SCHEMA_SHA256),
        ("update_policy", UPDATE_POLICY),
        ("alpha_short", ALPHA_SHORT),
        ("alpha_long", ALPHA_LONG),
        ("representation_dim", REPRESENTATION_DIM),
        ("contamination_safe", CONTAMINATION_SAFE),
        ("label_independent_history", True),
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
    if embedding_content_digest(
        np.stack([memory.past_observed_count, memory.past_update_count], axis=1)
    ) != manifest["history_count_sha256"]:
        raise M1MemoryError("M1 stream cache history counts do not match.")
    # Re-derive the chronology digest from the PERSISTED arrays rather than
    # trusting the manifest's own copy of it.
    rebuilt = canonical_sha256(
        {
            "order": "stream_then_start_sample",
            "rows": [
                [record, int(channel), int(start)]
                for record, channel, start in zip(
                    memory.record_ids,
                    memory.channel_indices.tolist(),
                    memory.start_samples.tolist(),
                )
            ],
        }
    )
    if rebuilt != manifest["ordered_chronology_sha256"]:
        raise M1MemoryError(
            "The chronology digest re-derived from the persisted stream cache "
            "arrays does not match the manifest."
        )
    if int(manifest["full_stream_row_count"]) != len(memory.stable_ids):
        raise M1MemoryError("M1 stream cache row count does not match.")
    if int(manifest["stream_count"]) != len(memory.streams):
        raise M1MemoryError("M1 stream cache stream count does not match.")
    standardizer_path = Path(cache_root) / STANDARDIZER_NAME
    if not standardizer_path.is_file():
        raise M1MemoryError(
            f"M1 stream cache at {directory} has no distance standardizer at "
            f"{standardizer_path}; human review is required."
        )
    standardizer = read_json(standardizer_path)
    if standardizer.get("standardizer_sha256") != manifest[
        "distance_standardizer_sha256"
    ]:
        raise M1MemoryError(
            "The persisted distance standardizer differs from the one this "
            "stream cache was built against."
        )
    M1DistanceStandardizer.from_dict(standardizer)
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


def subject_false_positive_evidence(
    labels: np.ndarray,
    scores: np.ndarray,
    subject_ids: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """Primary-background FPR and its subject-wise distribution.

    Frozen in M1 protocol section 16.1. This is SUPPORTING evidence: it never
    selects a threshold, never enters a weighted score and is never used for
    tuning. The threshold passed in is the arm's already-selected one.

    Quantile interpolation is named explicitly because conventions differ
    between libraries and the frozen definition must be reproducible.
    """
    outcomes = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    subjects = np.asarray([str(v) for v in subject_ids], dtype=np.str_)
    if not (outcomes.shape[0] == values.shape[0] == subjects.shape[0]):
        raise M1MemoryError("Subject-wise FPR inputs are not row-aligned.")

    negatives = outcomes == 0
    negative_count = int(np.sum(negatives))
    if negative_count == 0:
        raise M1MemoryError(
            "Primary validation carries no background-negative window; the "
            "subject-wise false-positive summary is undefined."
        )
    flagged = values >= float(threshold)
    pooled = float(np.sum(negatives & flagged) / negative_count)

    per_subject: dict[str, float] = {}
    for subject in sorted(set(subjects.tolist())):
        mask = negatives & (subjects == subject)
        support = int(np.sum(mask))
        if support == 0:
            continue  # no negative support: excluded by definition
        per_subject[subject] = float(np.sum(mask & flagged) / support)

    rates = np.sort(np.asarray(list(per_subject.values()), dtype=np.float64))
    q25 = float(np.quantile(rates, 0.25, method="linear"))
    q75 = float(np.quantile(rates, 0.75, method="linear"))
    return {
        "evidence_class": "m1_subject_false_positive_distribution",
        "evidence_status": "supporting",
        "partition": "validation",
        "population": "primary_validation_background_negative",
        "threshold": float(threshold),
        "threshold_source": "arm_selected_validation_threshold",
        "threshold_optimized_from_this_evidence": False,
        "weighted_score_used": False,
        "pooled_background_negative_fpr": pooled,
        "background_negative_count": negative_count,
        "contributing_subject_count": int(rates.shape[0]),
        "subject_fpr_median": float(np.median(rates)),
        "subject_fpr_q25": q25,
        "subject_fpr_q75": q75,
        "subject_fpr_iqr": q75 - q25,
        "subject_fpr_p90": float(np.quantile(rates, 0.90, method="linear")),
        "subject_fpr_max": float(np.max(rates)),
        "quantile_interpolation": "linear",
        "subject_false_positive_rates": per_subject,
        "subject_ids_are_reporting_keys_only": True,
    }


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
    subject_false_positives: dict[str, Any],
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
        "subject_false_positive_evidence": subject_false_positives,
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
        "subject_false_positive_evidence": lock["subject_false_positive_evidence"],
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


def validate_m1_stage1_results(
    run_root: Path, *, stream_cache_root: Path | None = None
) -> dict[str, Any]:
    """Re-verify the combined M1 Stage-1 result and everything it rests on.

    A PASS proves all three immutable arm locks, the exact frozen P1-B global
    control, the exact distance standardizer, both stream-cache identities and
    the absence of any test access. No scientific quantity is recomputed.
    """
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
        _require_exact(
            f"{arm} global control lock",
            lock.get("global_control_lock_sha256"),
            FROZEN_P1B_LOCK_SHA256,
        )
        _require_exact(
            f"{arm} P1 Stage-1 suite",
            lock.get("p1_stage1_suite_sha256"),
            FROZEN_P1_STAGE1_SUITE_SHA256,
        )
        if lock.get("boundary", {}).get("contamination_safe") is not False:
            raise M1MemoryError(f"M1 arm {arm} does not declare the M1 limitation.")

    control = dict(payload.get("global_control") or {})
    _require_exact(
        "global control experiment",
        control.get("experiment_id"),
        GLOBAL_CONTROL_EXPERIMENT_ID,
    )
    _require_exact(
        "global control lock", control.get("experiment_lock_sha256"),
        FROZEN_P1B_LOCK_SHA256,
    )
    if control.get("retrained_by_m1") is not False:
        raise M1MemoryError("The M1 suite claims the global control was retrained.")

    standardizer = dict(payload.get("distance_standardizer") or {})
    M1DistanceStandardizer.from_dict(standardizer)
    for field, expected in (
        ("p1_stage1_suite_sha256", FROZEN_P1_STAGE1_SUITE_SHA256),
        ("p1b_experiment_lock_sha256", FROZEN_P1B_LOCK_SHA256),
        ("physiology_transform_sha256", FROZEN_PHYSIOLOGY_TRANSFORM_SHA256),
    ):
        _require_exact(
            f"standardizer {field}",
            dict(standardizer.get("input_identities") or {}).get(field),
            expected,
        )
    if standardizer.get("fitted_on_partition") != "train":
        raise M1MemoryError("The M1 distance standardizer is not train-only.")

    caches = dict(payload.get("stream_caches") or {})
    if set(caches) != {"train", "validation"}:
        raise M1MemoryError(
            "The M1 Stage-1 result must bind both stream-cache partitions."
        )
    if stream_cache_root is not None:
        for partition, recorded in caches.items():
            store, manifest = load_stream_store(Path(stream_cache_root), partition)
            store.close()
            _require_exact(
                f"{partition} stream cache",
                manifest["stream_cache_sha256"],
                recorded["stream_cache_sha256"],
            )
    return payload



# --------------------------------------------------------------------------
# Bounded-memory production path
#
# Attempt 1 died with exit 137 after 6h41m, in a way strongly consistent with
# host memory exhaustion, because the earlier implementation held the whole
# corpus in Python objects before persisting anything. Everything below is
# row-aligned and disk-backed: peak memory is a function of chunk size, stream
# length and the SELECTED supervised populations, never of the full stream.
#
# The in-memory functions above are retained as a small, readable reference
# implementation for the equivalence tests. They are no longer on the canonical
# route.
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class M1SelectedMemory:
    """Memory columns for one bounded, already-selected evidence population.

    The evidence helpers only ever read these fields, so the full-stream
    `M1StreamMemory` never has to exist in RAM on the production route.
    """

    cold_start_bins: tuple[str, ...]
    past_observed_count: np.ndarray
    past_update_count: np.ndarray
    d_short: np.ndarray
    d_long: np.ndarray
    prototype_disagreement: np.ndarray


STAGING_PREFIX: Final = ".staging-"
STAGING_CLAIM_NAME: Final = "M1_STAGING_CLAIM.json"


def _ordered_stream_references(
    feature_root: Path, partition: str
) -> tuple[dict[StreamKey, tuple[B4WindowReference, ...]], list[B4WindowReference]]:
    """Frozen metadata in canonical stream-then-start-sample order."""
    references = load_b4_references(Path(feature_root), partition, primary_only=False)
    streams = build_causal_streams(references)
    ordered = [reference for key in sorted(streams) for reference in streams[key]]
    return streams, ordered


def _stream_slices(
    ordered: Sequence[B4WindowReference],
) -> list[tuple[StreamKey, int, int]]:
    """Contiguous `[begin, end)` row ranges, one per causal stream.

    Canonical order is stream-major, so every stream is already a contiguous
    slice of the store. That is what lets memory generation read one stream at a
    time instead of indexing a whole-corpus mapping.
    """
    slices: list[tuple[StreamKey, int, int]] = []
    begin = 0
    for position, reference in enumerate(ordered):
        key = stream_key(reference)
        if position and key != stream_key(ordered[position - 1]):
            slices.append((stream_key(ordered[position - 1]), begin, position))
            begin = position
    if ordered:
        slices.append((stream_key(ordered[-1]), begin, len(ordered)))
    return slices


def _sorted_id_index(store: M1RowStore) -> tuple[np.ndarray, np.ndarray]:
    """A vectorised stable-ID index.

    `np.argsort` over the identity column plus `searchsorted` replaces the
    2.2 M-entry Python dictionary the old path built. The cost is two int64/str
    arrays rather than millions of interned objects.
    """
    identifiers = np.asarray(store.array(STABLE_ID_FILE))
    order = np.argsort(identifiers, kind="stable")
    return identifiers[order], order


def _positions_for(
    sorted_ids: np.ndarray, order: np.ndarray, wanted: np.ndarray
) -> np.ndarray:
    """Row positions for a chunk of identifiers; -1 where absent."""
    keys = np.asarray(wanted, dtype=sorted_ids.dtype)
    slot = np.searchsorted(sorted_ids, keys)
    slot = np.clip(slot, 0, max(sorted_ids.shape[0] - 1, 0))
    hit = sorted_ids.shape[0] > 0
    matched = (sorted_ids[slot] == keys) if hit else np.zeros(keys.shape, dtype=bool)
    positions = np.where(matched, order[slot], -1)
    return positions.astype(np.int64)


def _write_identity_columns(
    store: M1RowStore, ordered: Sequence[B4WindowReference], *, chunk_rows: int
) -> None:
    stable = store.array(STABLE_ID_FILE)
    records = store.array(RECORD_ID_FILE)
    channels = store.array(CHANNEL_INDEX_FILE)
    starts = store.array(START_SAMPLE_FILE)
    for begin in range(0, len(ordered), chunk_rows):
        block = ordered[begin : begin + chunk_rows]
        end = begin + len(block)
        stable[begin:end] = np.asarray([r.stable_id for r in block], dtype=stable.dtype)
        records[begin:end] = np.asarray(
            [r.record_id for r in block], dtype=records.dtype
        )
        channels[begin:end] = np.asarray(
            [int(r.channel_index) for r in block], dtype=np.int64
        )
        starts[begin:end] = np.asarray(
            [int(r.start_sample) for r in block], dtype=np.int64
        )
    store.flush()


def _fill_embeddings(
    store: M1RowStore,
    ordered: Sequence[B4WindowReference],
    *,
    cache: Any,
    encoder,
    source: Path,
    partition: str,
    waveform_batches_for,
    chunk_rows: int,
) -> dict[str, Any]:
    """Write every 128-d embedding straight into its disk-backed row.

    Primary rows are copied from the frozen P1 cache. Extra rows are read
    through the reviewed waveform contract and encoded in batches of
    `WAVEFORM_BATCH_SIZE`; each batch is written to disk and released, so no
    `dict[str, ndarray]` of newly extracted rows ever exists.
    """
    representation = store.array(REPRESENTATION_FILE)
    cache_position = {key: index for index, key in enumerate(cache.stable_ids)}
    before = _model_state_digest(encoder)

    extra_rows: list[int] = []
    extra_refs: list[B4WindowReference] = []
    reads = 0
    extracted = 0
    extra_digest = StreamingContentDigest((0, EMBEDDING_DIM), np.float32)
    extra_ids: list[str] = []

    def flush_batch() -> None:
        nonlocal reads, extracted
        if not extra_refs:
            return
        batches = (
            waveform_batches_for(partition, tuple(r.stable_id for r in extra_refs))
            if waveform_batches_for is not None
            else canonical_waveform_batches(
                tuple(extra_refs), Path(source), batch_size=WAVEFORM_BATCH_SIZE
            )
        )
        produced: set[str] = set()
        cursor = 0
        for identifiers, waveforms, stats in batches:
            embeddings, _ = extract_frozen_embeddings(encoder, waveforms)
            block = embeddings.to(torch.float32).numpy()
            reads = max(reads, int(getattr(stats, "source_reads", reads)))
            for offset, key in enumerate(identifiers):
                identifier = str(key)
                if identifier in cache_position:
                    raise M1MemoryError(
                        f"Row {identifier} was re-extracted although it is "
                        "already in the frozen P1 cache; the overlap must be "
                        "reused, not duplicated."
                    )
                if identifier in produced:
                    raise M1MemoryError(
                        f"Row {identifier} was produced more than once."
                    )
                if identifier != extra_refs[cursor].stable_id:
                    raise M1MemoryError(
                        "The waveform iterator produced rows out of the "
                        "requested order; row alignment cannot be proven."
                    )
                produced.add(identifier)
                representation[extra_rows[cursor], :EMBEDDING_DIM] = block[offset]
                cursor += 1
                extracted += 1
        if cursor != len(extra_refs):
            raise M1MemoryError(
                f"The waveform iterator produced {cursor} of "
                f"{len(extra_refs)} requested rows."
            )
        extra_refs.clear()
        extra_rows.clear()

    for begin in range(0, len(ordered), chunk_rows):
        for offset, reference in enumerate(ordered[begin : begin + chunk_rows]):
            row = begin + offset
            position = cache_position.get(reference.stable_id)
            if position is not None:
                representation[row, :EMBEDDING_DIM] = cache.embeddings[position]
                continue
            extra_ids.append(reference.stable_id)
            extra_rows.append(row)
            extra_refs.append(reference)
            if len(extra_refs) >= WAVEFORM_BATCH_SIZE:
                flush_batch()
    flush_batch()
    store.flush()

    after = _model_state_digest(encoder)
    if before != after:
        raise M1MemoryError(
            "The locked B4-B encoder state changed during full-stream extraction."
        )
    if extracted != len(extra_ids):
        raise M1MemoryError("Extra-row extraction count does not match the plan.")

    # Digest the extracted rows by re-reading them from disk in bounded blocks.
    if extra_ids:
        sorted_ids, order = _sorted_id_index(store)
        positions = _positions_for(
            sorted_ids, order, np.asarray(extra_ids, dtype=sorted_ids.dtype)
        )
        extra_digest = StreamingContentDigest(
            (len(extra_ids), EMBEDDING_DIM), np.float32
        )
        for begin in range(0, len(extra_ids), chunk_rows):
            rows = positions[begin : begin + chunk_rows]
            extra_digest.update(
                np.asarray(representation[rows, :EMBEDDING_DIM], dtype=np.float32)
            )
    return {
        "primary_rows_reused": len(cache_position),
        "rows_newly_extracted": extracted,
        "extra_disjoint_from_primary_cache": True,
        "extra_ordered_stable_id_sha256": (
            streaming_ordered_stable_id_digest(iter(extra_ids)) if extra_ids else None
        ),
        "extra_embedding_content_sha256": (
            extra_digest.hexdigest() if extra_ids else None
        ),
        "waveform_source_reads": reads,
        "extraction_receipt": {
            "encoder_state_sha256_before": before,
            "encoder_state_sha256_after": after,
            "encoder_state_unchanged": True,
            "encoder_fine_tuned": False,
            "embedding_tap": EMBEDDING_TAP,
            "embedding_dim": EMBEDDING_DIM,
            "batch_size": WAVEFORM_BATCH_SIZE,
            "rows_extracted": extracted,
        },
    }


def _audit_primary_overlap(
    store: M1RowStore,
    ordered: Sequence[B4WindowReference],
    *,
    cache: Any,
    encoder,
    source: Path,
    partition: str,
    waveform_batches_for,
) -> dict[str, Any]:
    """Re-extract a deterministic primary sample and compare it to the cache."""
    by_id = {reference.stable_id: reference for reference in ordered}
    step = max(len(cache.stable_ids) // PRIMARY_AUDIT_ROWS, 1)
    sample = [
        by_id[key]
        for index, key in enumerate(cache.stable_ids)
        if index % step == 0 and key in by_id
    ][:PRIMARY_AUDIT_ROWS]
    if not sample:
        return {
            "re_extracted_primary_rows": 0,
            "re_extracted_primary_bitwise_identical": None,
            "re_extracted_primary_max_abs_deviation": None,
            "primary_audit_tolerance": PRIMARY_AUDIT_TOLERANCE,
        }
    cache_position = {key: index for index, key in enumerate(cache.stable_ids)}
    batches = (
        waveform_batches_for(partition, tuple(r.stable_id for r in sample))
        if waveform_batches_for is not None
        else canonical_waveform_batches(
            tuple(sample), Path(source), batch_size=WAVEFORM_BATCH_SIZE
        )
    )
    identical = True
    deviation = 0.0
    checked = 0
    for identifiers, waveforms, _stats in batches:
        embeddings, _ = extract_frozen_embeddings(encoder, waveforms)
        block = embeddings.to(torch.float32).numpy()
        for offset, key in enumerate(identifiers):
            frozen = cache.embeddings[cache_position[str(key)]]
            identical = identical and bool(np.array_equal(block[offset], frozen))
            deviation = max(
                deviation,
                float(
                    np.max(
                        np.abs(
                            block[offset].astype(np.float64) - frozen.astype(np.float64)
                        )
                    )
                ),
            )
            checked += 1
    if deviation > PRIMARY_AUDIT_TOLERANCE:
        raise M1MemoryError(
            "Re-extracted primary embeddings differ from the frozen P1 cache by "
            f"{deviation}, beyond the batch-grouping tolerance "
            f"{PRIMARY_AUDIT_TOLERANCE}. The overlap identity is not exact: the "
            "encoder, cache or waveform source differs."
        )
    return {
        "re_extracted_primary_rows": checked,
        "re_extracted_primary_bitwise_identical": identical,
        "re_extracted_primary_max_abs_deviation": deviation,
        "primary_audit_tolerance": PRIMARY_AUDIT_TOLERANCE,
    }


def _fill_physiology(
    store: M1RowStore,
    *,
    feature_root: Path,
    partition: str,
    transform: PhysiologyTransform,
    chunk_rows: int,
) -> dict[str, Any]:
    """Apply the exact frozen transform in bounded per-file chunks.

    The transform itself is the reviewed frozen artifact and is never refitted;
    only the plumbing changes. The old path built a whole-corpus
    `dict[str, ndarray]` of raw morphology, which is one of the allocations
    that exhausted memory.
    """
    evaluated = require_p1_partition(partition)
    representation = store.array(REPRESENTATION_FILE)
    sorted_ids, order = _sorted_id_index(store)
    columns = list(morphology_columns())
    filled = np.zeros(representation.shape[0], dtype=bool)
    files = 0

    for path in sorted((Path(feature_root) / evaluated).glob("*.npz")):
        files += 1
        with np.load(path, allow_pickle=False, mmap_mode="r") as archive:
            identifiers = np.asarray(archive["stable_ids"])
            features = archive["features"]
            for begin in range(0, identifiers.shape[0], chunk_rows):
                end = begin + chunk_rows
                keys = np.asarray(identifiers[begin:end], dtype=sorted_ids.dtype)
                positions = _positions_for(sorted_ids, order, keys)
                present = positions >= 0
                if not np.any(present):
                    continue
                raw = np.asarray(features[begin:end], dtype=np.float64)[:, columns]
                values = transform.transform(raw[present])
                representation[positions[present], EMBEDDING_DIM:] = values
                filled[positions[present]] = True
    store.flush()

    missing = int(np.sum(~filled))
    if missing:
        raise M1MemoryError(
            f"{missing} full-stream rows have no frozen morphology_v1 record."
        )
    return {
        "physiology_files_read": files,
        "physiology_dim": PHYSIOLOGY_DIM,
        "physiology_refitted": False,
        "physiology_transform_sha256": transform.as_dict()["transform_sha256"],
        "physiology_schema_sha256": MORPHOLOGY_SCHEMA_SHA256,
    }


def _generate_memory_into_store(
    store: M1RowStore,
    slices: Sequence[tuple[StreamKey, int, int]],
    standardizer: M1DistanceStandardizer,
) -> None:
    """Replay every causal stream, writing memory features straight to disk.

    One stream at a time: each is a contiguous row range, so the working set is
    one stream's representation rather than the corpus. Score-before-update is
    unchanged — `DualTimescaleMemory.observe` is the same reviewed object.
    """
    representation = store.array(REPRESENTATION_FILE)
    starts = store.array(START_SAMPLE_FILE)
    d_short = store.array(D_SHORT_FILE)
    d_long = store.array(D_LONG_FILE)
    observed = store.array(PAST_OBSERVED_FILE)
    updated = store.array(PAST_UPDATE_FILE)
    disagreement = store.array(DISAGREEMENT_FILE)
    ages = store.array(RECORDING_AGE_FILE)
    bins = store.array(COLD_START_BIN_FILE)
    prior = standardizer.prior_vector()

    for _key, begin, end in slices:
        memory = DualTimescaleMemory(prior)
        block = np.asarray(representation[begin:end], dtype=np.float64)
        standardized = standardizer.standardize(block)
        origin = int(starts[begin])
        for offset in range(end - begin):
            features = memory.observe(standardized[offset])
            row = begin + offset
            d_short[row] = features.d_short
            d_long[row] = features.d_long
            observed[row] = features.past_observed_count
            updated[row] = features.past_update_count
            disagreement[row] = features.prototype_disagreement
            age = (int(starts[row]) - origin) / SAMPLING_FREQUENCY_HZ
            ages[row] = age
            bins[row] = cold_start_bin(age)
    store.flush()


def _staging_directory(cache_root: Path, partition: str) -> Path:
    return Path(cache_root) / f"{STAGING_PREFIX}{partition}"


def scan_staging_directories(cache_root: Path) -> list[str]:
    """Report partial staging areas. They are never resumed or deleted."""
    root = Path(cache_root)
    if not root.exists():
        return []
    return sorted(
        str(path) for path in root.glob(f"{STAGING_PREFIX}*") if path.is_dir()
    )


def materialize_stream_store(
    partition: str,
    *,
    cache_root: Path,
    p1_cache_root: Path,
    feature_root: Path,
    source: Path,
    b4b_run_dir: Path,
    p1b_run_dir: Path,
    standardizer: M1DistanceStandardizer | None,
    manifest_fields: Mapping[str, Any],
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    _waveform_batches_for=None,
) -> tuple[dict[str, Any], M1DistanceStandardizer]:
    """Build one immutable, row-aligned, disk-backed stream cache.

    Order: identity columns -> embeddings -> physiology -> (train only) fit the
    standardizer -> causal memory -> manifest -> promote staging to canonical.
    Nothing is promoted until every digest is written, so a crash leaves a
    clearly-marked staging area rather than a plausible cache.
    """
    evaluated = require_p1_partition(partition)
    final = Path(cache_root) / evaluated
    if final.exists():
        raise M1MemoryError(
            f"An M1 stream cache already exists at {final}. A complete cache "
            "must be reused and a partial cache requires documented human "
            "review; it is never overwritten or automatically repaired."
        )
    staging = _staging_directory(cache_root, evaluated)
    if staging.exists():
        raise M1MemoryError(
            f"A partial M1 staging area exists at {staging}. It is never "
            "resumed, repaired or deleted automatically; human review is "
            "required."
        )

    transform = load_frozen_physiology_transform(p1b_run_dir)
    cache = load_p1_embedding_cache(Path(p1_cache_root), evaluated)
    streams, ordered = _ordered_stream_references(feature_root, evaluated)
    slices = _stream_slices(ordered)

    staging.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        staging / STAGING_CLAIM_NAME,
        {
            "artifact_class": "m1_stream_cache_staging_area",
            "is_a_valid_cache": False,
            "partition": evaluated,
            "schema": M1_STREAM_CACHE_SCHEMA,
            "resume_permitted": False,
            "automatic_repair_permitted": False,
            "automatic_deletion_permitted": False,
        },
    )
    spec = M1StoreSpec(rows=len(ordered), representation_dim=REPRESENTATION_DIM)
    store = M1RowStore(staging, spec, create=True)
    try:
        _write_identity_columns(store, ordered, chunk_rows=chunk_rows)
        encoder = load_official_b4b_encoder(Path(b4b_run_dir))
        audit = _fill_embeddings(
            store,
            ordered,
            cache=cache,
            encoder=encoder,
            source=source,
            partition=evaluated,
            waveform_batches_for=_waveform_batches_for,
            chunk_rows=chunk_rows,
        )
        if audit["rows_newly_extracted"]:
            audit.update(
                _audit_primary_overlap(
                    store,
                    ordered,
                    cache=cache,
                    encoder=encoder,
                    source=source,
                    partition=evaluated,
                    waveform_batches_for=_waveform_batches_for,
                )
            )
        else:
            audit.update(
                {
                    "re_extracted_primary_rows": 0,
                    "re_extracted_primary_bitwise_identical": None,
                    "re_extracted_primary_max_abs_deviation": None,
                    "primary_audit_tolerance": PRIMARY_AUDIT_TOLERANCE,
                }
            )
        del encoder
        audit.update(
            _fill_physiology(
                store,
                feature_root=feature_root,
                partition=evaluated,
                transform=transform,
                chunk_rows=chunk_rows,
            )
        )
        representation = store.array(REPRESENTATION_FILE)
        for begin in range(0, representation.shape[0], chunk_rows):
            block = np.asarray(representation[begin : begin + chunk_rows])
            if not np.all(np.isfinite(block)):
                raise M1MemoryError(
                    "A non-finite fused representation was produced despite the "
                    "frozen P1 transformation. M1 refuses rather than skipping "
                    "the window."
                )

        if standardizer is None:
            if evaluated != "train":
                raise M1MemoryError(
                    "The distance standardizer must be fitted from the train "
                    "stream before any validation memory is generated."
                )
            positions = locate_rows(store, cache.stable_ids)
            standardizer = build_distance_standardizer_from_rows(
                np.asarray(
                    store.gather(REPRESENTATION_FILE, positions), dtype=np.float64
                ),
                primary_train_stable_ids=cache.stable_ids,
                upstream_identities=manifest_fields["upstream_identities"],
            )
            write_json_atomic(
                Path(cache_root) / STANDARDIZER_NAME, standardizer.as_dict()
            )
            del positions

        _generate_memory_into_store(store, slices, standardizer)
        manifest = _bounded_stream_cache_manifest(
            store,
            partition=evaluated,
            streams=len(slices),
            record_ids=sorted({key[0] for key, _b, _e in slices}),
            channel_indices=sorted({int(key[1]) for key, _b, _e in slices}),
            audit=audit,
            standardizer_sha256=standardizer.as_dict()["standardizer_sha256"],
            manifest_fields=manifest_fields,
        )
        write_json_atomic(staging / STREAM_CACHE_MANIFEST_NAME, manifest)
        store.close()
        # The marker is rewritten rather than deleted: this module has no
        # artifact-deletion path at all, so a promoted cache carries its own
        # staging provenance instead.
        write_json_atomic(
            staging / STAGING_CLAIM_NAME,
            {
                "artifact_class": "m1_stream_cache_staging_area",
                "is_a_valid_cache": True,
                "promoted_to_canonical_cache": True,
                "partition": evaluated,
                "schema": M1_STREAM_CACHE_SCHEMA,
                "resume_permitted": False,
                "automatic_repair_permitted": False,
                "automatic_deletion_permitted": False,
            },
        )
        staging.rename(final)
        return manifest, standardizer
    except BaseException:
        store.close()
        # The staging area is deliberately left exactly as it is: no delete, no
        # repair, no resume. Human review decides what happens to it.
        raise


def _bounded_stream_cache_manifest(
    store: M1RowStore,
    *,
    partition: str,
    streams: int,
    record_ids: Sequence[str],
    channel_indices: Sequence[int],
    audit: Mapping[str, Any],
    standardizer_sha256: str,
    manifest_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every frozen identity, with all digests computed by streaming."""
    fields = dict(manifest_fields)
    manifest: dict[str, Any] = {
        "artifact_class": "m1_full_stream_memory_cache",
        "m1_stream_cache_schema": M1_STREAM_CACHE_SCHEMA,
        "storage": "row_aligned_memmapped_npy_directory",
        "partition": partition,
        "m1_protocol_sha256": M1_PROTOCOL_SHA256,
        "p1_protocol_sha256": P1_PROTOCOL_SHA256,
        "p1_retention_decision_sha256": P1_RETENTION_DECISION_SHA256,
        "p1_stage1_suite_sha256": fields["p1_stage1_suite_sha256"],
        "p1b_experiment_lock_sha256": fields["p1b_lock_sha256"],
        "encoder_checkpoint_sha256": B4B_CHECKPOINT_SHA256,
        "encoder_experiment_lock_sha256": B4B_EXPERIMENT_LOCK_SHA256,
        "embedding_tap": EMBEDDING_TAP,
        "physiology_transform_sha256": fields["physiology_transform_sha256"],
        "physiology_schema_sha256": MORPHOLOGY_SCHEMA_SHA256,
        "p1_embedding_cache_sha256": fields["embedding_cache_sha256"],
        "development_feature_integrity_sha256": fields["feature_integrity_sha256"],
        "development_source_integrity_sha256": fields["source_integrity_sha256"],
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "distance_standardizer_sha256": standardizer_sha256,
        "representation_dim": REPRESENTATION_DIM,
        "full_stream_row_count": int(store.spec.rows),
        "stream_count": int(streams),
        "record_ids": list(record_ids),
        "channel_indices": list(channel_indices),
        "ordered_stable_id_sha256": store.stable_id_digest(),
        "ordered_chronology_sha256": store.chronology_digest(),
        "representation_content_sha256": store.content_digest(REPRESENTATION_FILE),
        "d_short_content_sha256": store.content_digest(D_SHORT_FILE),
        "d_long_content_sha256": store.content_digest(D_LONG_FILE),
        "history_count_sha256": store.paired_content_digest(
            PAST_OBSERVED_FILE, PAST_UPDATE_FILE
        ),
        "primary_rows_reused": audit["primary_rows_reused"],
        "rows_newly_extracted": audit["rows_newly_extracted"],
        "primary_overlap_audit": dict(audit),
        "label_independent_history": True,
        "update_policy": UPDATE_POLICY,
        "contamination_safe": CONTAMINATION_SAFE,
        "alpha_short": ALPHA_SHORT,
        "alpha_long": ALPHA_LONG,
        "memory_features": ["d_short", "d_long"],
        "git_sha": fields["git_sha"],
        "git_dirty": fields["git_dirty"],
        "environment_dependency_digest": fields["dependency_digest"],
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "artifact_sha256": store.artifact_digests(),
    }
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    return manifest


def load_stream_store(
    cache_root: Path, partition: str, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> tuple[M1RowStore, dict[str, Any]]:
    """Open and fully re-verify a materialized store without loading it whole.

    Every refusal the reviewed loader enforced still applies; only the memory
    profile changes. Content digests are recomputed in bounded chunks and the
    chronology digest is re-derived from the persisted arrays.
    """
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
    if manifest.get("m1_stream_cache_schema") != M1_STREAM_CACHE_SCHEMA:
        raise M1MemoryError(
            f"M1 stream cache schema {manifest.get('m1_stream_cache_schema')!r} "
            f"is not the supported {M1_STREAM_CACHE_SCHEMA}."
        )
    for field, expected in (
        ("m1_protocol_sha256", M1_PROTOCOL_SHA256),
        ("p1_protocol_sha256", P1_PROTOCOL_SHA256),
        ("p1_retention_decision_sha256", P1_RETENTION_DECISION_SHA256),
        ("p1_stage1_suite_sha256", FROZEN_P1_STAGE1_SUITE_SHA256),
        ("p1b_experiment_lock_sha256", FROZEN_P1B_LOCK_SHA256),
        ("physiology_transform_sha256", FROZEN_PHYSIOLOGY_TRANSFORM_SHA256),
        ("p1_embedding_cache_sha256", FROZEN_P1_EMBEDDING_CACHE_SHA256[evaluated]),
        (
            "development_feature_integrity_sha256",
            FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256,
        ),
        (
            "development_source_integrity_sha256",
            FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256,
        ),
        ("encoder_checkpoint_sha256", B4B_CHECKPOINT_SHA256),
        ("encoder_experiment_lock_sha256", B4B_EXPERIMENT_LOCK_SHA256),
        ("split_sha256", B4_SPLIT_SHA256),
        ("feature_corpus_sha256", FEATURE_CORPUS_SHA256),
        ("physiology_schema_sha256", MORPHOLOGY_SCHEMA_SHA256),
        ("update_policy", UPDATE_POLICY),
        ("alpha_short", ALPHA_SHORT),
        ("alpha_long", ALPHA_LONG),
        ("representation_dim", REPRESENTATION_DIM),
        ("contamination_safe", CONTAMINATION_SAFE),
        ("label_independent_history", True),
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

    spec = M1StoreSpec(
        rows=int(manifest["full_stream_row_count"]),
        representation_dim=REPRESENTATION_DIM,
    )
    store = M1RowStore(directory, spec, create=False)
    for name, expected in dict(manifest["artifact_sha256"]).items():
        if sha256_file(directory / name) != expected:
            raise M1MemoryError(f"M1 store array {name} does not match its digest.")
    for field, name in (
        ("representation_content_sha256", REPRESENTATION_FILE),
        ("d_short_content_sha256", D_SHORT_FILE),
        ("d_long_content_sha256", D_LONG_FILE),
    ):
        if store.content_digest(name, chunk_rows=chunk_rows) != manifest[field]:
            raise M1MemoryError(f"M1 stream cache {field} does not match.")
    if (
        store.paired_content_digest(
            PAST_OBSERVED_FILE, PAST_UPDATE_FILE, chunk_rows=chunk_rows
        )
        != manifest["history_count_sha256"]
    ):
        raise M1MemoryError("M1 stream cache history counts do not match.")
    if store.stable_id_digest(chunk_rows=chunk_rows) != manifest[
        "ordered_stable_id_sha256"
    ]:
        raise M1MemoryError("M1 stream cache row order does not match its identity.")
    if store.chronology_digest(chunk_rows=chunk_rows) != manifest[
        "ordered_chronology_sha256"
    ]:
        raise M1MemoryError(
            "The chronology digest re-derived from the persisted stream cache "
            "arrays does not match the manifest."
        )
    standardizer_path = Path(cache_root) / STANDARDIZER_NAME
    if not standardizer_path.is_file():
        raise M1MemoryError(
            f"M1 stream cache at {directory} has no distance standardizer at "
            f"{standardizer_path}; human review is required."
        )
    standardizer = read_json(standardizer_path)
    if standardizer.get("standardizer_sha256") != manifest[
        "distance_standardizer_sha256"
    ]:
        raise M1MemoryError(
            "The persisted distance standardizer differs from the one this "
            "stream cache was built against."
        )
    M1DistanceStandardizer.from_dict(standardizer)
    return store, manifest


def build_distance_standardizer_from_rows(
    matrix: np.ndarray,
    *,
    primary_train_stable_ids: Sequence[str],
    upstream_identities: Mapping[str, Any],
) -> M1DistanceStandardizer:
    """Fit the frozen distance space from an already-gathered TRAIN matrix.

    The 374,452 x 146 primary TRAIN population is bounded and small relative to
    the full stream, so gathering exactly those rows into float64 is acceptable
    and keeps the fit bit-identical to the reviewed implementation.
    """
    required = (
        "p1_stage1_suite_sha256",
        "p1b_experiment_lock_sha256",
        "physiology_transform_sha256",
        "p1_train_embedding_cache_sha256",
        "encoder_checkpoint_sha256",
    )
    upstream = dict(upstream_identities)
    missing = [key for key in required if not upstream.get(key)]
    if missing:
        raise M1MemoryError(
            f"The distance standardizer must bind exact upstream identities; "
            f"{missing} are absent or null."
        )
    expected = EXPECTED_POPULATIONS["train"]["total"]
    if len(primary_train_stable_ids) != expected:
        raise M1MemoryError(
            f"The standardizer must be fitted on the frozen {expected} primary "
            f"TRAIN rows, received {len(primary_train_stable_ids)}."
        )
    return fit_distance_standardizer(
        np.asarray(matrix, dtype=np.float64),
        partition="train",
        input_identities={
            "m1_protocol_sha256": M1_PROTOCOL_SHA256,
            "p1_protocol_sha256": P1_PROTOCOL_SHA256,
            "p1_stage1_suite_sha256": upstream["p1_stage1_suite_sha256"],
            "p1b_experiment_lock_sha256": upstream["p1b_experiment_lock_sha256"],
            "physiology_transform_sha256": upstream["physiology_transform_sha256"],
            "p1_train_embedding_cache_sha256": upstream[
                "p1_train_embedding_cache_sha256"
            ],
            "encoder_checkpoint_sha256": upstream["encoder_checkpoint_sha256"],
            "ordered_stable_id_sha256": ordered_stable_id_digest(
                primary_train_stable_ids
            ),
            "representation_content_sha256": embedding_content_digest(
                np.asarray(matrix, dtype=np.float64)
            ),
        },
    )

# --------------------------------------------------------------------------
# Read-only preflight
# --------------------------------------------------------------------------


def scan_test_artifacts(*roots: Path | None) -> list[str]:
    """Scan for sealed-test artifacts using P1's conservative semantics.

    Reporting a hardcoded `False` here would make the firewall decorative, so
    the repository run tree and every supplied non-versioned root are actually
    walked.
    """
    found: set[str] = set()
    for path in Path(REPOSITORY_ROOT).glob("cardiosentinel-runs/**/TEST_*"):
        found.add(str(path))
    for root in roots:
        if root is None:
            continue
        directory = Path(root)
        if not directory.exists():
            continue
        for path in directory.glob("**/TEST_*"):
            found.add(str(path))
    return sorted(found)


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
            "partial": False,
        }
        if entry["present"]:
            try:
                store, manifest = load_stream_store(
                    Path(stream_cache_root), partition
                )
                store.close()
                entry["validated"] = True
                entry["schema"] = manifest["m1_stream_cache_schema"]
                entry["stream_cache_sha256"] = manifest["stream_cache_sha256"]
                entry["full_stream_row_count"] = manifest["full_stream_row_count"]
                entry["stream_count"] = manifest["stream_count"]
            except Exception as error:  # surfaced, never silently ignored
                entry["partial"] = True
                entry["error"] = f"{type(error).__name__}: {error}"
        elif entry["directory_exists"]:
            entry["partial"] = True
            entry["error"] = (
                "A partial stream cache directory exists without a manifest; "
                "human review is required. It is never deleted, repaired or "
                "overwritten."
            )
        caches[partition] = entry

    standardizer_path = Path(stream_cache_root) / STANDARDIZER_NAME
    validated_partitions = [k for k, v in caches.items() if v["validated"]]
    present_partitions = [k for k, v in caches.items() if v["directory_exists"]]
    partial_partitions = [k for k, v in caches.items() if v["partial"]]
    # Exactly one canonical partition materialized is NOT healthy initial
    # state: it is a half-finished run that must not be silently completed.
    one_sided = len(validated_partitions) == 1 or (
        len(present_partitions) == 1 and not partial_partitions
    )
    orphan_standardizer = standardizer_path.is_file() and len(
        validated_partitions
    ) != 2
    cache_state = {
        "standardizer_present": standardizer_path.is_file(),
        "validated_partitions": validated_partitions,
        "partial_partitions": partial_partitions,
        "one_partition_only": bool(one_sided),
        "orphan_standardizer": bool(orphan_standardizer),
    }
    staging = scan_staging_directories(Path(stream_cache_root))
    cache_state["staging_directories"] = staging
    test_artifacts = scan_test_artifacts(
        Path(run_root), Path(stream_cache_root), p1_run_root
    )

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
    if test_artifacts:
        status = "test_artifact_present_human_review_required"
    elif any(claimed.values()):
        status = "m1_arm_already_claimed"
    elif partial_partitions or staging:
        status = "partial_stream_cache_human_review_required"
    elif one_sided or orphan_standardizer:
        status = "partial_stream_cache_human_review_required"
    elif not gates_ready:
        status = "upstream_gates_incomplete"
    elif not caches_ready:
        status = "stream_cache_materialization_required"
    else:
        status = "ready_for_canonical_m1_stage1"

    report = {
        "preflight_class": "m1_stage1_readiness",
        "status": status,
        "healthy_initial_status": "stream_cache_materialization_required",
        "ready_for_canonical_m1_stage1": status == "ready_for_canonical_m1_stage1",
        "m1_protocol_sha256": M1_PROTOCOL_SHA256,
        "p1_retention_decision_sha256": P1_RETENTION_DECISION_SHA256,
        "representation_dim": REPRESENTATION_DIM,
        "memory": alphas,
        "boundary": m1_boundary_statement(),
        "execution_governance": {
            "prior_authorized_invocation_count": PRIOR_AUTHORIZED_INVOCATION_COUNT,
            "prior_failed_preclaim_attempt_documented": True,
            "prior_failed_attempt_document": ATTEMPT1_FAILURE_DOCUMENT,
            "prior_attempt_scientific_artifacts_created": False,
            "prior_attempt_arm_claims_created": False,
            "replacement_execution_requires_new_human_authorization": True,
        },
        "arm_claims": claimed,
        "stream_caches": caches,
        "stream_cache_state": cache_state,
        "m1_stream_cache_schema": M1_STREAM_CACHE_SCHEMA,
        "p1_evidence": p1_state,
        "encoder": encoder_state,
        "chronology": chronology,
        "validation_challenge": challenge_state,
        "development_integrity": integrity,
        "test_artifacts_present": bool(test_artifacts),
        "test_artifacts": test_artifacts,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "models_created": 0,
        "artifacts_created": 0,
        "read_only": True,
        "human_review_required": status.endswith("human_review_required"),
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "environment": environment,
        "environment_dependency_digest": dependency_digest,
    }
    # Computed only once the complete final report exists, so the future human
    # authorization can bind this exact digest.
    report["preflight_sha256"] = canonical_sha256(report)
    return report


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
        subject_false_positives = subject_false_positive_evidence(
            validation_labels, scores, validation_subject_ids, threshold
        )

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
                "subject_false_positive_evidence": subject_false_positives,
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
            subject_false_positives=subject_false_positives,
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
    _waveform_batches_for=None,
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
    feature_receipt = validate_development_feature_integrity(Path(feature_root))
    source_receipt = validate_development_source_integrity(
        Path(source), feature_receipt
    )
    transform_sha256 = load_frozen_physiology_transform(
        Path(p1_run_root) / P1B_EXPERIMENT_ID
    ).as_dict()["transform_sha256"]
    challenge = build_validation_challenge_index(Path(feature_root))
    embedding_caches = {
        partition: load_p1_embedding_cache(Path(cache_root), partition).manifest
        for partition in ("train", "validation")
    }
    upstream = require_frozen_upstream_identities(
        p1_suite=suite,
        p1b_lock={"experiment_lock_sha256": p1b_lock_sha256},
        physiology_transform_sha256=transform_sha256,
        embedding_caches=embedding_caches,
        encoder_lock=encoder_lock,
        feature_receipt=feature_receipt,
        source_receipt=source_receipt,
        challenge_selection_sha256=challenge.selection_sha256,
    )
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

    # --- full-stream memory caches (bounded, disk-backed) ---------------
    manifest_fields = {
        "upstream_identities": upstream,
        "p1_stage1_suite_sha256": upstream["p1_stage1_suite_sha256"],
        "p1b_lock_sha256": upstream["p1b_experiment_lock_sha256"],
        "physiology_transform_sha256": upstream["physiology_transform_sha256"],
        "feature_integrity_sha256": upstream["development_feature_integrity_sha256"],
        "source_integrity_sha256": upstream["development_source_integrity_sha256"],
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "dependency_digest": dependency_digest,
    }
    manifests: dict[str, dict[str, Any]] = {}
    standardizer: M1DistanceStandardizer | None = None

    for partition in ("train", "validation"):
        directory = Path(stream_cache_root) / partition
        if (directory / STREAM_CACHE_MANIFEST_NAME).is_file():
            store, manifest = load_stream_store(Path(stream_cache_root), partition)
            store.close()
            manifests[partition] = manifest
            continue
        manifest, standardizer = materialize_stream_store(
            partition,
            cache_root=Path(stream_cache_root),
            p1_cache_root=Path(cache_root),
            feature_root=Path(feature_root),
            source=Path(source),
            b4b_run_dir=Path(b4b_run_dir),
            p1b_run_dir=Path(p1_run_root) / P1B_EXPERIMENT_ID,
            standardizer=standardizer,
            manifest_fields={
                **manifest_fields,
                "embedding_cache_sha256": embedding_caches[partition]["cache_sha256"],
            },
            _waveform_batches_for=_waveform_batches_for,
        )
        # Re-open through the validating loader so the promoted cache is proven
        # from disk, then release every construction object before the next
        # partition begins.
        store, manifest = load_stream_store(Path(stream_cache_root), partition)
        store.close()
        manifests[partition] = manifest

    standardizer_payload = read_json(Path(stream_cache_root) / STANDARDIZER_NAME)

    # --- supervised membership (bounded, targeted) ----------------------
    train_cache = load_p1_embedding_cache(Path(cache_root), "train")
    validation_cache = load_p1_embedding_cache(Path(cache_root), "validation")
    challenge_ids = tuple(item.stable_id for item in challenge.references)
    if set(challenge_ids) & set(validation_cache.stable_ids):
        raise M1MemoryError(
            "The primary validation and challenge populations must stay "
            "disjoint; a shared row indicates a lookup built from the wrong "
            "population."
        )

    train_store, _ = load_stream_store(Path(stream_cache_root), "train")
    train_rows = locate_rows(train_store, train_cache.stable_ids)
    train_base = np.asarray(
        train_store.gather(REPRESENTATION_FILE, train_rows), dtype=np.float32
    )
    train_memory_columns = {
        "d_short": train_store.gather(D_SHORT_FILE, train_rows),
        "d_long": train_store.gather(D_LONG_FILE, train_rows),
    }
    train_store.close()
    del train_store, train_rows

    validation_store, _ = load_stream_store(Path(stream_cache_root), "validation")
    validation_rows = locate_rows(validation_store, validation_cache.stable_ids)
    challenge_rows = locate_rows(validation_store, challenge_ids)
    validation_base = np.asarray(
        validation_store.gather(REPRESENTATION_FILE, validation_rows), dtype=np.float32
    )
    challenge_base = np.asarray(
        validation_store.gather(REPRESENTATION_FILE, challenge_rows), dtype=np.float32
    )
    validation_memory_columns = {
        "d_short": validation_store.gather(D_SHORT_FILE, validation_rows),
        "d_long": validation_store.gather(D_LONG_FILE, validation_rows),
    }
    challenge_memory_columns = {
        "d_short": validation_store.gather(D_SHORT_FILE, challenge_rows),
        "d_long": validation_store.gather(D_LONG_FILE, challenge_rows),
    }
    validation_evidence_rows = M1SelectedMemory(
        cold_start_bins=tuple(
            str(value)
            for value in validation_store.gather(
                COLD_START_BIN_FILE, validation_rows
            )
        ),
        past_observed_count=np.asarray(
            validation_store.gather(PAST_OBSERVED_FILE, validation_rows),
            dtype=np.int64,
        ),
        past_update_count=np.asarray(
            validation_store.gather(PAST_UPDATE_FILE, validation_rows), dtype=np.int64
        ),
        d_short=np.asarray(validation_memory_columns["d_short"], dtype=np.float64),
        d_long=np.asarray(validation_memory_columns["d_long"], dtype=np.float64),
        prototype_disagreement=np.asarray(
            validation_store.gather(DISAGREEMENT_FILE, validation_rows),
            dtype=np.float64,
        ),
    )
    validation_store.close()
    del validation_store, validation_rows, challenge_rows

    def arm_matrix(experiment_id: str, base, columns) -> np.ndarray:
        selected = [columns[name] for name in M1_ARM_FEATURES[experiment_id]]
        return m1_arm_features(
            experiment_id, base, np.stack(selected, axis=1).astype(np.float32)
        )

    locks: dict[str, dict[str, Any]] = {}
    for experiment_id in M1_ARM_ORDER:
        locks[experiment_id] = run_m1_arm(
            experiment_id,
            run_root=Path(run_root),
            train_features=arm_matrix(
                experiment_id, train_base, train_memory_columns
            ),
            train_labels=train_cache.labels,
            validation_features=arm_matrix(
                experiment_id, validation_base, validation_memory_columns
            ),
            validation_labels=validation_cache.labels,
            validation_subject_ids=validation_cache.subject_ids,
            challenge_features=arm_matrix(
                experiment_id, challenge_base, challenge_memory_columns
            ),
            challenge_families=tuple(
                item.target_family for item in challenge.references
            ),
            challenge_subject_ids=tuple(
                item.subject_id for item in challenge.references
            ),
            validation_memory=validation_evidence_rows,
            validation_rows=np.arange(
                validation_evidence_rows.d_short.shape[0], dtype=np.int64
            ),
            train_cache={
                "stream_cache_sha256": manifests["train"]["stream_cache_sha256"],
                "full_stream_row_count": manifests["train"]["full_stream_row_count"],
                "supervised_rows": int(train_base.shape[0]),
            },
            validation_cache={
                "stream_cache_sha256": manifests["validation"][
                    "stream_cache_sha256"
                ],
                "full_stream_row_count": manifests["validation"][
                    "full_stream_row_count"
                ],
                "primary_rows": int(validation_base.shape[0]),
                "challenge_rows": int(challenge_base.shape[0]),
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
    return validate_m1_stage1_results(
        Path(run_root), stream_cache_root=Path(stream_cache_root)
    )
