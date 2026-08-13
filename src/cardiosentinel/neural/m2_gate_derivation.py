"""Read-only TRAIN-only reproduction of the frozen M2-v1 gate constants.

This script recomputes the G3 SQI bounds, the G4 normal-evidence threshold,
and the TRAIN-only sanity table recorded in
`docs/M2_GATE_DERIVATION_RECEIPT_V1.json`, using only already-materialized
TRAIN-partition artifacts:

* the M1-v2 stream memory store (`cardiosentinel-features/m1-stream-memory-v2`),
  which already carries the frozen retained M1L representation `z_t` and
  `d_long(t)` for every TRAIN row -- no B4-B encoder inference and no memory
  replay is performed here, the causal history was materialized once by the
  canonical M1-v2 run and is only read back;
* the frozen COMBINED_V1 feature corpus
  (`cardiosentinel-features/ltstdb-baseline-v1`), for the SQI and
  `morphology_valid` columns;
* the frozen retained M1L head checkpoint
  (`cardiosentinel-runs/phase5-m1-dual-memory-v2/M1L_long_memory_v2/model_selected.pt`),
  loaded read-only for a forward pass only -- no gradient, no optimizer step,
  no parameter is ever written.

No VALIDATION or TEST partition is referenced anywhere in this module. No
frozen document is written. This module only prints a comparison against the
frozen receipt and `m2_gate.py`; freezing a corrected receipt is a separate,
explicitly authorized step.

Every receipt-bound quantity is reproduced using the SAME arithmetic path
the original, uncommitted derivation session used (recovered read-only from
its Claude Code session transcript): `g3_sqi.combined_train_rejection_fraction`
via `1.0 - mean(pass_mask)`, `train_only_sanity.refusal_fractions.*` via a
direct `mean(fail_mask)`, and the G4 descriptive distribution via a dedicated
batch-4096 forward pass over only the PRIMARY TRAIN background-negative
population -- not as a subset of the full-timeline pass. These are not
simplifications of each other: they are float-nonidentical (subtraction
rounding, and batch-composition-dependent GEMM reduction order) even though
every one of them reflects the exact same underlying row-level evidence.
Reproducing the historical path bit-exactly, rather than a cleaner-looking
equivalent, is what lets this module use strict equality with zero
tolerance anywhere.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    read_json,
    require_nonversioned_path,
)
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.features.schema import COMBINED_V1, SIGNAL_V1
from cardiosentinel.neural import m2_gate as GATE
from cardiosentinel.neural.m1_experiment import (
    load_stream_store,
    validate_m1_lock,
    validate_m1_stage1_results,
)
from cardiosentinel.neural.m1_store import (
    CHANNEL_INDEX_FILE,
    D_LONG_FILE,
    OBSERVATION_STATE_FILE,
    RECORD_ID_FILE,
    RECORDING_AGE_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
    locate_rows,
)
from cardiosentinel.neural.p1_experiment import load_p1_embedding_cache
from cardiosentinel.neural.patient_memory import (
    M1L_EXPERIMENT_ID,
    OBSERVATION_AVAILABLE,
    REPOSITORY_ROOT,
    build_deterministic_m1_head,
    m1_arm_features,
)
from cardiosentinel.neural.physiology_fusion import P1_BATCH_SIZE
from cardiosentinel.neural.protocol import SAMPLING_FREQUENCY_HZ, WINDOW_SAMPLES
from cardiosentinel.neural.provenance import runtime_environment

DEFAULT_STREAM_CACHE_ROOT: Final = (
    REPOSITORY_ROOT / "cardiosentinel-features" / "m1-stream-memory-v2"
)
DEFAULT_FEATURE_ROOT: Final = (
    REPOSITORY_ROOT / "cardiosentinel-features" / "ltstdb-baseline-v1"
)
DEFAULT_P1_CACHE_ROOT: Final = (
    REPOSITORY_ROOT / "cardiosentinel-features" / "p1-b4b-embeddings-v1"
)
DEFAULT_M1_RUN_ROOT: Final = (
    REPOSITORY_ROOT / "cardiosentinel-runs" / "phase5-m1-dual-memory-v2"
)

COMBINED_NEEDED_COLUMNS: Final = GATE.G3_SQI_COLUMNS + (
    "finite_sample_fraction",
    "morphology_valid",
)

# The historical g4.json derivation scored the 280,839-row PRIMARY TRAIN
# background-negative population in one dedicated batched pass at this batch
# size, gathered in P1-embedding-cache row order. This is not a tunable
# knob: probing batch sizes {1, 4096, 8192, 280839} against the frozen
# receipt showed the descriptive minimum reproduces bit-exactly at 1, 4096
# and 8192, and only diverges (at the ULP-to-few-ULP level, via GEMM
# reduction-order sensitivity near sigmoid's tail) when the whole 280,839-row
# population is scored as a single batch. 4096 is reproduced here because it
# is what the original derivation actually used.
G4_HISTORICAL_BATCH_SIZE: Final = 4096

# `torch.set_num_threads()` is process-global and sticky. Other test modules
# in this repo (via `cardiosentinel.neural.resource_benchmark`, exercised by
# `test_candidate_experiment.py` / `test_validation_challenge.py`) pin
# intra-op threads to 1 for their own benchmark reproducibility, and that
# setting persists for the rest of a shared pytest process. Verified
# empirically: the frozen M1L head's forward pass is sensitive to intra-op
# thread count (it changes GEMM reduction order), which shifts the
# extreme-tail G4 descriptive minimum by a few ULP -- while every other
# receipt field (means/medians over much larger populations) is insensitive
# to it. Both the original derivation and every standalone run of this
# module observed 16, this machine's untouched PyTorch default; scoring
# always pins to it explicitly, then restores whatever was set before, so
# this module's output never depends on -- and never leaks into -- unrelated
# code sharing the same process.
M1L_INTRA_OP_THREADS: Final = 16


@contextlib.contextmanager
def _pinned_intra_op_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(M1L_INTRA_OP_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class M2GateDerivationError(RuntimeError):
    """Raised when a read-only reproduction of the M2 gate cannot proceed."""


# --------------------------------------------------------------------------
# Loading the frozen receipt this run is reproducing
# --------------------------------------------------------------------------


def load_frozen_receipt() -> dict[str, Any]:
    GATE.validate_m2_protocol()
    GATE.validate_m2_gate_receipt()
    return json.loads(GATE.M2_GATE_RECEIPT_PATH.read_text())


# --------------------------------------------------------------------------
# Stream cache (representation, d_long, identity columns) -- read-only
# --------------------------------------------------------------------------


def open_train_stream_store(cache_root: Path):
    store, manifest = load_stream_store(cache_root, "train")
    states = np.asarray(store.array(OBSERVATION_STATE_FILE))
    if not np.all(states == OBSERVATION_AVAILABLE):
        raise M2GateDerivationError(
            "This derivation assumes every TRAIN row is AVAILABLE, matching the "
            "frozen receipt's 'unavailable_exact_flat_row_count': 0; the cache "
            "on disk disagrees."
        )
    return store, manifest


# --------------------------------------------------------------------------
# COMBINED_V1 SQI + morphology columns, row-joined to the store's order
# --------------------------------------------------------------------------


def _combined_column_indices() -> dict[str, int]:
    names = COMBINED_V1.names
    return {name: names.index(name) for name in COMBINED_NEEDED_COLUMNS}


def _train_record_cache_paths(feature_root: Path) -> dict[str, Path]:
    root = require_nonversioned_path(Path(feature_root), "COMBINED_V1 feature root")
    manifest = read_json(root / FEATURE_MANIFEST_NAME)
    paths: dict[str, Path] = {}
    for entry in manifest.get("records", ()):
        if entry.get("partition") == "train" and entry.get("status") == "complete":
            cache_path = (root / str(entry["cache_path"])).resolve()
            cache_path.relative_to(root)
            paths[str(entry["record_id"])] = cache_path
    return paths


def join_sqi_and_morphology(
    store, manifest, feature_root: Path
) -> dict[str, np.ndarray]:
    """Row-align the frozen SQI + morphology columns to the store's causal order."""
    record_ids = np.asarray(store.array(RECORD_ID_FILE))
    stable_ids = np.asarray(store.array(STABLE_ID_FILE))
    rows = record_ids.shape[0]
    column_indices = _combined_column_indices()
    columns = {
        name: np.full(rows, np.nan, dtype=np.float64)
        for name in COMBINED_NEEDED_COLUMNS
    }

    record_paths = _train_record_cache_paths(feature_root)
    if set(record_paths) != set(manifest["record_ids"]):
        raise M2GateDerivationError(
            "The COMBINED_V1 feature corpus's TRAIN record set does not match "
            "the M1 stream cache's record list."
        )

    boundary = np.empty(rows, dtype=bool)
    boundary[0] = True
    boundary[1:] = record_ids[1:] != record_ids[:-1]
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], rows]

    for start, end in zip(starts, ends, strict=True):
        record_id = str(record_ids[start])
        with np.load(record_paths[record_id], allow_pickle=False) as cached:
            npz_ids = np.asarray(cached["stable_ids"], dtype=np.str_)
            npz_features = np.asarray(cached["features"], dtype=np.float64)
        lookup = {value: index for index, value in enumerate(npz_ids)}
        block_ids = stable_ids[start:end]
        try:
            positions = np.asarray([lookup[sid] for sid in block_ids], dtype=np.int64)
        except KeyError as error:
            raise M2GateDerivationError(
                f"A TRAIN stream row for record {record_id} has no COMBINED_V1 "
                f"match: {error}."
            ) from error
        for name, column in column_indices.items():
            columns[name][start:end] = npz_features[positions, column]

    for name, values in columns.items():
        if np.any(np.isnan(values)):
            raise M2GateDerivationError(
                f"COMBINED_V1 join left unmatched rows for {name!r}."
            )
    return columns


# --------------------------------------------------------------------------
# Frozen M1L head -- loaded read-only, forward pass only
# --------------------------------------------------------------------------


def load_frozen_m1l_head(run_root: Path, receipt: dict[str, Any]) -> torch.nn.Module:
    """Load the retained M1L head, re-verifying its lock and checkpoint digest.

    `validate_m1_lock` re-derives the lock's self-referential canonical-JSON
    digest and checks every artifact hash it records, so the checkpoint is
    proven bit-identical to the frozen retained run before it is ever loaded.
    """
    run_dir = Path(run_root) / M1L_EXPERIMENT_ID
    lock = validate_m1_lock(run_dir)

    lock_digest = lock["experiment_lock_sha256"]
    expected_lock = receipt["retained_m1l_lock_sha256"]
    if lock_digest != expected_lock:
        raise M2GateDerivationError(
            f"Retained M1L experiment lock digest {lock_digest} differs from "
            f"the receipt's {expected_lock}."
        )
    checkpoint_digest = lock["artifact_sha256"]["model_selected.pt"]
    expected_checkpoint = receipt["retained_m1l_checkpoint_sha256"]
    if checkpoint_digest != expected_checkpoint:
        raise M2GateDerivationError(
            f"Retained M1L checkpoint digest {checkpoint_digest} differs from "
            f"the receipt's {expected_checkpoint}."
        )

    head = build_deterministic_m1_head(M1L_EXPERIMENT_ID)
    state_dict = torch.load(run_dir / "model_selected.pt", map_location="cpu")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def score_m1l(
    head: torch.nn.Module, representation: np.ndarray, d_long: np.ndarray
) -> np.ndarray:
    memory = np.asarray(d_long, dtype=np.float64).reshape(-1, 1)
    features = m1_arm_features(M1L_EXPERIMENT_ID, representation, memory)
    outputs: list[np.ndarray] = []
    with _pinned_intra_op_threads(), torch.no_grad():
        for start in range(0, features.shape[0], P1_BATCH_SIZE):
            chunk = torch.from_numpy(features[start : start + P1_BATCH_SIZE])
            outputs.append(torch.sigmoid(head(chunk)).to(torch.float64).numpy())
    scores = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise M2GateDerivationError("The frozen M1L head produced a non-finite score.")
    return scores


# --------------------------------------------------------------------------
# PRIMARY TRAIN background-negative membership
# --------------------------------------------------------------------------


def primary_train_background_negative_mask(stable_ids: np.ndarray, cache) -> np.ndarray:
    """The 280,839-row PRIMARY TRAIN background-negative population.

    "PRIMARY TRAIN" here is the frozen 374,452-row sampled population used to
    fit the M1 distance standardizer and train every P1/M1 head (all TRAIN
    ischemic_positive rows plus a frozen downsample of background_negative),
    not the full unsampled 2,143,599-row primary-family TRAIN population.
    `load_p1_embedding_cache` is that exact frozen selection, with `label`
    already 0 for background_negative rows.
    """
    background = frozenset(
        stable_id
        for stable_id, label in zip(cache.stable_ids, cache.labels, strict=True)
        if int(label) == 0
    )
    return np.isin(stable_ids, np.asarray(sorted(background), dtype=stable_ids.dtype))


# --------------------------------------------------------------------------
# Every source/population/artifact identity currently bound by the receipt,
# other than `environment.dependency_digest` (which this run intentionally
# re-observes under the canonical runtime and is compared separately).
# --------------------------------------------------------------------------


def verify_identity_bindings(
    *,
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    cache,
    current_environment: dict[str, Any],
    m1_run_root: Path,
    stream_cache_root: Path,
) -> dict[str, Any]:
    mismatches: list[str] = []

    def _check(label: str, computed: Any, expected: Any) -> None:
        if computed != expected:
            mismatches.append(f"{label}: computed={computed!r} frozen={expected!r}")

    _check(
        "base_scientific_tree", manifest.get("git_sha"), receipt["base_scientific_tree"]
    )
    _check(
        "environment.amp_enabled",
        current_environment["amp_enabled"],
        receipt["environment"]["amp_enabled"],
    )
    _check(
        "environment.numpy",
        current_environment["numpy_version"],
        receipt["environment"]["numpy"],
    )
    _check(
        "environment.python",
        current_environment["python_version"],
        receipt["environment"]["python"],
    )
    _check(
        "environment.torch",
        current_environment["torch_version"],
        receipt["environment"]["torch"],
    )
    _check("environment.device", "cpu", receipt["environment"]["device"])

    identity = receipt["full_train_stream_identity"]
    _check(
        "full_train_stream_identity.records",
        len(manifest["record_ids"]),
        identity["records"],
    )
    _check(
        "full_train_stream_identity.rows",
        int(manifest["full_stream_row_count"]),
        identity["rows"],
    )
    _check(
        "full_train_stream_identity.streams",
        int(manifest["stream_count"]),
        identity["streams"],
    )

    _check(
        "m1_retention_decision_sha256",
        sha256_file(REPOSITORY_ROOT / "docs" / "M1_MEMORY_RETENTION_DECISION_V1.md"),
        receipt["m1_retention_decision_sha256"],
    )
    _check(
        "m1_v2_protocol_sha256",
        sha256_file(REPOSITORY_ROOT / "docs" / "M1_DUAL_MEMORY_PROTOCOL_V2.md"),
        receipt["m1_v2_protocol_sha256"],
    )
    stage1_payload = validate_m1_stage1_results(
        m1_run_root, stream_cache_root=stream_cache_root
    )
    _check(
        "m1_stage1_suite_sha256",
        stage1_payload["m1_stage1_suite_sha256"],
        receipt["m1_stage1_suite_sha256"],
    )

    positive = int(np.sum(np.asarray(cache.labels) == 1))
    subjects = len(set(cache.subject_ids))
    _check(
        "primary_train_identity.positive",
        positive,
        receipt["primary_train_identity"]["positive"],
    )
    _check(
        "primary_train_identity.rows",
        len(cache.labels),
        receipt["primary_train_identity"]["rows"],
    )
    _check(
        "primary_train_identity.subjects",
        subjects,
        receipt["primary_train_identity"]["subjects"],
    )
    _check(
        "primary_train_identity.p1_embedding_cache_sha256",
        cache.manifest["cache_sha256"],
        receipt["primary_train_identity"]["p1_embedding_cache_sha256"],
    )

    _check(
        "signal_v1_schema_sha256", SIGNAL_V1.sha256, receipt["signal_v1_schema_sha256"]
    )
    _check(
        "g4_normal_evidence.m1l_classification_threshold_for_non_equivalence_reference",
        GATE.M1L_CLASSIFICATION_THRESHOLD,
        receipt["g4_normal_evidence"][
            "m1l_classification_threshold_for_non_equivalence_reference"
        ],
    )

    return {"mismatches": mismatches, "all_bound": not mismatches}


# --------------------------------------------------------------------------
# G3 -- waveform SQI
# --------------------------------------------------------------------------


def _descriptive_distribution(
    values: np.ndarray, quantile_labels: dict[float, str]
) -> dict[str, float]:
    """min/max plus the given quantiles, labeled exactly as the receipt keys them.

    The frozen receipt is not internally consistent in this labeling: G3's 0.50
    entry is keyed "median" while G4's is keyed "q50". `quantile_labels` is
    passed explicitly per call site so this reproduces the receipt's actual
    keys rather than a normalized guess.
    """
    stats = {"min": float(np.min(values)), "max": float(np.max(values))}
    for q, label in quantile_labels.items():
        stats[label] = float(np.quantile(values, q, method="linear"))
    return stats


def derive_g3(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    finite_ok = columns["finite_sample_fraction"] == 1.0
    bounds: dict[str, float] = {}
    single_rejection: dict[str, float] = {}
    descriptive: dict[str, dict[str, float]] = {}
    per_column_fail = np.zeros(finite_ok.shape[0], dtype=bool)
    for name in GATE.G3_SQI_COLUMNS:
        values = columns[name]
        bound = float(
            np.quantile(values, GATE.G3_QUANTILE, method=GATE.G3_QUANTILE_METHOD)
        )
        bounds[name] = bound
        fails = values > bound
        single_rejection[name] = float(np.mean(fails))
        per_column_fail |= fails
        descriptive[name] = _descriptive_distribution(
            values, {0.50: "median", 0.95: "q95", 0.99: "q99"}
        )
    combined_fail = per_column_fail | ~finite_ok
    pass_mask = ~combined_fail
    # The historical derivation computed this field as `1.0 - mean(pass_mask)`
    # (sqi.json) while `train_only_sanity.refusal_fractions.sqi` was computed,
    # in a separate script (sanity.json), as `mean(fail_mask)` on an
    # independently-rebuilt but set-identical mask. The two are mathematically
    # equal but not float-identical (subtraction rounding), and the frozen
    # receipt already carries both values as originally produced -- so both
    # formulas are reproduced here explicitly rather than collapsed to one.
    return {
        "bounds": bounds,
        "single_feature_rejection_fraction": single_rejection,
        "combined_rejection_fraction": float(1.0 - np.mean(pass_mask)),
        "direct_fail_mean": float(np.mean(combined_fail)),
        "descriptive_distribution": descriptive,
        "pass_mask": pass_mask,
    }


# --------------------------------------------------------------------------
# G4 -- deterministic normal-evidence margin
# --------------------------------------------------------------------------


def score_background_negative_population(
    head: torch.nn.Module, store, cache
) -> np.ndarray:
    """Reproduce the historical g4.json scoring path exactly.

    A dedicated forward pass over just the 280,839-row PRIMARY TRAIN
    background-negative population, gathered via `locate_rows` in
    P1-embedding-cache row order and batched at `G4_HISTORICAL_BATCH_SIZE`.
    Scoring this same population as a subset of a full-timeline pass (as an
    earlier version of this module did) changes GEMM reduction order and
    shifts the descriptive minimum by a few ULP; this path is required for
    the receipt's G4 descriptive statistics to reproduce bit-exactly.
    """
    negative_ids = [
        stable_id
        for stable_id, label in zip(cache.stable_ids, cache.labels, strict=True)
        if int(label) == 0
    ]
    positions = locate_rows(store, negative_ids)
    base = np.asarray(store.gather(REPRESENTATION_FILE, positions), dtype=np.float32)
    d_long = np.asarray(store.gather(D_LONG_FILE, positions), dtype=np.float32).reshape(
        -1, 1
    )
    features = np.concatenate([base, d_long], axis=1).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise M2GateDerivationError(
            "G4 background-negative feature matrix contains a non-finite value."
        )
    outputs: list[np.ndarray] = []
    with _pinned_intra_op_threads(), torch.no_grad():
        for start in range(0, features.shape[0], G4_HISTORICAL_BATCH_SIZE):
            chunk = torch.from_numpy(features[start : start + G4_HISTORICAL_BATCH_SIZE])
            outputs.append(torch.sigmoid(head(chunk)).to(torch.float64).numpy())
    scores = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise M2GateDerivationError(
            "The frozen M1L head produced a non-finite G4 background-negative score."
        )
    return scores


def derive_g4(population_scores: np.ndarray) -> dict[str, Any]:
    threshold = float(
        np.quantile(
            population_scores,
            GATE.G4_DERIVATION_QUANTILE,
            method=GATE.G4_QUANTILE_METHOD,
        )
    )
    descriptive = _descriptive_distribution(
        population_scores,
        {
            0.10: "q10",
            0.25: "q25",
            0.50: "q50",
            0.75: "q75",
            0.90: "q90",
            0.95: "q95",
            0.99: "q99",
        },
    )
    return {
        "threshold": threshold,
        "population_rows": int(population_scores.shape[0]),
        "descriptive_distribution": descriptive,
    }


# --------------------------------------------------------------------------
# G5/G6 + causal refractory replay
# --------------------------------------------------------------------------


def causal_refractory_replay(
    store, g3_pass: np.ndarray, g4_pass: np.ndarray, g6_pass: np.ndarray
) -> dict[str, Any]:
    record_ids = np.asarray(store.array(RECORD_ID_FILE))
    channel_indices = np.asarray(store.array(CHANNEL_INDEX_FILE))
    start_samples = np.asarray(store.array(START_SAMPLE_FILE)).astype(np.float64)
    rows = record_ids.shape[0]

    available_time = (start_samples + WINDOW_SAMPLES) / SAMPLING_FREQUENCY_HZ
    trigger_time = np.where(
        ~g4_pass, available_time + GATE.REFRACTORY_DURATION_SECONDS, -np.inf
    )

    boundary = np.empty(rows, dtype=bool)
    boundary[0] = True
    boundary[1:] = (record_ids[1:] != record_ids[:-1]) | (
        channel_indices[1:] != channel_indices[:-1]
    )
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], rows]

    refractory_until_before = np.empty(rows, dtype=np.float64)
    for start, end in zip(starts, ends, strict=True):
        refractory_until_before[start] = -np.inf
        if end - start > 1:
            running_max = np.maximum.accumulate(trigger_time[start : end - 1])
            refractory_until_before[start + 1 : end] = running_max

    g5_pass = available_time >= refractory_until_before
    combined_pre_refractory = g3_pass & g4_pass & g6_pass
    admitted = combined_pre_refractory & g5_pass

    stream_count = int(starts.shape[0])
    per_stream_fraction = np.empty(stream_count, dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        per_stream_fraction[index] = float(np.mean(admitted[start:end]))

    return {
        "g5_pass_mask": g5_pass,
        "combined_pre_refractory_admission_fraction": float(
            np.mean(combined_pre_refractory)
        ),
        "final_update_fraction": float(np.mean(admitted)),
        "refractory_blocked_fraction": float(np.mean(~g5_pass)),
        # Matches the historical sanity.json formula exactly: a direct mean of
        # the (pre-refractory-admissible AND currently blocked) mask, not a
        # subtraction of two independently-rounded fractions.
        "refractory_refusal_fraction": float(
            np.mean(combined_pre_refractory & ~g5_pass)
        ),
        "stream_count": stream_count,
        "per_stream_update_fraction": {
            "min": float(np.min(per_stream_fraction)),
            "q10": float(np.quantile(per_stream_fraction, 0.10, method="linear")),
            "median": float(np.quantile(per_stream_fraction, 0.50, method="linear")),
            "q90": float(np.quantile(per_stream_fraction, 0.90, method="linear")),
            "max": float(np.max(per_stream_fraction)),
        },
        "admitted_mask": admitted,
    }


def cold_start_update_fraction(store, admitted: np.ndarray) -> dict[str, Any]:
    age = np.asarray(store.array(RECORDING_AGE_FILE)).astype(np.float64)
    bins = (
        ("0_5_minutes", 0.0, 300.0),
        ("5_60_minutes", 300.0, 3600.0),
        ("over_60_minutes", 3600.0, float("inf")),
    )
    result: dict[str, Any] = {}
    for name, lower, upper in bins:
        mask = (age >= lower) & (age < upper)
        result[name] = {
            "rows": int(np.count_nonzero(mask)),
            "update_fraction": float(np.mean(admitted[mask])) if np.any(mask) else None,
        }
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_derivation(
    *,
    stream_cache_root: Path = DEFAULT_STREAM_CACHE_ROOT,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    p1_cache_root: Path = DEFAULT_P1_CACHE_ROOT,
    m1_run_root: Path = DEFAULT_M1_RUN_ROOT,
) -> dict[str, Any]:
    receipt = load_frozen_receipt()

    store, manifest = open_train_stream_store(stream_cache_root)
    expected_identity = receipt["full_train_stream_identity"]
    if manifest["stream_cache_sha256"] != expected_identity["stream_cache_sha256"]:
        raise M2GateDerivationError(
            "TRAIN stream cache identity differs from the receipt."
        )
    if (
        manifest["ordered_chronology_sha256"]
        != expected_identity["ordered_chronology_sha256"]
    ):
        raise M2GateDerivationError("TRAIN chronology digest differs from the receipt.")

    stable_ids = np.asarray(store.array(STABLE_ID_FILE))
    representation = np.asarray(store.array(REPRESENTATION_FILE))
    d_long = np.asarray(store.array(D_LONG_FILE))

    columns = join_sqi_and_morphology(store, manifest, feature_root)
    g3 = derive_g3(columns)
    g6_pass = columns["morphology_valid"] == 1.0

    head = load_frozen_m1l_head(m1_run_root, receipt)
    scores = score_m1l(head, representation, d_long)

    cache = load_p1_embedding_cache(p1_cache_root, "train")
    background_negative_mask = primary_train_background_negative_mask(stable_ids, cache)
    expected_bg_rows = receipt["primary_train_identity"]["background_negative"]
    if int(np.count_nonzero(background_negative_mask)) != expected_bg_rows:
        raise M2GateDerivationError(
            "PRIMARY TRAIN background-negative row count "
            f"{int(np.count_nonzero(background_negative_mask))} differs from the "
            f"receipt's {expected_bg_rows}."
        )
    g4_population_scores = score_background_negative_population(head, store, cache)
    if g4_population_scores.shape[0] != expected_bg_rows:
        raise M2GateDerivationError(
            f"G4 background-negative scoring population "
            f"{g4_population_scores.shape[0]} differs from the receipt's "
            f"{expected_bg_rows}."
        )
    g4 = derive_g4(g4_population_scores)
    # Applied to the full-timeline score array (batch composition differs from
    # the dedicated G4 population pass, but the pass/fail *classification* is
    # invariant to that -- verified bit-exact for all 2,208,431 rows).
    g4_pass_mask = scores <= g4["threshold"]

    current_environment = runtime_environment("cpu", 0)
    identity = verify_identity_bindings(
        receipt=receipt,
        manifest=manifest,
        cache=cache,
        current_environment=current_environment,
        m1_run_root=m1_run_root,
        stream_cache_root=stream_cache_root,
    )

    replay = causal_refractory_replay(store, g3["pass_mask"], g4_pass_mask, g6_pass)
    cold_start = cold_start_update_fraction(store, replay["admitted_mask"])

    # Every entry below reproduces the exact historical sanity.json formula:
    # a direct mean of a fail/blocked mask, not `1 - pass_fraction`.
    refusal_fractions = {
        "sqi": g3["direct_fail_mean"],
        "normal_evidence": float(np.mean(~g4_pass_mask)),
        "morphology": float(np.mean(~g6_pass)),
        "refractory": replay["refractory_refusal_fraction"],
    }

    computed = {
        "rows": int(stable_ids.shape[0]),
        "g3_sqi": {
            "frozen_upper_bounds_q99": g3["bounds"],
            "single_feature_train_rejection_fraction": g3[
                "single_feature_rejection_fraction"
            ],
            "combined_train_rejection_fraction": g3["combined_rejection_fraction"],
            "descriptive_distribution": g3["descriptive_distribution"],
        },
        "g4_normal_evidence": {
            "normal_evidence_threshold": g4["threshold"],
            "population_rows": g4["population_rows"],
            "descriptive_distribution": g4["descriptive_distribution"],
        },
        "train_only_sanity": {
            "physical_available_fraction": 1.0,
            "g3_sqi_pass_fraction": float(np.mean(g3["pass_mask"])),
            "g4_normal_evidence_pass_fraction_where_score_exists": float(
                np.mean(g4_pass_mask)
            ),
            "g6_morphology_valid_fraction": float(np.mean(g6_pass)),
            "combined_pre_refractory_admission_fraction": replay[
                "combined_pre_refractory_admission_fraction"
            ],
            "final_m2g_update_fraction_after_causal_refractory": replay[
                "final_update_fraction"
            ],
            "refractory_blocked_fraction": replay["refractory_blocked_fraction"],
            "per_stream_update_fraction": {
                **replay["per_stream_update_fraction"],
                "streams": replay["stream_count"],
            },
            "cold_start_update_fraction": cold_start,
            "refusal_fractions": refusal_fractions,
        },
    }

    scientific_comparison = compare_to_frozen(computed, receipt)
    comparison = {
        "mismatches": scientific_comparison["mismatches"] + identity["mismatches"],
        "reproduced": scientific_comparison["reproduced"] and identity["all_bound"],
    }
    return {
        "receipt_sha256": GATE.M2_GATE_RECEIPT_SHA256,
        "protocol_sha256": GATE.M2_PROTOCOL_SHA256,
        "canonical_dependency_digest": current_environment["dependencies"][
            "installed_packages_sha256"
        ],
        "receipt_dependency_digest": receipt["environment"]["dependency_digest"],
        "computed": computed,
        "identity_bindings": identity,
        "frozen_receipt_excerpt": {
            "g3_sqi": receipt["g3_sqi"],
            "g4_normal_evidence": receipt["g4_normal_evidence"],
            "train_only_sanity": receipt["train_only_sanity"],
        },
        "comparison": comparison,
        "test_accessed": False,
        "validation_accessed": False,
    }


def compare_to_frozen(
    computed: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    """Exact-equality comparison; no tolerance is introduced anywhere.

    Every number here comes from a deterministic numpy operation over
    bit-identical frozen inputs (same checkpoint, same stream cache, same
    feature corpus, same numpy/torch build), so a genuine reproduction is
    bit-identical, not merely close. Any float mismatch, however small, is
    reported and fails reproduction.
    """
    mismatches: list[str] = []

    def _exact(label: str, computed_value: Any, expected_value: Any) -> None:
        if computed_value != expected_value:
            mismatches.append(
                f"{label}: computed={computed_value!r} frozen={expected_value!r}"
            )

    for name, bound in computed["g3_sqi"]["frozen_upper_bounds_q99"].items():
        _exact(
            f"g3_upper_bound[{name}]",
            bound,
            receipt["g3_sqi"]["frozen_upper_bounds_q99"][name],
        )
    for name, fraction in computed["g3_sqi"][
        "single_feature_train_rejection_fraction"
    ].items():
        _exact(
            f"g3_single_feature_rejection[{name}]",
            fraction,
            receipt["g3_sqi"]["single_feature_train_rejection_fraction"][name],
        )
    _exact(
        "g3_combined_rejection",
        computed["g3_sqi"]["combined_train_rejection_fraction"],
        receipt["g3_sqi"]["combined_train_rejection_fraction"],
    )
    for name, stats in computed["g3_sqi"]["descriptive_distribution"].items():
        expected_stats = receipt["g3_sqi"]["descriptive_distribution"][name]
        for stat_name, value in stats.items():
            _exact(
                f"g3_descriptive[{name}][{stat_name}]", value, expected_stats[stat_name]
            )

    _exact(
        "g4_threshold",
        computed["g4_normal_evidence"]["normal_evidence_threshold"],
        receipt["g4_normal_evidence"]["normal_evidence_threshold"],
    )
    _exact(
        "g4_population_rows",
        computed["g4_normal_evidence"]["population_rows"],
        receipt["g4_normal_evidence"]["population"]["rows"],
    )
    for stat_name, value in computed["g4_normal_evidence"][
        "descriptive_distribution"
    ].items():
        _exact(
            f"g4_descriptive[{stat_name}]",
            value,
            receipt["g4_normal_evidence"]["descriptive_distribution"][stat_name],
        )

    sanity = computed["train_only_sanity"]
    frozen_sanity = receipt["train_only_sanity"]
    sanity_keys = (
        "g3_sqi_pass_fraction",
        "g4_normal_evidence_pass_fraction_where_score_exists",
        "g6_morphology_valid_fraction",
        "combined_pre_refractory_admission_fraction",
        "final_m2g_update_fraction_after_causal_refractory",
        "refractory_blocked_fraction",
    )
    for key in sanity_keys:
        _exact(key, sanity[key], frozen_sanity[key])

    for key in ("min", "q10", "median", "q90", "max", "streams"):
        _exact(
            f"per_stream_update_fraction[{key}]",
            sanity["per_stream_update_fraction"][key],
            frozen_sanity["per_stream_update_fraction"][key],
        )

    for bin_name, bin_stats in sanity["cold_start_update_fraction"].items():
        expected_bin = frozen_sanity["cold_start_update_fraction"][bin_name]
        for stat_name in ("rows", "update_fraction"):
            _exact(
                f"cold_start[{bin_name}][{stat_name}]",
                bin_stats[stat_name],
                expected_bin[stat_name],
            )

    for key, value in sanity["refusal_fractions"].items():
        _exact(
            f"refusal_fractions[{key}]", value, frozen_sanity["refusal_fractions"][key]
        )

    return {"mismatches": mismatches, "reproduced": not mismatches}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream-cache-root", type=Path, default=DEFAULT_STREAM_CACHE_ROOT
    )
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--p1-cache-root", type=Path, default=DEFAULT_P1_CACHE_ROOT)
    parser.add_argument("--m1-run-root", type=Path, default=DEFAULT_M1_RUN_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full JSON report. Must not be inside "
        "this repository.",
    )
    args = parser.parse_args(argv)

    if args.output is not None:
        resolved_output = args.output.resolve()
        if (
            REPOSITORY_ROOT in resolved_output.parents
            or resolved_output == REPOSITORY_ROOT
        ):
            raise M2GateDerivationError(
                "--output must be outside the repository; this script never "
                "writes into the frozen tree."
            )

    try:
        report = run_derivation(
            stream_cache_root=args.stream_cache_root,
            feature_root=args.feature_root,
            p1_cache_root=args.p1_cache_root,
            m1_run_root=args.m1_run_root,
        )
    except (M2GateDerivationError, GATE.M2GateError) as error:
        print(f"M2 gate derivation refused: {error}", file=sys.stderr)
        return 1

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text)
    print(text)
    print(
        "\nREPRODUCED"
        if report["comparison"]["reproduced"]
        else "\nMISMATCH -- see comparison.mismatches above",
        file=sys.stderr,
    )
    return 0 if report["comparison"]["reproduced"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
