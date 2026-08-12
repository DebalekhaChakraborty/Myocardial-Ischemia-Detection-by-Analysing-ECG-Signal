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
"""

from __future__ import annotations

import argparse
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
from cardiosentinel.features.schema import COMBINED_V1
from cardiosentinel.neural import m2_gate as GATE
from cardiosentinel.neural.m1_experiment import load_stream_store, validate_m1_lock
from cardiosentinel.neural.m1_store import (
    CHANNEL_INDEX_FILE,
    D_LONG_FILE,
    OBSERVATION_STATE_FILE,
    RECORD_ID_FILE,
    RECORDING_AGE_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
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
    with torch.no_grad():
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


def primary_train_background_negative_mask(
    stable_ids: np.ndarray, p1_cache_root: Path
) -> np.ndarray:
    """The 280,839-row PRIMARY TRAIN background-negative population.

    "PRIMARY TRAIN" here is the frozen 374,452-row sampled population used to
    fit the M1 distance standardizer and train every P1/M1 head (all TRAIN
    ischemic_positive rows plus a frozen downsample of background_negative),
    not the full unsampled 2,143,599-row primary-family TRAIN population.
    `load_p1_embedding_cache` is that exact frozen selection, with `label`
    already 0 for background_negative rows.
    """
    cache = load_p1_embedding_cache(p1_cache_root, "train")
    background = frozenset(
        stable_id
        for stable_id, label in zip(cache.stable_ids, cache.labels, strict=True)
        if int(label) == 0
    )
    return np.isin(stable_ids, np.asarray(sorted(background), dtype=stable_ids.dtype))


# --------------------------------------------------------------------------
# G3 -- waveform SQI
# --------------------------------------------------------------------------


def derive_g3(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    finite_ok = columns["finite_sample_fraction"] == 1.0
    bounds: dict[str, float] = {}
    single_rejection: dict[str, float] = {}
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
    combined_fail = per_column_fail | ~finite_ok
    return {
        "bounds": bounds,
        "single_feature_rejection_fraction": single_rejection,
        "combined_rejection_fraction": float(np.mean(combined_fail)),
        "pass_mask": ~combined_fail,
    }


# --------------------------------------------------------------------------
# G4 -- deterministic normal-evidence margin
# --------------------------------------------------------------------------


def derive_g4(
    scores: np.ndarray, background_negative_mask: np.ndarray
) -> dict[str, Any]:
    population = scores[background_negative_mask]
    threshold = float(
        np.quantile(
            population, GATE.G4_DERIVATION_QUANTILE, method=GATE.G4_QUANTILE_METHOD
        )
    )
    pass_mask = scores <= threshold
    return {
        "threshold": threshold,
        "population_rows": int(population.shape[0]),
        "pass_mask": pass_mask,
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

    background_negative_mask = primary_train_background_negative_mask(
        stable_ids, p1_cache_root
    )
    expected_bg_rows = receipt["primary_train_identity"]["background_negative"]
    if int(np.count_nonzero(background_negative_mask)) != expected_bg_rows:
        raise M2GateDerivationError(
            "PRIMARY TRAIN background-negative row count "
            f"{int(np.count_nonzero(background_negative_mask))} differs from the "
            f"receipt's {expected_bg_rows}."
        )
    g4 = derive_g4(scores, background_negative_mask)

    replay = causal_refractory_replay(store, g3["pass_mask"], g4["pass_mask"], g6_pass)
    cold_start = cold_start_update_fraction(store, replay["admitted_mask"])

    refusal_fractions = {
        "sqi": g3["combined_rejection_fraction"],
        "normal_evidence": 1.0 - float(np.mean(g4["pass_mask"])),
        "morphology": 1.0 - float(np.mean(g6_pass)),
        "refractory": (
            replay["combined_pre_refractory_admission_fraction"]
            - replay["final_update_fraction"]
        ),
    }

    computed = {
        "rows": int(stable_ids.shape[0]),
        "g3_sqi": {
            "frozen_upper_bounds_q99": g3["bounds"],
            "single_feature_train_rejection_fraction": g3[
                "single_feature_rejection_fraction"
            ],
            "combined_train_rejection_fraction": g3["combined_rejection_fraction"],
        },
        "g4_normal_evidence": {
            "normal_evidence_threshold": g4["threshold"],
            "population_rows": g4["population_rows"],
        },
        "train_only_sanity": {
            "physical_available_fraction": 1.0,
            "g3_sqi_pass_fraction": float(np.mean(g3["pass_mask"])),
            "g4_normal_evidence_pass_fraction_where_score_exists": float(
                np.mean(g4["pass_mask"])
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

    comparison = compare_to_frozen(computed, receipt)
    return {
        "receipt_sha256": GATE.M2_GATE_RECEIPT_SHA256,
        "protocol_sha256": GATE.M2_PROTOCOL_SHA256,
        "computed": computed,
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
    mismatches: list[str] = []

    for name, bound in computed["g3_sqi"]["frozen_upper_bounds_q99"].items():
        expected = receipt["g3_sqi"]["frozen_upper_bounds_q99"][name]
        if not np.isclose(bound, expected, rtol=1e-9, atol=1e-12):
            mismatches.append(
                f"g3_upper_bound[{name}]: computed={bound} frozen={expected}"
            )

    combined = computed["g3_sqi"]["combined_train_rejection_fraction"]
    expected_combined = receipt["g3_sqi"]["combined_train_rejection_fraction"]
    if not np.isclose(combined, expected_combined, rtol=1e-6, atol=1e-9):
        mismatches.append(
            f"g3_combined_rejection: computed={combined} frozen={expected_combined}"
        )

    threshold = computed["g4_normal_evidence"]["normal_evidence_threshold"]
    expected_threshold = receipt["g4_normal_evidence"]["normal_evidence_threshold"]
    if not np.isclose(threshold, expected_threshold, rtol=1e-9, atol=1e-15):
        mismatches.append(
            f"g4_threshold: computed={threshold} frozen={expected_threshold}"
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
        computed_value = sanity[key]
        expected_value = frozen_sanity[key]
        if not np.isclose(computed_value, expected_value, rtol=1e-6, atol=1e-9):
            mismatches.append(
                f"{key}: computed={computed_value} frozen={expected_value}"
            )

    for key in ("min", "median", "max"):
        computed_value = sanity["per_stream_update_fraction"][key]
        expected_value = frozen_sanity["per_stream_update_fraction"][key]
        if not np.isclose(computed_value, expected_value, rtol=1e-6, atol=1e-9):
            mismatches.append(
                f"per_stream_update_fraction[{key}]: computed={computed_value} "
                f"frozen={expected_value}"
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
