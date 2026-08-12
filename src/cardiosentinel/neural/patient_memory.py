"""Strictly causal dual-timescale patient memory for M1.

This module owns the causal primitives promised by
`docs/M1_DUAL_MEMORY_PROTOCOL_V1.md`: the chronology contract, the train-only
distance standardizer, the dual-timescale prototype state, deterministic
full-stream feature generation, the M1 candidate heads, and the one-shot claim
helpers.

Three invariants are structural rather than conventional here:

* **Score before update.** `DualTimescaleMemory.observe` computes deviations
  and only then updates, so a window can never influence the prototype used to
  compute its own distance.
* **No label path.** Nothing in this module accepts a label, a target family, a
  score, a threshold or an event state. Memory admission therefore cannot be
  gated on evaluation metadata even by accident.
* **Streams are `(record_id, channel_index)`.** A record carries several
  simultaneous channels; merging them into one history would fabricate
  chronology that does not exist.

The heads reuse `P1FusionHead` rather than redeclaring the layer stack, so
"same head family as P1" is enforced by construction.
"""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from cardiosentinel.baseline.cache import require_nonversioned_path, write_json_atomic
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.metadata import B4WindowReference
from cardiosentinel.neural.physiology_fusion import (
    EMBEDDING_DIM,
    HEAD_DROPOUT,
    HEAD_HIDDEN_DIM,
    P1_BATCH_SIZE,
    P1_EARLY_STOPPING_DELTA,
    P1_EARLY_STOPPING_PATIENCE,
    P1_LEARNING_RATE,
    P1_MAX_EPOCHS,
    P1_SEED,
    P1_WEIGHT_DECAY,
    P1B_EXPERIMENT_ID,
    PHYSIOLOGY_DIM,
    P1FusionHead,
    require_p1_partition,
)
from cardiosentinel.neural.protocol import (
    REPOSITORY_ROOT,
    SAMPLING_FREQUENCY_HZ,
    STRIDE_SECONDS,
)

M1_PROTOCOL_V1_NAME: Final = "M1_DUAL_MEMORY_PROTOCOL_V1"
M1_PROTOCOL_V1_PATH: Final = REPOSITORY_ROOT / "docs" / f"{M1_PROTOCOL_V1_NAME}.md"
# M1-v1 is IMMUTABLE HISTORICAL EVIDENCE. It is superseded prospectively by
# M1-v2 and is never rewritten; its digest must keep validating.
M1_PROTOCOL_V1_SHA256: Final = (
    "08f71c5b54ebd0fcc9c1f26f05d7df2c5a1b0ca5253b8821435a65673ad65253"
)
M1_PHYSICAL_OBSERVATION_DECISION_SHA256: Final = (
    "ba9be6de0da7037e0d99b7c619aabbb09c44f84a32c04e2241a61d8277ed5ce7"
)
M1_ATTEMPT2_CENSUS_SHA256: Final = (
    "8170068ee3f40875428a28374c8bb1accf4b6fbfd3cc510195f6851f954ce1ee"
)
M1_ATTEMPT2_FAILURE_SHA256: Final = (
    "1bf9539f89d179e8cbf6adb7e578d9f78a9e990fbbf906e5ae3679b93ec1310a"
)

M1_PROTOCOL_NAME: Final = "M1_DUAL_MEMORY_PROTOCOL_V2"
M1_PROTOCOL_PATH: Final = REPOSITORY_ROOT / "docs" / f"{M1_PROTOCOL_NAME}.md"
M1_PROTOCOL_SHA256: Final = (
    "31a81358870cd23c2258cf4f307ab8c4dc7bf245bc4bf18a4d1f48fe2aada39c"
)
# Both superseded before any M1 evidence existed, and retained so a stale digest
# is recognised as such rather than merely rejected as unknown.
#
# 52eedc62...: stated the recording-age formula absolutely rather than
# stream-relative, left the standard-deviation convention unstated, and did not
# define the subject-wise false-positive summary.
# cc2e78e7...: described every LTSTDB record as carrying two channels, which
# real-environment read-only validation disproved. The stream key was already
# generic, so execution semantics were unaffected.
SUPERSEDED_M1_PROTOCOL_SHA256: Final = (
    "52eedc628d906ac02619264fc26cd4629e56f05d6c1916448d62a2844c9815f4",
    "cc2e78e720bbb55d3dd51e61a5ea6cd04c77cb77eef41508def3951361ccda61",
    # M1-v1: superseded PROSPECTIVELY, not before use. Two authorized
    # invocations ran under it and produced zero arm claims and zero results.
    M1_PROTOCOL_V1_SHA256,
)
P1_RETENTION_DECISION_SHA256: Final = (
    "7b403709fa0fb12eef65423d830c121fc3ada904266a1b47931d438f5e797d68"
)

# The P1 retention decision kept the complete 18-d physiology vector.
REPRESENTATION_DIM: Final = EMBEDDING_DIM + PHYSIOLOGY_DIM  # 146

M1S_EXPERIMENT_ID: Final = "M1S_short_memory_v2"
M1L_EXPERIMENT_ID: Final = "M1L_long_memory_v2"
M1D_EXPERIMENT_ID: Final = "M1D_dual_memory_v2"
# v1 arm identities are retained so a historical artifact can never be mistaken
# for a v2 one, and so a v1 directory is recognised rather than reused.
M1_V1_EXPERIMENT_IDS: Final = (
    "M1S_short_memory_v1",
    "M1L_long_memory_v1",
    "M1D_dual_memory_v1",
)
M1_EXPERIMENT_IDS: Final = (M1S_EXPERIMENT_ID, M1L_EXPERIMENT_ID, M1D_EXPERIMENT_ID)
M1_RUN_COLLECTION: Final = "phase5-m1-dual-memory-v2"
GLOBAL_CONTROL_EXPERIMENT_ID: Final = P1B_EXPERIMENT_ID

# Memory features, in the one frozen order used by every arm.
MEMORY_FEATURE_NAMES: Final = ("d_short", "d_long")
M1_ARM_FEATURES: Final = {
    M1S_EXPERIMENT_ID: ("d_short",),
    M1L_EXPERIMENT_ID: ("d_long",),
    M1D_EXPERIMENT_ID: ("d_short", "d_long"),
}

# Dual timescale. Half-lives are expressed in updates at the frozen 5 s stride.
MEMORY_STRIDE_SECONDS: Final = STRIDE_SECONDS
SHORT_HALF_LIFE_SECONDS: Final = 300.0
LONG_HALF_LIFE_SECONDS: Final = 3600.0
SHORT_HALF_LIFE_UPDATES: Final = 60
LONG_HALF_LIFE_UPDATES: Final = 720
ALPHA_SHORT: Final = 1.0 - 2.0 ** (-1.0 / SHORT_HALF_LIFE_UPDATES)
ALPHA_LONG: Final = 1.0 - 2.0 ** (-1.0 / LONG_HALF_LIFE_UPDATES)

# --------------------------------------------------------------------------
# Physical observation availability (M1-v2, the only scientific delta from v1)
# --------------------------------------------------------------------------
OBSERVATION_UNINITIALIZED: Final = 0
OBSERVATION_AVAILABLE: Final = 1
OBSERVATION_UNAVAILABLE_EXACT_FLAT: Final = 2
OBSERVATION_STATE_ENUM: Final = {
    "UNINITIALIZED_INVALID_FOR_COMPLETED_CACHE": OBSERVATION_UNINITIALIZED,
    "AVAILABLE": OBSERVATION_AVAILABLE,
    "UNAVAILABLE_EXACT_FLAT": OBSERVATION_UNAVAILABLE_EXACT_FLAT,
}
OBSERVATION_STATE_VERSION: Final = 1
PHYSICAL_OBSERVATION_CONTRACT: Final = (
    "An individually read 2500-sample single-channel physical mV segment that "
    "passes interval validity, header calibration, unit support, mV conversion "
    "and finiteness, but satisfies np.ptp(values) <= np.finfo(np.float64).eps, "
    "is a PHYSICALLY UNAVAILABLE SENSOR OBSERVATION. It receives no B4-B "
    "inference, no representation, no deviation score and no memory update, "
    "while retaining its timeline position and real elapsed time. Every other "
    "failure class remains fatal."
)


def exact_flat_unavailable(values: np.ndarray) -> bool:
    """The frozen physical-availability predicate.

    This is EXACTLY the existing B4 hard dynamic-variation criterion, applied
    to an already fully validated physical mV segment. It is deliberately not a
    near-flat, variance, amplitude, SQI or morphology threshold, and it never
    consults `morphology_valid`: a flat lead is decided from the samples alone.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise M1MemoryError(
                "Physical availability is decided per single channel."
            )
        array = array[:, 0]
    if array.ndim != 1 or array.size == 0:
        raise M1MemoryError("Physical availability needs a 1-D sample vector.")
    if not np.all(np.isfinite(array)):
        # Non-finite is a FATAL class, never reclassified as unavailable.
        raise M1MemoryError(
            "A non-finite waveform segment is a fatal integrity failure and "
            "must never be reclassified as a physically unavailable observation."
        )
    return bool(np.ptp(array) <= np.finfo(np.float64).eps)


UPDATE_POLICY: Final = "available_finite_observation_always_update"
UPDATE_POLICY_STATEMENT: Final = (
    "Every AVAILABLE finite fused observation updates both prototypes after its own "
    "features are recorded. This is intentionally NOT contamination-safe: an "
    "abnormal or confounded window may enter memory. M2 is required before any "
    "safe-adaptation or deployment-safe personalization claim."
)
CONTAMINATION_SAFE: Final = False

COLD_START_BINS: Final = (
    ("0_5_minutes", 0.0, 300.0),
    ("5_60_minutes", 300.0, 3600.0),
    ("over_60_minutes", 3600.0, float("inf")),
)

STANDARDIZER_NAME: Final = "M1_DISTANCE_STANDARDIZER.json"
STREAM_CACHE_MANIFEST_NAME: Final = "M1_STREAM_CACHE_MANIFEST.json"
STREAM_CACHE_ARRAY_NAME: Final = "m1_stream_memory.npz"
STREAM_CACHE_CLAIM_NAME: Final = "M1_STREAM_CACHE_CLAIM.json"
RUN_STATUS_NAME: Final = "RUN_STATUS.json"
ATTEMPT_STATUS_STARTED: Final = "STARTED"
ATTEMPT_STATUS_COMPLETE: Final = "COMPLETE"
ATTEMPT_STATUS_FAILED: Final = "FAILED_OR_INTERRUPTED"

StreamKey = tuple[str, int]


class M1MemoryError(RuntimeError):
    """Raised when a causal M1 step cannot proceed with full integrity."""


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


def validate_m1_protocol(path: Path = M1_PROTOCOL_PATH) -> str:
    """Verify the frozen M1 protocol byte-for-byte."""
    from cardiosentinel.data.provenance import sha256_file

    document = Path(path)
    if not document.is_file():
        raise M1MemoryError(f"Frozen M1 protocol is missing at {document}.")
    digest = sha256_file(document)
    if digest in SUPERSEDED_M1_PROTOCOL_SHA256:
        raise M1MemoryError(
            f"M1 protocol digest {digest} is a draft that was SUPERSEDED "
            "BEFORE USE; no M1 scientific evidence was generated under it. The "
            f"frozen protocol is {M1_PROTOCOL_SHA256}."
        )
    if digest != M1_PROTOCOL_SHA256:
        raise M1MemoryError(
            f"M1 protocol digest {digest} differs from the frozen "
            f"{M1_PROTOCOL_SHA256}. The protocol is immutable."
        )
    return digest


def validate_m1_protocol_v1(path: Path = M1_PROTOCOL_V1_PATH) -> str:
    """Verify the immutable M1-v1 protocol. It is history and must never move."""
    from cardiosentinel.data.provenance import sha256_file

    document = Path(path)
    if not document.is_file():
        raise M1MemoryError(f"Historical M1-v1 protocol is missing at {document}.")
    digest = sha256_file(document)
    if digest != M1_PROTOCOL_V1_SHA256:
        raise M1MemoryError(
            f"M1-v1 protocol digest {digest} differs from the frozen historical "
            f"{M1_PROTOCOL_V1_SHA256}. M1-v1 is immutable evidence."
        )
    return digest


def require_m1_experiment(experiment_id: str) -> str:
    if experiment_id not in M1_EXPERIMENT_IDS:
        raise M1MemoryError(f"Unknown M1 experiment {experiment_id!r}.")
    return experiment_id


def m1_alpha_identity() -> dict[str, Any]:
    """The frozen dual-timescale constants, asserted ordered on every call."""
    if not ALPHA_SHORT > ALPHA_LONG > 0.0:
        raise M1MemoryError("M1 requires alpha_short > alpha_long > 0.")
    return {
        "stride_seconds": MEMORY_STRIDE_SECONDS,
        "short_half_life_seconds": SHORT_HALF_LIFE_SECONDS,
        "long_half_life_seconds": LONG_HALF_LIFE_SECONDS,
        "short_half_life_updates": SHORT_HALF_LIFE_UPDATES,
        "long_half_life_updates": LONG_HALF_LIFE_UPDATES,
        "alpha_short": ALPHA_SHORT,
        "alpha_long": ALPHA_LONG,
        "alpha_rule": "alpha = 1 - 2 ** (-1 / half_life_updates)",
        "swept": False,
        "tuned": False,
    }


# --------------------------------------------------------------------------
# Chronology
# --------------------------------------------------------------------------


def stream_key(reference: B4WindowReference) -> StreamKey:
    """The independent causal state unit.

    Records may carry multiple simultaneous channels -- the frozen development
    corpus holds both 2-channel and 3-channel LTSTDB records, with observed
    indices {0, 1, 2}. The record alone is therefore not a sequential history:
    keying on it would interleave concurrent leads. Each
    (record_id, channel_index) pair is an independent causal state unit, and
    this key is generic over the integer index rather than assuming any
    particular channel count.
    """
    return (reference.record_id, int(reference.channel_index))


def build_causal_streams(
    references: Iterable[B4WindowReference],
) -> dict[StreamKey, tuple[B4WindowReference, ...]]:
    """Group development windows into ordered `(record, channel)` streams.

    `window_start_samples` is the causal order field. Acquisition wall-clock is
    unavailable, and `metadata_json.elapsed_seconds` is feature-generation
    timing rather than chronology, so it is deliberately not consulted.
    """
    grouped: dict[StreamKey, list[B4WindowReference]] = {}
    for reference in references:
        grouped.setdefault(stream_key(reference), []).append(reference)

    streams: dict[StreamKey, tuple[B4WindowReference, ...]] = {}
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: int(item.start_sample))
        starts = [int(item.start_sample) for item in ordered]
        if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
            raise M1MemoryError(
                f"Stream {key} does not have strictly increasing start samples; "
                "the causal order is ambiguous."
            )
        streams[key] = tuple(ordered)
    return streams


def ordered_chronology_digest(
    streams: Mapping[StreamKey, tuple[B4WindowReference, ...]],
) -> str:
    """Order-sensitive digest of `(record_id, channel_index, start_sample)`."""
    rows = [
        [key[0], key[1], int(reference.start_sample)]
        for key in sorted(streams)
        for reference in streams[key]
    ]
    return canonical_sha256({"order": "stream_then_start_sample", "rows": rows})


def cold_start_bin(recording_age_seconds: float) -> str:
    """Bin a window by recording age. Frozen before any M1 metric exists."""
    if not np.isfinite(recording_age_seconds) or recording_age_seconds < 0.0:
        raise M1MemoryError("Recording age must be finite and non-negative.")
    for name, low, high in COLD_START_BINS:
        if low <= recording_age_seconds < high:
            return name
    raise M1MemoryError(f"No cold-start bin covers {recording_age_seconds}.")


# --------------------------------------------------------------------------
# Distance standardizer
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M1DistanceStandardizer:
    """A TRAIN-ONLY global transform for the memory distance space.

    Fitted on the canonical frozen PRIMARY TRAIN fused representation only. It
    is a model statistic, so it stays tied to the prospectively defined
    supervised train population rather than to validation or challenge
    composition. There is no patient-specific normalization anywhere.
    """

    means: tuple[float, ...]
    scales: tuple[float, ...]
    prior: tuple[float, ...]
    zero_variance_dimensions: tuple[int, ...]
    fitted_rows: int
    fitted_population: str
    input_identities: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.means) != REPRESENTATION_DIM:
            raise M1MemoryError(
                f"M1 standardizer must describe {REPRESENTATION_DIM} dimensions."
            )
        if not (len(self.scales) == len(self.prior) == REPRESENTATION_DIM):
            raise M1MemoryError("M1 standardizer arrays are not aligned.")

    def standardize(self, values: np.ndarray) -> np.ndarray:
        """Map fused representations into the frozen distance space."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != REPRESENTATION_DIM:
            raise M1MemoryError(
                f"M1 standardizer expects [N, {REPRESENTATION_DIM}]."
            )
        if not np.all(np.isfinite(matrix)):
            raise M1MemoryError(
                "A non-finite fused representation reached the M1 distance "
                "space. M1 refuses rather than silently skipping the window."
            )
        return (matrix - np.asarray(self.means)) / np.asarray(self.scales)

    def prior_vector(self) -> np.ndarray:
        """The exact persisted global TRAIN prior used at every cold start."""
        return np.asarray(self.prior, dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transform": "train_only_global_standardization",
            "dimension": REPRESENTATION_DIM,
            "means": list(self.means),
            "scales": list(self.scales),
            "prior": list(self.prior),
            "prior_semantics": (
                "mean of the standardized frozen primary TRAIN rows; used "
                "verbatim at every stream cold start and never assumed zero"
            ),
            "zero_variance_dimensions": list(self.zero_variance_dimensions),
            "zero_variance_policy": "scale_set_to_one",
            "fitted_rows": self.fitted_rows,
            "fitted_population": self.fitted_population,
            "fitted_on_partition": "train",
            "fitted_on_full_stream": False,
            "validation_statistics_used": False,
            "patient_specific_normalization": False,
            "input_identities": dict(self.input_identities),
        }
        payload["standardizer_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> M1DistanceStandardizer:
        recorded = payload.get("standardizer_sha256")
        body = {k: v for k, v in payload.items() if k != "standardizer_sha256"}
        if recorded is None or recorded != canonical_sha256(body):
            raise M1MemoryError("M1 standardizer failed digest validation.")
        return cls(
            means=tuple(float(v) for v in payload["means"]),
            scales=tuple(float(v) for v in payload["scales"]),
            prior=tuple(float(v) for v in payload["prior"]),
            zero_variance_dimensions=tuple(
                int(v) for v in payload["zero_variance_dimensions"]
            ),
            fitted_rows=int(payload["fitted_rows"]),
            fitted_population=str(payload["fitted_population"]),
            input_identities=dict(payload["input_identities"]),
        )


def fit_distance_standardizer(
    values: np.ndarray,
    *,
    partition: str = "train",
    input_identities: Mapping[str, Any] | None = None,
    fitted_population: str = "frozen_primary_train_p1b_fused",
) -> M1DistanceStandardizer:
    """Fit the frozen distance transform on primary TRAIN rows only."""
    if require_p1_partition(partition) != "train":
        raise M1MemoryError(
            "The M1 distance standardizer is fitted on the frozen primary "
            "TRAIN population only."
        )
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != REPRESENTATION_DIM:
        raise M1MemoryError(f"M1 standardizer fit expects [N, {REPRESENTATION_DIM}].")
    if matrix.shape[0] == 0:
        raise M1MemoryError("M1 standardizer cannot be fitted on zero rows.")
    if not np.all(np.isfinite(matrix)):
        raise M1MemoryError("M1 standardizer fit received a non-finite value.")

    means = matrix.mean(axis=0)
    # Population standard deviation (ddof=0), as named in the protocol.
    deviations = matrix.std(axis=0, ddof=0)
    zero_variance = tuple(int(i) for i in np.flatnonzero(deviations == 0.0))
    scales = np.where(deviations == 0.0, 1.0, deviations)
    prior = ((matrix - means) / scales).mean(axis=0)
    return M1DistanceStandardizer(
        means=tuple(float(v) for v in means),
        scales=tuple(float(v) for v in scales),
        prior=tuple(float(v) for v in prior),
        zero_variance_dimensions=zero_variance,
        fitted_rows=int(matrix.shape[0]),
        fitted_population=fitted_population,
        input_identities=dict(input_identities or {}),
    )


# --------------------------------------------------------------------------
# Dual-timescale causal state
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    """The causal features exposed for one window, before its own update."""

    d_short: float
    d_long: float
    past_observed_count: int
    past_update_count: int
    prototype_disagreement: float


class DualTimescaleMemory:
    """Causal EMA prototypes for one `(record_id, channel_index)` stream.

    The state is scoped to a single stream by construction: a new instance is
    built at every recording/channel boundary, so no state can survive across
    recordings merely because two recordings share a subject.
    """

    __slots__ = ("_mu_short", "_mu_long", "_observed", "_updated")

    def __init__(self, prior: np.ndarray) -> None:
        vector = np.asarray(prior, dtype=np.float64)
        if vector.shape != (REPRESENTATION_DIM,):
            raise M1MemoryError(
                f"M1 cold-start prior must be [{REPRESENTATION_DIM}]."
            )
        if not np.all(np.isfinite(vector)):
            raise M1MemoryError("M1 cold-start prior must be finite.")
        self._mu_short = vector.copy()
        self._mu_long = vector.copy()
        self._observed = 0
        self._updated = 0

    @property
    def mu_short(self) -> np.ndarray:
        return self._mu_short.copy()

    @property
    def mu_long(self) -> np.ndarray:
        return self._mu_long.copy()

    @property
    def past_observed_count(self) -> int:
        return self._observed

    @property
    def past_update_count(self) -> int:
        return self._updated

    def _require_finite(self, standardized: np.ndarray) -> np.ndarray:
        vector = np.asarray(standardized, dtype=np.float64)
        if vector.shape != (REPRESENTATION_DIM,):
            raise M1MemoryError(
                f"M1 memory expects a [{REPRESENTATION_DIM}] observation."
            )
        if not np.all(np.isfinite(vector)):
            raise M1MemoryError(
                "Non-finite fused representation reached M1 memory. The frozen "
                "P1 transformation should make this impossible, so M1 refuses "
                "rather than silently skipping the observation."
            )
        return vector

    def deviations(self, standardized: np.ndarray) -> MemoryObservation:
        """Deviation features against state built only from earlier windows."""
        vector = self._require_finite(standardized)
        d_short = float(np.sqrt(np.mean((vector - self._mu_short) ** 2)))
        d_long = float(np.sqrt(np.mean((vector - self._mu_long) ** 2)))
        return MemoryObservation(
            d_short=d_short,
            d_long=d_long,
            past_observed_count=self._observed,
            past_update_count=self._updated,
            prototype_disagreement=float(
                np.sqrt(np.mean((self._mu_short - self._mu_long) ** 2))
            ),
        )

    def update(self, standardized: np.ndarray) -> None:
        """Admit a finite observation under finite-observation always-update.

        There is deliberately no label, score, threshold, uncertainty or event
        argument: M1-v1 cannot gate admission on any of them.
        """
        vector = self._require_finite(standardized)
        self._mu_short = (1.0 - ALPHA_SHORT) * self._mu_short + ALPHA_SHORT * vector
        self._mu_long = (1.0 - ALPHA_LONG) * self._mu_long + ALPHA_LONG * vector
        self._updated += 1

    def observe(self, standardized: np.ndarray) -> MemoryObservation:
        """Score strictly before updating.

        Deviations are computed against the pre-update prototypes and only then
        is the observation admitted, so a window can never influence the state
        used to compute its own distance.
        """
        features = self.deviations(standardized)
        self._observed += 1
        self.update(standardized)
        return features


# --------------------------------------------------------------------------
# Deterministic full-stream feature generation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M1StreamMemory:
    """Ordered causal memory features over the full development stream."""

    partition: str
    stable_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    channel_indices: np.ndarray
    start_samples: np.ndarray
    d_short: np.ndarray
    d_long: np.ndarray
    past_observed_count: np.ndarray
    past_update_count: np.ndarray
    prototype_disagreement: np.ndarray
    recording_age_seconds: np.ndarray
    cold_start_bins: tuple[str, ...]
    streams: tuple[StreamKey, ...]
    chronology_sha256: str

    def index(self) -> dict[str, int]:
        return {key: position for position, key in enumerate(self.stable_ids)}

    def memory_matrix(self, experiment_id: str) -> np.ndarray:
        """Assemble the arm's memory columns in the frozen feature order."""
        require_m1_experiment(experiment_id)
        columns = {"d_short": self.d_short, "d_long": self.d_long}
        selected = [columns[name] for name in M1_ARM_FEATURES[experiment_id]]
        return np.stack(selected, axis=1).astype(np.float32)


def generate_stream_memory(
    streams: Mapping[StreamKey, tuple[B4WindowReference, ...]],
    *,
    partition: str,
    representations: Mapping[str, np.ndarray],
    standardizer: M1DistanceStandardizer,
) -> M1StreamMemory:
    """Replay every causal stream and record past-only memory features.

    Every causally representable window participates, independent of its target
    family: `references` carries no label into this function and none is
    consulted. Removing a challenge row's *label* therefore cannot remove that
    *observation* from history.
    """
    evaluated = require_p1_partition(partition)
    prior = standardizer.prior_vector()

    stable_ids: list[str] = []
    record_ids: list[str] = []
    channels: list[int] = []
    starts: list[int] = []
    short: list[float] = []
    long: list[float] = []
    observed: list[int] = []
    updated: list[int] = []
    disagreement: list[float] = []
    ages: list[float] = []
    bins: list[str] = []

    for key in sorted(streams):
        rows = streams[key]
        if not rows:
            continue
        memory = DualTimescaleMemory(prior)
        origin = int(rows[0].start_sample)
        for reference in rows:
            vector = representations.get(reference.stable_id)
            if vector is None:
                raise M1MemoryError(
                    f"Stream window {reference.stable_id} has no frozen fused "
                    "representation."
                )
            standardized = standardizer.standardize(
                np.asarray(vector, dtype=np.float64)
            )[0]
            features = memory.observe(standardized)
            age = (int(reference.start_sample) - origin) / SAMPLING_FREQUENCY_HZ
            stable_ids.append(reference.stable_id)
            record_ids.append(reference.record_id)
            channels.append(int(reference.channel_index))
            starts.append(int(reference.start_sample))
            short.append(features.d_short)
            long.append(features.d_long)
            observed.append(features.past_observed_count)
            updated.append(features.past_update_count)
            disagreement.append(features.prototype_disagreement)
            ages.append(age)
            bins.append(cold_start_bin(age))

    return M1StreamMemory(
        partition=evaluated,
        stable_ids=tuple(stable_ids),
        record_ids=tuple(record_ids),
        channel_indices=np.asarray(channels, dtype=np.int64),
        start_samples=np.asarray(starts, dtype=np.int64),
        d_short=np.asarray(short, dtype=np.float64),
        d_long=np.asarray(long, dtype=np.float64),
        past_observed_count=np.asarray(observed, dtype=np.int64),
        past_update_count=np.asarray(updated, dtype=np.int64),
        prototype_disagreement=np.asarray(disagreement, dtype=np.float64),
        recording_age_seconds=np.asarray(ages, dtype=np.float64),
        cold_start_bins=tuple(bins),
        streams=tuple(sorted(streams)),
        chronology_sha256=ordered_chronology_digest(streams),
    )


# --------------------------------------------------------------------------
# Candidate heads
# --------------------------------------------------------------------------


def m1_input_dim(experiment_id: str) -> int:
    require_m1_experiment(experiment_id)
    return REPRESENTATION_DIM + len(M1_ARM_FEATURES[experiment_id])


def build_m1_head(experiment_id: str) -> P1FusionHead:
    """Construct one M1 arm head.

    `P1FusionHead` is reused rather than redeclared so "same head family as P1"
    holds by construction rather than by comment.
    """
    return P1FusionHead(m1_input_dim(experiment_id))


def build_deterministic_m1_head(experiment_id: str) -> P1FusionHead:
    """Construct an M1 head under a deterministic reseed."""
    torch.manual_seed(P1_SEED)
    return build_m1_head(experiment_id)


def m1_head_identity(experiment_id: str, head: P1FusionHead) -> dict[str, Any]:
    parameters = sum(p.numel() for p in head.parameters() if p.requires_grad)
    return {
        "experiment_id": require_m1_experiment(experiment_id),
        "input_dim": head.input_dim,
        "representation_dim": REPRESENTATION_DIM,
        "memory_features": list(M1_ARM_FEATURES[experiment_id]),
        "hidden_dim": HEAD_HIDDEN_DIM,
        "activation": "SiLU",
        "dropout": HEAD_DROPOUT,
        "output": "single_raw_logit",
        "trainable_parameter_count": parameters,
        "fp32_parameter_payload_bytes": parameters * 4,
        "patient_identifier_features": [],
        "learned_patient_embedding": False,
    }


def m1_training_configuration() -> dict[str, Any]:
    """The P1 training contract, reused identically by all three M1 arms."""
    return {
        "seed": P1_SEED,
        "loss": "BCEWithLogitsLoss(reduction=mean)",
        "optimizer": "AdamW",
        "learning_rate": P1_LEARNING_RATE,
        "weight_decay": P1_WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
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
        "threshold_rule": (
            "selected checkpoint, full primary validation, maximum F1; the "
            "highest observed score wins an exact tie"
        ),
        "calibration": None,
        "score_semantics": "uncalibrated model score, not a calibrated probability",
        "encoder": "frozen B4-B; not fine-tuned",
        "physiology_transform": "frozen P1 artifact; never refitted",
    }


def m1_boundary_statement() -> dict[str, Any]:
    """The explicit M1/M2 boundary carried into every M1 artifact."""
    return {
        "update_policy": UPDATE_POLICY,
        "update_policy_statement": UPDATE_POLICY_STATEMENT,
        "contamination_safe": CONTAMINATION_SAFE,
        "m2_required_before_safe_adaptation_claim": True,
        "physical_availability_decided_before_encoder": True,
        "physical_observation_contract": PHYSICAL_OBSERVATION_CONTRACT,
        "unavailable_row_produces_representation": False,
        "unavailable_row_produces_score": False,
        "unavailable_row_updates_memory": False,
        "unavailable_row_increments_counters": False,
        "unavailable_row_preserves_elapsed_time": True,
        "b4_input_contract_weakened": False,
        "alpha_time_rescaled": False,
        "label_gated_update": False,
        "score_gated_update": False,
        "uncertainty_admission": False,
        "event_state_admission": False,
        "conformal_admission": False,
        "rollback": False,
        "memory_resets_at_recording_channel_boundary": True,
        "cross_recording_state_carryover": False,
        "patient_identity_is_a_feature": False,
        "scope": (
            "M1-v1 is patient-adaptive WITHIN a continuous recording/lead "
            "stream and carries no learned state across separate recordings "
            "from the same subject."
        ),
    }


# --------------------------------------------------------------------------
# One-shot claim helpers
# --------------------------------------------------------------------------


def resolve_m1_run_dir(run_root: Path, experiment_id: str) -> Path:
    require_m1_experiment(experiment_id)
    root = require_nonversioned_path(Path(run_root), "M1 development evidence")
    return root / experiment_id


def claim_m1_run_directory(run_dir: Path, experiment_id: str) -> Path:
    """Atomically claim the one canonical attempt for an M1 arm.

    The directory is the claim. There is no force, overwrite, retry, reseed or
    delete path, and the claim is never released on failure.
    """
    require_m1_experiment(experiment_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise M1MemoryError(
            f"Canonical M1 experiment {experiment_id} has already been claimed "
            f"at {run_dir}. Automatic rerun, retry, selective rerun and "
            "fresh-seed restart are prohibited and require documented human "
            "review."
        ) from error
    descriptor = os.open(run_dir.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return run_dir


def write_m1_status(
    run_dir: Path, status: str, *, experiment_id: str, **fields: Any
) -> dict[str, Any]:
    """Write the M1 heartbeat for the calling experiment.

    `experiment_id` is required rather than defaulted, for the same reason the
    B4 and P1 runners require it: a default silently stamps one experiment's
    identity onto another's status file.
    """
    payload = {
        "experiment_id": require_m1_experiment(experiment_id),
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    write_json_atomic(Path(run_dir) / RUN_STATUS_NAME, payload)
    return payload


def record_m1_failure(
    run_dir: Path, experiment_id: str, error: BaseException
) -> dict[str, Any]:
    """Record a post-claim failure. The claim is never released."""
    return write_m1_status(
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


def m1_arm_features(
    experiment_id: str,
    representation: np.ndarray,
    memory: np.ndarray,
) -> np.ndarray:
    """Concatenate the original `z_t` with this arm's memory distances."""
    require_m1_experiment(experiment_id)
    base = np.asarray(representation, dtype=np.float32)
    columns = np.asarray(memory, dtype=np.float32)
    expected = len(M1_ARM_FEATURES[experiment_id])
    if base.ndim != 2 or base.shape[1] != REPRESENTATION_DIM:
        raise M1MemoryError(f"M1 representation must be [N, {REPRESENTATION_DIM}].")
    if columns.ndim != 2 or columns.shape != (base.shape[0], expected):
        raise M1MemoryError(f"M1 memory block must be [N, {expected}].")
    features = np.concatenate([base, columns], axis=1).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise M1MemoryError("M1 arm features contain a non-finite value.")
    return features


def select_rows(
    memory: M1StreamMemory, stable_ids: Sequence[str]
) -> np.ndarray:
    """Positions of an ordered ID subset inside the full-stream memory."""
    index = memory.index()
    missing = [key for key in stable_ids if key not in index]
    if missing:
        raise M1MemoryError(
            f"{len(missing)} requested rows are absent from the M1 full-stream "
            "memory."
        )
    return np.asarray([index[key] for key in stable_ids], dtype=np.int64)
