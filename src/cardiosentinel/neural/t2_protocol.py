"""Structural binding of the frozen T2 causal longitudinal temporal protocol.

This module records a *protocol*, not a computation. It freezes the T2 design
constants, generates the deterministic subject-disjoint internal TRAIN split,
and refuses inputs, orderings and configurations the protocol forbids.

Like `u1_protocol`, it imports **only the standard library**. There is no model,
no trainer, no optimiser, no scorer, no torch, no numpy, no waveform reader and
no run-artifact reader here, so protocol validation cannot reach real
development data even by accident -- and cannot reach TEST at all. Building and
fitting the two longitudinal candidates belongs to a separate reviewed execution
change set that does not exist yet.

The two things most worth understanding before reading further:

* **T2 is not B4-C.** B4-C modelled the *inside* of one 10-second window and was
  rejected. T2 models the sequence *across* successive windows at a 5-second
  stride. Different temporal scale, different scientific role; the B4-C
  rejection says nothing about state-space models here.
* **T2 is not T1.** T2 learns a temporal evidence score. T1 is the deterministic
  NORMAL/WATCH/EVENT/RECOVERY state machine that will later consume it. This
  module freezes the T1 *interface* and deliberately freezes no T1 threshold,
  duration or hysteresis value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Iterable, NamedTuple, Sequence

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]

T2_PROTOCOL_NAME: Final = "T2_LONGITUDINAL_TEMPORAL_PROTOCOL_V1"
T2_PROTOCOL_PATH: Final = REPOSITORY_ROOT / "docs" / f"{T2_PROTOCOL_NAME}.md"
T2_PROTOCOL_SHA256: Final = (
    "6510f831cd9cb7870f8cffba7eac1ffd5d95dea40272c279e6bcc75439cf3e25"
)

# ---------------------------------------------------------------------------
# Frozen upstream identities T2 consumes read-only (§30)
# ---------------------------------------------------------------------------
T2_STARTING_GIT_SHA: Final = "997df407376edcf585a68d019b26b02a7670c12b"

T2_SPLIT_PATH: Final = REPOSITORY_ROOT / "protocols" / "splits" / "ltstdb_v1.json"
T2_SPLIT_SHA256: Final = (
    "66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7"
)
T2_SPLIT_FILE_SHA256: Final = (
    "74f055dee370ab2742b2a5346eb37de4d3f6fccb011676b203b3eb339a62d714"
)

T2_U1_RETENTION_DECISION_SHA256: Final = (
    "9d8436f2b7d2c303aeeb03e438c60fb8110f7d06d0bbd589f5be65ea8f80cb7b"
)
T2_U1_RESULT_SHA256: Final = (
    "649631cbf5188731d006f533997cfe28df4f5acb79e7693514e86ad0cef0cb12"
)
T2_U1_EXPERIMENT_LOCK_SHA256: Final = (
    "7f4dd1505919e23a598773736dc57e2d1b4d360f496b45acdf2028ed0574b1b6"
)
T2_M2_RETENTION_DECISION_SHA256: Final = (
    "da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47"
)
T2_M2G_ARM_RESULT_SHA256: Final = (
    "a061d4d8c5211381c18baa228436bb9abc78b2f87f71fe4cab6ca71b2d15cf75"
)

# P1-B retained identities and the frozen representation schema
T2_P1_PROTOCOL_SHA256: Final = (
    "f48ffc66e52649d74a8286182d5e7220f78abdd6c12a7ebfe04f116b853337f1"
)
T2_P1_RETENTION_DECISION_SHA256: Final = (
    "7b403709fa0fb12eef65423d830c121fc3ada904266a1b47931d438f5e797d68"
)
T2_P1B_EXPERIMENT_LOCK_SHA256: Final = (
    "796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0"
)
T2_B4_PROTOCOL_SHA256: Final = (
    "f6f5e9ed728c86a9b2bd75b2327b9199f0e097b91387525a192c212e6771b28b"
)
T2_ENCODER_CHECKPOINT_SHA256: Final = (
    "b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9"
)
T2_PHYSIOLOGY_SCHEMA_SHA256: Final = (
    "13f60be400b5b957c1eb592bbafd8206d4d2855c1aa657a058671fb8d7cab434"
)
T2_PHYSIOLOGY_TRANSFORM_SHA256: Final = (
    "cc6bd3a353f0ac6cad342114ed96e135cbf3c61e2946f847d5b95358b6bd51a9"
)

T2_ENVIRONMENT_DEPENDENCY_DIGEST: Final = (
    "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
)

# ---------------------------------------------------------------------------
# The temporal input store (§1, §4). The M1 full stream memory cache is the
# ONLY admissible z_t source: it is the one artifact carrying the frozen
# 146-dimensional representation for FULL chronological streams on BOTH
# partitions, with record/channel/start-sample ordering keys and the physical
# availability state.
#
# The P1 embedding cache is deliberately NOT the source. Its TRAIN side holds
# 374,452 rows at exactly 3:1 negative sampling (280,839 = 3 x 93,613) -- a
# selection, not a timeline. Training T2 on it would silently destroy temporal
# continuity, which §9 forbids.
# ---------------------------------------------------------------------------
T2_INPUT_STORE_KIND: Final = "m1_full_stream_memory_cache"
T2_INPUT_STORE_SCHEMA: Final = 3
T2_INPUT_STORE_ROOT: Final = Path("cardiosentinel-features/m1-stream-memory-v2")

T2_TRAIN_STREAM_CACHE_SHA256: Final = (
    "d006c698017110bfd95774ca207036a820139779b95cf1b3f3a36c06efa779a4"
)
T2_VALIDATION_STREAM_CACHE_SHA256: Final = (
    "a3e39137a04ebebb3b97ef6c6c614339c990a6041cf649a0ba6e3c2d43baae18"
)
T2_TRAIN_REPRESENTATION_CONTENT_SHA256: Final = (
    "e52a566fbc285a7a9f92715752dee43c020faa3550aaeb660f5f400dee07b5d3"
)
T2_VALIDATION_REPRESENTATION_CONTENT_SHA256: Final = (
    "b26a2d9b6150e6518dc2bfb394427dc93ae48a7cc3de30adcc3fefcc9f1f53ba"
)
T2_TRAIN_P1_EMBEDDING_CACHE_SHA256: Final = (
    "0a5f021b89597d245a2afdc51fe1a65ba5cd6a090beba429f38bbccff8c372dd"
)
T2_VALIDATION_P1_EMBEDDING_CACHE_SHA256: Final = (
    "c533db3acfdfa1057c2ac9d8e77d011d3ac5f87fc7a872399227f94f526db0c3"
)

T2_TRAIN_FULL_STREAM_ROW_COUNT: Final = 2_208_431
T2_VALIDATION_FULL_STREAM_ROW_COUNT: Final = 492_904
T2_TRAIN_STREAM_COUNT: Final = 132
T2_VALIDATION_STREAM_COUNT: Final = 30

# The 3:1 selection that must never become the T2 training population.
T2_P1_TRAIN_SELECTION_ROW_COUNT: Final = 374_452
T2_NEGATIVE_SAMPLING_PERMITTED: Final = False
T2_FULL_CHRONOLOGICAL_POPULATION_REQUIRED: Final = True

# ---------------------------------------------------------------------------
# Representation (§4)
# ---------------------------------------------------------------------------
T2_EMBEDDING_DIM: Final = 128
T2_PHYSIOLOGY_DIM: Final = 18
T2_INPUT_DIM: Final = T2_EMBEDDING_DIM + T2_PHYSIOLOGY_DIM  # 146
T2_REPRESENTATION_DTYPE: Final = "float32"
T2_REPRESENTATION_IS_FROZEN: Final = True
T2_ENCODER_FINE_TUNED: Final = False

# Nothing on this list may reach the trainable T2 model in V1 (§4).
T2_FORBIDDEN_TRAINABLE_INPUTS: Final = (
    "u1_oof_calibrated_probability",
    "u1_calibrated_probability",
    "u1_uncertainty",
    "u_star_dev",
    "u_star_deploy",
    "future_window_label",
    "challenge_family_identity",
    "episode_identity",
    "future_episode_duration",
    "m2_gate_outcome",
    "test_derived_quantity",
)

# ---------------------------------------------------------------------------
# Window and stream semantics (§5, §6)
# ---------------------------------------------------------------------------
T2_WINDOW_LENGTH_SECONDS: Final = 10.0
T2_WINDOW_STRIDE_SECONDS: Final = 5.0
T2_STREAM_KEY_FIELDS: Final = ("record_id", "channel_index")
T2_STREAM_ORDER_FIELD: Final = "window_start_samples"
T2_BIDIRECTIONAL_PERMITTED: Final = False
T2_FUTURE_CONTEXT_PERMITTED: Final = False
T2_SHUFFLE_WITHIN_STREAM_PERMITTED: Final = False
T2_STATE_RESETS_AT_STREAM_BOUNDARY: Final = True
T2_STATE_CROSSES_RECORD: Final = False
T2_STATE_CROSSES_CHANNEL: Final = False
T2_STATE_CROSSES_SUBJECT: Final = False

# Availability enum, inherited verbatim from the M1 stream cache schema.
T2_OBSERVATION_AVAILABLE: Final = 1
T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT: Final = 2
T2_OBSERVATION_UNINITIALIZED: Final = 0
T2_SYNTHETIC_Z_PERMITTED: Final = False
T2_IMPUTATION_PERMITTED: Final = False
T2_FORWARD_FILL_PERMITTED: Final = False
T2_UNAVAILABLE_ROW_SCORED: Final = False
T2_UNAVAILABLE_ROW_TRAINED: Final = False
T2_UNAVAILABLE_ROW_UPDATES_HIDDEN_STATE: Final = False
T2_UNAVAILABLE_ROW_ADVANCES_TIMELINE: Final = True
T2_NEW_SQI_THRESHOLD_PERMITTED: Final = False

# ---------------------------------------------------------------------------
# Target (§7)
# ---------------------------------------------------------------------------
T2_TARGET: Final = "current_window_primary_ischemic_label"
T2_TARGET_AUTHORITY: Final = "ltstdb_stb_frozen_target_authority"
T2_FUTURE_TARGET_PERMITTED: Final = False
T2_CHALLENGE_ROWS_ARE_TRAINING_TARGETS: Final = False

# ---------------------------------------------------------------------------
# Internal TRAIN split (§8)
# ---------------------------------------------------------------------------
T2_INTERNAL_SPLIT_SEED_STRING: Final = "cardiosentinel-t2-internal-split-v1"
T2_INTERNAL_SPLIT_ALGORITHM: Final = "sha256_identity_ranked_subject_partition_v1"
T2_TRAIN_SUBJECT_COUNT: Final = 56
T2_FIT_SUBJECT_COUNT: Final = 48
T2_INTERNAL_DEV_SUBJECT_COUNT: Final = 8
T2_INTERNAL_SPLIT_SHA256: Final = (
    "54f8091ee7d4620ab6e24aaa32b121874b6a1610003e3df63f94f9727618e28e"
)
T2_SPLIT_USES_LABELS: Final = False
T2_SPLIT_USES_OUTCOMES: Final = False
T2_SPLIT_USES_PREVALENCE: Final = False

T2_INTERNAL_DEV_SUBJECTS: Final = (
    "ltstdb:s2008",
    "ltstdb:s2017",
    "ltstdb:s2042",
    "ltstdb:s2046",
    "ltstdb:s2049",
    "ltstdb:s2050",
    "ltstdb:s2063",
    "ltstdb:s2064",
)

# ---------------------------------------------------------------------------
# Loss and optimisation (§10, §14)
# ---------------------------------------------------------------------------
T2_LOSS: Final = "binary_cross_entropy_with_logits"
T2_POSITIVE_CLASS_WEIGHT_RULE: Final = "n_negative_over_n_positive_on_fit_partition"
T2_CLASS_WEIGHT_PARTITION: Final = "t2_fit_48_subjects"
T2_FOCAL_LOSS_COMPARISON_PERMITTED: Final = False
T2_LOSS_FAMILY_SEARCH_PERMITTED: Final = False
T2_VALIDATION_DERIVED_CLASS_WEIGHT_PERMITTED: Final = False

T2_OPTIMIZER: Final = "AdamW"
T2_LEARNING_RATE: Final = 3e-4
T2_WEIGHT_DECAY: Final = 1e-4
T2_MAX_EPOCHS: Final = 10
T2_GRADIENT_CLIP_NORM: Final = 1.0
T2_SEED: Final = 2026
T2_CHECKPOINT_CRITERION: Final = "internal_development_pooled_auprc"
T2_CHECKPOINT_TIE_BREAK: Final = "earlier_epoch"
T2_EARLY_STOPPING_PATIENCE_EPOCHS: Final = 3
T2_OUTER_VALIDATION_IN_EPOCH_SELECTION: Final = False

# ---------------------------------------------------------------------------
# Truncated backpropagation through time (§11)
# ---------------------------------------------------------------------------
T2_TBPTT_LENGTH: Final = 256
T2_TBPTT_HORIZON_SECONDS: Final = T2_TBPTT_LENGTH * T2_WINDOW_STRIDE_SECONDS  # 1280.0
T2_STATE_CARRIES_ACROSS_CHUNK: Final = True
T2_STATE_DETACHED_AT_CHUNK_BOUNDARY: Final = True
T2_GRADIENT_CROSSES_CHUNK_BOUNDARY: Final = False
T2_STATE_RESET_AT_CHUNK_BOUNDARY: Final = False

# ---------------------------------------------------------------------------
# Candidate arms and shared capacity envelope (§12, §13)
# ---------------------------------------------------------------------------
T2_ARM_GRU: Final = "causal_gru_longitudinal_v1"
T2_ARM_S4D: Final = "causal_s4d_longitudinal_v1"
T2_ARMS: Final = (T2_ARM_GRU, T2_ARM_S4D)
T2_S4D_FAMILY: Final = "s4d_inspired_diagonal_state_space"
T2_S4D_IS_MAMBA: Final = False
T2_EXTERNAL_SSM_PACKAGE_PERMITTED: Final = False
T2_FRAMEWORK: Final = "torch"

T2_INPUT_PROJECTION_DIM: Final = 64
T2_TEMPORAL_WIDTH: Final = 64
T2_TEMPORAL_LAYERS: Final = 2
T2_DROPOUT: Final = 0.10
T2_OUTPUT_DIM: Final = 1
T2_OUTPUT_SEMANTICS: Final = "single_current_window_logit"
T2_PARAMETER_RATIO_MIN: Final = 0.5
T2_PARAMETER_RATIO_MAX: Final = 2.0
T2_MODEL_SIZE_INCREASE_AFTER_RESULTS_PERMITTED: Final = False

# ---------------------------------------------------------------------------
# Evaluation (§15 - §24)
# ---------------------------------------------------------------------------
T2_OUTER_VALIDATION_ATTEMPTS: Final = 1
T2_AUTOMATIC_RETRY_PERMITTED: Final = False
T2_PRIMARY_SELECTION_METRIC: Final = "pooled_primary_validation_auprc"
T2_SECONDARY_SELECTION_METRIC: Final = "subject_macro_auprc"
T2_SELECTION_TIE_TOLERANCE: Final = 0.002
T2_SELECTION_FINAL_TIE_BREAK: Final = "smaller_faster_model"
T2_WEIGHTED_COMPOSITE_SCORE_PERMITTED: Final = False
T2_CHALLENGE_IS_SELECTION_INPUT: Final = False
T2_LATENCY_ADJUSTED_SCORE_PERMITTED: Final = False

T2_BINARY_THRESHOLD_RULE: Final = "exact_maximum_f1_highest_threshold_tie_break"
T2_BINARY_THRESHOLD_PARTITION: Final = "t2_internal_dev_8_subjects"
T2_THRESHOLD_LOCKED_BEFORE_OUTER_VALIDATION: Final = True
T2_OUTER_VALIDATION_MAY_ALTER_THRESHOLD: Final = False

T2_POOLED_METRICS: Final = (
    "auprc",
    "auroc",
    "f1",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "balanced_accuracy",
    "mcc",
)
T2_SUBJECT_MACRO_METRICS: Final = (
    "auprc",
    "auroc",
    "sensitivity",
    "specificity",
    "mcc",
)
T2_INFERENTIAL_UNIT: Final = "subject"
T2_WINDOWS_ARE_INDEPENDENT_EVIDENCE: Final = False

T2_BOOTSTRAP_REPLICATES: Final = 1000
T2_BOOTSTRAP_SEED: Final = 2026
T2_BOOTSTRAP_UNIT: Final = "subject"
T2_WINDOW_BOOTSTRAP_PERMITTED: Final = False
T2_BOOTSTRAP_CLAIM: Final = (
    "between_subject_variation_conditional_on_fitted_temporal_model"
)

T2_TEMPORAL_DESCRIPTIVE_METRICS: Final = (
    "positive_prediction_run_count",
    "median_positive_run_duration_seconds",
    "isolated_single_window_positive_fraction",
    "transition_count_per_hour",
    "prediction_persistence_around_labelled_ischemic_intervals",
)
T2_TEMPORAL_DESCRIPTIVE_IS_SELECTION_INPUT: Final = False

T2_COLD_START_STRATA: Final = ("0_5_minutes", "5_60_minutes", "over_60_minutes")
T2_INITIAL_HIDDEN_STATE: Final = "frozen_zero_state_at_stream_start"
T2_COLD_START_WARMUP_THRESHOLD_PERMITTED: Final = False
T2_COLD_START_REPAIR_PERMITTED: Final = False
T2_ALTERNATIVE_STATE_INITIALIZATION_PERMITTED: Final = False

T2_CHALLENGE_FAMILIES: Final = ("rate_related", "axis_shift", "conduction_change")
T2_CHALLENGE_MERGED_INTO_PRIMARY: Final = False
T2_CHALLENGE_TRAINED_ON: Final = False
T2_CONDUCTION_EVIDENCE_LEVEL: Final = "exploratory_descriptive"

# ---------------------------------------------------------------------------
# What T2 output is NOT (§18), and what stays undefined (§25 - §27)
# ---------------------------------------------------------------------------
T2_OUTPUT_IS_CALIBRATED_PROBABILITY: Final = False
T2_OUTPUT_IS_UNCERTAINTY: Final = False
T2_OUTPUT_IS_CONFORMAL_EVIDENCE: Final = False
T2_OUTPUT_SEMANTIC_NAME: Final = "causal_temporal_evidence_score"
T2_CALIBRATION_OF_T2_AUTHORISED: Final = False
T2_RETAINED_CALIBRATED_PROBABILITY_SOURCE: Final = "u1_oof_development_calibration"

T1_STATES: Final = ("NORMAL", "WATCH", "EVENT", "RECOVERY")
T1_IMPLEMENTED_HERE: Final = False
T1_PERMITTED_INPUTS: Final = (
    "frozen_detector_decision",
    "u1_oof_platt_calibrated_probability",
    "u1_calibrated_uncertainty",
    "m2g_causally_available_patient_adaptation_evidence",
    "selected_t2_temporal_evidence_score",
    "physical_availability_state",
    "elapsed_causal_time_or_state_duration",
)
T1_TRANSITION_THRESHOLD: Final = None
T1_PERSISTENCE_DURATION: Final = None
T1_HYSTERESIS_VALUE: Final = None
T1_EVENT_ONSET_RULE: Final = None
T1_RECOVERY_RULE: Final = None
T2_TRAINED_TO_EMIT_T1_STATES: Final = False

T2_ROUTING_DEFINED_HERE: Final = False
T2_ROUTE_THRESHOLD: Final = None
U1_SYMMETRIC_ROUTER_STILL_REJECTED: Final = True

# ---------------------------------------------------------------------------
# Firewalls (§28, §29)
# ---------------------------------------------------------------------------
T2_TEST_ACCESSED: Final = False
T2_SEALED_TEST_STATE: Final = "unopened"
T2_OUTER_VALIDATION_IS_UNSEEN_GENERALISATION: Final = False
T2_DEVELOPMENT_OPTIMISM_DISCLOSED: Final = True
T2_DEVELOPMENT_OPTIMISM_NOTE: Final = (
    "the outer VALIDATION partition was already used in upstream model, "
    "threshold and calibration development; the T2 internal TRAIN split reduces "
    "additional outer-validation tuning but T2 outer-VALIDATION results remain "
    "DEVELOPMENT evidence and are not unseen generalisation"
)


class T2ProtocolError(RuntimeError):
    """Raised when the frozen T2 protocol cannot be proven."""


class T2StreamKey(NamedTuple):
    """The unit across which T2 temporal state may carry."""

    record_id: str
    channel_index: int


class T2Row(NamedTuple):
    """One window position in a stream, as the protocol validates it."""

    record_id: str
    channel_index: int
    window_start_samples: int
    observation_state: int


class T2Chunk(NamedTuple):
    """One TBPTT chunk: gradients stop at its edge, causal state does not."""

    stream: T2StreamKey
    start_index: int
    stop_index: int
    carries_state_in: bool
    detaches_state_in: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def validate_t2_protocol_document(path: Path = T2_PROTOCOL_PATH) -> str:
    """Verify the frozen T2 protocol document byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise T2ProtocolError(f"T2 protocol document is missing at {document}.")
    digest = _sha256_file(document)
    if digest != T2_PROTOCOL_SHA256:
        raise T2ProtocolError(
            f"T2 protocol digest {digest} differs from the frozen "
            f"{T2_PROTOCOL_SHA256}. The protocol is immutable."
        )
    return digest


# ---------------------------------------------------------------------------
# Representation contract (§4)
# ---------------------------------------------------------------------------


def require_input_dimension(dimension: int) -> int:
    """T2 consumes the frozen 146-dimensional P1-B representation, exactly."""
    if int(dimension) != T2_INPUT_DIM:
        raise T2ProtocolError(
            f"T2 input dimension is {dimension}, not the frozen {T2_INPUT_DIM} "
            f"({T2_EMBEDDING_DIM} B4-B embedding + {T2_PHYSIOLOGY_DIM} retained "
            "physiology features). No other representation is authorised."
        )
    return T2_INPUT_DIM


def require_permitted_trainable_inputs(names: Iterable[str]) -> tuple[str, ...]:
    """Refuse every V1-forbidden trainable input by name.

    U1's calibrated probability and uncertainty are retained science, but TRAIN
    carries no equivalent frozen subject-disjoint calibration product. Feeding
    VALIDATION-only quantities to T2 would create a train/validation
    feature-definition mismatch, so they are refused here rather than later.
    """
    observed = tuple(str(name) for name in names)
    forbidden = sorted(set(observed) & set(T2_FORBIDDEN_TRAINABLE_INPUTS))
    if forbidden:
        raise T2ProtocolError(
            f"{forbidden} may not be a trainable T2-v1 input. U1 calibration "
            "remains retained and available to the later T1 / fusion layer."
        )
    return observed


# ---------------------------------------------------------------------------
# Stream and ordering contract (§5, §6)
# ---------------------------------------------------------------------------


def stream_key(row: T2Row) -> T2StreamKey:
    return T2StreamKey(row.record_id, int(row.channel_index))


def require_chronological_stream(rows: Sequence[T2Row]) -> tuple[T2Row, ...]:
    """Refuse shuffled time, duplicated positions and any non-causal ordering."""
    ordered = tuple(rows)
    if not ordered:
        raise T2ProtocolError("A T2 stream carries no rows.")
    keys = {stream_key(row) for row in ordered}
    if len(keys) != 1:
        raise T2ProtocolError(
            f"A T2 stream must be one {T2_STREAM_KEY_FIELDS} pair; got "
            f"{sorted(keys)}. Temporal state never crosses records or channels."
        )
    previous = None
    for row in ordered:
        position = int(row.window_start_samples)
        if previous is not None and position <= previous:
            raise T2ProtocolError(
                f"T2 requires strict chronological order by "
                f"{T2_STREAM_ORDER_FIELD}; {position} follows {previous}."
            )
        previous = position
    return ordered


def split_into_streams(rows: Sequence[T2Row]) -> dict[T2StreamKey, tuple[T2Row, ...]]:
    """Group rows into the streams T2 state may carry across, and validate each."""
    grouped: dict[T2StreamKey, list[T2Row]] = {}
    for row in rows:
        grouped.setdefault(stream_key(row), []).append(row)
    return {
        key: require_chronological_stream(values) for key, values in grouped.items()
    }


def state_reset_positions(rows: Sequence[T2Row]) -> tuple[int, ...]:
    """Indices where hidden state MUST reset: every new stream, and only those."""
    resets: list[int] = []
    previous: T2StreamKey | None = None
    for index, row in enumerate(rows):
        key = stream_key(row)
        if previous is None or key != previous:
            resets.append(index)
        previous = key
    return tuple(resets)


def require_no_future_access(
    current_index: int, accessed_indices: Iterable[int]
) -> None:
    """A causal model may read the present and the past. Nothing else."""
    future = sorted(index for index in accessed_indices if index > current_index)
    if future:
        raise T2ProtocolError(
            f"T2 is causal: window {current_index} may not read future windows "
            f"{future}. No bidirectional model and no future context."
        )


def is_available(row: T2Row) -> bool:
    return int(row.observation_state) == T2_OBSERVATION_AVAILABLE


def require_available_for_modelling(row: T2Row) -> T2Row:
    """A physically unavailable observation is never scored, trained or absorbed.

    The timeline position still advances and causal state still carries across
    it -- what must not happen is a synthetic representation, an imputed value,
    a forward fill, a target, or a hidden-state update from evidence that was
    never observed.
    """
    if not is_available(row):
        raise T2ProtocolError(
            f"Observation state {row.observation_state} is not AVAILABLE "
            f"({T2_OBSERVATION_AVAILABLE}). It receives no synthetic z, no "
            "imputation, no forward fill, no score, no target and no hidden-state "
            "update; its timeline position still advances."
        )
    return row


def modellable_rows(rows: Sequence[T2Row]) -> tuple[T2Row, ...]:
    """The rows T2 may score and train on: available observations only."""
    return tuple(row for row in rows if is_available(row))


# ---------------------------------------------------------------------------
# Internal TRAIN split (§8)
# ---------------------------------------------------------------------------


def assign_internal_split(
    subjects: Sequence[str],
    *,
    seed_string: str = T2_INTERNAL_SPLIT_SEED_STRING,
    fit_count: int = T2_FIT_SUBJECT_COUNT,
) -> dict[str, Any]:
    """Deterministically partition TRAIN subjects by identity digest alone.

    The ranking key is `sha256(f"{seed}:{subject}")`, so the assignment depends
    on nothing but the subject identity string and the frozen seed. No label, no
    prevalence, no episode count and no model outcome participates.
    """
    unique = sorted({str(subject) for subject in subjects})
    if len(unique) != len(list(subjects)):
        raise T2ProtocolError("The TRAIN subject list contains duplicates.")
    if len(unique) != T2_TRAIN_SUBJECT_COUNT:
        raise T2ProtocolError(
            f"The T2 internal split expects {T2_TRAIN_SUBJECT_COUNT} TRAIN "
            f"subjects; got {len(unique)}."
        )
    if not 0 < fit_count < len(unique):
        raise T2ProtocolError(f"Invalid fit subject count {fit_count}.")

    ranked = sorted(
        unique,
        key=lambda subject: (
            hashlib.sha256(f"{seed_string}:{subject}".encode()).hexdigest(),
            subject,
        ),
    )
    fit = sorted(ranked[:fit_count])
    internal_dev = sorted(ranked[fit_count:])
    payload = {
        "algorithm": T2_INTERNAL_SPLIT_ALGORITHM,
        "seed_string": seed_string,
        "fit_subjects": fit,
        "internal_dev_subjects": internal_dev,
        "fit_count": len(fit),
        "internal_dev_count": len(internal_dev),
    }
    return {**payload, "split_sha256": _canonical_digest(payload)}


def validate_internal_split(
    assignment: dict[str, Any],
    *,
    validation_subjects: Iterable[str],
    test_subjects: Iterable[str],
) -> dict[str, Any]:
    """Prove the internal split is disjoint and touches no outer partition."""
    fit = set(assignment["fit_subjects"])
    internal_dev = set(assignment["internal_dev_subjects"])
    overlap = sorted(fit & internal_dev)
    if overlap:
        raise T2ProtocolError(
            f"The T2 internal split is not subject-disjoint: {overlap}."
        )
    if len(fit) != T2_FIT_SUBJECT_COUNT or len(internal_dev) != (
        T2_INTERNAL_DEV_SUBJECT_COUNT
    ):
        raise T2ProtocolError(
            f"The T2 internal split must be "
            f"{T2_FIT_SUBJECT_COUNT}/{T2_INTERNAL_DEV_SUBJECT_COUNT}; got "
            f"{len(fit)}/{len(internal_dev)}."
        )
    outer_validation = set(str(subject) for subject in validation_subjects)
    sealed_test = set(str(subject) for subject in test_subjects)
    leaked_validation = sorted((fit | internal_dev) & outer_validation)
    if leaked_validation:
        raise T2ProtocolError(
            f"Outer VALIDATION subjects {leaked_validation} entered the T2 "
            "internal TRAIN split."
        )
    leaked_test = sorted((fit | internal_dev) & sealed_test)
    if leaked_test:
        raise T2ProtocolError(
            f"Sealed TEST subjects {leaked_test} entered the T2 internal split. "
            "TEST is refused by name."
        )
    return assignment


# ---------------------------------------------------------------------------
# Training population, loss and TBPTT (§9, §10, §11)
# ---------------------------------------------------------------------------


def require_full_chronological_population(
    *, offered_row_count: int, full_stream_row_count: int
) -> int:
    """Refuse a training population that has been thinned out of its timeline."""
    if int(offered_row_count) != int(full_stream_row_count):
        raise T2ProtocolError(
            f"T2 trains on full chronological streams: {offered_row_count} rows "
            f"were offered against {full_stream_row_count} available. Negative "
            "sampling is forbidden -- dropping windows destroys the temporal "
            "continuity T2 exists to model."
        )
    return int(offered_row_count)


def positive_class_weight(*, negative_count: int, positive_count: int) -> float:
    """The frozen imbalance weight, computed on the 48-subject fit partition."""
    if positive_count <= 0 or negative_count <= 0:
        raise T2ProtocolError(
            "The positive class weight needs both classes present in the T2 fit "
            "partition."
        )
    return float(negative_count) / float(positive_count)


def tbptt_chunks(
    stream: T2StreamKey, row_count: int, *, length: int = T2_TBPTT_LENGTH
) -> tuple[T2Chunk, ...]:
    """Chunk one stream for truncated backpropagation.

    The distinction this encodes is the whole point of §11: at a chunk boundary
    the carried state is **detached** so gradients stop, but the state itself
    continues. State resets only at a real stream boundary, never every 256
    windows.
    """
    if int(length) != T2_TBPTT_LENGTH:
        raise T2ProtocolError(
            f"The frozen TBPTT length is {T2_TBPTT_LENGTH} windows "
            f"({T2_TBPTT_HORIZON_SECONDS:.0f} s at a "
            f"{T2_WINDOW_STRIDE_SECONDS:.0f} s stride); {length} was requested."
        )
    if row_count <= 0:
        raise T2ProtocolError("A T2 stream carries no rows.")
    chunks: list[T2Chunk] = []
    for start in range(0, int(row_count), length):
        stop = min(start + length, int(row_count))
        first = start == 0
        chunks.append(
            T2Chunk(
                stream=stream,
                start_index=start,
                stop_index=stop,
                # The first chunk of a stream starts from the frozen zero state;
                # every later chunk inherits the previous chunk's state, detached.
                carries_state_in=not first,
                detaches_state_in=not first,
            )
        )
    return tuple(chunks)


# ---------------------------------------------------------------------------
# Candidates, capacity and selection (§12, §13, §16)
# ---------------------------------------------------------------------------


def require_arm(name: str) -> str:
    if name not in T2_ARMS:
        raise T2ProtocolError(
            f"{name!r} is not a frozen T2 candidate; exactly {list(T2_ARMS)} are."
        )
    return name


def require_capacity_envelope(parameter_counts: dict[str, int]) -> dict[str, Any]:
    """Neither candidate may win by being dramatically larger than the other."""
    missing = [arm for arm in T2_ARMS if arm not in parameter_counts]
    if missing:
        raise T2ProtocolError(f"Parameter counts are missing for {missing}.")
    gru = int(parameter_counts[T2_ARM_GRU])
    s4d = int(parameter_counts[T2_ARM_S4D])
    if gru <= 0 or s4d <= 0:
        raise T2ProtocolError("Trainable parameter counts must be positive.")
    ratio = s4d / gru
    if not T2_PARAMETER_RATIO_MIN <= ratio <= T2_PARAMETER_RATIO_MAX:
        raise T2ProtocolError(
            f"The candidates are outside the shared capacity envelope: "
            f"{T2_ARM_S4D} / {T2_ARM_GRU} = {ratio:.4f}, permitted "
            f"[{T2_PARAMETER_RATIO_MIN}, {T2_PARAMETER_RATIO_MAX}]."
        )
    return {
        "parameter_counts": {T2_ARM_GRU: gru, T2_ARM_S4D: s4d},
        "ratio_s4d_over_gru": ratio,
        "within_envelope": True,
    }


def select_t2_arm(
    *,
    pooled_auprc: dict[str, float],
    subject_macro_auprc: dict[str, float],
    parameter_counts: dict[str, int],
) -> dict[str, Any]:
    """The frozen three-step selection rule. Challenge evidence never enters it."""
    for source in (pooled_auprc, subject_macro_auprc, parameter_counts):
        missing = [arm for arm in T2_ARMS if arm not in source]
        if missing:
            raise T2ProtocolError(f"Selection input is missing {missing}.")
    gru, s4d = T2_ARM_GRU, T2_ARM_S4D
    difference = abs(float(pooled_auprc[gru]) - float(pooled_auprc[s4d]))
    if difference >= T2_SELECTION_TIE_TOLERANCE:
        selected = gru if pooled_auprc[gru] > pooled_auprc[s4d] else s4d
        basis = T2_PRIMARY_SELECTION_METRIC
    elif subject_macro_auprc[gru] != subject_macro_auprc[s4d]:
        selected = gru if subject_macro_auprc[gru] > subject_macro_auprc[s4d] else s4d
        basis = T2_SECONDARY_SELECTION_METRIC
    else:
        selected = gru if parameter_counts[gru] <= parameter_counts[s4d] else s4d
        basis = T2_SELECTION_FINAL_TIE_BREAK
    return {
        "selected_arm": selected,
        "selection_basis": basis,
        "pooled_auprc_difference": difference,
        "tie_tolerance": T2_SELECTION_TIE_TOLERANCE,
        "challenge_evidence_used": T2_CHALLENGE_IS_SELECTION_INPUT,
        "weighted_composite_used": T2_WEIGHTED_COMPOSITE_SCORE_PERMITTED,
    }


def require_threshold_partition(partition: str) -> str:
    """The binary threshold is frozen on internal-dev, before outer VALIDATION."""
    if partition != T2_BINARY_THRESHOLD_PARTITION:
        raise T2ProtocolError(
            f"The T2 binary threshold is frozen on "
            f"{T2_BINARY_THRESHOLD_PARTITION}, never on {partition!r}."
        )
    return partition


def require_t2_score_semantics(name: str) -> str:
    """A T2 sigmoid is a temporal model score, and may not be renamed upward."""
    forbidden = {
        "calibrated_probability",
        "calibrated_uncertainty",
        "confidence",
        "uncertainty",
        "conformal_evidence",
    }
    if name in forbidden:
        raise T2ProtocolError(
            f"A raw T2 score may not be called {name!r}. It is a "
            f"{T2_OUTPUT_SEMANTIC_NAME}; U1 remains the retained calibrated "
            "probability source for the frozen detector."
        )
    return name


def t2_protocol_identity() -> dict[str, Any]:
    """The prospective provenance bundle §30 requires, with nothing computed."""
    return {
        "protocol_class": "t2_longitudinal_temporal_protocol_v1",
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "starting_git_sha": T2_STARTING_GIT_SHA,
        "split_sha256": T2_SPLIT_SHA256,
        "split_file_sha256": T2_SPLIT_FILE_SHA256,
        "p1_protocol_sha256": T2_P1_PROTOCOL_SHA256,
        "p1_retention_decision_sha256": T2_P1_RETENTION_DECISION_SHA256,
        "p1b_experiment_lock_sha256": T2_P1B_EXPERIMENT_LOCK_SHA256,
        "b4_protocol_sha256": T2_B4_PROTOCOL_SHA256,
        "encoder_checkpoint_sha256": T2_ENCODER_CHECKPOINT_SHA256,
        "physiology_schema_sha256": T2_PHYSIOLOGY_SCHEMA_SHA256,
        "physiology_transform_sha256": T2_PHYSIOLOGY_TRANSFORM_SHA256,
        "train_stream_cache_sha256": T2_TRAIN_STREAM_CACHE_SHA256,
        "validation_stream_cache_sha256": T2_VALIDATION_STREAM_CACHE_SHA256,
        "train_representation_content_sha256": (T2_TRAIN_REPRESENTATION_CONTENT_SHA256),
        "validation_representation_content_sha256": (
            T2_VALIDATION_REPRESENTATION_CONTENT_SHA256
        ),
        "u1_retention_decision_sha256": T2_U1_RETENTION_DECISION_SHA256,
        "u1_result_sha256": T2_U1_RESULT_SHA256,
        "u1_experiment_lock_sha256": T2_U1_EXPERIMENT_LOCK_SHA256,
        "m2_retention_decision_sha256": T2_M2_RETENTION_DECISION_SHA256,
        "m2g_arm_result_sha256": T2_M2G_ARM_RESULT_SHA256,
        "internal_split_sha256": T2_INTERNAL_SPLIT_SHA256,
        "stream_ordering_rule": {
            "stream_key": list(T2_STREAM_KEY_FIELDS),
            "order_by": T2_STREAM_ORDER_FIELD,
            "bidirectional": T2_BIDIRECTIONAL_PERMITTED,
            "future_context": T2_FUTURE_CONTEXT_PERMITTED,
        },
        "availability_rule": {
            "available": T2_OBSERVATION_AVAILABLE,
            "unavailable_exact_flat": T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT,
            "synthetic_z_permitted": T2_SYNTHETIC_Z_PERMITTED,
            "updates_hidden_state": T2_UNAVAILABLE_ROW_UPDATES_HIDDEN_STATE,
            "advances_timeline": T2_UNAVAILABLE_ROW_ADVANCES_TIMELINE,
        },
        "input_dim": T2_INPUT_DIM,
        "tbptt_length": T2_TBPTT_LENGTH,
        "candidates": list(T2_ARMS),
        "optimizer": {
            "name": T2_OPTIMIZER,
            "learning_rate": T2_LEARNING_RATE,
            "weight_decay": T2_WEIGHT_DECAY,
            "max_epochs": T2_MAX_EPOCHS,
            "gradient_clip_norm": T2_GRADIENT_CLIP_NORM,
        },
        "seed": T2_SEED,
        "selection_rule": {
            "primary": T2_PRIMARY_SELECTION_METRIC,
            "secondary": T2_SECONDARY_SELECTION_METRIC,
            "tie_tolerance": T2_SELECTION_TIE_TOLERANCE,
            "final_tie_break": T2_SELECTION_FINAL_TIE_BREAK,
        },
        "environment_dependency_digest": T2_ENVIRONMENT_DEPENDENCY_DIGEST,
        "test_accessed": T2_TEST_ACCESSED,
        "sealed_test_state": T2_SEALED_TEST_STATE,
    }
