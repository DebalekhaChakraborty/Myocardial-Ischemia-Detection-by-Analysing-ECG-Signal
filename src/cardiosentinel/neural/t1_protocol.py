"""The frozen T1-v1 causal episode-state protocol.

T1 converts already-existing causal evidence into an interpretable, hysteretic
system state. It is a deterministic state machine, not a model: no parameter is
learned here, nothing is fitted, and every quantity a future execution harness
needs is either frozen below or derived by a rule frozen below.

**What T1 is not.** Not another neural model, not another temporal model, not
another calibrator, not a post-hoc smoother, not a relabelled T2 binary
threshold, not an LLM, and not an edge/cloud router.

**Structural binder only.** This module is standard library throughout. It holds
constants, pure structures, deterministic fold generation, the exact empirical
order-statistic rule, candidate enumeration, pure transition helpers, episode
grouping helpers and refusal functions. It constructs no model, opens no run
artifact, reads no partition, computes no metric against real data and mutates
no file. Nothing here can touch scientific state.

**The protocol is prospective.** Every threshold *rule*, persistence profile,
evidence formula, selection metric and tie-break below was frozen before any T1
state trace existed. Candidate thresholds are generated at execution time from
FIT-subject background negatives only; no absolute probability or temporal-score
threshold is hand-chosen here.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple, Sequence

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]

T1_PROTOCOL_NAME: Final = "T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1"
T1_PROTOCOL_PATH: Final = REPOSITORY_ROOT / "docs" / f"{T1_PROTOCOL_NAME}.md"
T1_PROTOCOL_SHA256: Final = (
    "ef044754020b1756ea7aae5fa1b747c5ba6fc0c8cd70d52e73185555897d70d4"
)

T1_STARTING_GIT_SHA: Final = "b3004da9dcd8e7462d69eac81eb82ca9da86b8cb"

# ---------------------------------------------------------------------------
# Frozen upstream identities T1 consumes read-only (§4)
#
# Every value is the exact immutable identity carried by the merged repository.
# T1 binds them so a future execution harness cannot silently consume a
# different upstream lineage.
# ---------------------------------------------------------------------------
T1_M2_RETENTION_DECISION_SHA256: Final = (
    "da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47"
)
T1_M2_RETAINED_ARM: Final = "M2-G"

T1_U1_PROTOCOL_SHA256: Final = (
    "d6235b477af278fe051822bdcccb54f985e4eceb0c6e92c1424f5e9d7d79b33b"
)
T1_U1_RETENTION_DECISION_SHA256: Final = (
    "9d8436f2b7d2c303aeeb03e438c60fb8110f7d06d0bbd589f5be65ea8f80cb7b"
)
T1_U1_OOF_EVIDENCE_STORE_SHA256: Final = (
    "b95f484c9a7b08447f5a5d4330528136e040cf05acb9e2f7e54305e20bdffcba"
)

T1_T2_PROTOCOL_SHA256: Final = (
    "6546086a55fe2c9c109f4121cdb6b42d4d53ce0112c9611eb895bd8c805cfefb"
)
T1_T2_RETENTION_DECISION_SHA256: Final = (
    "4846921135b0ac83ceb40a0db063c2e4a3b2520971f279abe4f0c517c4f7dd20"
)
T1_T2_RETAINED_ARM: Final = "causal_s4d_longitudinal_v1"
T1_T2_COMPARATOR_ARM: Final = "causal_gru_longitudinal_v1"
T1_T2_OUTER_RESULT_SHA256: Final = (
    "c58ed40dac753157b00ce6c70eb52fe903ecee72a5ef84e40932c1a80e259dbf"
)
T1_T2_ROW_EVIDENCE_CONTENT_SHA256: Final = (
    "2240ca683fbcb790609c47f4a82af85250abb281fbbb9751dc74607a4eb591ca"
)

T1_SPLIT_SHA256: Final = (
    "66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7"
)
T1_FEATURE_CORPUS_SHA256: Final = (
    "f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5"
)

# ---------------------------------------------------------------------------
# The full VALIDATION timeline T1 development runs over (§5)
# ---------------------------------------------------------------------------
T1_TIMELINE_ROW_COUNT: Final = 492_904
T1_TIMELINE_STREAM_COUNT: Final = 30
T1_TIMELINE_SUBJECT_COUNT: Final = 12
T1_STREAM_KEY: Final = ("record_id", "channel_index")
T1_TIMELINE_ORDER: Final = "start_sample"
T1_WINDOW_SAMPLES: Final = 2500
T1_STRIDE_SAMPLES: Final = 1250
T1_SAMPLING_FREQUENCY_HZ: Final = 250
T1_WINDOW_SECONDS: Final = T1_WINDOW_SAMPLES / T1_SAMPLING_FREQUENCY_HZ
T1_STRIDE_SECONDS: Final = T1_STRIDE_SAMPLES / T1_SAMPLING_FREQUENCY_HZ

# The timeline is never regenerated from waveform data; its identity comes from
# the already-retained T2 row-evidence lineage.
T1_TIMELINE_SOURCE: Final = "retained_t2_row_evidence_frozen_stream_lineage"
T1_TIMELINE_REGENERATED_FROM_WAVEFORM: Final = False

# Availability, reconciled across the two retained sources (§8).
T1_EXPECTED_SCORE_PRESENT_ROWS: Final = 492_898
T1_EXPECTED_UNAVAILABLE_ROWS: Final = 6

# ---------------------------------------------------------------------------
# The frozen detector operating point and calibration contract (§6, §7)
# ---------------------------------------------------------------------------
T1_DETECTOR_THRESHOLD: Final = 0.7554003000259399
T1_U1_CALIBRATOR_FAMILY: Final = "platt_logistic_on_recovered_logit"
T1_U1_CLAMP_DELTA: Final = 1e-7

# For DEVELOPMENT on these twelve subjects the subject-disjoint out-of-fold
# calibrator is the ONLY permitted one. The all-VALIDATION deployment calibrator
# was fitted on every one of them, so using it here would leak the held-out
# subject into its own calibrated probability.
T1_U1_OOF_CALIBRATOR_REQUIRED_FOR_DEVELOPMENT: Final = True
T1_U1_DEPLOYMENT_CALIBRATOR_PERMITTED_FOR_DEVELOPMENT: Final = False

# Applying an already-fitted held-out-subject calibrator to that subject's full
# timeline is deterministic arithmetic, not a new fit.
T1_U1_REFIT_PERMITTED: Final = False
T1_M2_REPLAY_PERMITTED: Final = False
T1_T2_REPLAY_PERMITTED: Final = False
T1_FULL_TIMELINE_CALIBRATION_CONTRACT: Final = (
    "apply_frozen_held_out_subject_platt_calibrator_to_every_scored_m2g_row"
)

# ---------------------------------------------------------------------------
# The T1 state space (§9)
# ---------------------------------------------------------------------------
T1_STATE_NORMAL: Final = "NORMAL"
T1_STATE_WATCH: Final = "WATCH"
T1_STATE_EVENT: Final = "EVENT"
T1_STATE_RECOVERY: Final = "RECOVERY"
T1_STATES: Final = (
    T1_STATE_NORMAL,
    T1_STATE_WATCH,
    T1_STATE_EVENT,
    T1_STATE_RECOVERY,
)
T1_INITIAL_STATE: Final = T1_STATE_NORMAL

# State never crosses a stream, and therefore never crosses a record, a channel
# or a subject. T1-v1 is per-stream; multi-channel patient fusion is undefined.
T1_STATE_CROSSES_STREAM: Final = False
T1_PER_STREAM: Final = True
T1_PATIENT_LEVEL_CHANNEL_FUSION_DEFINED: Final = False

# ---------------------------------------------------------------------------
# What one row may and may not offer the transition function (§7)
# ---------------------------------------------------------------------------
T1_ALLOWED_ROW_INPUTS: Final = (
    "stable_id",
    "m2g_detector_score",
    "detector_decision_d_t",
    "oof_calibrated_probability_p_t",
    "decision_error_uncertainty_u_t",
    "s4d_temporal_evidence_s_t",
    "score_present",
    "elapsed_stream_seconds",
    "elapsed_state_seconds",
)

T1_FORBIDDEN_TRANSITION_INPUTS: Final = (
    "label",
    "target_family",
    "subject_outcome",
    "episode_identity",
    "future_row",
    "future_score",
    "gru_score",
    "s4d_binary_decision",
    "t2_frozen_reporting_threshold",
    "u_star_dev",
    "u_star_deploy",
    "challenge_family_identity",
    "m2_gate_outcome",
    "m2_update_admitted",
    "test_derived_quantity",
)

# The retained S4D reporting threshold is T2 experiment evidence only. It is
# named here solely so a test can prove it never becomes a T1 policy value.
T1_T2_REPORTING_THRESHOLD_NOT_T1_POLICY: Final = 0.8972153067588806
T1_T2_THRESHOLD_IS_T1_POLICY: Final = False

# ---------------------------------------------------------------------------
# Prospective candidate threshold generation (§10)
# ---------------------------------------------------------------------------
Q_WATCH: Final = (0.90, 0.95)
Q_EVENT: Final = (0.99, 0.995)

# Candidate thresholds come from FIT-subject PRIMARY background negatives only.
T1_THRESHOLD_SOURCE_POPULATION: Final = "fit_subject_primary_background_negative"
T1_THRESHOLD_RULE: Final = "exact_empirical_order_statistic_ceil_q_n"
T1_THRESHOLD_TIE_ORDER: Final = ("value", "stable_id")
T1_THRESHOLD_INTERPOLATION_PERMITTED: Final = False
T1_THRESHOLD_USES_CHALLENGE_ROWS: Final = False
T1_THRESHOLD_USES_LABEL_WEIGHTING: Final = False

# ---------------------------------------------------------------------------
# The three frozen persistence profiles (§11)
# ---------------------------------------------------------------------------


class T1PersistenceProfile(NamedTuple):
    """One frozen duration profile. Counts are consecutive available rows."""

    name: str
    watch_clear_windows: int
    event_confirm_windows: int
    event_release_windows: int
    re_event_confirm_windows: int
    recovery_clear_windows: int
    cold_event_confirm_windows: int


T1_PROFILE_FAST: Final = T1PersistenceProfile(
    name="FAST",
    watch_clear_windows=2,
    event_confirm_windows=2,
    event_release_windows=2,
    re_event_confirm_windows=1,
    recovery_clear_windows=3,
    cold_event_confirm_windows=4,
)
T1_PROFILE_BALANCED: Final = T1PersistenceProfile(
    name="BALANCED",
    watch_clear_windows=3,
    event_confirm_windows=3,
    event_release_windows=3,
    re_event_confirm_windows=2,
    recovery_clear_windows=6,
    cold_event_confirm_windows=6,
)
T1_PROFILE_CONSERVATIVE: Final = T1PersistenceProfile(
    name="CONSERVATIVE",
    watch_clear_windows=6,
    event_confirm_windows=6,
    event_release_windows=6,
    re_event_confirm_windows=3,
    recovery_clear_windows=12,
    cold_event_confirm_windows=12,
)

# Ordered most cautious first: this is also the frozen tie-break preference.
T1_PERSISTENCE_PROFILES: Final = (
    T1_PROFILE_CONSERVATIVE,
    T1_PROFILE_BALANCED,
    T1_PROFILE_FAST,
)

# WATCH entry is immediate after a single WATCH-evidence row; there is no
# watch-confirmation duration to tune.
T1_WATCH_ENTRY_WINDOWS: Final = 1

T1_CANDIDATE_POLICY_COUNT: Final = (
    len(Q_WATCH) * len(Q_EVENT) * len(T1_PERSISTENCE_PROFILES)
)

# ---------------------------------------------------------------------------
# Cold start (§14)
# ---------------------------------------------------------------------------
T1_COLD_START_SECONDS: Final = 300.0
T1_COLD_START_REQUIRES_S4D: Final = False
T1_MATURE_REQUIRES_S4D: Final = True
T1_COLD_START_MODIFIES_T2: Final = False

# ---------------------------------------------------------------------------
# Development split (§17) and the optimism disclosure (§18)
# ---------------------------------------------------------------------------
T1_VALIDATION_SUBJECTS: Final = (
    "ltstdb:s2004",
    "ltstdb:s2005",
    "ltstdb:s2019",
    "ltstdb:s2020",
    "ltstdb:s2023",
    "ltstdb:s2031",
    "ltstdb:s2057",
    "ltstdb:s2058",
    "ltstdb:s2059",
    "ltstdb:s3068",
    "ltstdb:s3072",
    "ltstdb:s3073",
)
T1_FOLD_COUNT: Final = len(T1_VALIDATION_SUBJECTS)
T1_FOLD_DESIGN: Final = "leave_one_subject_out"
T1_HELD_OUT_LABELS_AVAILABLE_DURING_SELECTION: Final = False
T1_FOLD_RETRY_PERMITTED: Final = False
T1_FOLD_MANUAL_OVERRIDE_PERMITTED: Final = False

T1_EVIDENCE_CLASS: Final = (
    "cross_fitted_t1_development_evidence_conditional_on_frozen_upstream_components"
)
T1_IS_UNSEEN_GENERALIZATION: Final = False
T1_IS_EXTERNAL_VALIDATION: Final = False
T1_IS_INDEPENDENT_VALIDATION: Final = False
T1_IS_CLINICAL_VALIDATION: Final = False

# ---------------------------------------------------------------------------
# Episode, matching and selection semantics (§19-§23)
# ---------------------------------------------------------------------------
T1_EPISODE_CADENCE_SAMPLES: Final = T1_STRIDE_SAMPLES
T1_EPISODE_GAP_BRIDGING_PERMITTED: Final = False
T1_EPISODE_MINIMUM_DURATION_FILTER: Final = None
T1_EPISODE_ANNOTATION_REREAD: Final = False
T1_EPISODE_MATCHING: Final = "one_to_one_earliest_unmatched_overlapping_run"

T1_PRIMARY_SELECTION_METRIC: Final = "pooled_episode_f1"
T1_SECONDARY_SELECTION_METRIC: Final = "pooled_primary_window_mcc"
T1_THIRD_SELECTION_METRIC: Final = "false_event_onsets_per_physical_hour"
T1_FOURTH_SELECTION_METRIC: Final = "event_exposure_fraction"
T1_SELECTION_TOLERANCE: Final = 1e-6
T1_SELECTION_USES_CHALLENGE_EVIDENCE: Final = False
T1_SELECTION_USES_LATENCY: Final = False
T1_SELECTION_USES_WEIGHTED_COMPOSITE: Final = False

# Bootstrap (§25)
T1_BOOTSTRAP_REPLICATES: Final = 1000
T1_BOOTSTRAP_SEED: Final = 2026
T1_BOOTSTRAP_UNIT: Final = "subject"
T1_BOOTSTRAP_RESELECTS_POLICY: Final = False
T1_BOOTSTRAP_CLAIM_SCOPE: Final = (
    "between_subject_variation_conditional_on_the_cross_fitted_t1_procedure"
)

# Cold-start reporting strata (§26)
T1_COLD_START_STRATA: Final = ("0_5_minutes", "5_60_minutes", "over_60_minutes")

# Challenge families are reported, never routed on (§27)
T1_CHALLENGE_FAMILIES: Final = ("rate_related", "axis_shift", "conduction_change")
T1_CHALLENGE_IS_SELECTION_INPUT: Final = False
T1_CHALLENGE_IS_TRANSITION_INPUT: Final = False
T1_CONDUCTION_EVIDENCE_LEVEL: Final = "exploratory_descriptive"

# ---------------------------------------------------------------------------
# Firewalls (§30, §31, §32)
# ---------------------------------------------------------------------------
T1_TEST_ACCESSED: Final = False
T1_SEALED_TEST_STATE: Final = "unopened"

T1_ROUTING_DEFINED: Final = False
T1_U1_SYMMETRIC_ROUTER_REJECTED: Final = True
T1_LLM_PARTICIPATES_IN_STATE: Final = False


class T1ProtocolError(RuntimeError):
    """Raised when a T1 protocol rule is violated."""


def _sha256_file(path: Path) -> str:
    """The same streaming digest the other protocol modules use.

    Protocol modules stay standard-library only, so this mirrors
    `t2_protocol._sha256_file` rather than importing the persistence helper.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_t1_protocol_document(path: Path = T1_PROTOCOL_PATH) -> str:
    """Verify the frozen T1 protocol document byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise T1ProtocolError(f"T1 protocol document is missing at {document}.")
    digest = _sha256_file(document)
    if digest != T1_PROTOCOL_SHA256:
        raise T1ProtocolError(
            f"T1 protocol digest {digest} differs from the frozen "
            f"{T1_PROTOCOL_SHA256}. The protocol is immutable."
        )
    return digest


# ---------------------------------------------------------------------------
# Pure structures
# ---------------------------------------------------------------------------


class T1Thresholds(NamedTuple):
    """One candidate policy's four generated thresholds."""

    p_watch: float
    s_watch: float
    p_event: float
    s_event: float


@dataclass(frozen=True)
class T1CandidatePolicy:
    """One of the twelve prospective candidates, before thresholds exist."""

    q_watch: float
    q_event: float
    profile: T1PersistenceProfile

    @property
    def name(self) -> str:
        return f"qw{self.q_watch}_qe{self.q_event}_{self.profile.name}"


class T1Fold(NamedTuple):
    """One leave-one-subject-out development fold."""

    fold_index: int
    held_out_subject: str
    fit_subjects: tuple[str, ...]


class T1Row(NamedTuple):
    """Exactly what the transition function may see for one timeline position.

    There is deliberately no label, no target family and no future field. A row
    that cannot be scored carries `score_present=False` and no invented value.
    """

    stable_id: str
    score_present: bool
    detector_decision: bool | None
    calibrated_probability: float | None
    decision_error_uncertainty: float | None
    temporal_evidence: float | None
    elapsed_stream_seconds: float


# ---------------------------------------------------------------------------
# Deterministic fold generation (§17)
# ---------------------------------------------------------------------------


def t1_folds(subjects: Sequence[str] = T1_VALIDATION_SUBJECTS) -> tuple[T1Fold, ...]:
    """Leave-one-subject-out folds in frozen ascending subject order.

    Assignment depends on identity alone. Nothing about a subject's scores,
    labels or difficulty may influence which fold it lands in.
    """
    ordered = tuple(sorted(subjects))
    if len(set(ordered)) != len(ordered):
        raise T1ProtocolError("T1 fold subjects must be unique.")
    return tuple(
        T1Fold(
            fold_index=index,
            held_out_subject=held_out,
            fit_subjects=tuple(s for s in ordered if s != held_out),
        )
        for index, held_out in enumerate(ordered)
    )


def candidate_policies() -> tuple[T1CandidatePolicy, ...]:
    """The twelve prospective candidates, enumerated in frozen order."""
    return tuple(
        T1CandidatePolicy(q_watch=qw, q_event=qe, profile=profile)
        for qw in Q_WATCH
        for qe in Q_EVENT
        for profile in T1_PERSISTENCE_PROFILES
    )


# ---------------------------------------------------------------------------
# The exact empirical order statistic (§10)
# ---------------------------------------------------------------------------


def empirical_order_statistic(
    values: Sequence[float], stable_ids: Sequence[str], quantile: float
) -> float:
    """The frozen threshold rule: `k = ceil(q * N)`, 1-based, no interpolation.

    Ties are broken by `stable_id` so the result is independent of input order.
    A library quantile would interpolate between neighbours and would not be
    reproducible across versions; this is deliberately the raw order statistic.
    """
    if len(values) != len(stable_ids):
        raise T1ProtocolError("Values and stable ids must align one to one.")
    if not values:
        raise T1ProtocolError("An order statistic needs a non-empty population.")
    if not 0.0 < quantile <= 1.0:
        raise T1ProtocolError(f"Quantile {quantile!r} is outside (0, 1].")
    ordered = sorted(zip(values, stable_ids, strict=True))
    position = math.ceil(quantile * len(ordered))
    return float(ordered[position - 1][0])


# ---------------------------------------------------------------------------
# Evidence definitions (§12, §14)
# ---------------------------------------------------------------------------


def is_watch_evidence(row: T1Row, thresholds: T1Thresholds) -> bool:
    """Any one of the three signals is enough to raise attention."""
    _require_scored(row)
    return (
        bool(row.detector_decision)
        or float(row.calibrated_probability) >= thresholds.p_watch
        or float(row.temporal_evidence) >= thresholds.s_watch
    )


def is_event_evidence(row: T1Row, thresholds: T1Thresholds) -> bool:
    """EVENT needs agreement, and needs more of it once the stream is mature.

    Before `T1_COLD_START_SECONDS` the S4D temporal term is not required: T2's
    own outer evidence recorded zero thresholded sensitivity in the first five
    minutes, so demanding it there would make early EVENT unreachable by
    construction. This relaxes a T1 rule; it modifies no T2 state.
    """
    _require_scored(row)
    if row.elapsed_stream_seconds < T1_COLD_START_SECONDS:
        return (
            bool(row.detector_decision)
            and float(row.calibrated_probability) >= thresholds.p_event
        )
    return (
        bool(row.detector_decision)
        and float(row.calibrated_probability) >= thresholds.p_event
        and float(row.temporal_evidence) >= thresholds.s_event
    )


def is_normal_evidence(row: T1Row, thresholds: T1Thresholds) -> bool:
    """All three signals must be quiet for a row to argue for de-escalation."""
    _require_scored(row)
    return (
        not bool(row.detector_decision)
        and float(row.calibrated_probability) < thresholds.p_watch
        and float(row.temporal_evidence) < thresholds.s_watch
    )


def is_cold_start(row: T1Row) -> bool:
    return row.elapsed_stream_seconds < T1_COLD_START_SECONDS


def required_event_confirm_windows(row: T1Row, profile: T1PersistenceProfile) -> int:
    """Cold rows confirm on the cold budget, mature rows on the mature one."""
    if is_cold_start(row):
        return profile.cold_event_confirm_windows
    return profile.event_confirm_windows


def _require_scored(row: T1Row) -> None:
    if not row.score_present:
        raise T1ProtocolError(
            "An unavailable row has no evidence. It carries no probability, no "
            "uncertainty and no temporal score, and nothing may be invented for "
            "it; the caller holds state instead."
        )


def decision_error_uncertainty(
    detector_decision: bool, calibrated_probability: float
) -> float:
    """`u_t` is the calibrated probability that the detector's call is wrong."""
    p = float(calibrated_probability)
    return 1.0 - p if detector_decision else p


# ---------------------------------------------------------------------------
# Unavailable-row semantics (§8)
# ---------------------------------------------------------------------------

T1_UNAVAILABLE_HOLDS_STATE: Final = True
T1_UNAVAILABLE_RESETS_STREAKS: Final = True
T1_UNAVAILABLE_ADVANCES_STATE_TIME: Final = True
T1_UNAVAILABLE_PERMITS_TRANSITION: Final = False
T1_IMPUTATION_PERMITTED: Final = False
T1_FORWARD_FILL_PERMITTED: Final = False
T1_SYNTHETIC_ZERO_PERMITTED: Final = False


# ---------------------------------------------------------------------------
# Episode grouping (§19) and predicted runs (§20)
# ---------------------------------------------------------------------------


def group_reference_episodes(
    start_samples: Sequence[int], is_primary_positive: Sequence[bool]
) -> tuple[tuple[int, int], ...]:
    """Maximal runs of PRIMARY ischemic-positive rows at the exact cadence.

    Returned as `(begin_index, end_index_exclusive)` pairs. A non-positive row
    breaks a run, and so does any consecutive `start_sample` difference other
    than exactly one stride: an episode is a physically contiguous stretch, and
    a gap is never bridged.
    """
    if len(start_samples) != len(is_primary_positive):
        raise T1ProtocolError("Start samples and positives must align.")
    episodes: list[tuple[int, int]] = []
    begin: int | None = None
    for index, positive in enumerate(is_primary_positive):
        if positive:
            contiguous = (
                begin is not None
                and int(start_samples[index]) - int(start_samples[index - 1])
                == T1_EPISODE_CADENCE_SAMPLES
            )
            if begin is None or not contiguous:
                if begin is not None:
                    episodes.append((begin, index))
                begin = index
        elif begin is not None:
            episodes.append((begin, index))
            begin = None
    if begin is not None:
        episodes.append((begin, len(is_primary_positive)))
    return tuple(episodes)


def match_runs_to_episodes(
    episodes: Sequence[tuple[int, int]], runs: Sequence[tuple[int, int]]
) -> dict[int, int]:
    """One-to-one chronological matching, episodes ordered by onset.

    Each reference episode takes the earliest still-unmatched predicted run that
    overlaps it. A run spanning several episodes therefore matches only the
    first, leaving the rest unmatched unless another run detects them -- which
    is the intended penalty for an overmerged EVENT state.
    """
    matched: dict[int, int] = {}
    used: set[int] = set()
    for episode_index, (begin, end) in enumerate(
        sorted(episodes, key=lambda span: span[0])
    ):
        for run_index, (run_begin, run_end) in enumerate(runs):
            if run_index in used:
                continue
            if run_begin < end and begin < run_end:
                matched[episode_index] = run_index
                used.add(run_index)
                break
    return matched


# ---------------------------------------------------------------------------
# Policy selection ordering (§23)
# ---------------------------------------------------------------------------

T1_SELECTION_ORDER: Final = (
    "pooled_episode_f1_desc",
    "pooled_primary_window_mcc_desc",
    "false_event_onsets_per_hour_asc",
    "event_exposure_fraction_asc",
    "q_event_desc",
    "q_watch_desc",
    "persistence_profile_conservative_first",
)


def policy_sort_key(
    policy: T1CandidatePolicy,
    *,
    episode_f1: float,
    window_mcc: float,
    false_onsets_per_hour: float,
    event_exposure_fraction: float,
) -> tuple[Any, ...]:
    """The complete lexicographic selection order, as one sortable key.

    Smaller sorts first, so maximised terms are negated. The final three terms
    are the deterministic safety tie-break: no fold may be decided by dictionary
    order or by a human preference expressed after the numbers were seen.
    """
    return (
        -float(episode_f1),
        -float(window_mcc),
        float(false_onsets_per_hour),
        float(event_exposure_fraction),
        -float(policy.q_event),
        -float(policy.q_watch),
        T1_PERSISTENCE_PROFILES.index(policy.profile),
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def require_transition_input_permitted(name: str) -> str:
    """Refuse any field the transition function may never see."""
    if name in T1_FORBIDDEN_TRANSITION_INPUTS:
        raise T1ProtocolError(
            f"{name!r} may not reach the T1 transition function. A deployable "
            "causal state rule cannot depend on evaluation annotation, on a "
            "label, or on the future."
        )
    if name not in T1_ALLOWED_ROW_INPUTS:
        raise T1ProtocolError(
            f"{name!r} is not one of the frozen T1 row inputs: {T1_ALLOWED_ROW_INPUTS}."
        )
    return name


def require_development_calibrator(source: str) -> str:
    """Development on the twelve VALIDATION subjects is out-of-fold only."""
    if source != "u1_oof_development_calibration":
        raise T1ProtocolError(
            "T1 development must consume the subject-disjoint U1 out-of-fold "
            "calibration. The all-VALIDATION deployment calibrator was fitted on "
            "every one of these twelve subjects and would leak the held-out "
            "subject into its own probability."
        )
    return source


def require_state(state: str) -> str:
    if state not in T1_STATES:
        raise T1ProtocolError(f"{state!r} is not one of the four T1 states.")
    return state


# ---------------------------------------------------------------------------
# The exact transition specification (§15)
# ---------------------------------------------------------------------------


class T1Streaks(NamedTuple):
    """Consecutive-available-row counters, one per named condition."""

    event_confirm: int = 0
    watch_clear: int = 0
    event_release: int = 0
    re_event_confirm: int = 0
    recovery_clear: int = 0


T1_ZERO_STREAKS: Final = T1Streaks()


def next_state(
    state: str,
    streaks: T1Streaks,
    row: T1Row,
    thresholds: T1Thresholds,
    profile: T1PersistenceProfile,
) -> tuple[str, T1Streaks]:
    """One causal step. Pure: it reads the current row and nothing ahead of it.

    An unavailable row is not evidence of anything. State is held, every
    confirmation streak resets, and no transition may fire -- a gap must not be
    able to confirm an escalation or a release across itself.

    Escalation takes priority when a row could satisfy more than one internal
    condition, and any state change clears every counter so a streak can never
    survive into a state it was not accumulated in.
    """
    require_state(state)
    if not row.score_present:
        return state, T1_ZERO_STREAKS

    event_evidence = is_event_evidence(row, thresholds)
    normal_evidence = is_normal_evidence(row, thresholds)
    watch_evidence = is_watch_evidence(row, thresholds)

    if state == T1_STATE_NORMAL:
        confirm = streaks.event_confirm + 1 if event_evidence else 0
        if event_evidence and confirm >= required_event_confirm_windows(row, profile):
            return T1_STATE_EVENT, T1_ZERO_STREAKS
        if watch_evidence:
            # Immediate on one row; the streak survives so a WATCH entered by an
            # EVENT-evidence row keeps the confirmation it has already earned.
            return T1_STATE_WATCH, T1Streaks(event_confirm=confirm)
        return T1_STATE_NORMAL, T1_ZERO_STREAKS

    if state == T1_STATE_WATCH:
        if event_evidence:
            confirm = streaks.event_confirm + 1
            if confirm >= required_event_confirm_windows(row, profile):
                return T1_STATE_EVENT, T1_ZERO_STREAKS
            return T1_STATE_WATCH, T1Streaks(event_confirm=confirm)
        if normal_evidence:
            clear = streaks.watch_clear + 1
            if clear >= profile.watch_clear_windows:
                return T1_STATE_NORMAL, T1_ZERO_STREAKS
            return T1_STATE_WATCH, T1Streaks(watch_clear=clear)
        return T1_STATE_WATCH, T1_ZERO_STREAKS

    if state == T1_STATE_EVENT:
        if event_evidence:
            return T1_STATE_EVENT, T1_ZERO_STREAKS
        if normal_evidence:
            release = streaks.event_release + 1
            if release >= profile.event_release_windows:
                return T1_STATE_RECOVERY, T1_ZERO_STREAKS
            return T1_STATE_EVENT, T1Streaks(event_release=release)
        # Ambiguous rows neither release nor re-confirm; they hold EVENT.
        return T1_STATE_EVENT, T1Streaks(event_release=streaks.event_release)

    # RECOVERY. There is deliberately no automatic path back to WATCH: a
    # recovering stream either re-escalates on EVENT evidence or clears.
    if event_evidence:
        confirm = streaks.re_event_confirm + 1
        if confirm >= profile.re_event_confirm_windows:
            return T1_STATE_EVENT, T1_ZERO_STREAKS
        return T1_STATE_RECOVERY, T1Streaks(re_event_confirm=confirm)
    if normal_evidence:
        clear = streaks.recovery_clear + 1
        if clear >= profile.recovery_clear_windows:
            return T1_STATE_NORMAL, T1_ZERO_STREAKS
        return T1_STATE_RECOVERY, T1Streaks(recovery_clear=clear)
    return T1_STATE_RECOVERY, T1_ZERO_STREAKS


T1_TRANSITION_TABLE: Final = (
    ("NORMAL", "watch evidence", "WATCH", "immediate, one row"),
    ("NORMAL", "event evidence x confirm", "EVENT", "escalation has priority"),
    ("NORMAL", "no watch evidence", "NORMAL", "hold"),
    ("WATCH", "event evidence x confirm", "EVENT", "cold or mature budget"),
    ("WATCH", "normal evidence x watch_clear", "NORMAL", "de-escalation"),
    ("WATCH", "otherwise", "WATCH", "hold"),
    ("EVENT", "event evidence", "EVENT", "resets release streak"),
    ("EVENT", "normal evidence x event_release", "RECOVERY", "release"),
    ("EVENT", "ambiguous", "EVENT", "does not contribute to release"),
    ("RECOVERY", "event evidence x re_event_confirm", "EVENT", "re-escalation"),
    ("RECOVERY", "normal evidence x recovery_clear", "NORMAL", "full clear"),
    ("RECOVERY", "otherwise", "RECOVERY", "never automatically WATCH"),
    ("any", "unavailable row", "unchanged", "streaks reset, no transition"),
)
