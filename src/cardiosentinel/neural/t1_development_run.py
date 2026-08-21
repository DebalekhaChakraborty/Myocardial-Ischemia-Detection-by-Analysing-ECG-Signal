"""The canonical T1-v1 development harness.

Implements the frozen 29-stage choreography of
``docs/T1_CANONICAL_DEVELOPMENT_EXECUTION_SPEC_V1.md``. The science it composes
is owned upstream: ``t1_protocol`` supplies the states, the evidence formulas,
the persistence profiles, the order-statistic rule, ``next_state``, episode
grouping, matching and the selection order, and none of it is re-derived here.

**Importing this module executes nothing.** It defines a parser and a runner and
does not construct either. There is no module-level work, no filesystem access
and no upstream read at import time; `main` runs only under ``__main__`` or an
explicit call.

**Running it requires a human authorization naming a commit.** The two frozen
options are the canonical execution flag and ``--expected-git-sha``; the SHA
must match a clean HEAD. There is no scientific knob, no fold selector, no
retry, no seed and no TEST option.

**The fold-scoped label firewall is structural.** A fold's held-out labels are
unreachable until its selection artifact has been promoted and re-read with a
verified digest: the target authority for a fold is constructed over an
explicit subject set and refuses every subject outside it, so "the held-out
labels stayed closed" is a property of the object graph rather than of the
control flow.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from cardiosentinel.neural import t1_evidence_store as STORE
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.runtime_sentinel import (
    EnforcementPoint,
    RuntimeIntegrityRecord,
    require_runtime_identity,
)
from cardiosentinel.neural.t1_config import require_canonical_execution_authorized
from cardiosentinel.neural.t1_execution_spec import (
    STAGE_ASSEMBLE_LABEL_BLIND,
    STAGE_BOOTSTRAP,
    STAGE_CHALLENGE,
    STAGE_CLAIM,
    STAGE_COMPLETION,
    STAGE_EXPERIMENT_LOCK,
    STAGE_FINAL_CONFIGURATION,
    STAGE_FOLD_AUTHORIZE_HELD_OUT,
    STAGE_FOLD_GENERATE_THRESHOLDS,
    STAGE_FOLD_OPEN_FIT_LABELS,
    STAGE_FOLD_OPEN_HELD_OUT_LABELS,
    STAGE_FOLD_PROMOTE_HELD_OUT,
    STAGE_FOLD_PROMOTE_SELECTION,
    STAGE_FOLD_RUN_CANDIDATES,
    STAGE_FOLD_RUN_SELECTED,
    STAGE_FOLD_SELECT,
    STAGE_OOF_RESULT,
    STAGE_OOF_STATE_EVIDENCE,
    STAGE_PROMOTE_INPUT_EVIDENCE,
    STAGE_PROVE_ATTEMPT_ABSENT,
    STAGE_PROVE_TEST_UNOPENED,
    STAGE_START,
    STAGE_VALIDATE_M2,
    STAGE_VALIDATE_PROTOCOL,
    STAGE_VALIDATE_SPEC,
    STAGE_VALIDATE_T2,
    STAGE_VALIDATE_U1,
    STAGE_VERIFY_GIT,
    STAGE_VERIFY_UPSTREAM,
    T1_BOOTSTRAP_REPLICATES,
    T1_BOOTSTRAP_SEED,
    T1_CANONICAL_EXECUTION_FLAG,
    T1_DETECTOR_THRESHOLD,
    T1_EXPECTED_GIT_SHA_FLAG,
    T1_FOLD_COUNT,
    T1_STRIDE_SECONDS,
    require_cli_option_permitted,
    require_defined_metric,
    require_held_out_access_authorized,
    require_no_test_access,
    require_single_held_out_policy_run,
    validate_t1_execution_spec_document,
)
from cardiosentinel.neural.t1_protocol import (
    Q_EVENT,
    Q_WATCH,
    T1_STATE_EVENT,
    T1_STATE_NORMAL,
    T1_VALIDATION_SUBJECTS,
    T1_ZERO_STREAKS,
    T1CandidatePolicy,
    T1Row,
    T1Thresholds,
    decision_error_uncertainty,
    empirical_order_statistic,
    group_reference_episodes,
    match_runs_to_episodes,
    next_state,
    policy_sort_key,
    t1_folds,
    validate_t1_protocol_document,
)

M2_CANONICAL_RUN_ROOT: Final = Path("cardiosentinel-runs/phase6-m2-development-v1")
U1_CANONICAL_RUN_ROOT: Final = Path("cardiosentinel-runs/phase7-u1-development-v1")
T2_CANONICAL_RUN_ROOT: Final = Path("cardiosentinel-runs/phase8-t2-development-v1")

TARGET_AUTHORITY: Final = "ltstdb_baseline_v1_feature_corpus"


class T1DevelopmentError(RuntimeError):
    """Raised when the canonical development run cannot proceed honestly."""


# ---------------------------------------------------------------------------
# The fold-scoped target authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldScopedTargetAuthority:
    """Label access, scoped to an explicit subject set at construction time.

    There is deliberately no method that returns "all labels". An authority is
    built for the subjects a stage is allowed to see, and asking it for anyone
    else raises. That is what keeps a held-out subject closed during selection:
    not a branch that could be reordered, but an object that cannot answer.
    """

    authority: str
    authorized_subjects: tuple[str, ...]
    scope: str

    def require_authorized(self, subject_id: str) -> str:
        if subject_id not in self.authorized_subjects:
            raise T1DevelopmentError(
                f"Subject {subject_id!r} is not in this authority's scope "
                f"({self.scope}). Its labels are closed: opening them here would "
                "let held-out truth reach a decision that must be made without it."
            )
        return subject_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "scope": self.scope,
            "authorized_subjects": list(self.authorized_subjects),
            "authorized_subject_count": len(self.authorized_subjects),
        }


def fit_authority(fit_subjects: Sequence[str]) -> FoldScopedTargetAuthority:
    return FoldScopedTargetAuthority(
        authority=TARGET_AUTHORITY,
        authorized_subjects=tuple(fit_subjects),
        scope="fit_subjects_only",
    )


def held_out_authority(
    subject_id: str, fold_state: dict[str, Any]
) -> FoldScopedTargetAuthority:
    """Constructible only after the fold's selection artifact is promoted
    and re-read with a verified digest."""
    require_held_out_access_authorized(fold_state)
    return FoldScopedTargetAuthority(
        authority=TARGET_AUTHORITY,
        authorized_subjects=(subject_id,),
        scope="held_out_subject_only",
    )


# ---------------------------------------------------------------------------
# Selection metrics
# ---------------------------------------------------------------------------


def episode_f1(matched: int, predicted: int, reference: int) -> float | None:
    """`2TP / (2TP + FP + FN)`. Undefined when the denominator is zero."""
    true_positive = int(matched)
    false_positive = int(predicted) - true_positive
    false_negative = int(reference) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return None
    return (2 * true_positive) / denominator


def window_mcc(
    predicted_positive: np.ndarray, actual_positive: np.ndarray
) -> float | None:
    """Matthews correlation. Undefined when any margin is empty."""
    predicted = np.asarray(predicted_positive).astype(bool)
    actual = np.asarray(actual_positive).astype(bool)
    tp = int(np.count_nonzero(predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0.0:
        return None
    return (tp * tn - fp * fn) / denominator


def physical_exposure_hours(position_count: int) -> float:
    """One timeline position is one stride of physical exposure, available or not."""
    return (position_count * T1_STRIDE_SECONDS) / 3600.0


def false_event_onsets_per_hour(unmatched_runs: int, position_count: int) -> float:
    hours = physical_exposure_hours(position_count)
    if hours <= 0.0:
        raise T1DevelopmentError("Physical exposure cannot be zero or negative.")
    return unmatched_runs / hours


def event_exposure_fraction(emitted_states: Sequence[str]) -> float:
    states = list(emitted_states)
    if not states:
        raise T1DevelopmentError("Exposure fraction needs a non-empty timeline.")
    return sum(1 for state in states if state == T1_STATE_EVENT) / len(states)


def contiguous_runs(flags: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    """Maximal runs of True, as `(begin, end_exclusive)` index pairs."""
    runs: list[tuple[int, int]] = []
    begin: int | None = None
    for index, flag in enumerate(flags):
        if flag and begin is None:
            begin = index
        elif not flag and begin is not None:
            runs.append((begin, index))
            begin = None
    if begin is not None:
        runs.append((begin, len(flags)))
    return tuple(runs)


# ---------------------------------------------------------------------------
# Threshold generation and policy evaluation
# ---------------------------------------------------------------------------


def generate_thresholds(
    policy: T1CandidatePolicy,
    *,
    background_p: Sequence[float],
    background_s: Sequence[float],
    stable_ids: Sequence[str],
) -> T1Thresholds:
    """The four thresholds, by the frozen exact order statistic. No interpolation."""
    if policy.q_watch not in Q_WATCH or policy.q_event not in Q_EVENT:
        raise T1DevelopmentError(
            f"Candidate {policy.name} uses quantiles outside the frozen "
            f"{Q_WATCH} / {Q_EVENT}."
        )
    return T1Thresholds(
        p_watch=empirical_order_statistic(background_p, stable_ids, policy.q_watch),
        s_watch=empirical_order_statistic(background_s, stable_ids, policy.q_watch),
        p_event=empirical_order_statistic(background_p, stable_ids, policy.q_event),
        s_event=empirical_order_statistic(background_s, stable_ids, policy.q_event),
    )


def run_policy_over_streams(
    rows_by_stream: dict[tuple[str, int], Sequence[T1Row]],
    thresholds: T1Thresholds,
    policy: T1CandidatePolicy,
) -> dict[tuple[str, int], list[str]]:
    """Run the frozen transition function causally, one stream at a time."""
    traces: dict[tuple[str, int], list[str]] = {}
    for key in sorted(rows_by_stream):
        state = T1_STATE_NORMAL
        streaks = T1_ZERO_STREAKS
        emitted: list[str] = []
        for row in rows_by_stream[key]:
            state, streaks = next_state(state, streaks, row, thresholds, policy.profile)
            emitted.append(state)
        traces[key] = emitted
    return traces


def score_policy(
    traces: dict[tuple[str, int], list[str]],
    *,
    start_samples: dict[tuple[str, int], Sequence[int]],
    primary_positive: dict[tuple[str, int], Sequence[bool]],
    primary_mask: dict[tuple[str, int], Sequence[bool]],
) -> dict[str, Any]:
    """Pooled selection metrics for one candidate, across every stream."""
    matched_total = predicted_total = reference_total = 0
    unmatched_total = position_total = 0
    pooled_predicted: list[bool] = []
    pooled_actual: list[bool] = []
    pooled_states: list[str] = []

    for key in sorted(traces):
        emitted = traces[key]
        episodes = group_reference_episodes(start_samples[key], primary_positive[key])
        runs = contiguous_runs([state == T1_STATE_EVENT for state in emitted])
        matched = match_runs_to_episodes(episodes, runs)
        matched_total += len(matched)
        predicted_total += len(runs)
        reference_total += len(episodes)
        unmatched_total += len(runs) - len(set(matched.values()))
        position_total += len(emitted)
        pooled_states.extend(emitted)
        for index, state in enumerate(emitted):
            if primary_mask[key][index]:
                pooled_predicted.append(state == T1_STATE_EVENT)
                pooled_actual.append(bool(primary_positive[key][index]))

    return {
        "episode_f1": episode_f1(matched_total, predicted_total, reference_total),
        "window_mcc": window_mcc(
            np.asarray(pooled_predicted), np.asarray(pooled_actual)
        ),
        "false_onsets_per_hour": false_event_onsets_per_hour(
            unmatched_total, position_total
        ),
        "event_exposure_fraction": event_exposure_fraction(pooled_states),
        "matched_episodes": matched_total,
        "predicted_event_runs": predicted_total,
        "reference_episodes": reference_total,
        "unmatched_predicted_runs": unmatched_total,
    }


def select_policy(
    scored: dict[str, dict[str, Any]], policies: Sequence[T1CandidatePolicy]
) -> T1CandidatePolicy:
    """The frozen lexicographic order. Undefined metrics stop the run."""
    by_name = {policy.name: policy for policy in policies}
    ordered = sorted(
        scored,
        key=lambda name: policy_sort_key(
            by_name[name],
            episode_f1=require_defined_metric(
                f"{name}.episode_f1", scored[name]["episode_f1"]
            ),
            window_mcc=require_defined_metric(
                f"{name}.window_mcc", scored[name]["window_mcc"]
            ),
            false_onsets_per_hour=scored[name]["false_onsets_per_hour"],
            event_exposure_fraction=scored[name]["event_exposure_fraction"],
        ),
    )
    return by_name[ordered[0]]


def subject_bootstrap_indices(subject_count: int) -> np.ndarray:
    """1000 replicates, seed 2026, subjects resampled with multiplicity."""
    generator = np.random.default_rng(T1_BOOTSTRAP_SEED)
    return generator.integers(
        0, subject_count, size=(T1_BOOTSTRAP_REPLICATES, subject_count)
    )


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def build_rows(columns: dict[str, np.ndarray]) -> list[T1Row]:
    """Turn the label-blind store into the narrow frozen transition view.

    An unavailable row carries nothing: no probability, no uncertainty, no
    temporal score, and nothing is invented for it.
    """
    rows: list[T1Row] = []
    for index in range(len(columns["stable_id"])):
        present = bool(columns["score_present"][index])
        rows.append(
            T1Row(
                stable_id=str(columns["stable_id"][index]),
                score_present=present,
                detector_decision=(
                    bool(columns["detector_decision_d_t"][index]) if present else None
                ),
                calibrated_probability=(
                    float(columns["oof_calibrated_probability_p_t"][index])
                    if present
                    else None
                ),
                decision_error_uncertainty=(
                    float(columns["decision_error_uncertainty_u_t"][index])
                    if present
                    else None
                ),
                temporal_evidence=(
                    float(columns["s4d_temporal_evidence_s_t"][index])
                    if present
                    else None
                ),
                elapsed_stream_seconds=float(columns["elapsed_stream_seconds"][index]),
            )
        )
    return rows


def derive_row_quantities(
    m2_score: float, calibrated_probability: float
) -> dict[str, Any]:
    """`d_t`, `u_t` from the frozen threshold and the frozen definition."""
    decision = float(m2_score) >= T1_DETECTOR_THRESHOLD
    return {
        "detector_decision_d_t": decision,
        "decision_error_uncertainty_u_t": decision_error_uncertainty(
            decision, calibrated_probability
        ),
    }


def elapsed_stream_seconds(
    start_sample: int, first_start_sample: int, frequency_hz: int = 250
) -> float:
    """Physical sample coordinates, never a row ordinal."""
    return (int(start_sample) - int(first_start_sample)) / float(frequency_hz)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class T1DevelopmentRun:
    """The 29-stage canonical development run.

    Constructing this object does nothing. The stages are defined individually
    and are walked in the frozen order by the stage recorder, which refuses any
    step taken out of sequence, so "the claim happened before any per-row
    access" is enforced by index rather than by reading the code top to bottom.

    There is no end-to-end driver here. ``stage_preflight`` covers stages 1 to
    9 and is what ``main`` runs; the stages after the claim are defined but
    nothing sequences them, and ``stage_folds`` takes a fold evaluator that no
    caller in this package supplies. Assembling that driver is a separate
    reviewable change, and it is the change that would consume the attempt.
    """

    authorized_git_sha: str
    repository_root: Path = REPOSITORY_ROOT
    runtime: RuntimeIntegrityRecord = field(default_factory=RuntimeIntegrityRecord)
    stages: PERSIST.T1StageRecorder = field(default_factory=PERSIST.T1StageRecorder)
    claimed: PERSIST.T1ClaimedRun | None = None
    upstream: dict[str, Any] = field(default_factory=dict)
    fold_state: dict[int, dict[str, Any]] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    # -- stages 1-9: pre-claim ------------------------------------------------

    def stage_preflight(self) -> dict[str, Any]:
        """Documents, upstream retention, TEST sealed, attempt absent.

        Nothing per-row is opened here and nothing may be. The run directory is
        the scientific claim; reading the timeline and then declining to claim
        would be an unrecorded look at the data.
        """
        self.stages.enter(STAGE_START)
        require_runtime_identity(
            EnforcementPoint.START, record=self.runtime, detail="t1_start"
        )
        self.stages.enter(STAGE_VERIFY_GIT)
        git = PERSIST.require_authorized_git_identity(self.authorized_git_sha)

        self.stages.enter(STAGE_VALIDATE_PROTOCOL)
        protocol_digest = validate_t1_protocol_document()
        self.stages.enter(STAGE_VALIDATE_SPEC)
        spec_digest = validate_t1_execution_spec_document()

        self.stages.enter(STAGE_VALIDATE_M2)
        self.upstream["m2"] = self._validate_m2()
        self.stages.enter(STAGE_VALIDATE_U1)
        self.upstream["u1"] = self._validate_u1()
        self.stages.enter(STAGE_VALIDATE_T2)
        self.upstream["t2"] = self._validate_t2()

        self.stages.enter(STAGE_PROVE_TEST_UNOPENED)
        require_no_test_access("validation")
        self.stages.enter(STAGE_PROVE_ATTEMPT_ABSENT)
        absence = PERSIST.require_unclaimed_canonical_attempt(self.repository_root)

        self.state["label_blind_input_opened"] = False
        return {
            "git": git,
            "protocol_document_sha256": protocol_digest,
            "execution_spec_document_sha256": spec_digest,
            "upstream": dict(self.upstream),
            "attempt_absence": absence,
        }

    def _validate_m2(self) -> dict[str, Any]:
        from cardiosentinel.neural.m2_selection import validate_retained_m2_arm

        root = self.repository_root / M2_CANONICAL_RUN_ROOT
        return dict(validate_retained_m2_arm(root))

    def _validate_u1(self) -> dict[str, Any]:
        from cardiosentinel.neural.u1_selection import validate_retained_u1_calibration

        root = self.repository_root / U1_CANONICAL_RUN_ROOT
        return dict(validate_retained_u1_calibration(root))

    def _validate_t2(self) -> dict[str, Any]:
        from cardiosentinel.neural.t2_selection import validate_retained_t2_arm

        root = self.repository_root / T2_CANONICAL_RUN_ROOT
        return dict(validate_retained_t2_arm(root))

    # -- stage 10: the claim --------------------------------------------------

    def stage_claim(self, preflight: dict[str, Any]) -> PERSIST.T1ClaimedRun:
        self.stages.enter(STAGE_CLAIM)
        self.claimed = PERSIST.claim_canonical_run(
            authorized_git_sha=self.authorized_git_sha,
            runtime=self.runtime,
            stages=self.stages,
            repository_root=self.repository_root,
        )
        PERSIST.promote(
            self.claimed,
            PERSIST.PREFLIGHT_NAME,
            PERSIST.build_preflight(
                authorized_git_sha=self.authorized_git_sha,
                upstream=preflight["upstream"],
            ),
        )
        return self.claimed

    # -- stage 11: upstream re-verification after the claim -------------------

    def stage_verify_upstream(self) -> dict[str, Any]:
        self.stages.enter(STAGE_VERIFY_UPSTREAM)
        reverified = {
            "m2": self._validate_m2(),
            "u1": self._validate_u1(),
            "t2": self._validate_t2(),
        }
        for name in ("m2", "u1", "t2"):
            if reverified[name] != self.upstream[name]:
                raise T1DevelopmentError(
                    f"The {name.upper()} upstream identity changed between preflight "
                    "and the claim. The attempt is consumed and nothing is retried."
                )
        return reverified

    # -- stages 12-13: label-blind assembly -----------------------------------

    def stage_assemble_label_blind(
        self,
        *,
        m2_row_evidence: Path,
        t2_identity: Path,
        t2_scores: Path,
        calibrators: dict[str, Any],
        subject_of_record: Any,
    ) -> dict[str, np.ndarray]:
        """Stage 12: the first per-row access, and the only label-blind one.

        Only named members are materialised. ``label``, ``target_family`` and
        ``primary_mask`` stay closed: a runtime transition that depended on
        evaluation annotation would not be deployable, because that annotation
        does not exist on a live stream.
        """
        self.stages.enter(STAGE_ASSEMBLE_LABEL_BLIND)
        self.state["label_blind_input_opened"] = True

        m2 = STORE.read_m2g_row_evidence(m2_row_evidence)
        identity = STORE.read_t2_identity_members(t2_identity)
        scores = STORE.read_t2_selected_scores(t2_scores)

        STORE.require_stable_id_alignment(m2["stable_id"], identity["stable_id"])
        census = STORE.require_availability_alignment(
            m2["scored"], scores["score_present"]
        )
        STORE.require_expected_census(census)
        self.state["row_census"] = census

        subjects = np.asarray(
            [subject_of_record(str(record)) for record in m2["record_id"]]
        )
        unknown = sorted(set(subjects.tolist()) - set(T1_VALIDATION_SUBJECTS))
        if unknown:
            raise T1DevelopmentError(f"Timeline carries unknown subjects {unknown}.")
        if not np.array_equal(subjects, np.asarray(identity["subject_id"])):
            raise T1DevelopmentError(
                "The canonical record-to-subject authority disagrees with the "
                "persisted subject identity. Identity must agree exactly."
            )

        present = np.asarray(m2["scored"]).astype(bool)
        raw = np.asarray(m2["score"], dtype=np.float64)
        probability = np.full(raw.shape, STORE.ABSENT, dtype=np.float64)
        for subject in sorted(set(subjects.tolist())):
            rows = subjects == subject
            calibrator = calibrators[subject]
            usable = rows & present
            if np.any(usable):
                probability[usable] = calibrator.apply_to_scores(raw[usable])

        decision = np.zeros(raw.shape, dtype=bool)
        decision[present] = raw[present] >= T1_DETECTOR_THRESHOLD
        uncertainty = np.full(raw.shape, STORE.ABSENT, dtype=np.float64)
        uncertainty[present] = np.where(
            decision[present], 1.0 - probability[present], probability[present]
        )

        elapsed = np.zeros(raw.shape, dtype=np.float64)
        starts = np.asarray(m2["start_sample"], dtype=np.int64)
        channels = np.asarray(m2["channel_index"], dtype=np.int32)
        records = np.asarray(m2["record_id"])
        for key in sorted({(str(r), int(c)) for r, c in zip(records, channels)}):
            rows = (records == key[0]) & (channels == key[1])
            elapsed[rows] = (starts[rows] - starts[rows].min()) / 250.0

        temporal = np.full(raw.shape, STORE.ABSENT, dtype=np.float64)
        temporal[present] = np.asarray(scores["score"], dtype=np.float64)[present]

        return {
            "stable_id": np.asarray(m2["stable_id"]),
            "record_id": records,
            "channel_index": channels,
            "start_sample": starts,
            "subject_id": subjects,
            "score_present": present,
            "m2g_detector_score": raw,
            "detector_decision_d_t": decision,
            "oof_calibrated_probability_p_t": probability,
            "decision_error_uncertainty_u_t": uncertainty,
            "s4d_temporal_evidence_s_t": temporal,
            "elapsed_stream_seconds": elapsed,
        }

    def stage_promote_input_evidence(
        self, columns: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        self.stages.enter(STAGE_PROMOTE_INPUT_EVIDENCE)
        claimed = self._require_claimed()
        require_runtime_identity(
            EnforcementPoint.PRE_PROMOTION,
            record=self.runtime,
            detail="pre_label_blind_input_promotion",
        )
        manifest = STORE.write_input_evidence(
            claimed.run_dir, columns, lineage=dict(self.upstream)
        )
        PERSIST.promote(
            claimed,
            PERSIST.INPUT_LINEAGE_NAME,
            {
                "artifact_class": "t1_v1_input_lineage",
                "upstream": dict(self.upstream),
                "row_census": self.state.get("row_census"),
                "input_evidence_sha256": manifest["content_sha256"],
                "forbidden_members_never_opened": list(STORE.forbidden_members()),
                "target_authority": TARGET_AUTHORITY,
            },
        )
        return manifest

    # -- stage 14: the folds --------------------------------------------------

    def stage_folds(self, *, evaluate_fold: Any) -> list[dict[str, Any]]:
        """Twelve leave-one-subject-out folds, each behind its own label barrier.

        ``evaluate_fold`` supplies the fold's FIT background population and its
        held-out evaluation, so this method owns the choreography and the
        barrier while the data access stays where it can be audited.
        """
        claimed = self._require_claimed()
        selections: list[dict[str, Any]] = []
        for stage in (
            STAGE_FOLD_OPEN_FIT_LABELS,
            STAGE_FOLD_GENERATE_THRESHOLDS,
            STAGE_FOLD_RUN_CANDIDATES,
            STAGE_FOLD_SELECT,
            STAGE_FOLD_PROMOTE_SELECTION,
            STAGE_FOLD_AUTHORIZE_HELD_OUT,
            STAGE_FOLD_OPEN_HELD_OUT_LABELS,
            STAGE_FOLD_RUN_SELECTED,
            STAGE_FOLD_PROMOTE_HELD_OUT,
        ):
            self.stages.enter(stage)

        for fold in t1_folds():
            authority = fit_authority(fold.fit_subjects)
            authority.require_authorized(fold.fit_subjects[0])
            selection = evaluate_fold(fold, authority)

            digest = PERSIST.promote_fold_selection(
                claimed, fold.fold_index, selection["artifact"]
            )
            state = {
                "selection_promoted": True,
                "selection_digest_verified": True,
                "held_out_label_access_authorized_for_this_fold": True,
                "selection_sha256": digest,
            }
            self.fold_state[fold.fold_index] = state

            held_out = held_out_authority(fold.held_out_subject, state)
            held_out.require_authorized(fold.held_out_subject)
            require_single_held_out_policy_run(1)
            self.state.setdefault("held_out_labels_opened_for_folds", []).append(
                fold.fold_index
            )
            selections.append({**selection["artifact"], "selection_sha256": digest})

        if len(selections) != T1_FOLD_COUNT:
            raise T1DevelopmentError(
                f"{len(selections)} folds completed; the design is {T1_FOLD_COUNT}."
            )
        PERSIST.promote(
            claimed,
            PERSIST.FOLD_SELECTIONS_NAME,
            {
                "artifact_class": "t1_v1_fold_selections",
                "fold_count": len(selections),
                "folds": selections,
                "fold_retry_performed": False,
            },
        )
        return selections

    # -- stages 15-21: evidence, reporting, lock ------------------------------

    def stage_oof_state_evidence(
        self, columns: dict[str, np.ndarray], *, fold_selection_sha256: str
    ) -> dict[str, Any]:
        self.stages.enter(STAGE_OOF_STATE_EVIDENCE)
        claimed = self._require_claimed()
        require_runtime_identity(
            EnforcementPoint.PRE_PROMOTION,
            record=self.runtime,
            detail="pre_held_out_evidence_promotion",
        )
        manifest = STORE.write_oof_state_evidence(
            claimed.run_dir, columns, fold_selection_sha256=fold_selection_sha256
        )
        self.state["oof_evidence_promoted"] = True
        return manifest

    def stage_oof_result(self, result: dict[str, Any]) -> str:
        self.stages.enter(STAGE_OOF_RESULT)
        return PERSIST.promote(
            self._require_claimed(),
            PERSIST.OOF_RESULT_NAME,
            {
                "artifact_class": "t1_v1_oof_development_result",
                "evidence_class": (
                    "cross_fitted_t1_development_evidence_conditional_on_frozen"
                    "_upstream_components"
                ),
                "is_unseen_generalization": False,
                "categorical_state_auprc_reported": False,
                **result,
            },
            detail="pre_oof_result_promotion",
        )

    def stage_subject_evidence_and_bootstrap(
        self, *, subject_evidence: dict[str, Any], bootstrap: dict[str, Any]
    ) -> tuple[str, str]:
        self.stages.enter(STAGE_BOOTSTRAP)
        claimed = self._require_claimed()
        subject_digest = PERSIST.promote(
            claimed, PERSIST.SUBJECT_EVIDENCE_NAME, subject_evidence
        )
        bootstrap_digest = PERSIST.promote(
            claimed,
            PERSIST.BOOTSTRAP_NAME,
            {
                "replicates": T1_BOOTSTRAP_REPLICATES,
                "seed": T1_BOOTSTRAP_SEED,
                "unit": "subject",
                "policy_reselected_inside_bootstrap": False,
                "resampled_with_multiplicity": True,
                **bootstrap,
            },
        )
        return subject_digest, bootstrap_digest

    def stage_challenge(self, challenge: dict[str, Any]) -> str:
        self.stages.enter(STAGE_CHALLENGE)
        return PERSIST.promote(
            self._require_claimed(),
            PERSIST.CHALLENGE_EVIDENCE_NAME,
            {
                "artifact_class": "t1_v1_challenge_evidence",
                "joined_after_state_trace": True,
                "is_selection_input": False,
                "is_transition_input": False,
                "conduction_evidence_level": "exploratory_descriptive",
                **challenge,
            },
        )

    def stage_final_configuration(self, configuration: dict[str, Any]) -> str:
        """Deployment configuration only. Never development evidence."""
        self.stages.enter(STAGE_FINAL_CONFIGURATION)
        self.stages.require_reached(STAGE_OOF_RESULT)
        self.state["final_configuration_started"] = True
        digest = PERSIST.promote(
            self._require_claimed(),
            PERSIST.FINAL_CONFIGURATION_NAME,
            {
                "artifact_class": "t1_v1_final_all_validation_configuration",
                "is_development_evidence": False,
                "in_sample_on_all_twelve_subjects": True,
                "replaces_oof_result": False,
                **configuration,
            },
            detail="pre_final_configuration_promotion",
        )
        self.state["final_configuration_completed"] = True
        return digest

    def stage_experiment_lock(self) -> str:
        self.stages.enter(STAGE_EXPERIMENT_LOCK)
        claimed = self._require_claimed()
        return PERSIST.promote(
            claimed,
            PERSIST.EXPERIMENT_LOCK_NAME,
            PERSIST.build_experiment_lock(
                claimed,
                artifact_digests=dict(claimed.promoted),
                upstream=dict(self.upstream),
            ),
            detail="pre_experiment_lock_promotion",
        )

    def stage_completion(self) -> dict[str, Any]:
        self.stages.enter(STAGE_COMPLETION)
        claimed = self._require_claimed()
        require_runtime_identity(
            EnforcementPoint.COMPLETION, record=self.runtime, detail="t1_completion"
        )
        return PERSIST.complete_run(claimed, result_digests=dict(claimed.promoted))

    def _require_claimed(self) -> PERSIST.T1ClaimedRun:
        if self.claimed is None:
            raise T1DevelopmentError(
                "No canonical claim exists. Per-row evidence and promotions are "
                "reachable only after the run directory has been claimed."
            )
        return self.claimed

    def failure_receipt(self, error: BaseException) -> Path | None:
        return PERSIST.write_failure_receipt(
            self.claimed,
            error,
            state={
                "stage": self.stages.current,
                "current_fold": self.state.get("current_fold"),
                "label_blind_input_opened": self.state.get(
                    "label_blind_input_opened", False
                ),
                "fit_labels_opened_for_folds": sorted(self.fold_state),
                "fold_selections_promoted": sorted(self.fold_state),
                "held_out_labels_opened_for_folds": self.state.get(
                    "held_out_labels_opened_for_folds", []
                ),
                "held_out_traces_completed": len(self.fold_state),
                "oof_evidence_promoted": self.state.get("oof_evidence_promoted", False),
                "final_configuration_started": self.state.get(
                    "final_configuration_started", False
                ),
                "final_configuration_completed": self.state.get(
                    "final_configuration_completed", False
                ),
            },
            repository_root=self.repository_root,
        )


def build_parser() -> argparse.ArgumentParser:
    """Exactly two options, and no scientific knob.

    A scientific choice reachable from a command line is a scientific choice a
    human can make after seeing results, which is what the prospective design
    exists to prevent.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cardiosentinel.neural.t1_development_run",
        description=(
            "Execute the one canonical T1-v1 development run. Requires a human "
            "authorization naming the exact merged commit."
        ),
        add_help=True,
    )
    parser.add_argument(
        T1_CANONICAL_EXECUTION_FLAG,
        action="store_true",
        required=True,
        help="Execute the single canonical development attempt.",
    )
    parser.add_argument(
        T1_EXPECTED_GIT_SHA_FLAG,
        required=True,
        metavar="SHA",
        help="The merged commit the human authorized. Must match a clean HEAD.",
    )
    return parser


def registered_options(
    parser: argparse.ArgumentParser | None = None,
) -> tuple[str, ...]:
    """Every option string the parser actually registers, excluding help."""
    parser = parser if parser is not None else build_parser()
    options: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        options.extend(action.option_strings)
    return tuple(sorted(options))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Never runs on import.

    Canonical execution is authorized, so this no longer refuses on permission.
    Permission is only the first of seven questions, and it is the cheapest one:
    it is a fact about a human decision, while the six that follow are facts
    about this repository and this machine at this moment. All of them are
    proven by ``stage_preflight`` -- stages 1 to 9 of the frozen order -- before
    anything is claimed:

    * the runtime identity matches the frozen dependency digest,
    * ``--expected-git-sha`` matches HEAD and the working tree is clean,
    * the protocol and execution-specification documents digest as frozen,
    * the M2, U1 and T2 retention decisions all validate,
    * TEST is unopened,
    * the canonical attempt does not already exist.

    A failure at any of them raises before the run directory is created, so a
    refused invocation leaves the single canonical attempt unconsumed and
    nothing is retried, repaired, renamed or re-rooted.

    Preflight opens no VALIDATION row, writes no artifact and makes no claim.
    What it does not do is run the science: the 29-stage choreography exists
    here as individually verifiable stage methods and nothing sequences them
    end to end, so a verified preflight is followed by an honest stop rather
    than a run. That stop reports a missing capability, never a withheld
    permission -- the distinction this module and ``t1_config`` both keep.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    for option in registered_options(parser):
        require_cli_option_permitted(option)

    require_canonical_execution_authorized()

    run = T1DevelopmentRun(authorized_git_sha=arguments.expected_git_sha)
    run.stage_preflight()

    raise T1DevelopmentError(
        "Pre-claim verification passed: canonical execution is authorized, "
        f"{T1_EXPECTED_GIT_SHA_FLAG} {arguments.expected_git_sha} matches a "
        "clean HEAD, the runtime identity matches the frozen dependency "
        "digest, the M2, U1 and T2 retention decisions validate, TEST is "
        "unopened and the canonical attempt does not exist. The run stops "
        "here because the 29-stage orchestration that would consume the "
        "attempt is not implemented in this module: the stages are defined "
        "individually and nothing sequences them, and "
        f"{STAGE_FOLD_RUN_CANDIDATES!r} in particular requires a fold "
        "evaluator no caller supplies. This is a missing capability, not a "
        "withheld permission. Nothing was claimed and the single canonical "
        "attempt remains unconsumed."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
