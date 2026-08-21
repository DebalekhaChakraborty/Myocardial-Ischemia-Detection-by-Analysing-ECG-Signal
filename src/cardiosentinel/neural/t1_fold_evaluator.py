"""The canonical T1 fold evaluator: the scientific body, at last.

`T1NonExecutingFoldEvaluator` completed the collaborator graph and refused to
run. This module supplies the step it was standing in for: for one fold, it
generates the twelve candidate policies' thresholds from FIT-subject background
negatives, runs the frozen state machine under each, scores them by the frozen
selection metrics, picks one by the frozen lexicographic order, and -- after
the fold's selection artifact has been promoted and the barrier has opened --
runs that one policy once over the held-out subject and records the per-row
state trace the evidence store's schema requires.

**It computes no science of its own.** Every quantity comes from a frozen
component reached by composition: `t1_protocol.empirical_order_statistic` for
thresholds, `t1_protocol.next_state` for transitions, `t1_development_run`'s
`generate_thresholds`, `run_policy_over_streams`, `score_policy` and
`select_policy` for the sweep and the choice. There is no threshold literal, no
tuning knob, no seed and no alternative ordering in this file.

**It holds nothing.** No path, no frame, no target source, no cached subject
index. Everything arrives per call: the label-blind columns from stage 12, and
an already-scoped `FoldScopedEvaluationAuthority` built by the harness. That is
the shape `require_no_independent_access` demands, and it is why an evaluator
reference that escapes to an unauthorized caller is inert.

**How it decides which rows it may see.** Not by filtering the columns on
``subject_id`` -- that would be discovering membership for itself. The
authority's `targets_for_subject` returns one subject's ``stable_id`` tuple,
and those identifiers are the row set. A row the authority did not name is
unreachable, so the fold firewall is enforced by what the evaluator was handed
rather than by a predicate it applies to itself.

**Two phases, and the barrier between them is real.** `__call__` sees only the
eleven FIT subjects; it cannot reach the held-out one, because a FIT-scoped
authority refuses that subject. `evaluate_held_out` needs a HELD_OUT-scoped
authority, which `held_out_evaluation_authority` will only construct once the
fold's selection artifact is promoted and re-read with a verified digest. The
selected policy is therefore fixed in a promoted artifact before the held-out
subject's targets can be opened at all.

**What it still does not do.** It does not authorize execution, create a run
directory, consume the canonical attempt, reach TEST, refit U1, replay M2 or
T2, or alter any frozen configuration. Running it requires a caller that has
already claimed an authorized canonical attempt, and that caller does not
exist while `T1_EXECUTION_SPECIFICATION_AUTHORIZED` is False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np

from cardiosentinel.neural.t1_capability_gate import (
    T1CapabilityAttestation,
    attest,
)
from cardiosentinel.neural.t1_development_run import (
    build_rows,
    contiguous_runs,
    generate_thresholds,
    run_policy_over_streams,
    score_policy,
    select_policy,
)
from cardiosentinel.neural.t1_execution_spec import (
    T1_CANDIDATE_POLICIES_PER_FOLD,
    T1_HELD_OUT_POLICY_RUNS_PER_FOLD,
    T1_STRIDE_SECONDS,
    require_no_test_access,
)
from cardiosentinel.neural.t1_fold_authority import (
    SCOPE_FIT,
    SCOPE_HELD_OUT,
    FoldScopedEvaluationAuthority,
    T1SubjectTargets,
)
from cardiosentinel.neural.t1_protocol import (
    T1_STATE_EVENT,
    T1_STATE_NORMAL,
    T1_ZERO_STREAKS,
    T1Fold,
    T1Row,
    T1Thresholds,
    candidate_policies,
    group_reference_episodes,
    match_runs_to_episodes,
    next_state,
)

EVALUATOR_NAME: Final = "T1CanonicalFoldEvaluator"

# The eleven trace columns a held-out evaluation contributes to the OOF state
# evidence. The twelve label-blind columns come from stage 12 unchanged.
T1_HELD_OUT_TRACE_COLUMNS: Final = (
    "fold_index",
    "selected_policy_id",
    "p_watch",
    "s_watch",
    "p_event",
    "s_event",
    "emitted_state",
    "state_elapsed_seconds",
    "transition_from",
    "transition_to",
    "transition_occurred",
)

# The columns from stage 12 an evaluation reads. Named so a test can prove no
# label, mask or target family is among them.
T1_EVALUATOR_INPUT_COLUMNS: Final = (
    "stable_id",
    "record_id",
    "channel_index",
    "start_sample",
    "score_present",
    "detector_decision_d_t",
    "oof_calibrated_probability_p_t",
    "decision_error_uncertainty_u_t",
    "s4d_temporal_evidence_s_t",
    "elapsed_stream_seconds",
)

# The background population the frozen threshold rule names, spelled out so a
# reader can check it against protocol §7 without leaving this file:
# FIT subjects only, PRIMARY-eligible rows only, label-negative only, scored.
T1_THRESHOLD_POPULATION: Final = "fit_subject_primary_background_negative"


class T1FoldEvaluatorError(RuntimeError):
    """Raised when a fold evaluation cannot proceed honestly."""


# ---------------------------------------------------------------------------
# One fold's row view, assembled from what the authority named
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StreamRows:
    """One `(record_id, channel_index)` stream, in causal order."""

    rows: tuple[T1Row, ...]
    positions: tuple[int, ...]
    start_samples: tuple[int, ...]
    primary_positive: tuple[bool, ...]
    primary_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _FoldView:
    """Every stream the authority authorized, and nothing else."""

    streams: dict[tuple[str, int], _StreamRows]
    subjects: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return sum(len(stream.rows) for stream in self.streams.values())


def _row_index(columns: Mapping[str, Any]) -> dict[str, int]:
    """Position of every stable id in the label-blind timeline.

    Built once per call and never cached on the evaluator: a persistent index
    would be a frame the evaluator owns, and it owns nothing.
    """
    identifiers = [str(value) for value in np.asarray(columns["stable_id"])]
    index = {stable_id: position for position, stable_id in enumerate(identifiers)}
    if len(index) != len(identifiers):
        raise T1FoldEvaluatorError(
            "The label-blind timeline repeats a stable_id, so a row could be "
            "attributed to two subjects. Identity must be unique."
        )
    return index


def _require_input_columns(columns: Mapping[str, Any]) -> None:
    missing = [name for name in T1_EVALUATOR_INPUT_COLUMNS if name not in columns]
    if missing:
        raise T1FoldEvaluatorError(
            f"The label-blind timeline is missing {missing}. A fold evaluation "
            "reads the stage 12 columns and assembles none of its own."
        )
    for forbidden in ("label", "primary_mask", "target_family", "primary_positive"):
        if forbidden in columns:
            raise T1FoldEvaluatorError(
                f"The label-blind timeline carries {forbidden!r}. Targets reach "
                "an evaluation only through an authority, never through the "
                "columns; a column that carried them would be a way around it."
            )


def _build_fold_view(
    columns: Mapping[str, Any],
    authority: FoldScopedEvaluationAuthority,
    subjects: Sequence[str],
) -> _FoldView:
    """Assemble the streams for exactly the subjects the authority authorizes.

    Membership is never inferred from the columns. Each subject's targets are
    requested from the authority, and the ``stable_id`` tuple it returns is the
    row set; anything not named there is unreachable from here.
    """
    _require_input_columns(columns)
    index = _row_index(columns)
    records = np.asarray(columns["record_id"])
    channels = np.asarray(columns["channel_index"])
    starts = np.asarray(columns["start_sample"], dtype=np.int64)
    rows = build_rows(dict(columns))

    grouped: dict[tuple[str, int], list[tuple[int, int, bool, bool]]] = {}
    for subject in sorted(subjects):
        targets = authority.targets_for_subject(subject)
        _require_targets_shape(targets, subject)
        for offset, stable_id in enumerate(targets.stable_id):
            position = index.get(str(stable_id))
            if position is None:
                raise T1FoldEvaluatorError(
                    f"The authority named row {stable_id!r} for {subject!r}, "
                    "which the label-blind timeline does not contain. The two "
                    "views of one fold must describe the same rows."
                )
            key = (str(records[position]), int(channels[position]))
            grouped.setdefault(key, []).append(
                (
                    int(starts[position]),
                    position,
                    bool(targets.primary_positive[offset]),
                    bool(targets.primary_mask[offset]),
                )
            )

    streams: dict[tuple[str, int], _StreamRows] = {}
    for key in sorted(grouped):
        # Causal order is physical order: sorted by start_sample, never by the
        # order the authority happened to return rows in.
        ordered = sorted(grouped[key], key=lambda item: item[0])
        streams[key] = _StreamRows(
            rows=tuple(rows[position] for _, position, _, _ in ordered),
            positions=tuple(position for _, position, _, _ in ordered),
            start_samples=tuple(start for start, _, _, _ in ordered),
            primary_positive=tuple(positive for _, _, positive, _ in ordered),
            primary_mask=tuple(mask for _, _, _, mask in ordered),
        )
    if not streams:
        raise T1FoldEvaluatorError(
            "The authority authorized no rows, so there is nothing to evaluate."
        )
    return _FoldView(streams=streams, subjects=tuple(sorted(subjects)))


def _require_targets_shape(targets: T1SubjectTargets, subject: str) -> None:
    if not isinstance(targets, T1SubjectTargets):  # pragma: no cover - authority checks
        raise T1FoldEvaluatorError(
            f"Targets for {subject!r} arrived as {type(targets).__name__}."
        )
    if len(targets) == 0:
        raise T1FoldEvaluatorError(f"Subject {subject!r} contributed no rows.")


# ---------------------------------------------------------------------------
# Threshold generation population (§7)
# ---------------------------------------------------------------------------


def background_negative_population(view: _FoldView) -> dict[str, Any]:
    """FIT-subject PRIMARY background negatives, and only those.

    Three conditions, all required by §7 and none of them optional: the row is
    PRIMARY-eligible, its reference label is negative, and it carries a score.
    Dropping any one of them would generate a threshold from a different
    population and quietly change the science.
    """
    values_p: list[float] = []
    values_s: list[float] = []
    identifiers: list[str] = []
    for key in sorted(view.streams):
        stream = view.streams[key]
        for offset, row in enumerate(stream.rows):
            if not stream.primary_mask[offset]:
                continue
            if stream.primary_positive[offset]:
                continue
            if not row.score_present:
                continue
            values_p.append(float(row.calibrated_probability))
            values_s.append(float(row.temporal_evidence))
            identifiers.append(row.stable_id)
    if not identifiers:
        raise T1FoldEvaluatorError(
            "This fold has no FIT-subject PRIMARY background negatives, so the "
            "frozen order statistic has no population. A threshold cannot be "
            "generated and nothing is substituted for one."
        )
    return {
        "population": T1_THRESHOLD_POPULATION,
        "background_p": tuple(values_p),
        "background_s": tuple(values_s),
        "stable_ids": tuple(identifiers),
        "row_count": len(identifiers),
    }


# ---------------------------------------------------------------------------
# The per-row trace the evidence store schema requires
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TracedStream:
    """One stream's emitted states with the per-row detail the store needs."""

    emitted: tuple[str, ...]
    state_elapsed_seconds: tuple[float, ...]
    transition_from: tuple[str, ...]
    transition_to: tuple[str, ...]
    transition_occurred: tuple[bool, ...]


def trace_stream(
    rows: Sequence[T1Row], thresholds: T1Thresholds, profile: Any
) -> _TracedStream:
    """Step one stream, recording what changed at each row.

    A strict refinement of `run_policy_over_streams`: the same primitive, the
    same order, the same arguments, and additionally the state-before, the
    state-after and the time held in the current state. A test asserts the
    emitted sequences are identical, so this records more without deciding
    anything differently.

    ``state_elapsed_seconds`` counts one stride per timeline position, whether
    or not that position carried a score: time passes and the state is carried
    across an unavailable row, so excluding it would understate the exposure.
    """
    state = T1_STATE_NORMAL
    streaks = T1_ZERO_STREAKS
    elapsed = 0.0
    emitted: list[str] = []
    held: list[float] = []
    came_from: list[str] = []
    went_to: list[str] = []
    changed: list[bool] = []
    for row in rows:
        before = state
        state, streaks = next_state(before, streaks, row, thresholds, profile)
        occurred = state != before
        elapsed = 0.0 if occurred else elapsed + T1_STRIDE_SECONDS
        emitted.append(state)
        held.append(elapsed)
        came_from.append(before)
        went_to.append(state)
        changed.append(occurred)
    return _TracedStream(
        emitted=tuple(emitted),
        state_elapsed_seconds=tuple(held),
        transition_from=tuple(came_from),
        transition_to=tuple(went_to),
        transition_occurred=tuple(changed),
    )


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T1CanonicalFoldEvaluator:
    """The one scientific fold evaluator. Stateless, deterministic, complete.

    Frozen and slotted with no fields at all: there is deliberately nowhere to
    put a path, a frame, a source or a cached index, so the class cannot grow a
    route around the authorities by accident.
    """

    def __call__(
        self,
        fold: T1Fold,
        authority: FoldScopedEvaluationAuthority,
        columns: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Selection: twelve candidates on the eleven FIT subjects, one winner.

        The held-out subject is unreachable here. `_build_fold_view` asks the
        authority for each FIT subject in turn, and a FIT-scoped authority
        refuses the held-out one, so the barrier is enforced by the object
        rather than by this function remembering to respect it.
        """
        self._require_scope(authority, SCOPE_FIT, fold)
        view = _build_fold_view(columns, authority, fold.fit_subjects)
        background = background_negative_population(view)

        rows_by_stream = {key: view.streams[key].rows for key in view.streams}
        start_samples = {key: view.streams[key].start_samples for key in view.streams}
        positives = {key: view.streams[key].primary_positive for key in view.streams}
        masks = {key: view.streams[key].primary_mask for key in view.streams}

        policies = candidate_policies()
        if len(policies) != T1_CANDIDATE_POLICIES_PER_FOLD:  # pragma: no cover
            raise T1FoldEvaluatorError(
                f"{len(policies)} candidates enumerated; the design is "
                f"{T1_CANDIDATE_POLICIES_PER_FOLD}."
            )

        thresholds_by_name: dict[str, T1Thresholds] = {}
        scored: dict[str, dict[str, Any]] = {}
        for policy in policies:
            thresholds = generate_thresholds(
                policy,
                background_p=background["background_p"],
                background_s=background["background_s"],
                stable_ids=background["stable_ids"],
            )
            traces = run_policy_over_streams(rows_by_stream, thresholds, policy)
            thresholds_by_name[policy.name] = thresholds
            scored[policy.name] = score_policy(
                traces,
                start_samples=start_samples,
                primary_positive=positives,
                primary_mask=masks,
            )

        selected = select_policy(scored, policies)
        chosen = thresholds_by_name[selected.name]
        return {
            "artifact": {
                "artifact_class": "t1_v1_fold_selection",
                "fold_index": int(fold.fold_index),
                "held_out_subject": fold.held_out_subject,
                "fit_subject_count": len(fold.fit_subjects),
                "evaluator": EVALUATOR_NAME,
                "selected_policy_id": selected.name,
                "q_watch": float(selected.q_watch),
                "q_event": float(selected.q_event),
                "persistence_profile": selected.profile.name,
                "p_watch": float(chosen.p_watch),
                "s_watch": float(chosen.s_watch),
                "p_event": float(chosen.p_event),
                "s_event": float(chosen.s_event),
                "threshold_population": background["population"],
                "threshold_population_row_count": background["row_count"],
                "candidate_count": len(policies),
                "candidate_metrics": {
                    name: dict(sorted(metrics.items()))
                    for name, metrics in sorted(scored.items())
                },
                "selection_scope": authority.as_dict(),
                "held_out_labels_opened": False,
                "test_accessed": False,
            }
        }

    def evaluate_held_out(
        self,
        fold: T1Fold,
        authority: FoldScopedEvaluationAuthority,
        columns: Mapping[str, Any],
        selection: Mapping[str, Any],
    ) -> dict[str, Any]:
        """One policy, one subject, one run, after the barrier.

        The policy is read out of the promoted selection artifact rather than
        recomputed: recomputing it here would make the promoted artifact a
        description of the decision instead of the decision itself, and the two
        could disagree.
        """
        self._require_scope(authority, SCOPE_HELD_OUT, fold)
        thresholds, profile, policy_id = _policy_from_selection(selection)
        view = _build_fold_view(columns, authority, (fold.held_out_subject,))

        trace_columns: dict[str, list[Any]] = {
            name: [] for name in T1_HELD_OUT_TRACE_COLUMNS
        }
        positions: list[int] = []
        matched_total = predicted_total = reference_total = 0
        unmatched_total = 0
        onset_latency: list[float] = []
        confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        for key in sorted(view.streams):
            stream = view.streams[key]
            traced = trace_stream(stream.rows, thresholds, profile)
            episodes = group_reference_episodes(
                stream.start_samples, stream.primary_positive
            )
            runs = contiguous_runs(
                [state == T1_STATE_EVENT for state in traced.emitted]
            )
            matched = match_runs_to_episodes(episodes, runs)
            matched_total += len(matched)
            predicted_total += len(runs)
            reference_total += len(episodes)
            unmatched_total += len(runs) - len(set(matched.values()))
            onset_latency.extend(
                _onset_latency(episodes, runs, matched, stream.start_samples)
            )

            positions.extend(stream.positions)
            for offset in range(len(stream.rows)):
                trace_columns["fold_index"].append(int(fold.fold_index))
                trace_columns["selected_policy_id"].append(policy_id)
                trace_columns["p_watch"].append(float(thresholds.p_watch))
                trace_columns["s_watch"].append(float(thresholds.s_watch))
                trace_columns["p_event"].append(float(thresholds.p_event))
                trace_columns["s_event"].append(float(thresholds.s_event))
                trace_columns["emitted_state"].append(traced.emitted[offset])
                trace_columns["state_elapsed_seconds"].append(
                    float(traced.state_elapsed_seconds[offset])
                )
                trace_columns["transition_from"].append(traced.transition_from[offset])
                trace_columns["transition_to"].append(traced.transition_to[offset])
                trace_columns["transition_occurred"].append(
                    bool(traced.transition_occurred[offset])
                )
                if stream.primary_mask[offset]:
                    predicted_positive = traced.emitted[offset] == T1_STATE_EVENT
                    actual = bool(stream.primary_positive[offset])
                    key_name = (
                        "tp"
                        if predicted_positive and actual
                        else "fp"
                        if predicted_positive
                        else "fn"
                        if actual
                        else "tn"
                    )
                    confusion[key_name] += 1

        return {
            "fold_index": int(fold.fold_index),
            "held_out_subject": fold.held_out_subject,
            "selected_policy_id": policy_id,
            "policy_runs": T1_HELD_OUT_POLICY_RUNS_PER_FOLD,
            "row_positions": tuple(positions),
            "trace_columns": {
                name: tuple(values) for name, values in trace_columns.items()
            },
            "episode_evidence": {
                "reference_episodes": reference_total,
                "predicted_event_runs": predicted_total,
                "matched_episodes": matched_total,
                "unmatched_predicted_runs": unmatched_total,
            },
            "onset_latency_seconds": tuple(onset_latency),
            "primary_confusion": dict(confusion),
            "evaluation_scope": authority.as_dict(),
            "test_accessed": False,
        }

    def t1_execution_capability(self) -> T1CapabilityAttestation:
        """Declare, to the pre-claim gate, that this can finish a run.

        Every branch below the entry points returns a result or raises on a
        genuinely impossible input; nothing here refuses because a body is
        missing. That is the whole difference between this object and the one
        it replaces.
        """
        return attest(
            "evaluate_fold",
            provider=type(self).__name__,
            executes=True,
            reason=(
                "Generates the twelve candidates' thresholds from FIT-subject "
                "background negatives, runs the frozen transition under each, "
                "selects by the frozen lexicographic order, and traces the "
                "selected policy once over the held-out subject."
            ),
        )

    @staticmethod
    def _require_scope(
        authority: FoldScopedEvaluationAuthority, scope: str, fold: T1Fold
    ) -> None:
        if not isinstance(authority, FoldScopedEvaluationAuthority):
            raise T1FoldEvaluatorError(
                "A fold evaluation takes a FoldScopedEvaluationAuthority, not "
                f"{type(authority).__name__}. A looser type is how a frame gets "
                "in through a scoped door."
            )
        if authority.scope != scope:
            raise T1FoldEvaluatorError(
                f"This phase needs a {scope!r} authority and was handed "
                f"{authority.scope!r}. The two scopes exist so selection cannot "
                "see the held-out subject and evaluation cannot re-select."
            )
        if authority.fold_index != fold.fold_index:
            raise T1FoldEvaluatorError(
                f"The authority scopes fold {authority.fold_index} and the fold "
                f"is {fold.fold_index}. One evaluation is one fold."
            )
        require_no_test_access(authority.partition)


def _policy_from_selection(
    selection: Mapping[str, Any],
) -> tuple[T1Thresholds, Any, str]:
    """Recover the promoted policy. Refuses anything it cannot fully identify."""
    required = (
        "selected_policy_id",
        "p_watch",
        "s_watch",
        "p_event",
        "s_event",
        "persistence_profile",
    )
    missing = [name for name in required if name not in selection]
    if missing:
        raise T1FoldEvaluatorError(
            f"The promoted fold selection is missing {missing}, so the held-out "
            "run cannot name the policy it is running."
        )
    policy_id = str(selection["selected_policy_id"])
    named = {policy.name: policy for policy in candidate_policies()}
    if policy_id not in named:
        raise T1FoldEvaluatorError(
            f"{policy_id!r} is not one of the twelve frozen candidates."
        )
    policy = named[policy_id]
    if policy.profile.name != str(selection["persistence_profile"]):
        raise T1FoldEvaluatorError(
            f"The selection names profile {selection['persistence_profile']!r} "
            f"for candidate {policy_id!r}, whose profile is {policy.profile.name!r}."
        )
    return (
        T1Thresholds(
            p_watch=float(selection["p_watch"]),
            s_watch=float(selection["s_watch"]),
            p_event=float(selection["p_event"]),
            s_event=float(selection["s_event"]),
        ),
        policy.profile,
        policy_id,
    )


def _onset_latency(
    episodes: Sequence[tuple[int, int]],
    runs: Sequence[tuple[int, int]],
    matched: Mapping[int, int],
    start_samples: Sequence[int],
) -> list[float]:
    """Seconds from each matched episode's onset to its run's onset.

    Measured in physical sample coordinates, never in row ordinals, and only
    for episodes a run actually matched: an undetected episode has no latency,
    and recording one as zero would read as an instant detection.
    """
    ordered = sorted(range(len(episodes)), key=lambda i: episodes[i][0])
    latencies: list[float] = []
    for position, episode_index in enumerate(ordered):
        run_index = matched.get(position)
        if run_index is None:
            continue
        episode_begin = episodes[episode_index][0]
        run_begin = runs[run_index][0]
        latencies.append(
            (int(start_samples[run_begin]) - int(start_samples[episode_begin])) / 250.0
        )
    return latencies


def evaluator_capability() -> dict[str, Any]:
    """What this layer provides, as data a receipt can carry."""
    return {
        "evaluator": EVALUATOR_NAME,
        "execution_enabled": True,
        "candidate_policies_per_fold": T1_CANDIDATE_POLICIES_PER_FOLD,
        "held_out_policy_runs_per_fold": T1_HELD_OUT_POLICY_RUNS_PER_FOLD,
        "threshold_population": T1_THRESHOLD_POPULATION,
        "trace_columns": list(T1_HELD_OUT_TRACE_COLUMNS),
        "reads_datasets_independently": False,
        "reads_labels_independently": False,
        "holds_a_target_source": False,
        "test_accessed": False,
    }


def widen_with_held_out_traces(
    columns: Mapping[str, Any], held_out_traces: Mapping[int, Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    """Place each fold's held-out trace at the rows it was produced for.

    Every row in the label-blind timeline belongs to exactly one subject and is
    that subject's held-out row in exactly one fold, so the twelve traces tile
    the timeline without overlap. Both halves of that sentence are checked: a
    row written twice and a row never written are each refused, because either
    would mean the cross-fitted evidence was not actually cross-fitted.

    This widens; it does not compute. The trace values come from the held-out
    evaluations unchanged, and the twelve label-blind columns pass through
    untouched.
    """
    row_count = len(np.asarray(columns["stable_id"]))
    written = np.zeros(row_count, dtype=np.int32)
    staged: dict[str, list[Any]] = {
        name: [None] * row_count for name in T1_HELD_OUT_TRACE_COLUMNS
    }

    for fold_index in sorted(held_out_traces):
        trace = held_out_traces[fold_index]
        positions = tuple(trace["row_positions"])
        values = trace["trace_columns"]
        for name in T1_HELD_OUT_TRACE_COLUMNS:
            if len(values[name]) != len(positions):
                raise T1FoldEvaluatorError(
                    f"Fold {fold_index} trace column {name!r} has "
                    f"{len(values[name])} values for {len(positions)} rows."
                )
        for offset, position in enumerate(positions):
            if not 0 <= position < row_count:
                raise T1FoldEvaluatorError(
                    f"Fold {fold_index} names row {position}, outside the "
                    f"{row_count}-row timeline."
                )
            written[position] += 1
            for name in T1_HELD_OUT_TRACE_COLUMNS:
                staged[name][position] = values[name][offset]

    twice = int(np.count_nonzero(written > 1))
    never = int(np.count_nonzero(written == 0))
    if twice or never:
        raise T1FoldEvaluatorError(
            f"The held-out traces do not tile the timeline: {twice} rows were "
            f"evaluated more than once and {never} were never held out. Every "
            "row is held out exactly once."
        )

    widened: dict[str, np.ndarray] = {
        name: np.asarray(columns[name]) for name in columns
    }
    for name in T1_HELD_OUT_TRACE_COLUMNS:
        widened[name] = np.asarray(staged[name])
    return widened
