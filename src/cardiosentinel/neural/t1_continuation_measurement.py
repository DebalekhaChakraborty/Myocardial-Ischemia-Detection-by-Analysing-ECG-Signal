"""The measurement continuation: persisted predictions + held-out labels.

    frozen predictions + held-out labels -> measurement

This module recovers the three quantities the consumed attempt computed and lost
-- per-fold PRIMARY confusion counts, episode evidence and onset latencies --
without regenerating anything. It reads `emitted_state` from the promoted OOF
state store, joins the held-out labels for one fold at a time, and groups and
matches episodes through the **frozen protocol functions the consumed attempt
itself used**. The continuation introduces no new science; it supplies one new
input to code that is byte-identical to what ran before.

**What is deliberately not imported.** `contiguous_runs` lives in
`t1_development_run` and `_onset_latency` in `t1_fold_evaluator`, and both of
those modules bind the state machine and the fold evaluator. Importing either to
borrow a nine-line helper would hand the continuation the entire forbidden
surface, which is exactly what amendment §13.6 Layer 1 exists to prevent. So both
are re-implemented here, and `test_t1_continuation_measurement` proves the
re-implementations agree with the originals exactly, on the real trace and on
randomised inputs. That converts duplicated science into proven-identical
science, and leaves two digest-pinned files untouched.

**Layer 3 of the §13.6 proof lives here.** The consumed trace is verified by
array digest before a single row is used, and every fold's thresholds and
selected policy id are compared with that fold's promoted selection artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from cardiosentinel.neural.t1_continuation_gate import (
    ContinuationCounters,
    T1ContinuationCapabilityError,
)
from cardiosentinel.neural.t1_continuation_spec import (
    PREDECESSOR_FOLD_SELECTIONS,
    PREDECESSOR_OOF_ARRAY_SHA256,
    fold_selection_relative_path,
)
from cardiosentinel.neural.t1_evidence_store import read_store
from cardiosentinel.neural.t1_protocol import (
    group_reference_episodes,
    match_runs_to_episodes,
)

OOF_MANIFEST_NAME: Final = "T1_OOF_STATE_EVIDENCE.json"

#: The state whose contiguous runs are the predicted detections.
T1_STATE_EVENT_VALUE: Final = "EVENT"

#: Sampling rate used to convert onset sample offsets to seconds. Matches the
#: divisor the consumed attempt used in `t1_fold_evaluator._onset_latency`.
SAMPLES_PER_SECOND: Final = 250.0

#: Columns the continuation reads from the promoted store. Every one is
#: label-free by construction; the store attests `contains_label: false`.
CONSUMED_TRACE_COLUMNS: Final = (
    "stable_id",
    "record_id",
    "channel_index",
    "start_sample",
    "subject_id",
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

#: Thresholds compared per fold against the promoted selection (Layer 3).
THRESHOLD_COLUMNS: Final = ("p_watch", "s_watch", "p_event", "s_event")


class T1ContinuationMeasurementError(RuntimeError):
    """Raised when the consumed trace does not match its promoted selections."""


# ---------------------------------------------------------------------------
# Re-implemented helpers -- proven equivalent, never imported
# ---------------------------------------------------------------------------


def contiguous_runs(flags: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    """Maximal runs of True, as `(begin, end_exclusive)` index pairs.

    Behaviourally identical to `t1_development_run.contiguous_runs`, which the
    continuation may not import. Equivalence is asserted by test, not assumed.
    """
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


def onset_latency_seconds(
    episodes: Sequence[tuple[int, int]],
    runs: Sequence[tuple[int, int]],
    matched: Mapping[int, int],
    start_samples: Sequence[int],
) -> list[float]:
    """Seconds from each matched episode's onset to its run's onset.

    Measured in physical sample coordinates, never row ordinals, and only for
    episodes a run actually matched: an undetected episode has no latency, and
    recording one as zero would read as an instant detection.

    `matched` is keyed by position in onset order, matching what
    `match_runs_to_episodes` returns. Behaviourally identical to
    `t1_fold_evaluator._onset_latency`, which the continuation may not import.
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
            (int(start_samples[run_begin]) - int(start_samples[episode_begin]))
            / SAMPLES_PER_SECOND
        )
    return latencies


# ---------------------------------------------------------------------------
# The consumed trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumedTrace:
    """The predecessor's promoted state trace, digest-verified. Never regenerated."""

    columns: dict[str, np.ndarray]
    array_sha256: str
    content_sha256: str
    row_count: int

    def fold_mask(self, fold_index: int) -> np.ndarray:
        return self.columns["fold_index"] == fold_index


def consume_oof_state_trace(
    attempt_dir: Path, counters: ContinuationCounters | None = None
) -> ConsumedTrace:
    """Read the promoted OOF state trace. Reads only; regenerates nothing.

    `read_store` verifies the array against the digest the manifest promoted, so
    a trace that changed on disk is refused before a row is used. The result is
    the same trace the scientific claim already rests on -- there is no second
    trace that could differ from the first.
    """
    if counters is not None and counters.state_machine_invocations:
        raise T1ContinuationCapabilityError(
            "The state machine ran before the trace was consumed."
        )
    manifest = json.loads((attempt_dir / OOF_MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("contains_label") is not False:
        raise T1ContinuationMeasurementError(
            "The consumed state store does not attest contains_label: false."
        )
    columns = read_store(attempt_dir, OOF_MANIFEST_NAME, columns=CONSUMED_TRACE_COLUMNS)
    if manifest["array_sha256"] != PREDECESSOR_OOF_ARRAY_SHA256:
        raise T1ContinuationMeasurementError(
            f"Consumed trace array digest {manifest['array_sha256']} is not the "
            f"bound {PREDECESSOR_OOF_ARRAY_SHA256}."
        )
    return ConsumedTrace(
        columns=columns,
        array_sha256=manifest["array_sha256"],
        content_sha256=manifest["content_sha256"],
        row_count=int(manifest["row_count"]),
    )


def require_trace_matches_selections(
    trace: ConsumedTrace, attempt_dir: Path
) -> dict[int, dict[str, Any]]:
    """Layer 3: every fold's thresholds and policy are the promoted ones.

    Proves the continuation measures against the decision that was actually
    taken, rather than one it re-derived. A mismatch means the trace and the
    promoted artifact disagree about what ran, which is unrecoverable here.
    """
    verified: dict[int, dict[str, Any]] = {}
    mismatches: list[str] = []

    for fold_index, (subject, policy_id, _digest) in sorted(
        PREDECESSOR_FOLD_SELECTIONS.items()
    ):
        mask = trace.fold_mask(fold_index)
        if not mask.any():
            mismatches.append(f"fold {fold_index:02d}: no rows in the consumed trace")
            continue
        selection = json.loads(
            (attempt_dir / fold_selection_relative_path(fold_index)).read_text(
                encoding="utf-8"
            )
        )
        record: dict[str, Any] = {
            "fold_index": fold_index,
            "held_out_subject": subject,
            "row_count": int(mask.sum()),
        }

        observed_policy = np.unique(trace.columns["selected_policy_id"][mask])
        if len(observed_policy) != 1 or str(observed_policy[0]) != policy_id:
            mismatches.append(
                f"fold {fold_index:02d}: trace policy {observed_policy.tolist()} "
                f"!= promoted {policy_id!r}"
            )
        else:
            record["selected_policy_id"] = policy_id

        for column in THRESHOLD_COLUMNS:
            observed = np.unique(trace.columns[column][mask])
            promoted = float(selection[column])
            if len(observed) != 1 or float(observed[0]) != promoted:
                mismatches.append(
                    f"fold {fold_index:02d}: trace {column} {observed.tolist()} "
                    f"!= promoted {promoted}"
                )
            else:
                record[column] = promoted

        observed_subjects = np.unique(trace.columns["subject_id"][mask]).tolist()
        if observed_subjects != [subject]:
            mismatches.append(
                f"fold {fold_index:02d}: trace subjects {observed_subjects} "
                f"!= held-out {[subject]}"
            )
        verified[fold_index] = record

    if mismatches:
        raise T1ContinuationMeasurementError(
            "The consumed trace disagrees with the promoted fold selections. "
            "The continuation measures the decision that was taken; it does "
            "not re-derive one.\n  " + "\n  ".join(mismatches)
        )
    return verified


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldMeasurement:
    """One fold's recovered evidence. The three quantities that were lost."""

    fold_index: int
    held_out_subject: str
    selected_policy_id: str
    thresholds: dict[str, float]
    primary_confusion: dict[str, int]
    episode_evidence: dict[str, int]
    onset_latency_seconds: list[float]
    stream_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "held_out_subject": self.held_out_subject,
            "selected_policy_id": self.selected_policy_id,
            "thresholds": dict(sorted(self.thresholds.items())),
            "primary_confusion": dict(sorted(self.primary_confusion.items())),
            "episode_evidence": dict(sorted(self.episode_evidence.items())),
            "onset_latency_seconds": list(self.onset_latency_seconds),
            "stream_count": self.stream_count,
        }


def measure_fold(
    trace: ConsumedTrace,
    fold_index: int,
    held_out_labels: Mapping[str, Sequence[bool]],
    verified: Mapping[int, Mapping[str, Any]],
    counters: ContinuationCounters,
) -> FoldMeasurement:
    """Measure one fold from the persisted states and that fold's held-out labels.

    `held_out_labels` maps `stable_id` to `(primary_mask, primary_positive)`
    pairs keyed by the two names below. Labels arrive per fold, under the
    selection already promoted for that fold, and are never used to choose
    anything -- only to score states that were emitted before any label was
    opened.
    """
    subject, policy_id, _ = PREDECESSOR_FOLD_SELECTIONS[fold_index]
    mask = trace.fold_mask(fold_index)
    record = verified[fold_index]

    stable_ids = trace.columns["stable_id"][mask]
    records = trace.columns["record_id"][mask]
    channels = trace.columns["channel_index"][mask]
    starts = trace.columns["start_sample"][mask]
    states = trace.columns["emitted_state"][mask]

    primary_mask = np.asarray(
        [bool(held_out_labels["primary_mask"][str(sid)]) for sid in stable_ids]
    )
    primary_positive = np.asarray(
        [bool(held_out_labels["primary_positive"][str(sid)]) for sid in stable_ids]
    )

    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    matched_total = predicted_total = reference_total = unmatched_total = 0
    latencies: list[float] = []

    # Streams are grouped exactly as the consumed attempt grouped them, and
    # visited in sorted order so the result does not depend on dict ordering.
    keys = sorted({(str(r), int(c)) for r, c in zip(records, channels)})
    for record_id, channel_index in keys:
        in_stream = np.asarray(
            [
                str(r) == record_id and int(c) == channel_index
                for r, c in zip(records, channels)
            ]
        )
        order = np.argsort(starts[in_stream], kind="stable")
        stream_starts = starts[in_stream][order]
        stream_states = states[in_stream][order]
        stream_mask = primary_mask[in_stream][order]
        stream_positive = primary_positive[in_stream][order]

        episodes = group_reference_episodes(
            [int(s) for s in stream_starts],
            [bool(p) for p in stream_positive],
        )
        runs = contiguous_runs(
            [str(state) == T1_STATE_EVENT_VALUE for state in stream_states]
        )
        matched = match_runs_to_episodes(episodes, runs)

        matched_total += len(matched)
        predicted_total += len(runs)
        reference_total += len(episodes)
        unmatched_total += len(runs) - len(set(matched.values()))
        latencies.extend(
            onset_latency_seconds(
                episodes, runs, matched, [int(s) for s in stream_starts]
            )
        )

        for offset in range(len(stream_states)):
            if not stream_mask[offset]:
                continue
            predicted_positive = str(stream_states[offset]) == T1_STATE_EVENT_VALUE
            actual = bool(stream_positive[offset])
            if predicted_positive and actual:
                confusion["tp"] += 1
            elif predicted_positive and not actual:
                confusion["fp"] += 1
            elif not predicted_positive and actual:
                confusion["fn"] += 1
            else:
                confusion["tn"] += 1

    # Nothing above may have touched a forbidden entry point.
    counters.require_all_zero()

    return FoldMeasurement(
        fold_index=fold_index,
        held_out_subject=subject,
        selected_policy_id=policy_id,
        thresholds={column: float(record[column]) for column in THRESHOLD_COLUMNS},
        primary_confusion=confusion,
        episode_evidence={
            "reference_episodes": reference_total,
            "predicted_event_runs": predicted_total,
            "matched_episodes": matched_total,
            "unmatched_predicted_runs": unmatched_total,
        },
        onset_latency_seconds=latencies,
        stream_count=len(keys),
    )
