"""Future outer-VALIDATION evaluator and descriptive metrics -- execution refused.

Every entry point that would touch VALIDATION calls
`require_outer_validation_authorized()` **first**, before path resolution,
before the representation memmap and before any label read. The activation
constant lives in `t2_persistence`, is `False`, and has no setter, flag or
environment variable. This module exists so the semantics can be reviewed before
scientific exposure, not so they can be run.

The metric functions below are pure and are exercised synthetically. They
compute nothing about the real corpus in this change set.
"""

from __future__ import annotations

import json
from typing import Any, Final, Sequence

import numpy as np

from cardiosentinel.baseline.metrics import (
    binary_metrics,
    subject_bootstrap_confidence_intervals,
    subject_macro_metrics,
)
from cardiosentinel.neural.t2_persistence import (
    ARM_SELECTION_PENDING,
    require_outer_validation_authorized,
)
from cardiosentinel.neural.t2_protocol import (
    T2_ARMS,
    T2_BOOTSTRAP_CLAIM,
    T2_BOOTSTRAP_REPLICATES,
    T2_BOOTSTRAP_SEED,
    T2_CHALLENGE_FAMILIES,
    T2_COLD_START_STRATA,
    T2_POOLED_METRICS,
    T2_SUBJECT_MACRO_METRICS,
    T2_VALIDATION_PRIMARY_ROW_COUNT,
    T2_WINDOW_STRIDE_SECONDS,
    select_t2_arm,
)

OUTER_VALIDATION_RESULT_CLASS: Final = "t2_v1_outer_validation_result"


class T2EvaluationError(RuntimeError):
    """Raised when T2 evaluation semantics are violated."""


# ---------------------------------------------------------------------------
# The disabled canonical evaluator
# ---------------------------------------------------------------------------


def execute_canonical_outer_validation(
    run_dir: Any, *, validation_root: Any = None, corpus_manifest: Any = None
) -> dict[str, Any]:
    """The one-shot outer-VALIDATION route. Refuses while unauthorized.

    The refusal is the first statement: no argument is inspected, no path is
    resolved and no VALIDATION array is opened before it fires. Only once the
    activation state is `True` does it reach `_outer_validation_worker`, whose
    body is already written and reviewed here so a future activation change set
    changes the switch, not the science.
    """
    require_outer_validation_authorized()
    return _outer_validation_worker(  # pragma: no cover - gate is False
        run_dir, validation_root=validation_root, corpus_manifest=corpus_manifest
    )


def open_validation_timeline(*_args: Any, **kwargs: Any) -> Any:
    """Would open the VALIDATION timeline. Refuses before touching the store."""
    require_outer_validation_authorized()
    return _open_validation_timeline(  # pragma: no cover - gate is False
        kwargs.get("root")
    )


def load_validation_labels(*_args: Any, **kwargs: Any) -> Any:
    """Would read VALIDATION labels. Refuses first."""
    require_outer_validation_authorized()
    timeline = _open_validation_timeline(  # pragma: no cover - gate is False
        kwargs.get("root")
    )
    return _load_validation_targets(  # pragma: no cover - gate is False
        timeline, manifest_path=kwargs.get("corpus_manifest")
    )


OUTER_VALIDATION_ENTRY_POINTS: Final = (
    execute_canonical_outer_validation,
    open_validation_timeline,
    load_validation_labels,
)


# ---------------------------------------------------------------------------
# The execution body, written now so it can be reviewed before exposure
#
# Everything below the gate. A future activation change set flips
# `T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED` and updates the authorized merged
# SHA; it does not get to invent evaluation logic once the TRAIN numbers are
# known. These functions are private, are never called by the public entry
# points while the gate is False, and are exercised only against synthetic
# fixtures.
# ---------------------------------------------------------------------------


def _open_validation_timeline(root: Any = None) -> Any:
    """Open the VALIDATION timeline through the one byte-level verified route."""
    from cardiosentinel.neural.t2_timeline import T2Timeline

    return T2Timeline("validation", root=root)


def _load_validation_targets(timeline: Any, *, manifest_path: Any = None) -> Any:
    """Resolve VALIDATION rows to their persisted frozen target families."""
    from cardiosentinel.neural.t2_timeline import resolve_timeline_target_families

    return resolve_timeline_target_families(timeline, manifest_path=manifest_path)


def _outer_validation_worker(
    run_dir: Any,
    *,
    validation_root: Any = None,
    corpus_manifest: Any = None,
) -> dict[str, Any]:
    """One complete outer-VALIDATION attempt over both retained candidates.

    The frozen order, and the reasons each step is where it is:

    1. read and completely verify the TRAIN experiment lock and both checkpoint
       locks -- the evaluation is bound to promoted bytes, not to a directory;
    2. open the VALIDATION timeline through the same byte-level verified loader
       TRAIN used, and resolve its target authority the same way;
    3. for each arm in frozen order, load the checkpoint by digest and run
       **one** full causal pass per arm over the whole VALIDATION timeline;
    4. derive PRIMARY and CHALLENGE evidence from **that same pass** -- a
       separate challenge pass would give the challenge rows a different state
       history and would no longer describe the same model;
    5. score at the arm's own frozen internal-dev threshold, which VALIDATION
       may not alter;
    6. delegate arm selection to `t2_protocol.select_t2_arm`, which is the only
       place the rule lives.

    The unavailable exact-flat VALIDATION rows are state no-ops: they are
    skipped by the reader before the forward pass, so they receive no score and
    leave the carried state untouched.
    """
    from pathlib import Path

    from cardiosentinel.neural import t2_persistence as persistence
    from cardiosentinel.neural.m1_store import COLD_START_BIN_FILE
    from cardiosentinel.neural.t2_models import build_t2_model
    from cardiosentinel.neural.t2_timeline import FAMILY_NAME
    from cardiosentinel.neural.t2_training import T2TimelineReader, score_streams

    directory = Path(run_dir)
    verification = persistence.validate_canonical_t2_attempt(
        directory.parent, directory.name
    )
    training_lock = json.loads(
        (directory / persistence.EXPERIMENT_LOCK_NAME).read_text()
    )
    thresholds = dict(training_lock["internal_dev_thresholds"])

    timeline = _open_validation_timeline(validation_root)
    try:
        family_codes, target_identity = _load_validation_targets(
            timeline, manifest_path=corpus_manifest
        )
        reader = T2TimelineReader(timeline, family_codes)
        streams = timeline.streams()
        families = np.asarray(FAMILY_NAME, dtype="<U32")

        per_arm: dict[str, Any] = {}
        pooled_auprc_by_arm: dict[str, float] = {}
        macro_auprc_by_arm: dict[str, float] = {}
        bootstrap: dict[str, Any] = {}
        descriptors: dict[str, Any] = {}
        for arm in T2_ARMS:
            checkpoint_lock = persistence.read_checkpoint_lock(directory, arm)
            state = persistence.load_checkpoint(
                directory / persistence.CHECKPOINT_NAME[arm],
                expected_sha256=checkpoint_lock["checkpoint_sha256"],
            )
            model = build_t2_model(arm)
            model.load_state_dict(state, strict=True)
            model.eval()
            threshold = float(thresholds[arm]["threshold"])

            # ONE pass. PRIMARY and CHALLENGE evidence are both read out of it.
            scored = score_streams(model, reader, streams)
            row_families = families[family_codes[scored.positions]]
            primary = scored.direct_loss
            challenge = np.isin(row_families, np.asarray(T2_CHALLENGE_FAMILIES_RAW))

            per_arm[arm] = {
                "architecture": arm,
                "checkpoint_sha256": checkpoint_lock["checkpoint_sha256"],
                "internal_dev_threshold": threshold,
                "threshold_altered_by_outer_validation": False,
                "scored_row_count": int(scored.scores.size),
                "primary_row_count": int(primary.sum()),
                "single_causal_pass": True,
                "same_pass_supplies_primary_and_challenge": True,
                "unavailable_rows_scored": 0,
                "pooled": pooled_evidence(
                    scored.labels[primary].tolist(),
                    scored.scores[primary].tolist(),
                    threshold,
                ),
                "subject_macro": subject_macro_evidence(
                    scored.subjects[primary].tolist(),
                    scored.labels[primary].tolist(),
                    scored.scores[primary].tolist(),
                    threshold,
                ),
                "cold_start": cold_start_strata_evidence(
                    np.asarray(timeline.store.array(COLD_START_BIN_FILE))[
                        scored.positions
                    ][primary].tolist(),
                    scored.labels[primary].tolist(),
                    scored.scores[primary].tolist(),
                    threshold,
                ),
                "challenge": challenge_family_evidence(
                    _challenge_family_labels(row_families[challenge]),
                    scored.labels[challenge].tolist(),
                    scored.scores[challenge].tolist(),
                    threshold,
                ),
            }
            bootstrap[arm] = subject_bootstrap_evidence(
                scored.subjects[primary].tolist(),
                scored.labels[primary].tolist(),
                scored.scores[primary].tolist(),
                threshold,
            )
            descriptors[arm] = temporal_descriptors(
                (scored.scores[primary] >= threshold).astype(int).tolist(),
                labels=scored.labels[primary].tolist(),
            )
            pooled_auprc_by_arm[arm] = float(per_arm[arm]["pooled"]["auprc"])
            # `subject_macro_metrics` reports each metric as
            # `{value, contributing_subject_count, non_contributing_subject_count}`;
            # the selection rule consumes the mean, and the contributing counts
            # travel with it so a macro built from fewer subjects is visible.
            macro_auprc_by_arm[arm] = float(
                per_arm[arm]["subject_macro"]["auprc"]["value"]
            )

        decision = select_from_outer_validation(
            pooled_auprc=pooled_auprc_by_arm,
            subject_macro_auprc=macro_auprc_by_arm,
            parameter_counts={
                arm: int(training_lock["trainable_parameters"][arm]) for arm in T2_ARMS
            },
        )
        result = {
            "artifact_class": OUTER_VALIDATION_RESULT_CLASS,
            "training_experiment_lock_sha256": (
                training_lock["experiment_lock_sha256"]
            ),
            "training_attempt_verification": verification,
            "checkpoint_sha256": dict(training_lock["checkpoint_sha256"]),
            "checkpoint_lock_sha256": dict(training_lock["checkpoint_lock_sha256"]),
            "internal_dev_thresholds": thresholds,
            "t2_protocol_sha256": training_lock["t2_protocol_sha256"],
            "t2_execution_spec_sha256": training_lock["t2_execution_spec_sha256"],
            "validation_stream_cache_sha256": (
                timeline.manifest["stream_cache_sha256"]
            ),
            "validation_timeline_identity": timeline.identity(),
            "target_authority_identity": target_identity,
            "primary_population_identity": {
                "row_count": int(target_identity["primary_row_count"]),
                "ischemic_positive": int(target_identity["ischemic_positive"]),
                "background_negative": int(target_identity["background_negative"]),
            },
            "challenge_population_identity": {
                "row_count": int(target_identity["challenge_row_count"]),
                "merged_into_primary": False,
                **dict(CHALLENGE_CAUSAL_SEMANTICS),
            },
            "per_arm_evidence": per_arm,
            "subject_bootstrap": bootstrap,
            "temporal_descriptors": descriptors,
            "selection_decision": decision,
            "selected_arm": decision["selected_arm"],
            "latency_used_in_selection": False,
            "challenge_used_in_selection": False,
            "attempts_permitted": 1,
            "automatic_retry_performed": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        }
        return validate_outer_validation_result(result)
    finally:
        timeline.close()


# The frozen corpus family names behind the three challenge reporting families.
T2_CHALLENGE_FAMILIES_RAW: Final = (
    "rate_related_confounder",
    "axis_shift_confounder",
    "conduction_change_confounder",
)
_CHALLENGE_REPORTING_NAME: Final = dict(
    zip(T2_CHALLENGE_FAMILIES_RAW, T2_CHALLENGE_FAMILIES, strict=True)
)


def _challenge_family_labels(raw_families: np.ndarray) -> list[str]:
    """Map persisted corpus family names onto the frozen reporting names."""
    return [_CHALLENGE_REPORTING_NAME[str(value)] for value in raw_families]


# ---------------------------------------------------------------------------
# The future result schema, validated synthetically
# ---------------------------------------------------------------------------

REQUIRED_OUTER_VALIDATION_FIELDS: Final = (
    "artifact_class",
    "training_experiment_lock_sha256",
    "checkpoint_sha256",
    "internal_dev_thresholds",
    "t2_protocol_sha256",
    "t2_execution_spec_sha256",
    "validation_stream_cache_sha256",
    "primary_population_identity",
    "challenge_population_identity",
    "per_arm_evidence",
    "subject_bootstrap",
    "temporal_descriptors",
    "selection_decision",
    "selected_arm",
    "test_accessed",
    "sealed_test_state",
)


def validate_outer_validation_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate the future result's shape. No real values exist yet."""
    if result.get("artifact_class") != OUTER_VALIDATION_RESULT_CLASS:
        raise T2EvaluationError(
            f"Unknown outer-validation class {result.get('artifact_class')!r}."
        )
    missing = [name for name in REQUIRED_OUTER_VALIDATION_FIELDS if name not in result]
    if missing:
        raise T2EvaluationError(f"The outer-validation result is missing {missing}.")
    for arm in T2_ARMS:
        if arm not in result["per_arm_evidence"]:
            raise T2EvaluationError(f"No outer-validation evidence for {arm}.")
    if result.get("test_accessed") is not False:
        raise T2EvaluationError("An outer-validation result records TEST access.")
    if result.get("sealed_test_state") != "unopened":
        raise T2EvaluationError("The B4 sealed test must remain unopened.")
    primary = result["primary_population_identity"]
    if int(primary.get("row_count", -1)) != T2_VALIDATION_PRIMARY_ROW_COUNT:
        raise T2EvaluationError(
            f"The PRIMARY population must be {T2_VALIDATION_PRIMARY_ROW_COUNT} "
            f"rows; got {primary.get('row_count')}."
        )
    return result


def select_from_outer_validation(
    *,
    pooled_auprc: dict[str, float],
    subject_macro_auprc: dict[str, float],
    parameter_counts: dict[str, int],
) -> dict[str, Any]:
    """Delegate to the frozen protocol rule; never reimplement it here."""
    return select_t2_arm(
        pooled_auprc=pooled_auprc,
        subject_macro_auprc=subject_macro_auprc,
        parameter_counts=parameter_counts,
    )


def training_selection_status() -> dict[str, Any]:
    """After TRAIN-only execution both arms remain candidates."""
    return {
        "arm_selection_status": ARM_SELECTION_PENDING,
        "selected_arm": None,
        "candidates": list(T2_ARMS),
        "selection_requires": "one_shot_outer_validation",
    }


# ---------------------------------------------------------------------------
# Pooled / subject-macro / bootstrap evidence
# ---------------------------------------------------------------------------


def pooled_evidence(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    """The frozen pooled metric set at the arm's internal-dev threshold."""
    metrics = binary_metrics(labels, scores, threshold)
    return {name: metrics[name] for name in T2_POOLED_METRICS}


def _as_arrays(
    subjects: Sequence[str], labels: Sequence[int], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The baseline helpers take `(labels, scores, subjects)` as numpy arrays."""
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(subjects, dtype=np.str_),
    )


def subject_macro_evidence(
    subjects: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """Subject-macro metrics; the subject is the inferential unit, never a window."""
    label_array, score_array, subject_array = _as_arrays(subjects, labels, scores)
    macro = subject_macro_metrics(label_array, score_array, subject_array, threshold)
    return {name: macro.get(name) for name in T2_SUBJECT_MACRO_METRICS}


def subject_bootstrap_evidence(
    subjects: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """1000 subject resamples, seed 2026. Windows are never bootstrapped."""
    label_array, score_array, subject_array = _as_arrays(subjects, labels, scores)
    intervals = subject_bootstrap_confidence_intervals(
        label_array,
        score_array,
        subject_array,
        threshold,
        replicates=T2_BOOTSTRAP_REPLICATES,
        seed=T2_BOOTSTRAP_SEED,
    )
    return {
        "evidence_class": "t2_subject_bootstrap",
        "replicates": T2_BOOTSTRAP_REPLICATES,
        "seed": T2_BOOTSTRAP_SEED,
        "unit": "subject",
        "window_bootstrap_performed": False,
        "model_refitted_per_replicate": False,
        "claim_scope": T2_BOOTSTRAP_CLAIM,
        "intervals": intervals,
    }


# ---------------------------------------------------------------------------
# Temporal descriptive evidence -- descriptive only, never a selection input
# ---------------------------------------------------------------------------


def _runs(predictions: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous positive runs as `(start, stop)` half-open index pairs."""
    flags = np.asarray(predictions).astype(bool)
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def temporal_descriptors(
    predictions: Sequence[int],
    *,
    stride_seconds: float = T2_WINDOW_STRIDE_SECONDS,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """The five frozen descriptive temporal statistics.

    Descriptive only: these can never choose a checkpoint, choose an arm or
    alter a threshold, and nothing in this module lets them.

    **What `prediction_persistence_around_labelled_ischemic_intervals` is.** It
    is the fraction of labelled-positive *windows* that were predicted positive
    -- a window-level descriptive quantity conditional on the labelled-positive
    population, and nothing more. It is **not** an episode onset/offset
    persistence measurement: it says nothing about where in an episode a
    detection lands, how long after onset it arrives, or whether it survives to
    offset, because it never groups windows into episodes at all. Formal
    episode reasoning is T1's, and no episode metric is invented here.
    """
    flags = np.asarray(predictions).astype(bool)
    runs = _runs(flags)
    lengths = [stop - start for start, stop in runs]
    durations = sorted(length * float(stride_seconds) for length in lengths)
    isolated = sum(1 for length in lengths if length == 1)
    transitions = int(np.count_nonzero(np.diff(flags.astype(np.int8)) != 0))
    elapsed_hours = (flags.size * float(stride_seconds)) / 3600.0
    persistence = None
    if labels is not None:
        truth = np.asarray(labels).astype(bool)
        if truth.shape != flags.shape:
            raise T2EvaluationError("Labels and predictions must be aligned.")
        positive = int(truth.sum())
        persistence = None if positive == 0 else float(flags[truth].mean())
    return {
        "evidence_class": "t2_temporal_descriptors",
        "is_selection_input": False,
        "may_alter_threshold": False,
        "positive_prediction_run_count": len(runs),
        "median_positive_run_duration_seconds": (
            None if not durations else float(np.median(durations))
        ),
        "isolated_single_window_positive_fraction": (
            None if not runs else isolated / len(runs)
        ),
        "transition_count_per_hour": (
            None if elapsed_hours == 0 else transitions / elapsed_hours
        ),
        "prediction_persistence_around_labelled_ischemic_intervals": persistence,
        "prediction_persistence_definition": (
            "fraction_of_labelled_positive_windows_predicted_positive"
        ),
        "prediction_persistence_unit": "window",
        "prediction_persistence_conditioning_population": ("labelled_positive_windows"),
        "prediction_persistence_is_episode_onset_offset_measurement": False,
        "episode_grouping_performed": False,
        "formal_episode_reasoning_belongs_to": "t1",
    }


# ---------------------------------------------------------------------------
# Cold start and challenge reporting mechanics
# ---------------------------------------------------------------------------


def cold_start_strata_evidence(
    bins: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """Inherited strata, reported. No warmup threshold and no repair."""
    bins_array = np.asarray(bins)
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    strata: dict[str, Any] = {}
    for stratum in T2_COLD_START_STRATA:
        selected = bins_array == stratum
        count = int(selected.sum())
        if count == 0:
            strata[stratum] = {"row_count": 0, "metrics": None}
            continue
        strata[stratum] = {
            "row_count": count,
            "metrics": binary_metrics(
                labels_array[selected].tolist(),
                scores_array[selected].tolist(),
                threshold,
            ),
        }
    return {
        "evidence_class": "t2_cold_start_evidence",
        "warmup_threshold_applied": False,
        "cold_start_repair_applied": False,
        "alternative_state_initialization": False,
        "strata": strata,
    }


# The precise causal-context semantics. A broad `trained_on: false` would be
# FALSE: an AVAILABLE challenge `z_t` is label-blind causal context and can
# influence a later PRIMARY training loss through the carried state. What is
# true is narrower, and each clause below is separately true.
CHALLENGE_CAUSAL_SEMANTICS: Final = {
    "direct_training_loss_received": False,
    "challenge_identity_model_input": False,
    "challenge_label_model_input": False,
    "may_be_label_blind_causal_context": True,
    "checkpoint_selection_input": False,
    "arm_selection_input": False,
}


def challenge_family_evidence(
    families: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """False-positive behaviour per challenge family at the frozen threshold.

    Note what this does **not** say. There is deliberately no `trained_on`
    field: a challenge row receives no direct loss, but the model does consume
    its representation as causal context, so its `z_t` can move a later PRIMARY
    row's loss through the carried state. `CHALLENGE_CAUSAL_SEMANTICS` states
    the six things that are actually true instead of one thing that is not.
    """
    families_array = np.asarray(families)
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    subsets: dict[str, Any] = {}
    for family in T2_CHALLENGE_FAMILIES:
        selected = families_array == family
        count = int(selected.sum())
        predicted = scores_array[selected] >= threshold
        subsets[family] = {
            "row_count": count,
            "false_positive_count": int(predicted.sum()),
            "false_positive_rate": (None if count == 0 else float(predicted.mean())),
            "evidence_level": (
                "exploratory_descriptive"
                if family == "conduction_change"
                else "quantitative_secondary"
            ),
            "is_selection_input": False,
            "merged_into_primary": False,
            **dict(CHALLENGE_CAUSAL_SEMANTICS),
        }
        if int(labels_array[selected].sum()) and family != "conduction_change":
            subsets[family]["label_positive_present"] = True
    return {
        "evidence_class": "t2_challenge_evidence",
        "is_selection_input": False,
        "merged_into_primary": False,
        **dict(CHALLENGE_CAUSAL_SEMANTICS),
        "subsets": subsets,
    }
