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


def execute_canonical_outer_validation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """The future one-shot outer-VALIDATION route. Currently refuses.

    The refusal is the first statement: no argument is inspected, no path is
    resolved and no VALIDATION array is opened before it fires.
    """
    require_outer_validation_authorized()
    raise T2EvaluationError(  # pragma: no cover - unreachable while unauthorized
        "Outer VALIDATION was authorized but its execution body is intentionally "
        "absent until the activation change set supplies it."
    )


def open_validation_timeline(*_args: Any, **_kwargs: Any) -> Any:
    """Would open the VALIDATION timeline. Refuses before touching the store."""
    require_outer_validation_authorized()
    raise T2EvaluationError(  # pragma: no cover - unreachable while unauthorized
        "VALIDATION timeline access is not authorized."
    )


def load_validation_labels(*_args: Any, **_kwargs: Any) -> Any:
    """Would read VALIDATION labels. Refuses first."""
    require_outer_validation_authorized()
    raise T2EvaluationError(  # pragma: no cover - unreachable while unauthorized
        "VALIDATION label access is not authorized."
    )


OUTER_VALIDATION_ENTRY_POINTS: Final = (
    execute_canonical_outer_validation,
    open_validation_timeline,
    load_validation_labels,
)


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


def challenge_family_evidence(
    families: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """False-positive behaviour per challenge family at the frozen threshold."""
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
            "trained_on": False,
        }
        if int(labels_array[selected].sum()) and family != "conduction_change":
            subsets[family]["label_positive_present"] = True
    return {
        "evidence_class": "t2_challenge_evidence",
        "is_selection_input": False,
        "merged_into_primary": False,
        "subsets": subsets,
    }
