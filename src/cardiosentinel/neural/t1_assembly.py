"""The T1 label-bearing assembly collaborators.

`T1ExecutionCollaborators` names seven callables the driver threads between
stages. This module supplies them. Each one is a **composition layer**: it
arranges values the frozen components already produce into the shape the
specification requires, and derives no science of its own.

**What "label-bearing" means here, and what it does not.** These collaborators
sit downstream of the fold barrier, so the values they arrange include held-out
outcomes. None of them opens a label: targets arrive only as arguments, from a
caller that obtained them through `FoldScopedEvaluationAuthority`. There is no
reader, no path and no archive in this module, and a test asserts all three.

**Nothing here runs.** Assembling requires per-fold traces that only a canonical
execution produces, and canonical execution is not authorized. Every
collaborator refuses an input it was not given honestly, and the refusals are
what this module's tests exercise.

Spec sections implemented, in order: §7 subject identity authority, §18 OOF
state-evidence store, §19 OOF development result, §21 subject evidence and
bootstrap, §22 challenge reporting, §23 final all-VALIDATION configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np

from cardiosentinel.data.ltstdb import subject_id_for_record
from cardiosentinel.neural.t1_development_run import (
    contiguous_runs,
    episode_f1,
    false_event_onsets_per_hour,
    physical_exposure_hours,
    subject_bootstrap_indices,
    window_mcc,
)
from cardiosentinel.neural.t1_execution_spec import (
    T1_BOOTSTRAP_REPLICATES,
    T1_BOOTSTRAP_RESAMPLES_WITH_MULTIPLICITY,
    T1_BOOTSTRAP_RESELECTS_POLICY,
    T1_BOOTSTRAP_SEED,
    T1_BOOTSTRAP_UNIT,
    T1_CATEGORICAL_STATE_AUPRC_REPORTED,
    T1_CHALLENGE_IS_SELECTION_INPUT,
    T1_CHALLENGE_IS_THRESHOLD_GENERATION_INPUT,
    T1_CHALLENGE_IS_TRANSITION_INPUT,
    T1_CHALLENGE_JOIN_AFTER_STATE_TRACE,
    T1_FINAL_CONFIGURATION_IS_DEVELOPMENT_EVIDENCE,
    T1_FINAL_CONFIGURATION_OVERWRITES_OOF_RESULT,
    T1_FOLD_COUNT,
    T1_OOF_STATE_EVIDENCE_COLUMNS,
    T1_STRIDE_SECONDS,
    T1_SUBJECT_IDENTITY_DERIVED_FROM_LABEL,
    T1_SUBJECT_IDENTITY_IS_TRANSITION_FEATURE,
    require_defined_metric,
    require_no_test_access,
)
from cardiosentinel.neural.t1_protocol import (
    T1_STATE_EVENT,
    T1_STATE_NORMAL,
    T1_STATE_RECOVERY,
    T1_STATE_WATCH,
    T1_STATES,
    T1_VALIDATION_SUBJECTS,
)

CHALLENGE_FAMILIES: Final = ("RATE", "AXIS", "CONDUCTION")
CHALLENGE_EVIDENCE_LEVEL: Final = {
    "RATE": "quantitative_secondary",
    "AXIS": "quantitative_secondary",
    "CONDUCTION": "exploratory_descriptive",
}

# The six flows §19 names. Any other ordered pair is not a reported flow.
REPORTED_STATE_FLOWS: Final = (
    (T1_STATE_NORMAL, T1_STATE_WATCH),
    (T1_STATE_WATCH, T1_STATE_EVENT),
    (T1_STATE_WATCH, T1_STATE_NORMAL),
    (T1_STATE_EVENT, T1_STATE_RECOVERY),
    (T1_STATE_RECOVERY, T1_STATE_EVENT),
    (T1_STATE_RECOVERY, T1_STATE_NORMAL),
)

FINAL_CONFIGURATION_FIELDS: Final = (
    "q_watch",
    "q_event",
    "p_watch",
    "s_watch",
    "p_event",
    "s_event",
    "persistence_profile",
)


class T1AssemblyError(RuntimeError):
    """Raised when an assembly input is missing, misordered or dishonest."""


# ---------------------------------------------------------------------------
# §7 Subject identity authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T1SubjectAuthority:
    """The canonical record-to-subject authority, and nothing more.

    Identity selects a state namespace and a calibrator. It is never a
    transition feature, never derived from a label, and never an input to a
    prediction -- the frozen constants say so and this object refuses to be
    used as anything else, because it returns only a namespace string.
    """

    def __call__(self, record_id: str) -> str:
        return self.subject_of_record(record_id)

    def subject_of_record(self, record_id: str) -> str:
        """Resolve one record to its subject, then prove roster membership."""
        name = str(record_id).strip()
        require_no_test_access(name)
        if not name:
            raise T1AssemblyError("A record identifier cannot be empty.")
        subject = subject_id_for_record(name)
        if subject not in T1_VALIDATION_SUBJECTS:
            raise T1AssemblyError(
                f"Record {record_id!r} resolves to {subject!r}, which is not in "
                "the frozen T1 validation roster. Identity is checked against "
                "the roster rather than trusted from the data."
            )
        return subject

    def require_agreement(self, record_id: str, persisted_subject: str) -> str:
        """§7: a T2-persisted identity must agree exactly. Disagreement STOPs."""
        canonical = self.subject_of_record(record_id)
        if str(persisted_subject) != canonical:
            raise T1AssemblyError(
                f"The canonical authority resolves {record_id!r} to "
                f"{canonical!r} but the persisted identity says "
                f"{persisted_subject!r}. Exact agreement is required; a "
                "disagreement is a hard stop, not a preference."
            )
        return canonical

    def as_dict(self) -> dict[str, Any]:
        return {
            "authority": "subject_id_for_record",
            "is_transition_feature": T1_SUBJECT_IDENTITY_IS_TRANSITION_FEATURE,
            "derived_from_label": T1_SUBJECT_IDENTITY_DERIVED_FROM_LABEL,
            "selects": ["state_namespace", "u1_loso_calibrator"],
            "roster_size": len(T1_VALIDATION_SUBJECTS),
        }


def subject_of_record() -> T1SubjectAuthority:
    """The `subject_of_record` collaborator."""
    return T1SubjectAuthority()


# ---------------------------------------------------------------------------
# Shared input discipline
# ---------------------------------------------------------------------------


def _require_columns(columns: Mapping[str, Any], required: Sequence[str]) -> None:
    if not isinstance(columns, Mapping):
        raise T1AssemblyError(
            f"Assembly takes a column mapping, not {type(columns).__name__}."
        )
    missing = sorted(set(required) - set(columns))
    if missing:
        raise T1AssemblyError(
            f"Assembly input is missing columns {missing}. A partial input "
            "would produce a well-formed artifact describing a run that did "
            "not happen."
        )


def _require_complete_folds(selections: Sequence[Mapping[str, Any]]) -> None:
    if len(selections) != T1_FOLD_COUNT:
        raise T1AssemblyError(
            f"{len(selections)} fold selections were supplied; the frozen "
            f"design is {T1_FOLD_COUNT}. Assembling from an incomplete set "
            "would report cross-fitted evidence that is not cross-fitted."
        )
    indices = sorted(int(s["fold_index"]) for s in selections)
    if indices != list(range(T1_FOLD_COUNT)):
        raise T1AssemblyError(
            f"Fold indices {indices} are not the frozen 0..{T1_FOLD_COUNT - 1}."
        )


def _ordered_subjects(columns: Mapping[str, Any]) -> tuple[str, ...]:
    """Deterministic subject order: frozen roster order, never observed order."""
    present = {str(s) for s in np.asarray(columns["subject_id"]).tolist()}
    unknown = sorted(present - set(T1_VALIDATION_SUBJECTS))
    if unknown:
        raise T1AssemblyError(f"Timeline carries unknown subjects {unknown}.")
    return tuple(s for s in T1_VALIDATION_SUBJECTS if s in present)


# ---------------------------------------------------------------------------
# §18 OOF state-evidence columns
# ---------------------------------------------------------------------------


def assemble_oof_state_columns(
    *, columns: Mapping[str, Any], selections: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    """Widen the label-blind input with each row's fold-scoped state trace.

    The evidence store's frozen column list is the schema, and this returns
    exactly it: the twelve label-blind input columns unchanged, plus the eleven
    trace columns the held-out evaluation produced. No forbidden column is
    added, and the store refuses one anyway.
    """
    _require_columns(columns, T1_OOF_STATE_EVIDENCE_COLUMNS)
    _require_complete_folds(selections)
    assembled = {
        name: np.asarray(columns[name]) for name in T1_OOF_STATE_EVIDENCE_COLUMNS
    }
    widths = {len(value) for value in assembled.values()}
    if len(widths) != 1:
        raise T1AssemblyError(
            f"OOF state columns are ragged: observed row counts {sorted(widths)}."
        )
    folds = np.asarray(assembled["fold_index"])
    seen = sorted({int(v) for v in folds.tolist()})
    if seen != list(range(T1_FOLD_COUNT)):
        raise T1AssemblyError(
            f"The state trace covers folds {seen}, not the frozen "
            f"0..{T1_FOLD_COUNT - 1}. Every row is held out exactly once."
        )
    states = {str(s) for s in np.asarray(assembled["emitted_state"]).tolist()}
    unknown = sorted(states - set(T1_STATES))
    if unknown:
        raise T1AssemblyError(f"The trace emits non-states {unknown}.")
    return assembled


# ---------------------------------------------------------------------------
# §19 OOF development result
# ---------------------------------------------------------------------------


def _state_burden(emitted: Sequence[str]) -> dict[str, float]:
    total = len(emitted)
    if total == 0:
        raise T1AssemblyError("State burden needs a non-empty trace.")
    return {state: sum(1 for s in emitted if s == state) / total for state in T1_STATES}


def _state_flows(emitted: Sequence[str]) -> dict[str, int]:
    counts = {f"{a}->{b}": 0 for a, b in REPORTED_STATE_FLOWS}
    for previous, current in zip(emitted, emitted[1:]):
        key = f"{previous}->{current}"
        if key in counts:
            counts[key] += 1
    return counts


def assemble_oof_result(
    *,
    oof_columns: Mapping[str, Any],
    selections: Sequence[Mapping[str, Any]],
    episode_evidence: Mapping[str, Any],
    onset_latency_seconds: Sequence[float],
    primary_confusion: Mapping[str, int],
) -> dict[str, Any]:
    """The protocol-defined T1 development evidence, and exactly that.

    Episode counts, onset latency and the PRIMARY confusion margins arrive from
    the held-out traces rather than being recomputed here: recomputing them
    would be a second implementation of the frozen matching rule, and a second
    implementation is a second answer. What this does is arrange them, derive
    the ratios the specification names from the counts already supplied, and
    refuse an undefined metric rather than let it become a zero.
    """
    _require_columns(oof_columns, ("emitted_state", "subject_id"))
    _require_complete_folds(selections)
    for field in ("reference_episodes", "predicted_event_runs", "matched_episodes"):
        if field not in episode_evidence:
            raise T1AssemblyError(f"Episode evidence is missing {field!r}.")

    reference = int(episode_evidence["reference_episodes"])
    predicted = int(episode_evidence["predicted_event_runs"])
    matched = int(episode_evidence["matched_episodes"])
    if matched > reference or matched > predicted:
        raise T1AssemblyError(
            f"{matched} matched episodes cannot exceed {reference} reference or "
            f"{predicted} predicted runs."
        )

    emitted = [str(s) for s in np.asarray(oof_columns["emitted_state"]).tolist()]
    positions = len(emitted)
    tp = int(primary_confusion["true_positive"])
    fp = int(primary_confusion["false_positive"])
    fn = int(primary_confusion["false_negative"])
    tn = int(primary_confusion["true_negative"])

    latency = np.asarray(list(onset_latency_seconds), dtype=np.float64)
    unmatched = predicted - matched

    result = {
        "episode": {
            "reference_episodes": reference,
            "predicted_event_runs": predicted,
            "matched_episodes": matched,
            "episode_precision": (matched / predicted) if predicted else None,
            "episode_sensitivity": (matched / reference) if reference else None,
            "episode_f1": episode_f1(matched, predicted, reference),
        },
        "onset_latency_seconds": {
            "median": float(np.median(latency)) if latency.size else None,
            "iqr": (
                [float(np.percentile(latency, 25)), float(np.percentile(latency, 75))]
                if latency.size
                else None
            ),
            "p90": float(np.percentile(latency, 90)) if latency.size else None,
            "matched_onsets": int(latency.size),
        },
        "primary_window": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
            "sensitivity": (tp / (tp + fn)) if (tp + fn) else None,
            "specificity": (tn / (tn + fp)) if (tn + fp) else None,
            "ppv": (tp / (tp + fp)) if (tp + fp) else None,
            "npv": (tn / (tn + fn)) if (tn + fn) else None,
            "balanced_accuracy": (
                ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
                if (tp + fn) and (tn + fp)
                else None
            ),
            "mcc": window_mcc(
                np.repeat([True, True, False, False], [tp, fp, fn, tn]),
                np.repeat([True, False, True, False], [tp, fp, fn, tn]),
            ),
        },
        "state_burden": _state_burden(emitted),
        "transitions_per_hour": (
            sum(_state_flows(emitted).values()) / physical_exposure_hours(positions)
        ),
        "state_flows": _state_flows(emitted),
        "descriptive": {
            "overmerged_event_runs": int(
                episode_evidence.get("overmerged_event_runs", 0)
            ),
            "reference_episodes_split_across_runs": int(
                episode_evidence.get("reference_episodes_split_across_runs", 0)
            ),
            "unmatched_predicted_event_runs": unmatched,
            "false_onsets_per_hour": false_event_onsets_per_hour(unmatched, positions),
        },
        "categorical_state_auprc_reported": T1_CATEGORICAL_STATE_AUPRC_REPORTED,
        "physical_exposure_hours": physical_exposure_hours(positions),
        "stride_seconds": T1_STRIDE_SECONDS,
        "fold_count": len(selections),
    }
    require_defined_metric("oof.episode_f1", result["episode"]["episode_f1"])
    return result


# ---------------------------------------------------------------------------
# §21 Subject evidence and bootstrap
# ---------------------------------------------------------------------------


def assemble_subject_evidence(
    *, oof_columns: Mapping[str, Any], per_subject: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Per-subject evidence in frozen roster order. Subject is the unit."""
    _require_columns(oof_columns, ("subject_id", "emitted_state"))
    subjects = _ordered_subjects(oof_columns)
    missing = sorted(set(subjects) - set(per_subject))
    if missing:
        raise T1AssemblyError(f"Subject evidence is missing subjects {missing}.")
    return {
        "artifact_class": "t1_v1_subject_evidence",
        "inferential_unit": T1_BOOTSTRAP_UNIT,
        "subject_order": list(subjects),
        "subject_count": len(subjects),
        "subjects": {name: dict(per_subject[name]) for name in subjects},
        "identity_is_transition_feature": T1_SUBJECT_IDENTITY_IS_TRANSITION_FEATURE,
    }


def assemble_bootstrap(
    *, oof_columns: Mapping[str, Any], subject_statistic: Mapping[str, float]
) -> dict[str, Any]:
    """1000 replicates, seed 2026, subjects resampled with multiplicity.

    The replicate indices come from the frozen `subject_bootstrap_indices`, so
    the resampling design is not re-derived here. A subject drawn twice
    contributes its already-frozen OOF trace twice; no fold is rerun and no
    policy is re-derived. Undefined replicates are preserved as undefined.
    """
    subjects = _ordered_subjects(oof_columns)
    missing = sorted(set(subjects) - set(subject_statistic))
    if missing:
        raise T1AssemblyError(f"Bootstrap input lacks a statistic for {missing}.")
    indices = subject_bootstrap_indices(len(subjects))
    if indices.shape != (T1_BOOTSTRAP_REPLICATES, len(subjects)):
        raise T1AssemblyError(
            f"Bootstrap indices are {indices.shape}, not "
            f"({T1_BOOTSTRAP_REPLICATES}, {len(subjects)})."
        )
    values = np.asarray([float(subject_statistic[name]) for name in subjects])
    replicates = [
        None if np.isnan(values[row]).any() else float(np.mean(values[row]))
        for row in indices
    ]
    defined = [value for value in replicates if value is not None]
    return {
        "artifact_class": "t1_v1_subject_bootstrap",
        "replicates": T1_BOOTSTRAP_REPLICATES,
        "seed": T1_BOOTSTRAP_SEED,
        "unit": T1_BOOTSTRAP_UNIT,
        "policy_reselected_inside_bootstrap": T1_BOOTSTRAP_RESELECTS_POLICY,
        "resampled_with_multiplicity": T1_BOOTSTRAP_RESAMPLES_WITH_MULTIPLICITY,
        "subject_order": list(subjects),
        "defined_replicates": len(defined),
        "undefined_replicates": len(replicates) - len(defined),
        "percentile_2_5": float(np.percentile(defined, 2.5)) if defined else None,
        "percentile_97_5": float(np.percentile(defined, 97.5)) if defined else None,
        "claim_scope": (
            "between_subject_variation_conditional_on_the_cross_fitted_t1_"
            "development_procedure"
        ),
    }


# ---------------------------------------------------------------------------
# §22 Challenge reporting
# ---------------------------------------------------------------------------


def assemble_challenge(
    *, oof_columns: Mapping[str, Any], challenge_rows: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    """Family identity joined after the state trace, never before it.

    Challenge membership is annotation, not a transition input, not a
    threshold-generation input and not a selection input. The frozen constants
    say so and are carried into the artifact so a reader cannot infer otherwise.
    """
    _require_columns(oof_columns, ("emitted_state",))
    if not T1_CHALLENGE_JOIN_AFTER_STATE_TRACE:  # pragma: no cover - frozen True
        raise T1AssemblyError("The protocol no longer joins after the trace; stop.")
    unknown = sorted(set(challenge_rows) - set(CHALLENGE_FAMILIES))
    if unknown:
        raise T1AssemblyError(f"Unknown challenge families {unknown}.")

    emitted = [str(s) for s in np.asarray(oof_columns["emitted_state"]).tolist()]
    event_runs = contiguous_runs([state == T1_STATE_EVENT for state in emitted])
    onsets = {begin for begin, _ in event_runs}

    families: dict[str, Any] = {}
    for family in CHALLENGE_FAMILIES:
        rows = sorted({int(index) for index in challenge_rows.get(family, ())})
        outside = [index for index in rows if not 0 <= index < len(emitted)]
        if outside:
            raise T1AssemblyError(
                f"Challenge family {family!r} names rows outside the trace: "
                f"{outside[:5]}."
            )
        watch = sum(1 for index in rows if emitted[index] == T1_STATE_WATCH)
        event = sum(1 for index in rows if emitted[index] == T1_STATE_EVENT)
        families[family] = {
            "row_count": len(rows),
            "watch_rows": watch,
            "watch_fraction": (watch / len(rows)) if rows else None,
            "event_rows": event,
            "event_fraction": (event / len(rows)) if rows else None,
            "event_onsets_on_challenge_rows": sum(1 for i in rows if i in onsets),
            "evidence_level": CHALLENGE_EVIDENCE_LEVEL[family],
        }
    return {
        "artifact_class": "t1_v1_challenge_evidence",
        "families": families,
        "joined_after_state_trace": T1_CHALLENGE_JOIN_AFTER_STATE_TRACE,
        "is_selection_input": T1_CHALLENGE_IS_SELECTION_INPUT,
        "is_transition_input": T1_CHALLENGE_IS_TRANSITION_INPUT,
        "is_threshold_generation_input": T1_CHALLENGE_IS_THRESHOLD_GENERATION_INPUT,
    }


# ---------------------------------------------------------------------------
# §23 Final all-VALIDATION configuration
# ---------------------------------------------------------------------------


def assemble_final_configuration(
    *,
    oof_columns: Mapping[str, Any],
    selections: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
    oof_result_promoted: bool = False,
) -> dict[str, Any]:
    """Deployment configuration only. Never development evidence.

    Refuses unless the OOF result has already been promoted: §23 permits this
    selection *only after* that, and a configuration assembled first could be
    mistaken for the result it must never replace.
    """
    _require_columns(oof_columns, ("subject_id",))
    _require_complete_folds(selections)
    if not oof_result_promoted:
        raise T1AssemblyError(
            "The final all-VALIDATION configuration may be assembled only after "
            "the OOF result has been promoted and verified. It is deployment "
            "configuration, and assembling it first would let in-sample "
            "all-VALIDATION numbers stand where development evidence belongs."
        )
    missing = sorted(set(FINAL_CONFIGURATION_FIELDS) - set(configuration))
    if missing:
        raise T1AssemblyError(f"Final configuration is missing {missing}.")
    subjects = _ordered_subjects(oof_columns)
    if len(subjects) != T1_FOLD_COUNT:
        raise T1AssemblyError(
            f"The final configuration is selected over all {T1_FOLD_COUNT} "
            f"subjects; {len(subjects)} are present."
        )
    return {
        "artifact_class": "t1_v1_final_all_validation_configuration",
        **{name: configuration[name] for name in FINAL_CONFIGURATION_FIELDS},
        "subject_order": list(subjects),
        "in_sample_on_all_twelve_subjects": True,
        "is_development_evidence": T1_FINAL_CONFIGURATION_IS_DEVELOPMENT_EVIDENCE,
        "replaces_oof_result": T1_FINAL_CONFIGURATION_OVERWRITES_OOF_RESULT,
        "purpose": "deployment_or_separately_authorised_test_only",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def assembly_capability() -> dict[str, Any]:
    """What this layer provides, as data a receipt can carry."""
    return {
        "collaborators": [
            "subject_of_record",
            "assemble_oof_state_columns",
            "assemble_oof_result",
            "assemble_subject_evidence",
            "assemble_bootstrap",
            "assemble_challenge",
            "assemble_final_configuration",
        ],
        "opens_labels": False,
        "reads_datasets": False,
        "computes_predictions": False,
        "creates_artifacts": False,
        "test_accessed": False,
    }
