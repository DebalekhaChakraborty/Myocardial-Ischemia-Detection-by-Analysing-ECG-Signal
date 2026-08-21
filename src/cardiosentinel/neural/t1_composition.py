"""The canonical composition root.

Everything the canonical run needs now exists, and nothing assembled it. This
module is that assembly and only that: it resolves frozen artifacts from
repository-defined locations, reconstructs the frozen U1 fits without refitting
them, binds the collaborators the driver threads, and hands the result to
`T1CanonicalDevelopmentExecutor`.

**It performs no science.** No metric is computed here, no threshold derived,
no policy chosen, no evidence transformed. Every scientific quantity is
produced by a component that already owns it -- the evaluator, the assembly
layer, the final-configuration selector -- and this module's entire job is to
put them in each other's reach. Tests assert it calls none of the frozen
selection helpers.

**Where the artifacts come from.** Canonical locations, derived from the run
roots the harness already validates in preflight. There is no path parameter,
no artifact-root override and no latest-run discovery: a run that could be
pointed at a different artifact is a run whose provenance is an argument.

**Late binding, and why it is not a trick.** Four collaborators need values
that do not exist when the graph is built -- they are produced during the run
itself, behind the fold barriers. They are bound as read-through views over the
run object rather than as literals, so they resolve at the moment the driver
calls them and not before. The alternative is a composition root that runs the
science to obtain its own inputs, which is exactly what it must not do.

**It authorizes nothing.** `T1_EXECUTION_SPECIFICATION_AUTHORIZED` is never
read or written here. Building the graph is not permission to run it, and
`execute` asks the permission gate first regardless of what this module built.
"""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

from cardiosentinel.neural import t1_assembly as ASSEMBLY
from cardiosentinel.neural.t1_canonical_driver import T1ExecutionCollaborators
from cardiosentinel.neural.t1_development_run import (
    M2_CANONICAL_RUN_ROOT,
    T2_CANONICAL_RUN_ROOT,
    U1_CANONICAL_RUN_ROOT,
    T1DevelopmentRun,
)
from cardiosentinel.neural.t1_execution_spec import (
    STAGE_OOF_RESULT,
    T1_M2_ROW_EVIDENCE_NAME,
    T1_REQUIRED_M2_RETAINED_ARM,
    T1_REQUIRED_T2_RETAINED_ARM,
    T1_REQUIRED_U1_FAMILY,
    T1_T2_IDENTITY_NAME,
    T1_U1_FOLD_COUNT,
    T1_U1_FOLD_MANIFEST_NAME,
    require_no_test_access,
)
from cardiosentinel.neural.t1_final_configuration import (
    final_validation_authority,
    select_final_validation_configuration,
)
from cardiosentinel.neural.t1_fold_evaluation import T1CorpusTargetSource
from cardiosentinel.neural.t1_fold_evaluator import T1CanonicalFoldEvaluator
from cardiosentinel.neural.t1_protocol import T1_VALIDATION_SUBJECTS
from cardiosentinel.neural.u1_calibration import U1Calibrator

COMPOSITION_NAME: Final = "T1CanonicalComposition"

# The retained arms name their own artifacts. Written as a derivation rather
# than a literal so a different retained arm cannot silently keep reading the
# old file.
M2_EVIDENCE_ATTEMPT: Final = "m2-v1-development-two-arm-recovery2__evidence"
T2_OUTER_ATTEMPT: Final = "t2-v1-outer-validation"
U1_ATTEMPT: Final = "u1-v1-development"
T2_SCORE_FILE_BY_ARM: Final = {
    "causal_s4d_longitudinal_v1": "t2_outer_scores_s4d.npz",
    "causal_gru_longitudinal_v1": "t2_outer_scores_gru.npz",
}


class T1CompositionError(RuntimeError):
    """Raised when the canonical graph cannot be composed honestly."""


# ---------------------------------------------------------------------------
# Frozen artifact resolution
# ---------------------------------------------------------------------------


def canonical_artifact_paths(repository_root: Path) -> dict[str, Path]:
    """Where the frozen upstream artifacts live. Not a parameter, a derivation.

    Each one is named from the retention decision that selected it, so a
    resolution that drifted from the retained arm would be a resolution that
    could not be built at all.
    """
    arm = str(T1_REQUIRED_T2_RETAINED_ARM)
    if arm not in T2_SCORE_FILE_BY_ARM:
        raise T1CompositionError(
            f"The retained T2 arm {arm!r} names no score artifact here."
        )
    paths = {
        "m2_row_evidence": (
            repository_root
            / M2_CANONICAL_RUN_ROOT
            / M2_EVIDENCE_ATTEMPT
            / str(T1_REQUIRED_M2_RETAINED_ARM)
            / T1_M2_ROW_EVIDENCE_NAME
        ),
        "t2_identity": (
            repository_root
            / T2_CANONICAL_RUN_ROOT
            / T2_OUTER_ATTEMPT
            / "row_evidence"
            / T1_T2_IDENTITY_NAME
        ),
        "t2_selected_scores": (
            repository_root
            / T2_CANONICAL_RUN_ROOT
            / T2_OUTER_ATTEMPT
            / "row_evidence"
            / T2_SCORE_FILE_BY_ARM[arm]
        ),
        "u1_fold_manifest": (
            repository_root
            / U1_CANONICAL_RUN_ROOT
            / U1_ATTEMPT
            / T1_U1_FOLD_MANIFEST_NAME
        ),
    }
    for name, path in paths.items():
        require_no_test_access(path.stem)
        if not path.exists():
            raise T1CompositionError(
                f"The frozen {name} artifact is absent at {path}. A canonical "
                "run composes from artifacts that already exist; it does not "
                "search for an alternative, fall back to a different attempt "
                "or create one."
            )
    return paths


def load_oof_calibrators(fold_manifest: Path) -> dict[str, U1Calibrator]:
    """The twelve frozen U1 out-of-fold fits, reconstructed and never refitted.

    Each fold's calibrator was fitted on the eleven subjects that fold did not
    hold out, so keying by `held_out_subject` gives every subject the
    calibrator that never saw it. Reconstructing the recorded parameters is
    arithmetic; fitting is not performed and the module that could fit is not
    imported.
    """
    require_no_test_access(fold_manifest.stem)
    payload = json.loads(fold_manifest.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != T1_U1_FOLD_COUNT:
        raise T1CompositionError(
            f"The U1 fold manifest carries {len(folds or ())} folds, not "
            f"{T1_U1_FOLD_COUNT}."
        )
    calibrators: dict[str, U1Calibrator] = {}
    for entry in folds:
        subject = str(entry["held_out_subject"])
        fitted = entry.get("fitted", {}).get(str(T1_REQUIRED_U1_FAMILY))
        if fitted is None:
            raise T1CompositionError(
                f"Fold {entry.get('fold_index')} carries no "
                f"{T1_REQUIRED_U1_FAMILY!r} fit. The retained family is not "
                "substituted for another."
            )
        if subject in calibrators:
            raise T1CompositionError(
                f"{subject!r} is held out by more than one U1 fold, so its "
                "calibration is not out-of-fold."
            )
        calibrators[subject] = U1Calibrator(
            family=str(fitted["family"]),
            a=float(fitted["a"]),
            b=float(fitted["b"]),
            clamp_delta=float(fitted["clamp_delta"]),
            fit_row_count=int(fitted["fit_row_count"]),
            fit_subjects=tuple(str(s) for s in fitted["fit_subjects"]),
            optimizer=dict(fitted["optimizer"]),
        )
        if subject in calibrators[subject].fit_subjects:
            raise T1CompositionError(
                f"The calibrator for {subject!r} was fitted on {subject!r}. "
                "Development calibration is out-of-fold."
            )
    missing = sorted(set(T1_VALIDATION_SUBJECTS) - set(calibrators))
    if missing:
        raise T1CompositionError(f"No U1 calibrator was fitted for {missing}.")
    return calibrators


# ---------------------------------------------------------------------------
# Read-through views over values the run produces
# ---------------------------------------------------------------------------


class _LazyMapping(MappingABC):
    """A mapping resolved when it is read, not when it is bound.

    The four collaborators below need values the run produces behind its own
    barriers. Binding a literal would mean computing them here, which is the
    one thing a composition root must not do.
    """

    __slots__ = ("_produce", "_what")

    def __init__(self, produce, what: str) -> None:
        self._produce = produce
        self._what = what

    def _resolved(self) -> MappingABC:
        value = self._produce()
        if value is None:
            raise T1CompositionError(
                f"{self._what} was read before the run produced it."
            )
        return value

    def __getitem__(self, key: str) -> Any:
        return self._resolved()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._resolved())

    def __len__(self) -> int:
        return len(self._resolved())


class _StageReached:
    """Truthy only once a named stage has actually been entered."""

    __slots__ = ("_run", "_stage")

    def __init__(self, run: T1DevelopmentRun, stage: str) -> None:
        self._run = run
        self._stage = stage

    def __bool__(self) -> bool:
        return self._stage in tuple(self._run.stages.entered)


# The evaluator's counter names, mapped to the names the assembly layer reads.
# Module scope and explicit, so the two vocabularies are visible to a reader of
# this file rather than discoverable only by running the twenty-fourth stage.
PRIMARY_CONFUSION_KEYS: Final = {
    "tp": "true_positive",
    "fp": "false_positive",
    "tn": "true_negative",
    "fn": "false_negative",
}


def _pooled_episode_evidence(run: T1DevelopmentRun) -> dict[str, int]:
    """Sum the per-fold episode counts the evaluator already produced."""
    totals = {
        "reference_episodes": 0,
        "predicted_event_runs": 0,
        "matched_episodes": 0,
        "unmatched_predicted_runs": 0,
    }
    for trace in run.held_out_traces.values():
        for key in totals:
            totals[key] += int(trace["episode_evidence"][key])
    return totals


def _pooled_confusion(run: T1DevelopmentRun) -> dict[str, int]:
    """Pool the per-fold PRIMARY confusion and translate it once, here.

    Two vocabularies exist and both are correct in their own layer. The fold
    evaluator counts with `tp/fp/tn/fn`, which is what a counter is called next
    to the loop that increments it; the assembly layer reads
    `true_positive/...`, which is what a reported margin is called in an
    evidence document. This function is the single place they meet, so the
    translation belongs here, written out rather than implied.

    A missing count is a refusal, never a zero. A silently defaulted margin is
    an evidence value that nothing produced, and the difference between "no
    true positives" and "the producer stopped supplying true positives" is the
    difference this layer exists to keep.
    """
    totals = dict.fromkeys(PRIMARY_CONFUSION_KEYS, 0)
    for index, trace in run.held_out_traces.items():
        confusion = trace["primary_confusion"]
        missing = [key for key in PRIMARY_CONFUSION_KEYS if key not in confusion]
        if missing:
            raise T1CompositionError(
                f"Fold {index} reported PRIMARY confusion without {missing}. The "
                f"evaluator supplies {sorted(PRIMARY_CONFUSION_KEYS)} and the "
                f"assembly layer reads "
                f"{sorted(PRIMARY_CONFUSION_KEYS.values())}; a count that is "
                "absent is refused rather than defaulted to zero."
            )
        for key in totals:
            totals[key] += int(confusion[key])
    return {
        reported: totals[counted]
        for counted, reported in PRIMARY_CONFUSION_KEYS.items()
    }


def _pooled_latency(run: T1DevelopmentRun) -> tuple[float, ...]:
    values: list[float] = []
    for index in sorted(run.held_out_traces):
        values.extend(
            float(value)
            for value in run.held_out_traces[index]["onset_latency_seconds"]
        )
    return tuple(values)


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T1CanonicalComposition:
    """Resolved artifacts and the collaborators built from them."""

    repository_root: Path
    paths: dict[str, Path]
    calibrators: dict[str, U1Calibrator]

    def as_dict(self) -> dict[str, Any]:
        return {
            "composition": COMPOSITION_NAME,
            "artifacts": {name: str(path) for name, path in self.paths.items()},
            "calibrator_count": len(self.calibrators),
            "calibrator_family": str(T1_REQUIRED_U1_FAMILY),
            "refit_performed": False,
            "test_accessed": False,
        }


def resolve_canonical_composition(repository_root: Path) -> T1CanonicalComposition:
    """Resolve every frozen dependency, or refuse naming the missing one."""
    paths = canonical_artifact_paths(repository_root)
    return T1CanonicalComposition(
        repository_root=repository_root,
        paths=paths,
        calibrators=load_oof_calibrators(paths["u1_fold_manifest"]),
    )


def build_canonical_collaborators(
    run: T1DevelopmentRun, composition: T1CanonicalComposition
) -> T1ExecutionCollaborators:
    """Bind the complete execution graph. Computes nothing, runs nothing."""
    identity = composition.paths["t2_identity"]
    source = T1CorpusTargetSource(identity)

    def _final_configuration() -> dict[str, Any]:
        columns = getattr(run, "label_blind_columns", None)
        if columns is None:
            raise T1CompositionError(
                "The final all-VALIDATION configuration was requested before "
                "the label-blind timeline existed."
            )
        return select_final_validation_configuration(
            columns=columns,
            authority=final_validation_authority(source=source),
        )["configuration"]

    return T1ExecutionCollaborators(
        m2_row_evidence=composition.paths["m2_row_evidence"],
        t2_identity=identity,
        t2_selected_scores=composition.paths["t2_selected_scores"],
        calibrators=composition.calibrators,
        target_source=source,
        subject_of_record=ASSEMBLY.subject_of_record(),
        evaluate_fold=T1CanonicalFoldEvaluator(),
        assemble_oof_state_columns=ASSEMBLY.assemble_oof_state_columns,
        assemble_oof_result=ASSEMBLY.assemble_oof_result(
            episode_evidence=_LazyMapping(
                lambda: _pooled_episode_evidence(run), "Pooled episode evidence"
            ),
            onset_latency_seconds=_LazyLatency(run),
            primary_confusion=_LazyMapping(
                lambda: _pooled_confusion(run), "Pooled PRIMARY confusion"
            ),
        ),
        assemble_subject_evidence=ASSEMBLY.assemble_subject_evidence(
            held_out_traces=run.held_out_traces
        ),
        assemble_bootstrap=ASSEMBLY.assemble_bootstrap(
            held_out_traces=run.held_out_traces
        ),
        assemble_challenge=ASSEMBLY.assemble_challenge(t2_identity=identity),
        assemble_final_configuration=ASSEMBLY.assemble_final_configuration(
            configuration=_LazyMapping(
                _final_configuration, "The final all-VALIDATION configuration"
            ),
            oof_result_promoted=_StageReached(run, STAGE_OOF_RESULT),
        ),
    )


class _LazyLatency:
    """The pooled onset latencies, resolved when read."""

    __slots__ = ("_run",)

    def __init__(self, run: T1DevelopmentRun) -> None:
        self._run = run

    def __iter__(self) -> Iterator[float]:
        return iter(_pooled_latency(self._run))

    def __len__(self) -> int:
        return len(_pooled_latency(self._run))

    def __getitem__(self, index):
        return _pooled_latency(self._run)[index]


def composition_capability() -> dict[str, Any]:
    """What this layer provides, as data a receipt can carry."""
    return {
        "composition": COMPOSITION_NAME,
        "resolves_frozen_artifacts": True,
        "performs_scientific_computation": False,
        "accepts_a_path_parameter": False,
        "authorizes_execution": False,
        "refit_performed": False,
        "test_accessed": False,
    }
