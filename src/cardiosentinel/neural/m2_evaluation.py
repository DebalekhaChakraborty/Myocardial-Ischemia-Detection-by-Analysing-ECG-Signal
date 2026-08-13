"""Post-replay M2 evaluation: the ONLY place annotations are permitted.

This module runs strictly AFTER a label-blind replay has already produced its
scores, decisions and prototype trajectory. Annotations enter here to *define
evaluation strata* and nothing else; they cannot reach the replay, because the
replay-side modules neither name an annotation quantity nor import this module
(`m2_execution.assert_label_firewall()` enforces that direction structurally).

**Alignment is by immutable row identity, never by position.** Equal sequence
length proves nothing: a permuted annotation vector of the same length would be
silently accepted by a positional join. Every annotation therefore carries the
frozen `(record_id, channel_index, start_sample)` identity, which maps
one-to-one onto the frozen `stable_id`, and each bundle performs a
deterministic keyed join that rejects duplicates, missing identities, extra
identities and any mismatch.

**Three populations, three contracts.** PRIMARY and CHALLENGE are separate
structures with separate annotation types and separate frozen authorities (see
`m2_populations`), because forcing them into one permissive record is exactly
how a challenge confounder acquires an invented binary label or a headline
metric silently widens its denominator to the full causal timeline.

**Unscored rows are refused, not dropped.** This reproduces the frozen M1
governance rule verbatim (`m1_experiment.require_available_rows`): a
score-bearing population containing a row with no prediction is a refusal, not
a silent exclusion -- "the row is never dropped from a metric, no prediction is
invented and no denominator is altered automatically."

**No threshold is selected here.** Every thresholded path is guarded by
`require_frozen_m1l_classification_threshold`, so no evaluation can silently
run at a non-frozen operating point.

**Nothing in this module is executed against VALIDATION under the current
authorization**, and no retention decision is expressed anywhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from cardiosentinel.neural.m2_evidence import (
    PrototypeTrajectory,
    interval_drift_evidence,
    summarize_admission,
)
from cardiosentinel.neural.m2_policy import M2RowEvidence, require_m2_arm
from cardiosentinel.neural.m2_populations import (
    CHALLENGE_AUTHORITY,
    POPULATION_CHALLENGE,
    POPULATION_PRIMARY,
    POPULATION_REPLAY,
    POPULATION_STRESS,
    PRIMARY_AUTHORITY,
    REPLAY_AUTHORITY,
)
from cardiosentinel.neural.m2_scorer import (
    M1L_CLASSIFICATION_THRESHOLD,
    NORMAL_EVIDENCE_THRESHOLD,
)
from cardiosentinel.neural.protocol import DATASET, WINDOW_SAMPLES

CONDUCTION_EVIDENCE_STATUS: Final = "exploratory_descriptive"
QUANTITATIVE_CHALLENGE_STATUS: Final = "quantitative_secondary"

EvaluationKey = tuple[str, int, int]
"""`(record_id, channel_index, start_sample)` -- immutable frozen row identity."""


class M2EvaluationError(RuntimeError):
    """Raised when post-replay evaluation cannot proceed with integrity."""


# --------------------------------------------------------------------------
# Frozen threshold enforcement -- one helper, applied on every thresholded path
# --------------------------------------------------------------------------


def require_frozen_m1l_classification_threshold(threshold: float) -> float:
    """Refuse any operating point other than the frozen retained M1L threshold.

    Shared deliberately rather than restated per call site: subtly different
    per-function checks are exactly how a non-frozen operating point reaches a
    claim-bearing metric.
    """
    value = float(threshold)
    if value != M1L_CLASSIFICATION_THRESHOLD:
        raise M2EvaluationError(
            f"M2 evaluation requires the frozen retained M1L classification "
            f"threshold {M1L_CLASSIFICATION_THRESHOLD!r}; received {value!r}. "
            "No new threshold may be selected, and the M2 normal-evidence "
            "margin is never a classification threshold."
        )
    if value == NORMAL_EVIDENCE_THRESHOLD:
        raise M2EvaluationError(
            "The M2 normal-evidence margin must never be used as a "
            "classification threshold."
        )
    return value


def _observed_population_digest(keys: Iterable[EvaluationKey]) -> str:
    """The frozen ordered-stable-ID digest of an evaluation population.

    Internal on purpose. It DESCRIBES a population; it can never AUTHORISE one.
    The canonical reference comes from
    `M2InputBundle.canonical_input_population_identity()`, which proves against
    the frozen stream-cache manifest -- so the rows being evaluated can never
    author the standard they are judged against.
    """
    from cardiosentinel.neural.p1_experiment import ordered_stable_id_digest

    return ordered_stable_id_digest([stable_id_for_key(key) for key in sorted(keys)])


def evaluation_key(row: M2RowEvidence) -> EvaluationKey:
    """The immutable evaluation identity of one evidence row."""
    return (row.record_id, int(row.channel_index), int(row.start_sample))


def stable_id_for_key(key: EvaluationKey) -> str:
    """The frozen `stable_id` this evaluation key corresponds to.

    The correspondence is exact and one-to-one: the frozen identity is
    `dataset:record:channel:start:end` with `end = start + WINDOW_SAMPLES`, so
    the evaluation key determines it completely and vice versa.
    """
    record_id, channel_index, start_sample = key
    return (
        f"{DATASET}:{record_id}:{int(channel_index)}:{int(start_sample)}:"
        f"{int(start_sample) + WINDOW_SAMPLES}"
    )


@dataclass(frozen=True, slots=True)
class M2PrimaryAnnotation:
    """One PRIMARY metric annotation, bound to an immutable row identity.

    Carries a binary label because the primary population IS the classification
    denominator. `subject_id` exists for subject-macro and subject-FPR
    reporting only, and `cold_start_bin` for the frozen recording-age strata.
    None of them is ever a runtime predictive or gating input: all three are
    absent from `M2TimelineRow`, from `evaluate_gate` and from the replay path.

    Deliberately NOT interchangeable with `M2ChallengeAnnotation`. Forcing both
    into one permissive record is how a challenge row acquires an invented
    binary label.
    """

    record_id: str
    channel_index: int
    start_sample: int
    label: int
    subject_id: str
    cold_start_bin: str

    def __post_init__(self) -> None:
        if int(self.label) not in (0, 1):
            raise M2EvaluationError(
                f"A primary annotation carries the binary label 0 or 1; "
                f"received {self.label!r}."
            )

    @property
    def key(self) -> EvaluationKey:
        return (self.record_id, int(self.channel_index), int(self.start_sample))

    @property
    def stable_id(self) -> str:
        return stable_id_for_key(self.key)


@dataclass(frozen=True, slots=True)
class M2ChallengeAnnotation:
    """One CHALLENGE metric annotation. It has NO binary label, by design.

    A challenge confounder row is not an ischemia-positive/negative
    classification row, so no primary label exists for it and none is invented.
    The structure simply has nowhere to put one.
    """

    record_id: str
    channel_index: int
    start_sample: int
    target_family: str
    subject_id: str

    @property
    def key(self) -> EvaluationKey:
        return (self.record_id, int(self.channel_index), int(self.start_sample))

    @property
    def stable_id(self) -> str:
        return stable_id_for_key(self.key)


# --------------------------------------------------------------------------
# Annotation tables: compact, purpose-specific, and structurally incompatible
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2PrimaryAnnotationTable:
    """PRIMARY annotations as compact arrays. Carries labels."""

    stable_ids: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    cold_start_bins: np.ndarray

    def __post_init__(self) -> None:
        widths = {
            self.stable_ids.shape[0],
            self.labels.shape[0],
            self.subject_ids.shape[0],
            self.cold_start_bins.shape[0],
        }
        if len(widths) != 1:
            raise M2EvaluationError("Primary annotation columns are not row-aligned.")
        if len(set(self.stable_ids.tolist())) != self.stable_ids.shape[0]:
            raise M2EvaluationError("Duplicate primary annotation identity.")
        unexpected = sorted(set(np.asarray(self.labels).tolist()) - {0, 1})
        if unexpected:
            raise M2EvaluationError(
                f"Primary annotations carry non-binary labels {unexpected}."
            )

    @classmethod
    def from_rows(cls, rows: Iterable[M2PrimaryAnnotation]) -> M2PrimaryAnnotationTable:
        collected = list(rows)
        for row in collected:
            if not isinstance(row, M2PrimaryAnnotation):
                raise M2EvaluationError(
                    f"A primary annotation table accepts M2PrimaryAnnotation "
                    f"only; received {type(row).__name__}. A challenge "
                    "confounder never enters the classification denominator."
                )
        return cls(
            stable_ids=np.asarray([r.stable_id for r in collected], dtype=np.str_),
            labels=np.asarray([int(r.label) for r in collected], dtype=np.int64),
            subject_ids=np.asarray(
                [str(r.subject_id) for r in collected], dtype=np.str_
            ),
            cold_start_bins=np.asarray(
                [str(r.cold_start_bin) for r in collected], dtype=np.str_
            ),
        )


@dataclass(frozen=True, slots=True)
class M2ChallengeAnnotationTable:
    """CHALLENGE annotations as compact arrays. There is no label column."""

    stable_ids: np.ndarray
    target_families: np.ndarray
    subject_ids: np.ndarray

    def __post_init__(self) -> None:
        widths = {
            self.stable_ids.shape[0],
            self.target_families.shape[0],
            self.subject_ids.shape[0],
        }
        if len(widths) != 1:
            raise M2EvaluationError("Challenge annotation columns are not row-aligned.")
        if len(set(self.stable_ids.tolist())) != self.stable_ids.shape[0]:
            raise M2EvaluationError("Duplicate challenge annotation identity.")

    @classmethod
    def from_rows(
        cls, rows: Iterable[M2ChallengeAnnotation]
    ) -> M2ChallengeAnnotationTable:
        collected = list(rows)
        for row in collected:
            if not isinstance(row, M2ChallengeAnnotation):
                raise M2EvaluationError(
                    f"A challenge annotation table accepts M2ChallengeAnnotation "
                    f"only; received {type(row).__name__}. A labelled primary row "
                    "is not a challenge confounder."
                )
        return cls(
            stable_ids=np.asarray([r.stable_id for r in collected], dtype=np.str_),
            target_families=np.asarray(
                [str(r.target_family) for r in collected], dtype=np.str_
            ),
            subject_ids=np.asarray(
                [str(r.subject_id) for r in collected], dtype=np.str_
            ),
        )


def _as_annotation_table(annotations: Any, kind: str) -> Any:
    """Accept either a compact table or an iterable of row objects."""
    expected = (
        M2PrimaryAnnotationTable if kind == "primary" else M2ChallengeAnnotationTable
    )
    if isinstance(annotations, expected):
        return annotations
    other = (
        M2ChallengeAnnotationTable if kind == "primary" else M2PrimaryAnnotationTable
    )
    if isinstance(annotations, other):
        raise M2EvaluationError(
            f"A {kind} bundle was given a {type(annotations).__name__}. The two "
            "populations are different denominators and are never interchanged."
        )
    return expected.from_rows(annotations)


def _as_score_table(arm: str, scores: Any) -> Any:
    """Accept a compact `M2ScoreTable` or a bounded sequence of evidence rows.

    The canonical route always passes a table: at validation scale the row
    objects are exactly the corpus-scale retention the M1 incident forbade.
    """
    if hasattr(scores, "scores_for"):
        if getattr(scores, "arm", arm) != arm:
            raise M2EvaluationError(
                f"The score table belongs to arm {scores.arm!r}, not {arm!r}."
            )
        return scores
    from cardiosentinel.neural.m2_evidence_store import (
        M2EvidenceStoreError,
        M2ScoreTable,
    )

    rows = list(scores)
    if not rows:
        raise M2EvaluationError("No replay evidence was supplied.")
    keys = [evaluation_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise M2EvaluationError(
            "Duplicate evidence identity; the evaluated population would be ambiguous."
        )
    try:
        return M2ScoreTable(
            arm=require_m2_arm(arm),
            stable_ids=np.asarray(
                [stable_id_for_key(key) for key in keys], dtype=np.str_
            ),
            scores=np.asarray(
                [
                    np.nan if row.decision.score is None else float(row.decision.score)
                    for row in rows
                ],
                dtype=np.float64,
            ),
            scored=np.asarray(
                [row.decision.score is not None for row in rows], dtype=np.bool_
            ),
        )
    except M2EvidenceStoreError as error:
        raise M2EvaluationError(str(error)) from error


def _scores_for(table: Any, stable_ids: Sequence[str]) -> np.ndarray:
    """Align scores to an exact identity list, as one evaluation error type."""
    from cardiosentinel.neural.m2_evidence_store import M2EvidenceStoreError

    try:
        return table.scores_for(list(stable_ids))
    except M2EvidenceStoreError as error:
        raise M2EvaluationError(str(error)) from error


def _require_exact_population(
    observed: np.ndarray,
    authority_stable_ids: Sequence[str],
    *,
    population_name: str,
) -> None:
    """The evaluated rows must be EXACTLY the frozen population, no more, no less."""
    seen = set(observed.tolist())
    expected = set(str(value) for value in authority_stable_ids)
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise M2EvaluationError(
            f"The evaluated {population_name} population is not the frozen "
            f"population: {len(missing)} frozen rows absent (beginning "
            f"{missing[:3]}) and {len(extra)} rows present that the frozen "
            f"authority does not contain (beginning {extra[:3]}). The "
            "denominator is never widened or narrowed to fit the evidence."
        )


def _authority_column(
    stable_ids: Sequence[str], values: Sequence[Any]
) -> dict[str, Any]:
    return dict(zip((str(v) for v in stable_ids), values, strict=True))


@dataclass(frozen=True, slots=True)
class M2PrimaryBundle:
    """PRIMARY metric rows, aligned by identity and held as compact arrays.

    This is the classification denominator, and its membership comes from the
    frozen P1 validation population -- never from an M2 score.
    """

    arm: str
    stable_ids: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    cold_start_bins: np.ndarray
    authority: str
    authority_identity: dict[str, Any]

    @property
    def row_count(self) -> int:
        return int(self.stable_ids.shape[0])

    def population_identity(self) -> dict[str, Any]:
        from cardiosentinel.neural.p1_experiment import ordered_stable_id_digest

        payload = dict(self.authority_identity)
        payload.update(
            {
                "evaluated_rows": self.row_count,
                "evaluated_ordered_stable_id_sha256": ordered_stable_id_digest(
                    sorted(self.stable_ids.tolist())
                ),
                "identity_key": "(record_id, channel_index, start_sample)",
                "identity_corresponds_to_frozen_stable_id": True,
                "positional_join_used": False,
                "matches_frozen_authority_exactly": True,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class M2ChallengeBundle:
    """CHALLENGE metric rows. There is no `labels` field, deliberately."""

    arm: str
    stable_ids: np.ndarray
    scores: np.ndarray
    target_families: np.ndarray
    subject_ids: np.ndarray
    authority: str
    authority_identity: dict[str, Any]

    @property
    def row_count(self) -> int:
        return int(self.stable_ids.shape[0])

    def population_identity(self) -> dict[str, Any]:
        from cardiosentinel.neural.p1_experiment import ordered_stable_id_digest

        payload = dict(self.authority_identity)
        payload.update(
            {
                "evaluated_rows": self.row_count,
                "evaluated_ordered_stable_id_sha256": ordered_stable_id_digest(
                    sorted(self.stable_ids.tolist())
                ),
                "identity_key": "(record_id, channel_index, start_sample)",
                "identity_corresponds_to_frozen_stable_id": True,
                "positional_join_used": False,
                "matches_frozen_authority_exactly": True,
                "binary_labels_invented": False,
            }
        )
        return payload


def build_primary_bundle(
    arm: str,
    scores: Any,
    annotations: Any,
    *,
    primary_population: Any,
) -> M2PrimaryBundle:
    """Join PRIMARY scores to labels by identity, under the frozen P1 authority.

    The annotation table never gets to define the population: the evaluated
    rows must be exactly the frozen P1 validation population, and every label
    and subject must agree with it. Alignment is by stable identity in one
    deterministic order, so a permuted annotation table carries no meaning and
    equal length proves nothing.
    """
    evaluated_arm = require_m2_arm(arm)
    if getattr(primary_population, "source", None) != PRIMARY_AUTHORITY:
        raise M2EvaluationError(
            "The primary metric population must come from the frozen P1 "
            "validation authority "
            "(m2_populations.primary_evaluation_population); received source "
            f"{getattr(primary_population, 'source', None)!r}."
        )
    table = _as_annotation_table(annotations, "primary")
    _require_exact_population(
        table.stable_ids, primary_population.stable_ids, population_name="primary"
    )

    order = np.argsort(table.stable_ids, kind="stable")
    stable_ids = table.stable_ids[order]
    labels = np.asarray(table.labels, dtype=np.int64)[order]
    subject_ids = table.subject_ids[order]
    cold_start_bins = table.cold_start_bins[order]

    frozen_labels = _authority_column(
        primary_population.stable_ids, primary_population.labels
    )
    frozen_subjects = _authority_column(
        primary_population.stable_ids, primary_population.subject_ids
    )
    for stable_id, label, subject in zip(
        stable_ids.tolist(), labels.tolist(), subject_ids.tolist(), strict=True
    ):
        if int(label) != int(frozen_labels[stable_id]):
            raise M2EvaluationError(
                f"Primary annotation {stable_id} carries label {label}, but the "
                f"frozen P1 validation population records "
                f"{frozen_labels[stable_id]}. Labels are never reassigned "
                "during M2 evaluation."
            )
        if str(subject) != str(frozen_subjects[stable_id]):
            raise M2EvaluationError(
                f"Primary annotation {stable_id} carries subject {subject!r}, "
                f"but the frozen population records "
                f"{frozen_subjects[stable_id]!r}."
            )
    return M2PrimaryBundle(
        arm=evaluated_arm,
        stable_ids=stable_ids,
        scores=_scores_for(_as_score_table(evaluated_arm, scores), stable_ids.tolist()),
        labels=labels,
        subject_ids=subject_ids,
        cold_start_bins=cold_start_bins,
        authority=PRIMARY_AUTHORITY,
        authority_identity=primary_population.identity(),
    )


def build_challenge_bundle(
    arm: str,
    scores: Any,
    annotations: Any,
    *,
    challenge_population: Any,
) -> M2ChallengeBundle:
    """Join CHALLENGE scores by identity, under the frozen challenge selection."""
    evaluated_arm = require_m2_arm(arm)
    if getattr(challenge_population, "source", None) != CHALLENGE_AUTHORITY:
        raise M2EvaluationError(
            "The challenge metric population must come from the frozen "
            "validation challenge selection "
            "(m2_populations.challenge_evaluation_population); received source "
            f"{getattr(challenge_population, 'source', None)!r}."
        )
    table = _as_annotation_table(annotations, "challenge")
    _require_exact_population(
        table.stable_ids, challenge_population.stable_ids, population_name="challenge"
    )

    order = np.argsort(table.stable_ids, kind="stable")
    stable_ids = table.stable_ids[order]
    families = table.target_families[order]
    subject_ids = table.subject_ids[order]

    frozen_families = _authority_column(
        challenge_population.stable_ids, challenge_population.target_families
    )
    frozen_subjects = _authority_column(
        challenge_population.stable_ids, challenge_population.subject_ids
    )
    for stable_id, family, subject in zip(
        stable_ids.tolist(), families.tolist(), subject_ids.tolist(), strict=True
    ):
        if str(family) != str(frozen_families[stable_id]):
            raise M2EvaluationError(
                f"Challenge annotation {stable_id} carries family {family!r}, "
                f"but the frozen selection records {frozen_families[stable_id]!r}."
            )
        if str(subject) != str(frozen_subjects[stable_id]):
            raise M2EvaluationError(
                f"Challenge annotation {stable_id} carries subject {subject!r}, "
                f"but the frozen selection records {frozen_subjects[stable_id]!r}."
            )
    return M2ChallengeBundle(
        arm=evaluated_arm,
        stable_ids=stable_ids,
        scores=_scores_for(_as_score_table(evaluated_arm, scores), stable_ids.tolist()),
        target_families=families,
        subject_ids=subject_ids,
        authority=CHALLENGE_AUTHORITY,
        authority_identity=challenge_population.identity(),
    )


def require_primary_bundle(bundle: Any, purpose: str) -> M2PrimaryBundle:
    """Refuse anything but a frozen-authority PRIMARY bundle."""
    if not isinstance(bundle, M2PrimaryBundle):
        raise M2EvaluationError(
            f"{purpose} is computed over the PRIMARY metric population only; "
            f"received {type(bundle).__name__}. The full replay population and "
            "the challenge population are different denominators."
        )
    if bundle.authority != PRIMARY_AUTHORITY:
        raise M2EvaluationError(
            f"{purpose} requires the frozen P1 validation authority; the bundle "
            f"carries {bundle.authority!r}."
        )
    return bundle


def require_challenge_bundle(bundle: Any, purpose: str) -> M2ChallengeBundle:
    """Refuse anything but a frozen-authority CHALLENGE bundle."""
    if not isinstance(bundle, M2ChallengeBundle):
        raise M2EvaluationError(
            f"{purpose} is computed over the CHALLENGE metric population only; "
            f"received {type(bundle).__name__}. It must never run over the "
            "primary population."
        )
    if bundle.authority != CHALLENGE_AUTHORITY:
        raise M2EvaluationError(
            f"{purpose} requires the frozen validation challenge authority; the "
            f"bundle carries {bundle.authority!r}."
        )
    return bundle


# --------------------------------------------------------------------------
# Thresholded evaluation: every path guarded, every path bound to ONE population
# --------------------------------------------------------------------------


def window_evidence(
    bundle: M2PrimaryBundle,
    *,
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Pooled and subject-macro window discrimination over the PRIMARY rows."""
    from cardiosentinel.neural.p1_experiment import p1_validation_evidence

    frozen = require_frozen_m1l_classification_threshold(threshold)
    primary = require_primary_bundle(bundle, "window_evidence")
    payload = p1_validation_evidence(
        primary.labels, primary.scores, primary.subject_ids, frozen
    )
    payload["evidence_class"] = "m2_window_evidence"
    payload["arm"] = primary.arm
    payload["threshold_source"] = "frozen_retained_m1l_classification_threshold"
    payload["threshold_selected_here"] = False
    payload["population"] = POPULATION_PRIMARY
    payload["population_identity"] = primary.population_identity()
    return payload


def background_false_positive_evidence(
    bundle: M2PrimaryBundle,
    *,
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Background FPR and its subject distribution, over the PRIMARY rows.

    The frozen M1 definition needs background-negative rows, which exist only
    in the primary classification population. Running it over the full replay
    timeline or the challenge selection would silently change its denominator.
    """
    from cardiosentinel.neural.m1_experiment import subject_false_positive_evidence

    frozen = require_frozen_m1l_classification_threshold(threshold)
    primary = require_primary_bundle(bundle, "background_false_positive_evidence")
    payload = subject_false_positive_evidence(
        primary.labels, primary.scores, primary.subject_ids, frozen
    )
    payload["arm"] = primary.arm
    payload["population"] = POPULATION_PRIMARY
    payload["population_identity"] = primary.population_identity()
    payload["threshold_source"] = "frozen_retained_m1l_classification_threshold"
    payload["threshold_selected_here"] = False
    return payload


def challenge_false_positive_evidence(
    bundle: M2ChallengeBundle,
    *,
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Challenge FPR over the CHALLENGE rows, never over the primary rows.

    Challenge-target precedence is preserved exactly as the frozen production
    metric defines it: an ischemic-positive row is never removed merely because
    a challenge context also applies to it. That precedence lives upstream in
    the frozen selection, which is why this path consumes that selection rather
    than re-deriving membership here.
    """
    from cardiosentinel.neural.p1_experiment import p1_challenge_evidence

    frozen = require_frozen_m1l_classification_threshold(threshold)
    challenge = require_challenge_bundle(bundle, "challenge_false_positive_evidence")
    payload = p1_challenge_evidence(
        challenge.target_families, challenge.scores, challenge.subject_ids, frozen
    )
    payload["evidence_class"] = "m2_challenge_evidence"
    payload["arm"] = challenge.arm
    payload["threshold_source"] = "frozen_retained_m1l_classification_threshold"
    payload["threshold_selected_here"] = False
    payload["conduction_change_evidence_status"] = CONDUCTION_EVIDENCE_STATUS
    payload["ischemic_positive_rows_removed_for_challenge_context"] = False
    payload["binary_labels_invented_for_challenge_rows"] = False
    payload["population"] = POPULATION_CHALLENGE
    payload["population_identity"] = challenge.population_identity()
    return payload


def false_alarm_evidence(
    *,
    primary_bundle: M2PrimaryBundle,
    challenge_bundle: M2ChallengeBundle,
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """The one top-level false-alarm section, explicit about its TWO denominators.

    Background/subject FPR comes from the PRIMARY metric population; challenge
    FPR comes from the CHALLENGE metric population. Each subsection carries its
    own validated population identity, so the result never implies that one
    denominator served both.
    """
    frozen = require_frozen_m1l_classification_threshold(threshold)
    background = background_false_positive_evidence(primary_bundle, threshold=frozen)
    challenge = challenge_false_positive_evidence(challenge_bundle, threshold=frozen)
    if background["population_identity"] == challenge["population_identity"]:
        raise M2EvaluationError(
            "The background and challenge false-alarm subsections report the "
            "same population identity; they must be computed over the distinct "
            "frozen primary and challenge populations."
        )
    return {
        "evidence_class": "m2_false_alarm_evidence",
        "arm": primary_bundle.arm,
        "threshold": frozen,
        "threshold_source": "frozen_retained_m1l_classification_threshold",
        "threshold_selected_here": False,
        "background_and_subject_fpr": background,
        "challenge_fpr": challenge,
        "conduction_change_evidence_status": CONDUCTION_EVIDENCE_STATUS,
        "ischemic_positive_rows_removed_for_challenge_context": False,
        "single_denominator_served_both": False,
        "background_population_identity": background["population_identity"],
        "challenge_population_identity": challenge["population_identity"],
    }


def cold_start_stratified_evidence(
    bundle: M2PrimaryBundle,
    *,
    threshold: float = M1L_CLASSIFICATION_THRESHOLD,
) -> dict[str, Any]:
    """Frozen recording-age strata over the PRIMARY rows, matching frozen M1.

    Headline cold-start evidence is never computed over the full replay
    timeline: the frozen M1 route stratifies the primary validation metric rows.
    """
    from cardiosentinel.baseline.metrics import binary_metrics
    from cardiosentinel.neural.patient_memory import COLD_START_BINS

    frozen = require_frozen_m1l_classification_threshold(threshold)
    primary = require_primary_bundle(bundle, "cold_start_stratified_evidence")
    labels = primary.labels
    scores = primary.scores
    bins = primary.cold_start_bins

    strata: dict[str, Any] = {}
    for name, _low, _high in COLD_START_BINS:
        mask = bins == name
        count = int(np.sum(mask))
        entry: dict[str, Any] = {"window_count": count, "evidence_status": "supporting"}
        if count:
            entry["metrics"] = binary_metrics(labels[mask], scores[mask], frozen)
        strata[name] = entry
    return {
        "evidence_class": "m2_cold_start_evidence",
        "arm": primary.arm,
        "threshold": frozen,
        "threshold_source": "frozen_retained_m1l_classification_threshold",
        "threshold_selected_here": False,
        "post_hoc_early_threshold_defined": False,
        "population": POPULATION_PRIMARY,
        "inherited_limitation": (
            "M1's zero sensitivity in the 0-5 minute bin at the frozen "
            "thresholds is inherited by every M2 arm and is not addressed by "
            "this protocol; gating can only make early adaptation more "
            "conservative"
        ),
        "strata": strata,
        "population_identity": primary.population_identity(),
    }


def policy_evidence(
    evidence: Sequence[M2RowEvidence], *, replay_population: Any | None = None
) -> dict[str, Any]:
    """Update-admission and refusal accounting. Label-free by construction.

    Bound to the FULL REPLAY population, not to a metric denominator: update
    admission happened over the entire causal timeline, including rows that
    never enter the primary or challenge metrics.
    """
    # Accepts a bounded sequence of rows, or the streaming accumulator the
    # canonical route folds each stream into. Both produce the identical
    # payload; only the memory profile differs.
    summary = (
        evidence.summary()
        if hasattr(evidence, "summary")
        else summarize_admission(evidence)
    )
    summary["evidence_class"] = "m2_policy_evidence"
    summary["memory_admission_threshold"] = NORMAL_EVIDENCE_THRESHOLD
    summary["classification_threshold_used_for_admission"] = False
    summary["population"] = POPULATION_REPLAY
    if replay_population is not None:
        if getattr(replay_population, "source", None) != REPLAY_AUTHORITY:
            raise M2EvaluationError(
                "Policy evidence binds the verified FULL REPLAY authority; "
                f"received source {getattr(replay_population, 'source', None)!r}."
            )
        summary["population_identity"] = replay_population.identity()
    return summary


# --------------------------------------------------------------------------
# Prototype contamination, bound to the correct stream
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2StressInterval:
    """An annotation-defined stress interval, bound to ONE stream.

    An interval belongs to a specific `(record_id, channel_index)` trajectory.
    Overlapping timestamps on another stream are a different patient-channel
    history and must never be evaluated against it.
    """

    record_id: str
    channel_index: int
    family: str
    start_time: float
    end_time: float

    @property
    def stream_key(self) -> tuple[str, int]:
        return (self.record_id, int(self.channel_index))


def contamination_evidence(
    trajectories: dict[tuple[str, int], PrototypeTrajectory],
    *,
    stress_intervals: Sequence[M2StressInterval],
    replay_population: Any | None = None,
    stress_selection_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prototype drift for annotation-defined intervals, per stream.

    Each interval is evaluated only against its own stream's trajectory. The
    trajectory itself was produced by a label-blind replay before any
    annotation was consulted; intervals only select points on a fixed
    trajectory and can never alter it. Missing support is excluded with a
    reason and never fabricated, and no recovery threshold is defined.
    """
    results = []
    for interval in stress_intervals:
        key = interval.stream_key
        if key not in trajectories:
            raise M2EvaluationError(
                f"Stress interval for stream {key} has no matching prototype "
                "trajectory; an interval may never be applied to another "
                "stream merely because its timestamps overlap."
            )
        entry = interval_drift_evidence(
            trajectories[key],
            stress_start_time=float(interval.start_time),
            stress_end_time=float(interval.end_time),
        )
        entry["family"] = interval.family
        entry["record_id"] = interval.record_id
        entry["channel_index"] = int(interval.channel_index)
        entry["evidence_status"] = (
            CONDUCTION_EVIDENCE_STATUS
            if interval.family == "conduction_change"
            else QUANTITATIVE_CHALLENGE_STATUS
        )
        results.append(entry)
    payload = {
        "evidence_class": "m2_prototype_contamination_evidence",
        "trajectory_produced_label_blind": True,
        "annotations_applied_after_replay": True,
        "intervals_bound_to_stream_identity": True,
        "recovery_threshold_defined": False,
        "follow_up_fabricated": False,
        "population": POPULATION_STRESS,
        "trajectory_population": POPULATION_REPLAY,
        "intervals": results,
    }
    if replay_population is not None:
        if getattr(replay_population, "source", None) != REPLAY_AUTHORITY:
            raise M2EvaluationError(
                "Prototype contamination binds the verified FULL REPLAY "
                "authority for its trajectory; received source "
                f"{getattr(replay_population, 'source', None)!r}."
            )
        payload["replay_population_identity"] = replay_population.identity()
    if stress_selection_identity is not None:
        payload["stress_interval_selection_identity"] = dict(stress_selection_identity)
    return payload


def arm_evaluation(arm: str, evidence: Sequence[M2RowEvidence]) -> dict[str, Any]:
    """The label-free half of an arm's evaluation, safe to compute anywhere."""
    return {
        "arm": require_m2_arm(arm),
        "policy_evidence": policy_evidence(evidence),
        "window_evidence": None,
        "false_alarm_evidence": None,
        "cold_start_evidence": None,
        "contamination_evidence": None,
        "label_joined_sections_populated": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
