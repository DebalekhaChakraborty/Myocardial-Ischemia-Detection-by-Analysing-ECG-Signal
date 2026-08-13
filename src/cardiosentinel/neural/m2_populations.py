"""The four distinct M2-v1 evidence populations, each with its own authority.

The frozen M1/P1 route separates *what the memory saw* from *what a metric is
computed over*, and M2 must reproduce that distinction exactly rather than let
one population stand in for another:

===========================  ==========================================
Population                   Authority
===========================  ==========================================
FULL REPLAY                  the verified full VALIDATION M1 stream cache
PRIMARY METRIC               the frozen P1 VALIDATION population/cache
CHALLENGE METRIC             `build_validation_challenge_index(...)`
STRESS INTERVAL              `m2_stress_intervals` (source-defined only)
===========================  ==========================================

**FULL REPLAY** carries causal memory evolution, the M2-0/M2-G update policy,
policy evidence and the prototype trajectory. It must contain every frozen
timeline row needed to preserve causal history -- challenge, quality, boundary
and censored rows are *not* dropped from it merely because they do not belong
in the primary discrimination denominator.

**PRIMARY METRIC** carries AUPRC, AUROC, sensitivity, specificity, PPV, MCC,
subject-macro evidence, background FPR, the subject-level FPR distribution and
cold-start evidence. Its membership comes from the frozen P1 validation
population -- never from an M2 score -- and challenge-only, quality-only,
boundary and censored rows never enter its classification denominator.

**CHALLENGE METRIC** carries rate-related and axis-shift challenge FPR
(`quantitative_secondary`) and conduction-change FP/N (`exploratory_descriptive`,
one subject). No binary primary label is invented for a challenge-only row.

Primary and challenge are proven **disjoint**, and both are proven **exact
subsets of the full replay population by stable identity**. A population is
never authored by the rows being evaluated: each authority verifies against a
frozen upstream identity before it is issued.

Nothing here selects a threshold, changes a denominator, or consults an M2
score, gate decision or prototype.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from cardiosentinel.neural.p1_experiment import (
    EXPECTED_POPULATIONS,
    ordered_stable_id_digest,
)
from cardiosentinel.neural.validation_challenge import (
    CHALLENGE_EXPECTED_COUNTS,
    CHALLENGE_FAMILIES,
    CHALLENGE_SELECTION_SHA256,
    CHALLENGE_TOTAL_WINDOWS,
    PRIMARY_FAMILY_SET,
)

DEVELOPMENT_PARTITION: Final = "validation"
"""The ONLY partition that may carry canonical M2 development evidence."""

PRIMARY_VALIDATION_POPULATION: Final = EXPECTED_POPULATIONS["validation"]

REPLAY_AUTHORITY: Final = "verified_full_input_bundle"
PRIMARY_AUTHORITY: Final = "frozen_p1_validation_population"
CHALLENGE_AUTHORITY: Final = "frozen_validation_challenge_selection"

POPULATION_REPLAY: Final = "full_replay"
POPULATION_PRIMARY: Final = "primary_metric"
POPULATION_CHALLENGE: Final = "challenge_metric"
POPULATION_STRESS: Final = "stress_interval"


class M2PopulationError(RuntimeError):
    """Raised when a population's authority or containment cannot be proven."""


# --------------------------------------------------------------------------
# PRIMARY metric population
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2PrimaryEvaluationPopulation:
    """The frozen PRIMARY VALIDATION population, the classification denominator.

    Issued only by `primary_evaluation_population()`, which proves the frozen
    counts and the cache's own manifest digest first. Membership is fixed
    upstream by P1/M1 and can never be derived from an M2 score.
    """

    partition: str
    stable_ids: tuple[str, ...]
    labels: tuple[int, ...]
    subject_ids: tuple[str, ...]
    counts: dict[str, int]
    cache_sha256: str
    source: str = PRIMARY_AUTHORITY

    def __post_init__(self) -> None:
        if self.partition != DEVELOPMENT_PARTITION:
            raise M2PopulationError(
                f"The primary metric population is {DEVELOPMENT_PARTITION!r}; "
                f"received {self.partition!r}."
            )
        lengths = {len(self.stable_ids), len(self.labels), len(self.subject_ids)}
        if len(lengths) != 1:
            raise M2PopulationError("Primary population columns are not row-aligned.")
        if len(set(self.stable_ids)) != len(self.stable_ids):
            raise M2PopulationError("The primary population has duplicate stable IDs.")

    @property
    def row_count(self) -> int:
        return len(self.stable_ids)

    def ordered_stable_id_sha256(self) -> str:
        return ordered_stable_id_digest(sorted(self.stable_ids))

    def identity(self) -> dict[str, Any]:
        return {
            "population": POPULATION_PRIMARY,
            "partition": self.partition,
            "authority": self.source,
            "authority_detail": (
                "frozen P1 validation embedding cache; membership fixed by "
                "P1/M1 upstream, never derived from an M2 score"
            ),
            "row_count": self.row_count,
            "counts": dict(self.counts),
            "ordered_stable_id_sha256": self.ordered_stable_id_sha256(),
            "p1_embedding_cache_sha256": self.cache_sha256,
            "membership_derived_from_m2_scores": False,
            "binary_labels_present": True,
        }


def verify_primary_population(
    *,
    stable_ids: Sequence[str],
    labels: Sequence[int],
    subject_ids: Sequence[str],
    cache_sha256: str,
    partition: str = DEVELOPMENT_PARTITION,
    expected_counts: dict[str, int] | None = None,
) -> M2PrimaryEvaluationPopulation:
    """Prove the frozen primary counts, then issue the authority token.

    Kept separate from the loader so the contract is testable without opening
    any real partition.
    """
    expected = dict(expected_counts or PRIMARY_VALIDATION_POPULATION)
    outcomes = np.asarray(labels, dtype=np.int64)
    unexpected = sorted(set(outcomes.tolist()) - {0, 1})
    if unexpected:
        raise M2PopulationError(
            f"The primary population carries non-binary labels {unexpected}."
        )
    observed = {
        "total": int(outcomes.shape[0]),
        "positive": int(np.sum(outcomes == 1)),
        "negative": int(np.sum(outcomes == 0)),
        "subjects": int(len({str(v) for v in subject_ids})),
    }
    if observed != expected:
        raise M2PopulationError(
            f"The primary metric population {observed} differs from the frozen "
            f"identity {expected}. The denominator is never adjusted to fit."
        )
    return M2PrimaryEvaluationPopulation(
        partition=partition,
        stable_ids=tuple(str(v) for v in stable_ids),
        labels=tuple(int(v) for v in outcomes.tolist()),
        subject_ids=tuple(str(v) for v in subject_ids),
        counts=observed,
        cache_sha256=str(cache_sha256),
    )


def primary_evaluation_population(cache_root: Path) -> M2PrimaryEvaluationPopulation:
    """Load the frozen P1 VALIDATION population and issue its authority.

    Opens development data. It is never called during route construction and
    only ever runs inside an authorized canonical development execution.
    """
    from cardiosentinel.neural.p1_experiment import load_p1_embedding_cache

    cache = load_p1_embedding_cache(Path(cache_root), DEVELOPMENT_PARTITION)
    return verify_primary_population(
        stable_ids=cache.stable_ids,
        labels=cache.labels.tolist(),
        subject_ids=cache.subject_ids,
        cache_sha256=str(cache.manifest["cache_sha256"]),
    )


# --------------------------------------------------------------------------
# CHALLENGE metric population
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2ChallengeEvaluationPopulation:
    """The frozen validation challenge selection. Carries NO binary label."""

    partition: str
    stable_ids: tuple[str, ...]
    target_families: tuple[str, ...]
    subject_ids: tuple[str, ...]
    counts: dict[str, dict[str, int]]
    selection_sha256: str
    source: str = CHALLENGE_AUTHORITY

    def __post_init__(self) -> None:
        if self.partition != DEVELOPMENT_PARTITION:
            raise M2PopulationError(
                f"The challenge metric population is {DEVELOPMENT_PARTITION!r}; "
                f"received {self.partition!r}."
            )
        lengths = {
            len(self.stable_ids),
            len(self.target_families),
            len(self.subject_ids),
        }
        if len(lengths) != 1:
            raise M2PopulationError("Challenge population columns are not row-aligned.")
        if len(set(self.stable_ids)) != len(self.stable_ids):
            raise M2PopulationError(
                "The challenge population has duplicate stable IDs."
            )
        intruders = sorted(set(self.target_families) - set(CHALLENGE_FAMILIES))
        if intruders:
            raise M2PopulationError(
                f"Challenge population contains non-challenge families {intruders}."
            )

    @property
    def row_count(self) -> int:
        return len(self.stable_ids)

    def identity(self) -> dict[str, Any]:
        return {
            "population": POPULATION_CHALLENGE,
            "partition": self.partition,
            "authority": self.source,
            "authority_detail": (
                "build_validation_challenge_index(...) and its frozen selection digest"
            ),
            "row_count": self.row_count,
            "counts": {family: dict(v) for family, v in self.counts.items()},
            "challenge_selection_sha256": self.selection_sha256,
            "ordered_stable_id_sha256": ordered_stable_id_digest(
                sorted(self.stable_ids)
            ),
            "binary_labels_invented": False,
            "membership_derived_from_m2_scores": False,
        }


def verify_challenge_population(
    *,
    stable_ids: Sequence[str],
    target_families: Sequence[str],
    subject_ids: Sequence[str],
    selection_sha256: str,
    counts: dict[str, dict[str, int]],
    partition: str = DEVELOPMENT_PARTITION,
    expected_selection_sha256: str = CHALLENGE_SELECTION_SHA256,
    expected_counts: dict[str, dict[str, int]] | None = None,
    expected_total: int = CHALLENGE_TOTAL_WINDOWS,
) -> M2ChallengeEvaluationPopulation:
    """Prove the frozen challenge selection, then issue the authority token."""
    expected = expected_counts or CHALLENGE_EXPECTED_COUNTS
    if str(selection_sha256) != str(expected_selection_sha256):
        raise M2PopulationError(
            f"Challenge selection digest {selection_sha256} differs from the "
            f"frozen identity {expected_selection_sha256}."
        )
    if len(stable_ids) != int(expected_total):
        raise M2PopulationError(
            f"Challenge population holds {len(stable_ids)} windows; the frozen "
            f"selection holds {expected_total}."
        )
    for family in CHALLENGE_FAMILIES:
        if counts.get(family) != expected[family]:
            raise M2PopulationError(
                f"Challenge family {family} counts {counts.get(family)} differ "
                f"from the frozen identity {expected[family]}."
            )
    return M2ChallengeEvaluationPopulation(
        partition=partition,
        stable_ids=tuple(str(v) for v in stable_ids),
        target_families=tuple(str(v) for v in target_families),
        subject_ids=tuple(str(v) for v in subject_ids),
        counts={family: dict(counts[family]) for family in CHALLENGE_FAMILIES},
        selection_sha256=str(selection_sha256),
    )


def challenge_evaluation_population(
    feature_root: Path,
) -> M2ChallengeEvaluationPopulation:
    """Rebuild the frozen validation challenge selection and issue its authority.

    Opens development metadata. Never called during route construction.
    """
    from cardiosentinel.neural.validation_challenge import (
        build_validation_challenge_index,
    )

    index = build_validation_challenge_index(Path(feature_root))
    return verify_challenge_population(
        stable_ids=[item.stable_id for item in index.references],
        target_families=[item.target_family for item in index.references],
        subject_ids=[item.subject_id for item in index.references],
        selection_sha256=index.selection_sha256,
        counts=index.counts,
    )


# --------------------------------------------------------------------------
# Containment: both metric populations live inside the full replay population
# --------------------------------------------------------------------------


def prove_population_containment(
    *,
    replay_population: Any,
    replay_stable_ids: Iterable[str],
    primary: M2PrimaryEvaluationPopulation,
    challenge: M2ChallengeEvaluationPopulation,
) -> dict[str, Any]:
    """Prove primary and challenge are disjoint subsets of the full replay set.

    `replay_stable_ids` is authenticated against the replay token's own frozen
    digest before it is trusted, so the containment standard cannot be authored
    by the rows being checked.

    The authenticated ID set is built once, shared by both proofs and released
    when this function returns -- it is never retained per arm, and no row
    object is materialised.
    """
    source = getattr(replay_population, "source", None)
    if source != REPLAY_AUTHORITY:
        raise M2PopulationError(
            f"Containment requires the verified full replay authority; received "
            f"source {source!r}."
        )
    ordered = [str(value) for value in replay_stable_ids]
    digest = ordered_stable_id_digest(ordered)
    expected_digest = getattr(replay_population, "ordered_stable_id_sha256", None)
    if digest != expected_digest:
        raise M2PopulationError(
            f"The supplied replay stable IDs digest to {digest}, not the "
            f"canonical replay identity {expected_digest}."
        )
    if len(ordered) != int(getattr(replay_population, "row_count", -1)):
        raise M2PopulationError(
            "The supplied replay stable IDs do not match the canonical replay "
            "row count."
        )

    universe = frozenset(ordered)
    primary_ids = frozenset(primary.stable_ids)
    challenge_ids = frozenset(challenge.stable_ids)

    for name, subset in (
        (POPULATION_PRIMARY, primary_ids),
        (POPULATION_CHALLENGE, challenge_ids),
    ):
        outside = subset - universe
        if outside:
            raise M2PopulationError(
                f"{len(outside)} {name} rows are absent from the full replay "
                f"population, beginning {sorted(outside)[:3]}. A metric row "
                "with no causal replay history is never evaluated."
            )
    overlap = primary_ids & challenge_ids
    if overlap:
        raise M2PopulationError(
            f"{len(overlap)} rows are in BOTH the primary and challenge "
            f"populations, beginning {sorted(overlap)[:3]}. The frozen M1 route "
            "keeps them disjoint."
        )
    return {
        "proof_class": "m2_population_containment",
        "replay_row_count": len(ordered),
        "replay_ordered_stable_id_sha256": digest,
        "primary_row_count": len(primary_ids),
        "challenge_row_count": len(challenge_ids),
        "primary_is_subset_of_replay": True,
        "challenge_is_subset_of_replay": True,
        "primary_and_challenge_disjoint": True,
        "primary_family_set": sorted(PRIMARY_FAMILY_SET),
        "challenge_family_set": sorted(CHALLENGE_FAMILIES),
        "checked_by": "stable_identity",
    }
