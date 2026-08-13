"""Source-defined M2 stress-interval selection, per the frozen human decision.

`docs/M2_STRESS_INTERVAL_ELIGIBILITY_DECISION_V1.md` freezes which longitudinal
stress families may enter the frozen §7.2 prototype-drift metric: only those
whose START AND END are both explicitly source-defined by the LTSTDB annotation
semantics. Duration is never invented for an instantaneous source marker.

Eligible -- the source supplies both boundaries:

* `STEvent(event_subtype="ischemic")`           -> `[onset_sample, end_sample]`
* `STEvent(event_subtype="heart_rate_related")` -> `[onset_sample, end_sample]`
* `SignalQualityInterval(state="unreadable")`   -> `[start_sample, end_sample]`

Not estimable -- an instantaneous marker with no paired end annotation:

* `AnnotationMarker(category="st_shift", subtype="axis_related")`
* `AnnotationMarker(category="st_shift", subtype="conduction_related")`
* `AnnotationMarker(category="noise", subtype="point_noise")`

Those three are reported as `not_estimable_from_source_defined_LTSTDB_intervals`
with an explicit canonical reason. They are NEVER given a zero drift value, and
no stress end, drift-at-stress-end, +5 minute origin or +30 minute origin is
fabricated for them. Zero drift would be a measurement, and no measurement was
possible.

**This module deliberately does not import `MARKER_VICINITY_SECONDS`.** That
±30 s radius is frozen for axis-shift window-level FPR challenge membership
only; repurposing it as a stress duration would silently redefine "drift at
stress end" and the residual origins, which the frozen protocol never claimed.
There is likewise no marker-to-next-marker rule, no marker-to-stream-end rule,
and no merge-gap, dilation, persistence or recovery constant anywhere here.

**Time coordinate.** A source interval converts to seconds as `sample / fs` on
the record's own 250 Hz clock. A prototype trajectory is stamped with
`available_time = (start_sample + WINDOW_SAMPLES) / fs`, which is the real
elapsed record time at which that window's last sample was observed. Both are
therefore the same physical clock, and the comparison needs no offset,
realignment or tolerance.

**Selection is blind to M2.** It consumes frozen source annotations only --
never a score, gate decision, prototype, trajectory or arm result -- so
membership cannot be influenced by any development outcome. It runs strictly on
the post-replay evaluation side, and the replay-side firewall in
`m2_execution.assert_label_firewall()` proves the replay cannot import it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.m2_evaluation import M2StressInterval
from cardiosentinel.neural.protocol import DATASET, SAMPLING_FREQUENCY_HZ

DECISION_DOCUMENT: Final = "docs/M2_STRESS_INTERVAL_ELIGIBILITY_DECISION_V1.md"
DECISION_SHA256: Final = (
    "078acb3d72a11513010c88a03b0143a2be43da5da807c72d3d7433f98031f8f6"
)

PRIMARY_ANNOTATION_SET: Final = "stb"

FAMILY_ISCHEMIC: Final = "ischemic"
FAMILY_RATE: Final = "heart_rate_related"
FAMILY_UNREADABLE: Final = "unreadable_quality"

SOURCE_DEFINED_FAMILIES: Final = (FAMILY_ISCHEMIC, FAMILY_RATE, FAMILY_UNREADABLE)
"""The only families with a source-defined start AND end. Frozen by decision."""

ELIGIBLE_ST_SUBTYPES: Final = (FAMILY_ISCHEMIC, FAMILY_RATE)

UNREADABLE_STATE: Final = "unreadable"

NOT_ESTIMABLE: Final = "not_estimable_from_source_defined_LTSTDB_intervals"

EXCLUDED_MARKER_FAMILIES: Final = {
    "axis_shift": "axis_shift_marker_has_no_source_defined_interval",
    "conduction_change": "conduction_change_marker_has_no_source_defined_interval",
    "point_noise": "point_noise_marker_has_no_source_defined_interval",
}

MARKER_SUBTYPE_TO_FAMILY: Final = {
    "axis_related": "axis_shift",
    "conduction_related": "conduction_change",
    "point_noise": "point_noise",
}

CENSORED_EXCLUSION_REASON: Final = (
    "source_censored_interval_lacks_a_source_defined_boundary"
)

SOURCE_SEMANTICS_ST: Final = "ltstdb_st_episode_onset_to_end"
SOURCE_SEMANTICS_UNREADABLE: Final = "ltstdb_paired_unreadable_start_to_end"


class M2StressSelectionError(RuntimeError):
    """Raised when stress-interval selection cannot proceed with integrity."""


# --------------------------------------------------------------------------
# One interval, carrying its full source identity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2SourceStressInterval:
    """A stress interval whose start AND end both come from the source."""

    dataset: str
    annotation_set: str
    record_id: str
    subject_id: str
    channel_index: int
    family: str
    start_sample: int
    end_sample: int
    source_semantics: str
    source_defined_interval: bool = True

    def __post_init__(self) -> None:
        if self.family not in SOURCE_DEFINED_FAMILIES:
            raise M2StressSelectionError(
                f"{self.family!r} has no source-defined interval; the frozen "
                f"decision permits only {list(SOURCE_DEFINED_FAMILIES)}."
            )
        if not self.source_defined_interval:
            raise M2StressSelectionError(
                "Only source-defined intervals may enter drift evaluation."
            )
        if int(self.end_sample) <= int(self.start_sample):
            raise M2StressSelectionError(
                f"Stress interval {self.identity_key} is empty or inverted. A "
                "source-defined interval is never repaired, padded or extended."
            )

    @property
    def stream_key(self) -> tuple[str, int]:
        """The frozen `(record_id, channel_index)` trajectory identity."""
        return (self.record_id, int(self.channel_index))

    @property
    def identity_key(self) -> tuple[str, str, str, int, str, int, int]:
        """The canonical ordering and uniqueness key."""
        return (
            self.dataset,
            self.annotation_set,
            self.record_id,
            int(self.channel_index),
            self.family,
            int(self.start_sample),
            int(self.end_sample),
        )

    def as_identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "annotation_set": self.annotation_set,
            "record_id": self.record_id,
            "subject_id": self.subject_id,
            "channel_index": int(self.channel_index),
            "family": self.family,
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
            "source_semantics": self.source_semantics,
            "source_defined_interval": True,
        }

    def as_evaluation_interval(
        self, *, sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ
    ) -> M2StressInterval:
        """Convert to the evaluation-layer interval on the real-time axis."""
        return M2StressInterval(
            record_id=self.record_id,
            channel_index=int(self.channel_index),
            family=self.family,
            start_time=int(self.start_sample) / float(sampling_frequency_hz),
            end_time=int(self.end_sample) / float(sampling_frequency_hz),
        )


# --------------------------------------------------------------------------
# Source-eligibility extraction
# --------------------------------------------------------------------------


def _reject_censored(candidates: Iterable[Any], *, argument: str) -> None:
    """Refuse a source-censored interval rather than silently skipping it.

    A `SourceCensoredInterval` carries a `reason` and no `event_subtype`/
    `state`, so an eligibility filter would drop it without comment. Losing it
    quietly is exactly the failure this module exists to prevent, so passing
    one is an error and it must be declared as a censored exclusion instead.
    """
    for candidate in candidates:
        if type(candidate).__name__ == "SourceCensoredInterval":
            raise M2StressSelectionError(
                f"A SourceCensoredInterval was passed as {argument}. It lacks a "
                "source-defined boundary and can never become a stress "
                "interval; declare it via source_censored_intervals so the "
                "exclusion is audited rather than silently dropped."
            )


def intervals_from_st_events(events: Iterable[Any]) -> list[M2SourceStressInterval]:
    """Ischemic and rate-related ST episodes, using their own onset and end.

    A complete `STEvent` always carries a source onset and a source end: the
    ingestion layer routes a censored episode to `SourceCensoredInterval`
    instead. The explicit refusal below therefore never fires in practice, and
    exists so a missing boundary can never be papered over.
    """
    _reject_censored(events, argument="st_events")
    selected: list[M2SourceStressInterval] = []
    for event in events:
        if str(getattr(event, "event_subtype", "")) not in ELIGIBLE_ST_SUBTYPES:
            continue
        onset = getattr(event, "onset_sample", None)
        end = getattr(event, "end_sample", None)
        if onset is None or end is None:
            raise M2StressSelectionError(
                f"ST episode on {event.record_id!r} lacks a source "
                "onset or end and can never become a stress interval; no "
                "boundary is inferred."
            )
        lead = getattr(event, "lead", None)
        if lead is None:
            raise M2StressSelectionError(
                f"ST episode on {event.record_id!r} has no lead and cannot be "
                "bound to one causal stream; an interval is never broadcast "
                "across leads."
            )
        selected.append(
            M2SourceStressInterval(
                dataset=DATASET,
                annotation_set=str(event.annotation_source),
                record_id=str(event.record_id),
                subject_id=str(event.subject_id),
                channel_index=int(lead),
                family=str(event.event_subtype),
                start_sample=int(onset),
                end_sample=int(end),
                source_semantics=SOURCE_SEMANTICS_ST,
            )
        )
    return selected


def intervals_from_quality(
    quality_intervals: Iterable[Any],
) -> list[M2SourceStressInterval]:
    """LTSTDB unreadable-quality intervals from paired `(urdX` / `urdX)`.

    Only the `unreadable` state has a paired source start and end. Point `noi`
    markers are not longitudinal quality intervals and never appear here, so
    this family must not be described as covering artifact or noise generally.
    """
    _reject_censored(quality_intervals, argument="quality_intervals")
    selected: list[M2SourceStressInterval] = []
    for interval in quality_intervals:
        if str(getattr(interval, "state", "")) != UNREADABLE_STATE:
            continue
        lead = getattr(interval, "lead", None)
        if lead is None:
            raise M2StressSelectionError(
                f"Unreadable interval on {interval.record_id!r} has no lead and "
                "cannot be bound to one causal stream; an interval is never "
                "broadcast across leads."
            )
        selected.append(
            M2SourceStressInterval(
                dataset=DATASET,
                annotation_set=str(interval.annotation_source),
                record_id=str(interval.record_id),
                subject_id=str(interval.subject_id),
                channel_index=int(lead),
                family=FAMILY_UNREADABLE,
                start_sample=int(interval.start_sample),
                end_sample=int(interval.end_sample),
                source_semantics=SOURCE_SEMANTICS_UNREADABLE,
            )
        )
    return selected


# --------------------------------------------------------------------------
# Source-eligibility exclusion audit (NOT follow-up exclusion)
# --------------------------------------------------------------------------


def marker_exclusion_audit(markers: Iterable[Any]) -> dict[str, Any]:
    """The explicit audit for families with no source-defined interval.

    Each family reports `eligible_drift_intervals = 0` with a stable canonical
    reason and `drift_value_produced: false`. The observed marker count is a
    count of source annotations, not a drift measurement.
    """
    counts = dict.fromkeys(EXCLUDED_MARKER_FAMILIES, 0)
    for marker in markers:
        family = MARKER_SUBTYPE_TO_FAMILY.get(str(getattr(marker, "subtype", "")))
        if family is not None:
            counts[family] += 1
    return {
        family: {
            "observed_source_markers": counts[family],
            "eligible_drift_intervals": 0,
            "status": NOT_ESTIMABLE,
            "reason": reason,
            "drift_value_produced": False,
            "zero_drift_asserted": False,
            "stress_end_fabricated": False,
        }
        for family, reason in sorted(EXCLUDED_MARKER_FAMILIES.items())
    }


def censored_exclusion_audit(censored: Iterable[Any]) -> dict[str, Any]:
    """The audit for episodes the source itself left without a boundary."""
    by_reason: dict[str, int] = {}
    for interval in censored:
        reason = str(getattr(interval, "reason", "unspecified_source_censoring"))
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return {
        "observed_source_censored_intervals": sum(by_reason.values()),
        "eligible_drift_intervals": 0,
        "status": NOT_ESTIMABLE,
        "reason": CENSORED_EXCLUSION_REASON,
        "boundary_fabricated": False,
        "observed_by_source_reason": dict(sorted(by_reason.items())),
    }


# --------------------------------------------------------------------------
# The deterministic selection and its identity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2StressSelection:
    """The deterministic post-replay stress selection and its bound identity."""

    intervals: tuple[M2SourceStressInterval, ...]
    excluded_marker_families: dict[str, Any]
    source_censored_exclusions: dict[str, Any]
    sampling_frequency_hz: float

    def for_stream(
        self, stream_key: tuple[str, int]
    ) -> tuple[M2SourceStressInterval, ...]:
        """The intervals belonging to exactly one `(record_id, channel)`."""
        return tuple(item for item in self.intervals if item.stream_key == stream_key)

    def family_counts(self) -> dict[str, int]:
        return {
            family: sum(1 for item in self.intervals if item.family == family)
            for family in SOURCE_DEFINED_FAMILIES
        }

    def evaluation_intervals(self) -> tuple[M2StressInterval, ...]:
        """The selection on the trajectory's real elapsed-time axis."""
        return tuple(
            item.as_evaluation_interval(
                sampling_frequency_hz=self.sampling_frequency_hz
            )
            for item in self.intervals
        )

    def ordered_interval_identities(self) -> list[dict[str, Any]]:
        """The canonical ordered identities the selection digest is taken over."""
        return [item.as_identity() for item in self.intervals]

    def selection_digest(self) -> str:
        """A deterministic digest over the canonical ordered identities.

        Taken over the ordered identity list bound to the decision that admitted
        it. It is not self-referential: the digest is never part of the body it
        hashes, matching the M1/P1/B4 repository lock convention.
        """
        return canonical_sha256(
            {
                "selection_class": "m2_v1_source_defined_stress_selection",
                "decision_document": DECISION_DOCUMENT,
                "decision_sha256": DECISION_SHA256,
                "dataset": DATASET,
                "source_defined_families": list(SOURCE_DEFINED_FAMILIES),
                "ordered_interval_identities": self.ordered_interval_identities(),
            }
        )

    def identity(self) -> dict[str, Any]:
        """The `stress_interval_selection_identity` block for result and lock."""
        return {
            "selection_class": "m2_v1_source_defined_stress_selection",
            "decision_document": DECISION_DOCUMENT,
            "decision_sha256": DECISION_SHA256,
            "dataset": DATASET,
            "sampling_frequency_hz": float(self.sampling_frequency_hz),
            "source_defined_families": list(SOURCE_DEFINED_FAMILIES),
            "eligible_interval_count": len(self.intervals),
            "family_counts": self.family_counts(),
            "ordered_interval_identities": self.ordered_interval_identities(),
            "stress_interval_selection_sha256": self.selection_digest(),
            "excluded_marker_families": self.excluded_marker_families,
            "source_censored_exclusions": self.source_censored_exclusions,
            "exclusion_stage": "source_interval_eligibility",
            "follow_up_exclusions_recorded_separately": True,
            "marker_vicinity_reused_as_stress_duration": False,
            "persistence_duration_invented": False,
            "merge_gap_applied": False,
            "interval_broadcast_across_leads": False,
            "selection_influenced_by_m2_outputs": False,
            "selection_performed_after_label_blind_replay": True,
        }


def build_stress_selection(
    *,
    st_events: Sequence[Any] = (),
    quality_intervals: Sequence[Any] = (),
    markers: Sequence[Any] = (),
    source_censored_intervals: Sequence[Any] = (),
    sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ,
    annotation_set: str = PRIMARY_ANNOTATION_SET,
) -> M2StressSelection:
    """Build the deterministic source-defined stress selection.

    Consumes frozen source annotations only. It never receives an M2 score,
    gate decision, prototype, trajectory or arm result, so which intervals are
    selected cannot be influenced by any development outcome.
    """
    selected = [
        *intervals_from_st_events(st_events),
        *intervals_from_quality(quality_intervals),
    ]
    foreign = sorted({item.annotation_set for item in selected} - {str(annotation_set)})
    if foreign:
        raise M2StressSelectionError(
            f"Stress intervals came from annotation sets {foreign}, not the "
            f"canonical {annotation_set!r}. Mixing annotation sets would make "
            "the selection identity ambiguous."
        )
    # One canonical order, so the selection digest is deterministic regardless
    # of the order in which records were parsed.
    selected.sort(key=lambda item: item.identity_key)
    duplicates = len(selected) - len({item.identity_key for item in selected})
    if duplicates:
        raise M2StressSelectionError(
            f"{duplicates} duplicate stress-interval identities. The selection "
            "would double-count a stress episode; it is not deduplicated "
            "automatically."
        )
    return M2StressSelection(
        intervals=tuple(selected),
        excluded_marker_families=marker_exclusion_audit(markers),
        source_censored_exclusions=censored_exclusion_audit(source_censored_intervals),
        sampling_frequency_hz=float(sampling_frequency_hz),
    )


def build_stress_selection_from_parsed(
    parsed: Iterable[Any],
    *,
    sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ,
    annotation_set: str = PRIMARY_ANNOTATION_SET,
) -> M2StressSelection:
    """Build the selection from `ParsedAnnotations` objects, one per record."""
    events: list[Any] = []
    intervals: list[Any] = []
    markers: list[Any] = []
    censored: list[Any] = []
    for record in parsed:
        events.extend(record.events)
        intervals.extend(record.quality_intervals)
        markers.extend(record.markers)
        censored.extend(record.source_censored_intervals)
    return build_stress_selection(
        st_events=events,
        quality_intervals=intervals,
        markers=markers,
        source_censored_intervals=censored,
        sampling_frequency_hz=sampling_frequency_hz,
        annotation_set=annotation_set,
    )
