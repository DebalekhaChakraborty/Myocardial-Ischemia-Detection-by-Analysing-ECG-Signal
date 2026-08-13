"""Source-defined M2 stress-interval selection.

Synthetic annotation fixtures only. No VALIDATION stream cache is opened, no
VALIDATION rows are enumerated, no `.stb` file is read, no VALIDATION interval
count is computed, neither M2 arm is run and no M2 metric is calculated. The
sealed TEST partition is untouched.

The fixtures are constructed from the real
`cardiosentinel.data.models` dataclasses, so the eligibility rules are tested
against the actual parsed representation rather than a convenient stand-in.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from cardiosentinel.data.models import (
    AnnotationMarker,
    SignalQualityInterval,
    SourceCensoredInterval,
    STEvent,
)
from cardiosentinel.neural import m2_stress_intervals as SI
from cardiosentinel.neural.m2_evaluation import M2StressInterval, contamination_evidence
from cardiosentinel.neural.m2_evidence import PrototypeTrajectory
from cardiosentinel.neural.m2_execution import assert_label_firewall
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.protocol import SAMPLING_FREQUENCY_HZ

MODULE_PATH = REPOSITORY_ROOT / "src/cardiosentinel/neural/m2_stress_intervals.py"


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def st_event(
    *,
    record_id: str = "s20011",
    subject_id: str = "subject-1",
    lead: int = 0,
    subtype: str = "ischemic",
    onset: int = 10_000,
    end: int = 40_000,
    annotation_source: str = "stb",
) -> STEvent:
    peak = (onset + end) // 2
    return STEvent(
        dataset_id="ltstdb",
        record_id=record_id,
        subject_id=subject_id,
        lead=lead,
        event_family="st_episode",
        event_subtype=subtype,
        onset_sample=onset,
        peak_sample=peak,
        end_sample=end,
        onset_seconds=onset / SAMPLING_FREQUENCY_HZ,
        peak_seconds=peak / SAMPLING_FREQUENCY_HZ,
        end_seconds=end / SAMPLING_FREQUENCY_HZ,
        peak_deviation_uv=-150.0,
        direction="depression",
        annotation_source=annotation_source,
        annotation_definition=f"ltstdb.{annotation_source}",
        is_primary_definition=annotation_source == "stb",
        original_annotations=(),
    )


def quality_interval(
    *,
    record_id: str = "s20011",
    subject_id: str = "subject-1",
    lead: int | None = 1,
    start: int = 60_000,
    end: int = 65_000,
    state: str = "unreadable",
) -> SignalQualityInterval:
    return SignalQualityInterval(
        record_id=record_id,
        subject_id=subject_id,
        lead=lead,
        start_sample=start,
        end_sample=end,
        state=state,
        annotation_source="stb",
        original_annotations=(),
    )


def marker(
    *,
    subtype: str,
    record_id: str = "s20011",
    subject_id: str = "subject-1",
    lead: int | None = 0,
    sample: int = 25_000,
) -> AnnotationMarker:
    category = {
        "axis_related": "st_shift",
        "conduction_related": "st_shift",
        "point_noise": "noise",
    }.get(subtype, "reference")
    return AnnotationMarker(
        record_id=record_id,
        subject_id=subject_id,
        lead=lead,
        sample=sample,
        category=category,
        subtype=subtype,
        annotation_source="stb",
        original_annotation=None,
    )


def censored(
    *,
    reason: str = "right_censored_ischemic_episode",
    record_id: str = "s20011",
    start: int = 90_000,
    end: int = 120_000,
) -> SourceCensoredInterval:
    return SourceCensoredInterval(
        record_id=record_id,
        subject_id="subject-1",
        lead=0,
        start_sample=start,
        end_sample=end,
        reason=reason,
        annotation_source="stb",
        original_annotations=(),
    )


# --------------------------------------------------------------------------
# 1-3. The three source-defined families ARE selected, with source boundaries
# --------------------------------------------------------------------------


def test_ischemic_episode_uses_its_own_source_onset_and_end() -> None:
    selection = SI.build_stress_selection(st_events=[st_event()])
    (interval,) = selection.intervals
    assert interval.family == "ischemic"
    assert (interval.start_sample, interval.end_sample) == (10_000, 40_000)
    assert interval.source_semantics == SI.SOURCE_SEMANTICS_ST
    assert interval.source_defined_interval is True


def test_heart_rate_related_episode_is_eligible() -> None:
    selection = SI.build_stress_selection(
        st_events=[st_event(subtype="heart_rate_related")]
    )
    (interval,) = selection.intervals
    assert interval.family == "heart_rate_related"
    assert (interval.start_sample, interval.end_sample) == (10_000, 40_000)


def test_unreadable_quality_interval_uses_its_paired_source_bounds() -> None:
    selection = SI.build_stress_selection(quality_intervals=[quality_interval()])
    (interval,) = selection.intervals
    assert interval.family == "unreadable_quality"
    assert (interval.start_sample, interval.end_sample) == (60_000, 65_000)
    assert interval.source_semantics == SI.SOURCE_SEMANTICS_UNREADABLE


def test_only_the_unreadable_quality_state_is_selected() -> None:
    """Point `noi` is not a longitudinal quality interval (decision §6)."""
    selection = SI.build_stress_selection(
        quality_intervals=[quality_interval(state="noise")]
    )
    assert selection.intervals == ()


# --------------------------------------------------------------------------
# 4-8. The three marker families produce NO interval and NO drift value
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subtype", "family"),
    [
        ("axis_related", "axis_shift"),
        ("conduction_related", "conduction_change"),
        ("point_noise", "point_noise"),
    ],
)
def test_point_marker_families_yield_no_stress_interval(
    subtype: str, family: str
) -> None:
    selection = SI.build_stress_selection(markers=[marker(subtype=subtype)])
    assert selection.intervals == ()
    audit = selection.excluded_marker_families[family]
    assert audit["observed_source_markers"] == 1
    assert audit["eligible_drift_intervals"] == 0
    assert audit["status"] == SI.NOT_ESTIMABLE
    assert audit["reason"] == SI.EXCLUDED_MARKER_FAMILIES[family]


def test_exclusion_audit_carries_exactly_the_three_canonical_reasons() -> None:
    audit = SI.marker_exclusion_audit([])
    assert sorted(audit) == ["axis_shift", "conduction_change", "point_noise"]
    assert {entry["reason"] for entry in audit.values()} == {
        "axis_shift_marker_has_no_source_defined_interval",
        "conduction_change_marker_has_no_source_defined_interval",
        "point_noise_marker_has_no_source_defined_interval",
    }


def test_excluded_families_never_receive_a_zero_drift_value() -> None:
    """`eligible_drift_intervals = 0` is a count, not a measured drift of 0."""
    audit = SI.marker_exclusion_audit(
        [marker(subtype="axis_related"), marker(subtype="conduction_related")]
    )
    for entry in audit.values():
        assert entry["drift_value_produced"] is False
        assert entry["zero_drift_asserted"] is False
        assert entry["stress_end_fabricated"] is False
        # The audit is exactly a count, a status, a reason and negative flags.
        # A drift magnitude would be a float, and no float is present.
        assert set(entry) == {
            "observed_source_markers",
            "eligible_drift_intervals",
            "status",
            "reason",
            "drift_value_produced",
            "zero_drift_asserted",
            "stress_end_fabricated",
        }
        assert not any(isinstance(value, float) for value in entry.values())


def test_markers_are_counted_but_never_promoted_by_volume() -> None:
    many = [marker(subtype="axis_related", sample=1_000 * i) for i in range(1, 51)]
    selection = SI.build_stress_selection(markers=many)
    assert selection.intervals == ()
    assert (
        selection.excluded_marker_families["axis_shift"]["observed_source_markers"]
        == 50
    )
    assert (
        selection.excluded_marker_families["axis_shift"]["eligible_drift_intervals"]
        == 0
    )


# --------------------------------------------------------------------------
# 9-11. The forbidden duration inventions are structurally absent
# --------------------------------------------------------------------------


def test_marker_vicinity_seconds_is_not_imported_as_a_stress_duration() -> None:
    """±30 s is frozen for axis-shift FPR membership only (decision §2)."""
    tree = ast.parse(MODULE_PATH.read_text())
    names: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    assert "MARKER_VICINITY_SECONDS" not in names
    # The constant lives in the challenge-metric protocol; the stress selector
    # must not reach that module at all, by any name.
    assert "cardiosentinel.evaluation.protocol" not in modules
    assert not any(module.startswith("cardiosentinel.evaluation") for module in modules)


def test_module_defines_no_duration_constant_at_all() -> None:
    """No merge gap, dilation, persistence or recovery constant may exist."""
    tree = ast.parse(MODULE_PATH.read_text())
    numeric_constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, (int, float))
        and not isinstance(node.value.value, bool)
    }
    assert numeric_constants == {}


def test_two_distant_markers_do_not_create_a_marker_to_marker_interval() -> None:
    """No persistence is inferred from annotation spacing (decision §3)."""
    selection = SI.build_stress_selection(
        markers=[
            marker(subtype="axis_related", sample=1_000),
            marker(subtype="axis_related", sample=900_000),
        ]
    )
    assert selection.intervals == ()
    assert selection.identity()["persistence_duration_invented"] is False
    assert selection.identity()["marker_vicinity_reused_as_stress_duration"] is False


def test_a_lone_marker_does_not_extend_to_the_end_of_the_stream() -> None:
    selection = SI.build_stress_selection(
        markers=[marker(subtype="conduction_related")]
    )
    assert selection.intervals == ()
    assert selection.identity()["eligible_interval_count"] == 0


# --------------------------------------------------------------------------
# 12-15. Interval identity and the deterministic selection digest
# --------------------------------------------------------------------------


def test_interval_identity_carries_every_required_field() -> None:
    selection = SI.build_stress_selection(st_events=[st_event()])
    (identity,) = selection.ordered_interval_identities()
    assert identity == {
        "dataset": "ltstdb",
        "annotation_set": "stb",
        "record_id": "s20011",
        "subject_id": "subject-1",
        "channel_index": 0,
        "family": "ischemic",
        "start_sample": 10_000,
        "end_sample": 40_000,
        "source_semantics": SI.SOURCE_SEMANTICS_ST,
        "source_defined_interval": True,
    }


def test_selection_digest_is_invariant_to_input_order() -> None:
    events = [
        st_event(record_id="s20021", onset=5_000, end=9_000),
        st_event(record_id="s20011", lead=1, subtype="heart_rate_related"),
        st_event(record_id="s20011"),
    ]
    intervals = [quality_interval(), quality_interval(record_id="s20031", lead=0)]
    forward = SI.build_stress_selection(st_events=events, quality_intervals=intervals)
    reversed_ = SI.build_stress_selection(
        st_events=list(reversed(events)), quality_intervals=list(reversed(intervals))
    )
    assert forward.selection_digest() == reversed_.selection_digest()
    assert forward.ordered_interval_identities() == (
        reversed_.ordered_interval_identities()
    )


def test_selection_digest_changes_when_a_source_boundary_changes() -> None:
    base = SI.build_stress_selection(st_events=[st_event()])
    moved = SI.build_stress_selection(st_events=[st_event(end=40_001)])
    assert base.selection_digest() != moved.selection_digest()


def test_selection_digest_is_not_self_referential() -> None:
    """The digest is never a member of the body it hashes (lock convention)."""
    selection = SI.build_stress_selection(st_events=[st_event()])
    identity = selection.identity()
    assert identity["stress_interval_selection_sha256"] == selection.selection_digest()
    assert "stress_interval_selection_sha256" not in set(
        selection.ordered_interval_identities()[0]
    )


def test_identity_binds_the_frozen_human_decision() -> None:
    identity = SI.build_stress_selection().identity()
    assert identity["decision_document"] == (
        "docs/M2_STRESS_INTERVAL_ELIGIBILITY_DECISION_V1.md"
    )
    assert identity["decision_sha256"] == SI.DECISION_SHA256
    assert identity["source_defined_families"] == [
        "ischemic",
        "heart_rate_related",
        "unreadable_quality",
    ]


def test_decision_document_digest_matches_the_committed_file() -> None:
    import hashlib

    path = REPOSITORY_ROOT / "docs/M2_STRESS_INTERVAL_ELIGIBILITY_DECISION_V1.md"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == SI.DECISION_SHA256


# --------------------------------------------------------------------------
# 16-19. Refusals: never repair, never deduplicate, never broadcast
# --------------------------------------------------------------------------


def test_duplicate_interval_identities_are_refused_not_deduplicated() -> None:
    with pytest.raises(SI.M2StressSelectionError, match="duplicate"):
        SI.build_stress_selection(st_events=[st_event(), st_event()])


def test_an_inverted_or_empty_interval_is_refused_not_repaired() -> None:
    with pytest.raises(SI.M2StressSelectionError, match="never repaired"):
        SI.build_stress_selection(st_events=[st_event(onset=40_000, end=40_000)])


def test_a_family_without_a_source_defined_interval_cannot_be_constructed() -> None:
    with pytest.raises(SI.M2StressSelectionError, match="no source-defined interval"):
        SI.M2SourceStressInterval(
            dataset="ltstdb",
            annotation_set="stb",
            record_id="s20011",
            subject_id="subject-1",
            channel_index=0,
            family="axis_shift",
            start_sample=0,
            end_sample=7_500,
            source_semantics="invented",
        )


def test_an_interval_without_a_lead_is_never_broadcast_across_channels() -> None:
    with pytest.raises(SI.M2StressSelectionError, match="never broadcast"):
        SI.build_stress_selection(quality_intervals=[quality_interval(lead=None)])


def test_mixing_annotation_sets_is_refused() -> None:
    with pytest.raises(SI.M2StressSelectionError, match="annotation sets"):
        SI.build_stress_selection(st_events=[st_event(annotation_source="sta")])


# --------------------------------------------------------------------------
# 20-21. Source-censored intervals: audited, never given a boundary
# --------------------------------------------------------------------------


def test_a_source_censored_interval_is_refused_rather_than_silently_skipped() -> None:
    with pytest.raises(SI.M2StressSelectionError, match="SourceCensoredInterval"):
        SI.build_stress_selection(st_events=[censored()])


def test_censored_intervals_are_audited_without_fabricating_a_boundary() -> None:
    selection = SI.build_stress_selection(
        st_events=[st_event()],
        source_censored_intervals=[
            censored(),
            censored(record_id="s20021"),
            censored(reason="right_censored_unreadable_interval"),
        ],
    )
    audit = selection.source_censored_exclusions
    assert len(selection.intervals) == 1
    assert audit["observed_source_censored_intervals"] == 3
    assert audit["eligible_drift_intervals"] == 0
    assert audit["status"] == SI.NOT_ESTIMABLE
    assert audit["reason"] == SI.CENSORED_EXCLUSION_REASON
    assert audit["boundary_fabricated"] is False
    assert audit["observed_by_source_reason"] == {
        "right_censored_ischemic_episode": 2,
        "right_censored_unreadable_interval": 1,
    }


# --------------------------------------------------------------------------
# 22-25. Separation, time axis, and the label firewall
# --------------------------------------------------------------------------


def test_source_eligibility_is_separate_from_follow_up_exclusion() -> None:
    """Decision §7: an eligible interval may still lack causal follow-up."""
    selection = SI.build_stress_selection(st_events=[st_event()])
    identity = selection.identity()
    assert identity["exclusion_stage"] == "source_interval_eligibility"
    assert identity["follow_up_exclusions_recorded_separately"] is True

    trajectory = PrototypeTrajectory(
        times=np.array([40.0, 41.0], dtype=np.float64),
        prototypes=np.zeros((2, 4), dtype=np.float64),
    )
    evidence = contamination_evidence(
        {("s20011", 0): trajectory},
        stress_intervals=list(selection.evaluation_intervals()),
    )
    (entry,) = evidence["intervals"]
    # The interval was source-ELIGIBLE; it is still excluded here for missing
    # causal support, and that is a different exclusion recorded separately.
    assert entry["family"] == "ischemic"
    assert evidence["follow_up_fabricated"] is False


def test_evaluation_intervals_use_the_real_elapsed_record_time_axis() -> None:
    selection = SI.build_stress_selection(st_events=[st_event()])
    (interval,) = selection.evaluation_intervals()
    assert isinstance(interval, M2StressInterval)
    assert interval.start_time == 10_000 / SAMPLING_FREQUENCY_HZ
    assert interval.end_time == 40_000 / SAMPLING_FREQUENCY_HZ
    assert interval.stream_key == ("s20011", 0)


def test_intervals_are_bound_to_one_stream_not_to_a_record() -> None:
    selection = SI.build_stress_selection(
        st_events=[st_event(lead=0), st_event(lead=1)]
    )
    assert len(selection.for_stream(("s20011", 0))) == 1
    assert len(selection.for_stream(("s20011", 1))) == 1
    assert selection.for_stream(("s20021", 0)) == ()


def test_selection_consumes_no_m2_output() -> None:
    """Membership cannot be influenced by a score, decision or prototype."""
    tree = ast.parse(MODULE_PATH.read_text())
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        module.endswith(("m2_policy", "m2_scorer", "m2_execution", "m2_gate"))
        for module in imported
    )
    assert (
        SI.build_stress_selection().identity()["selection_influenced_by_m2_outputs"]
        is False
    )


def test_the_replay_side_may_not_import_the_stress_selection_module() -> None:
    report = assert_label_firewall()
    assert "m2_stress_intervals" in report["post_replay_modules"]
    assert report["replay_imports_evaluation"] is False
    for name in ("m2_policy.py", "m2_execution.py"):
        source = (REPOSITORY_ROOT / "src/cardiosentinel/neural" / name).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").endswith("m2_stress_intervals")


def test_no_validation_or_test_partition_is_referenced_by_the_module() -> None:
    text = MODULE_PATH.read_text()
    assert "validation" not in text.lower()
    assert "sealed" not in text.lower()


def test_parsed_annotations_entry_point_unpacks_every_family() -> None:
    class FakeParsed:
        events = (st_event(),)
        quality_intervals = (quality_interval(),)
        markers = (marker(subtype="point_noise"),)
        source_censored_intervals = (censored(),)

    selection = SI.build_stress_selection_from_parsed([FakeParsed()])
    assert len(selection.intervals) == 2
    assert selection.family_counts() == {
        "ischemic": 1,
        "heart_rate_related": 0,
        "unreadable_quality": 1,
    }
    assert (
        selection.excluded_marker_families["point_noise"]["observed_source_markers"]
        == 1
    )
    assert (
        selection.source_censored_exclusions["observed_source_censored_intervals"] == 1
    )
