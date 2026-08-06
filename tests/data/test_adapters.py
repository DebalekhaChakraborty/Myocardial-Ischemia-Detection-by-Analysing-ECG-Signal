"""SYNTHETIC annotation fixtures only; no physiological data is used here."""

import pytest

from cardiosentinel.data import edb, ltstdb
from cardiosentinel.data.models import (
    AnnotationSample,
    AnnotationValidationError,
    DatasetRecord,
)
from cardiosentinel.data.validation import validate_dataset


def synthetic_record(dataset_id: str = "edb", signal_count: int = 2) -> DatasetRecord:
    """Return metadata that is explicitly synthetic and contains no ECG samples."""
    return DatasetRecord(
        dataset_id,
        "1.0.0",
        "synthetic",
        "synthetic-subject",
        250.0,
        ("I", "II", "III")[:signal_count],
        ("I", "II", "III")[:signal_count],
        signal_count,
        10_000,
        40.0,
        "synthetic",
        {},
        {"fixture": "SYNTHETIC"},
    )


def test_edb_reference_and_axis_events_remain_distinct() -> None:
    parsed = edb.parse_annotations(
        synthetic_record(),
        (
            AnnotationSample(100, "s", aux_note="(ST0-\x00"),
            AnnotationSample(200, "s", aux_note="AST0-200"),
            AnnotationSample(300, "s", aux_note="ST0-)"),
            AnnotationSample(400, '"', aux_note="(st1+"),
            AnnotationSample(500, '"', aux_note="ast1+120"),
            AnnotationSample(600, '"', aux_note="st1+)"),
        ),
    )
    assert [event.event_family for event in parsed.events] == [
        "st_change",
        "axis_shift",
    ]
    assert [event.event_subtype for event in parsed.events] == [
        "reference_st_change",
        "apparent_st_change",
    ]
    assert [event.peak_deviation_uv for event in parsed.events] == [-200, 120]
    assert parsed.quality_intervals == ()


def test_edb_quality_state_uses_only_noise_annotations() -> None:
    parsed = edb.parse_annotations(
        synthetic_record(),
        (
            AnnotationSample(100, "N", subtype=0),
            AnnotationSample(150, "n", subtype=0),
            AnnotationSample(200, "~", subtype=0x23),
        ),
    )
    assert [interval.state for interval in parsed.quality_intervals] == [
        "noisy",
        "unreadable",
    ]


def test_edb_malformed_sequence_fails() -> None:
    with pytest.raises(AnnotationValidationError, match="Unmatched EDB peak"):
        edb.parse_annotations(
            synthetic_record(), (AnnotationSample(10, "s", aux_note="AST0-100"),)
        )


def test_documented_subject_mappings_do_not_leak_across_records() -> None:
    assert edb.subject_id_for_record("e0118") == edb.subject_id_for_record("e0122")
    assert edb.subject_id_for_record("e0118") != edb.subject_id_for_record("e0116")
    assert ltstdb.subject_id_for_record("s20271") == ltstdb.subject_id_for_record(
        "s20274"
    )
    assert ltstdb.subject_id_for_record("s20271") != ltstdb.subject_id_for_record(
        "s20281"
    )


def test_ltstdb_protocols_and_subtypes_remain_separate() -> None:
    record = synthetic_record("ltstdb", 3)
    ischemic = ltstdb.parse_annotations(
        record,
        (
            AnnotationSample(10, "s", aux_note="(st0-100"),
            AnnotationSample(20, "s", aux_note="ast0-200"),
            AnnotationSample(30, "s", aux_note="st0-49)"),
        ),
        "stb",
    )
    rate_related = ltstdb.parse_annotations(
        record,
        (
            AnnotationSample(40, "s", channel=1, aux_note="(rtst1+100"),
            AnnotationSample(50, "s", channel=1, aux_note="artst1+200"),
            AnnotationSample(60, "s", channel=1, aux_note="rtst1+49)"),
        ),
        "sta",
    )
    assert ischemic.events[0].event_subtype == "ischemic"
    assert ischemic.events[0].is_primary_definition is True
    assert ischemic.events[0].peak_deviation_uv == -200.0
    assert ischemic.events[0].direction == "depression"
    assert rate_related.events[0].event_subtype == "heart_rate_related"
    assert rate_related.events[0].peak_deviation_uv == 200.0
    assert rate_related.events[0].direction == "elevation"
    assert rate_related.events[0].annotation_definition == "ltstdb.sta"
    assert rate_related.events[0].is_primary_definition is False


def test_ltstdb_non_ischemic_markers_and_unreadable_interval() -> None:
    parsed = ltstdb.parse_annotations(
        synthetic_record("ltstdb", 3),
        (
            AnnotationSample(10, "s", aux_note="sst0"),
            AnnotationSample(20, "s", channel=1, aux_note="sccst1"),
            AnnotationSample(30, "s", channel=1, aux_note="noi1-187"),
            AnnotationSample(40, "s", aux_note="(urd0"),
            AnnotationSample(50, "s", aux_note="urd0)"),
            AnnotationSample(60, "s", aux_note="GRST0"),
            AnnotationSample(70, "s", channel=1, aux_note="LRST1+20"),
        ),
        "stc",
    )
    assert [marker.subtype for marker in parsed.markers] == [
        "axis_related",
        "conduction_related",
        "point_noise",
        "global",
        "local",
    ]
    assert parsed.quality_intervals[0].state == "unreadable"


def test_invalid_lead_and_timing_fail() -> None:
    record = synthetic_record("ltstdb", 2)
    with pytest.raises(
        AnnotationValidationError, match="Invalid LTSTDB episode timing"
    ):
        ltstdb.parse_annotations(
            record,
            (
                AnnotationSample(10, "s", aux_note="(st2+100"),
                AnnotationSample(20, "s", aux_note="ast2+200"),
                AnnotationSample(30, "s", aux_note="st2+49)"),
            ),
            "stb",
        )


def test_unknown_potentially_st_related_annotation_blocks_validation() -> None:
    record = synthetic_record("ltstdb", 2)
    parsed = ltstdb.parse_annotations(
        record, (AnnotationSample(10, "s", aux_note="scst1"),), "stb"
    )
    report = validate_dataset(
        (record,),
        (),
        (),
        1,
        1,
        "ltstdb.stb",
        (),
        (parsed,),
    )
    assert report.is_valid is False
    assert report.summary["unknown_relevant_annotation_count"] == 1


def test_edb_wrong_symbol_is_not_silently_interpreted() -> None:
    record = synthetic_record()
    parsed = edb.parse_annotations(
        record,
        (
            AnnotationSample(10, '"', aux_note="(ST0+"),
            AnnotationSample(20, '"', aux_note="AST0+100"),
            AnnotationSample(30, '"', aux_note="ST0+)"),
        ),
    )
    assert parsed.events == ()
    assert len(parsed.unknown_relevant_annotations) == 3


def test_edb_known_non_st_comments_and_t_wave_forms_are_ignored_explicitly() -> None:
    parsed = edb.parse_annotations(
        synthetic_record(),
        (
            AnnotationSample(10, '"', aux_note="BUTTON"),
            AnnotationSample(20, '"', aux_note="TS"),
            AnnotationSample(30, "T", aux_note="(T0+\x13"),
        ),
    )
    assert parsed.ignored_known_annotation_count == 3
    assert parsed.unknown_annotations == ()


def test_ltstdb_left_censored_peak_and_end_are_warned_not_fabricated() -> None:
    parsed = ltstdb.parse_annotations(
        synthetic_record("ltstdb", 2),
        (
            AnnotationSample(10, "s", aux_note="ast1-232"),
            AnnotationSample(20, "s", aux_note="st1-50)"),
        ),
        "stb",
    )
    assert parsed.events == ()
    assert "left-censored" in parsed.warnings[0]


def test_ltstdb_unmatched_peak_remains_a_blocking_error() -> None:
    with pytest.raises(AnnotationValidationError, match="Unmatched LTSTDB peak"):
        ltstdb.parse_annotations(
            synthetic_record("ltstdb", 2),
            (AnnotationSample(10, "s", aux_note="ast1-232"),),
            "stb",
        )
