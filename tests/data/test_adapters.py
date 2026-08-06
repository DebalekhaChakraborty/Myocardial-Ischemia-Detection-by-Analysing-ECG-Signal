"""SYNTHETIC annotation fixtures only; no physiological data is used here."""

import pytest

from cardiosentinel.data import edb, ltstdb
from cardiosentinel.data.models import (
    AnnotationSample,
    AnnotationValidationError,
    DatasetRecord,
)


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
    assert parsed.quality_intervals == ()


def test_edb_quality_state_uses_only_noise_annotations() -> None:
    parsed = edb.parse_annotations(
        synthetic_record(),
        (
            AnnotationSample(100, "N", subtype=0),
            AnnotationSample(200, "~", subtype=0x12),
        ),
    )
    assert [interval.state for interval in parsed.quality_intervals] == [
        "unreadable",
        "noisy",
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
            AnnotationSample(10, "s", aux_note="(st0 -0.100"),
            AnnotationSample(20, "s", aux_note="ast0 -0.200"),
            AnnotationSample(30, "s", aux_note="st0 -0.050)"),
        ),
        "stb",
    )
    rate_related = ltstdb.parse_annotations(
        record,
        (
            AnnotationSample(40, "s", aux_note="(rtst1 0.100"),
            AnnotationSample(50, "s", aux_note="artst1 0.200"),
            AnnotationSample(60, "s", aux_note="rtst1 0.050)"),
        ),
        "sta",
    )
    assert ischemic.events[0].event_subtype == "ischemic"
    assert ischemic.events[0].is_primary_definition is True
    assert rate_related.events[0].event_subtype == "heart_rate_related"
    assert rate_related.events[0].annotation_definition == "ltstdb.sta"


def test_ltstdb_non_ischemic_markers_and_unreadable_interval() -> None:
    parsed = ltstdb.parse_annotations(
        synthetic_record("ltstdb", 3),
        (
            AnnotationSample(10, "s", aux_note="sst0"),
            AnnotationSample(20, "s", aux_note="scst1"),
            AnnotationSample(30, "s", aux_note="(urd2"),
            AnnotationSample(40, "s", aux_note="urd2)"),
        ),
        "stc",
    )
    assert [marker.subtype for marker in parsed.markers] == [
        "axis_related",
        "conduction_related",
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
                AnnotationSample(10, "s", aux_note="(st2 0.1"),
                AnnotationSample(20, "s", aux_note="ast2 0.2"),
                AnnotationSample(30, "s", aux_note="st2 0.1)"),
            ),
            "stb",
        )
