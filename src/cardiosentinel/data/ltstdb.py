"""Long-Term ST Database v1.0.0 adapter with separate episode protocols."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Iterable

from cardiosentinel.data.models import (
    AnnotationMarker,
    AnnotationSample,
    AnnotationValidationError,
    DatasetRecord,
    ParsedAnnotations,
    SignalQualityInterval,
    STEvent,
)

DATASET_ID: Final = "ltstdb"
DATASET_VERSION: Final = "1.0.0"
EXPECTED_RECORD_COUNT: Final = 86
EXPECTED_SUBJECT_COUNT: Final = 80
PRIMARY_ANNOTATION: Final = "stb"
ANNOTATION_SETS: Final = ("sta", "stb", "stc")
_RECORD = re.compile(r"^s[23]\d{4}$")
_START = re.compile(
    r"^\((?P<rate>rt)?st(?P<lead>[0-2])\s*(?P<value>[+-]?\d+(?:\.\d+)?)$"
)
_PEAK = re.compile(r"^a(?P<rate>rt)?st(?P<lead>[0-2])\s*(?P<value>[+-]?\d+(?:\.\d+)?)$")
_END = re.compile(r"^(?P<rate>rt)?st(?P<lead>[0-2])\s*(?P<value>[+-]?\d+(?:\.\d+)?)\)$")
_UNREADABLE_START = re.compile(r"^\(urd(?P<lead>[0-2])$")
_UNREADABLE_END = re.compile(r"^urd(?P<lead>[0-2])\)$")
_SHIFT = re.compile(r"^s(?P<conduction>c)?st(?P<lead>[0-2])$")
_NOISE = re.compile(r"^noi(?P<lead>[0-2])(?:\s*(?P<value>[+-]?\d+(?:\.\d+)?))?$")
_REFERENCE = re.compile(r"^(?P<kind>GRST|LRST)(?P<lead>[0-2])")


def subject_id_for_record(record_id: str) -> str:
    """Use the documented LTSTDB relationship: only the final digit varies by record."""
    if not _RECORD.fullmatch(record_id):
        raise AnnotationValidationError(
            f"Invalid LTSTDB record identifier {record_id}."
        )
    return f"ltstdb:{record_id[:-1]}"


def parse_annotations(
    record: DatasetRecord,
    annotations: Iterable[AnnotationSample],
    annotation_set: str,
) -> ParsedAnnotations:
    """Preserve episode, shift, reference, noise, and unreadable semantics exactly."""
    if annotation_set not in ANNOTATION_SETS:
        raise AnnotationValidationError(
            f"Unknown LTSTDB annotation set {annotation_set}."
        )
    events: list[STEvent] = []
    markers: list[AnnotationMarker] = []
    intervals: list[SignalQualityInterval] = []
    open_events: dict[tuple[int, str], list[AnnotationSample]] = {}
    open_unreadable: dict[int, AnnotationSample] = {}
    for annotation in annotations:
        text = annotation.aux_note.strip(" \t\r\n\x00")
        start, peak, end = (
            _START.fullmatch(text),
            _PEAK.fullmatch(text),
            _END.fullmatch(text),
        )
        unreadable_start, unreadable_end = (
            _UNREADABLE_START.fullmatch(text),
            _UNREADABLE_END.fullmatch(text),
        )
        shift, noise, reference = (
            _SHIFT.fullmatch(text),
            _NOISE.fullmatch(text),
            _REFERENCE.fullmatch(text),
        )
        if start:
            key = (
                int(start["lead"]),
                "heart_rate_related" if start["rate"] else "ischemic",
            )
            if key in open_events:
                raise AnnotationValidationError(
                    f"Duplicate LTSTDB start marker at {annotation.sample}."
                )
            open_events[key] = [annotation]
        elif peak:
            key = (
                int(peak["lead"]),
                "heart_rate_related" if peak["rate"] else "ischemic",
            )
            if key not in open_events or len(open_events[key]) != 1:
                raise AnnotationValidationError(
                    f"Unmatched LTSTDB peak marker at {annotation.sample}."
                )
            open_events[key].append(annotation)
        elif end:
            key = (
                int(end["lead"]),
                "heart_rate_related" if end["rate"] else "ischemic",
            )
            sequence = open_events.pop(key, None)
            if sequence is None or len(sequence) != 2:
                raise AnnotationValidationError(
                    f"Unmatched LTSTDB end marker at {annotation.sample}."
                )
            onset, peak_annotation = sequence
            if (
                key[0] >= record.signal_count
                or not onset.sample < peak_annotation.sample < annotation.sample
            ):
                raise AnnotationValidationError(
                    f"Invalid LTSTDB episode timing in {record.record_id}."
                )
            peak_match = _PEAK.fullmatch(peak_annotation.aux_note.strip())
            peak_uv = float(peak_match["value"]) * 1000 if peak_match else None
            events.append(
                STEvent(
                    DATASET_ID,
                    record.record_id,
                    record.subject_id,
                    key[0],
                    "st_episode",
                    key[1],
                    onset.sample,
                    peak_annotation.sample,
                    annotation.sample,
                    onset.sample / record.sampling_frequency_hz,
                    peak_annotation.sample / record.sampling_frequency_hz,
                    annotation.sample / record.sampling_frequency_hz,
                    peak_uv,
                    _direction(peak_uv),
                    annotation_set,
                    f"ltstdb.{annotation_set}",
                    annotation_set == PRIMARY_ANNOTATION,
                    (onset, peak_annotation, annotation),
                )
            )
        elif unreadable_start:
            lead = int(unreadable_start["lead"])
            if lead in open_unreadable:
                raise AnnotationValidationError(
                    f"Duplicate unreadable start at {annotation.sample}."
                )
            open_unreadable[lead] = annotation
        elif unreadable_end:
            lead = int(unreadable_end["lead"])
            onset = open_unreadable.pop(lead, None)
            if onset is None or not onset.sample < annotation.sample:
                raise AnnotationValidationError(
                    f"Unmatched unreadable end at {annotation.sample}."
                )
            intervals.append(
                SignalQualityInterval(
                    record.record_id,
                    record.subject_id,
                    lead,
                    onset.sample,
                    annotation.sample,
                    "unreadable",
                    annotation_set,
                    (onset, annotation),
                )
            )
        elif shift:
            markers.append(
                AnnotationMarker(
                    record.record_id,
                    record.subject_id,
                    int(shift["lead"]),
                    annotation.sample,
                    "st_shift",
                    "conduction_related" if shift["conduction"] else "axis_related",
                    annotation_set,
                    annotation,
                )
            )
        elif noise:
            markers.append(
                AnnotationMarker(
                    record.record_id,
                    record.subject_id,
                    int(noise["lead"]),
                    annotation.sample,
                    "noise",
                    "point_noise",
                    annotation_set,
                    annotation,
                )
            )
        elif reference:
            markers.append(
                AnnotationMarker(
                    record.record_id,
                    record.subject_id,
                    int(reference["lead"]),
                    annotation.sample,
                    "reference",
                    "global" if reference["kind"] == "GRST" else "local",
                    annotation_set,
                    annotation,
                )
            )
    if open_events or open_unreadable:
        raise AnnotationValidationError("Unclosed LTSTDB annotation sequence.")
    return ParsedAnnotations(tuple(events), tuple(intervals), tuple(markers))


def _direction(value: float | None) -> str | None:
    if value is None or value == 0:
        return None
    return "elevation" if value > 0 else "depression"


def read_record(root: Path, record_id: str) -> DatasetRecord:
    """Read header metadata, including actual sampling rate and signal names."""
    import wfdb

    header = wfdb.rdheader(str(root / record_id))
    signal_names = tuple(header.sig_name)
    return DatasetRecord(
        DATASET_ID,
        DATASET_VERSION,
        record_id,
        subject_id_for_record(record_id),
        float(header.fs),
        signal_names,
        tuple(signal_names),
        header.n_sig,
        header.sig_len,
        header.sig_len / float(header.fs),
        record_id,
        {"comments": tuple(header.comments)},
        {"source": "PhysioNet LTSTDB v1.0.0"},
    )


def read_annotations(
    root: Path, record: DatasetRecord, annotation_set: str = PRIMARY_ANNOTATION
) -> ParsedAnnotations:
    """Read one selected ST episode definition; alternative sets are never mixed."""
    import wfdb

    raw = wfdb.rdann(str(root / record.record_id), annotation_set)
    annotations = tuple(
        AnnotationSample(int(sample), symbol, int(subtype), int(channel), aux)
        for sample, symbol, subtype, channel, aux in zip(
            raw.sample, raw.symbol, raw.subtype, raw.chan, raw.aux_note, strict=True
        )
    )
    return parse_annotations(record, annotations, annotation_set)
