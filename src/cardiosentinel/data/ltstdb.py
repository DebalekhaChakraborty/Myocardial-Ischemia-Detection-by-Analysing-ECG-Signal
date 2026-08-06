"""Long-Term ST Database v1.0.0 adapter with separate episode protocols."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Iterable

from cardiosentinel.data.models import (
    AnnotationClassification,
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
    r"^\((?P<rate>rt)?st(?P<lead>[0-2])(?P<value>[+-]\d+)$"
)
_PEAK = re.compile(r"^a(?P<rate>rt)?st(?P<lead>[0-2])(?P<value>[+-]\d+)$")
_END = re.compile(r"^(?P<rate>rt)?st(?P<lead>[0-2])(?P<value>[+-]\d+)\)$")
_UNREADABLE_START = re.compile(r"^\(urd(?P<lead>[0-2])$")
_UNREADABLE_END = re.compile(r"^urd(?P<lead>[0-2])\)$")
_SHIFT = re.compile(r"^s(?P<conduction>cc)?st(?P<lead>[0-2])$")
_NOISE = re.compile(r"^noi(?P<lead>[0-2])(?P<value>[+-]\d+)$")
_GLOBAL_REFERENCE = re.compile(r"^GRST(?P<lead>[0-2])$")
_LOCAL_REFERENCE = re.compile(r"^LRST(?P<lead>[0-2])(?P<value>[+-]\d+)$")


def subject_id_for_record(record_id: str) -> str:
    """Use the documented LTSTDB relationship: only the final digit varies by record."""
    if not _RECORD.fullmatch(record_id):
        raise AnnotationValidationError(
            f"Invalid LTSTDB record identifier {record_id}."
        )
    return f"ltstdb:{record_id[:-1]}"


def _text(annotation: AnnotationSample) -> str:
    return annotation.aux_note.strip(" \t\r\n\x00")


def _lead(match: re.Match[str]) -> int:
    """Return the explicit lead number encoded by the source annotation."""
    return int(match["lead"])


def classify_annotations(
    annotations: Iterable[AnnotationSample],
) -> tuple[AnnotationClassification, ...]:
    """Classify all LTSTDB annotations without silently discarding a form."""
    classifications: list[AnnotationClassification] = []
    for annotation in annotations:
        text = _text(annotation)
        recognized = any(
            pattern.fullmatch(text)
            for pattern in (
                _START,
                _PEAK,
                _END,
                _UNREADABLE_START,
                _UNREADABLE_END,
                _SHIFT,
                _NOISE,
                _GLOBAL_REFERENCE,
                _LOCAL_REFERENCE,
            )
        )
        if recognized and annotation.symbol == "s":
            classifications.append(
                AnnotationClassification(
                    annotation, "recognized_relevant", "ltstdb.st_annotation"
                )
            )
        elif recognized:
            classifications.append(
                AnnotationClassification(
                    annotation,
                    "unknown",
                    "LTSTDB ST text requires WFDB symbol 's'",
                    True,
                )
            )
        elif not text and annotation.symbol in {"N", "+"}:
            classifications.append(
                AnnotationClassification(
                    annotation, "recognized_irrelevant", "ltstdb.empty_non_st"
                )
            )
        else:
            classifications.append(
                AnnotationClassification(
                    annotation,
                    "unknown",
                    "unrecognized LTSTDB annotation",
                    annotation.symbol == "s" or "st" in text.lower(),
                )
            )
    return tuple(classifications)


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
    annotations = tuple(annotations)
    classifications = classify_annotations(annotations)
    invalid_channels = [
        annotation.channel
        for annotation in annotations
        if not 0 <= annotation.channel < record.signal_count
    ]
    if invalid_channels:
        raise AnnotationValidationError(
            f"Invalid LTSTDB WFDB channel values in {record.record_id}: "
            f"{sorted(set(invalid_channels))}."
        )
    events: list[STEvent] = []
    markers: list[AnnotationMarker] = []
    intervals: list[SignalQualityInterval] = []
    open_events: dict[tuple[int, str], list[AnnotationSample]] = {}
    left_censored_peaks: dict[tuple[int, str], AnnotationSample] = {}
    open_unreadable: dict[int, AnnotationSample] = {}
    warnings: list[str] = []
    for annotation, classification in zip(annotations, classifications, strict=True):
        if classification.disposition != "recognized_relevant":
            continue
        text = _text(annotation)
        start, peak, end = (
            _START.fullmatch(text),
            _PEAK.fullmatch(text),
            _END.fullmatch(text),
        )
        unreadable_start, unreadable_end = (
            _UNREADABLE_START.fullmatch(text),
            _UNREADABLE_END.fullmatch(text),
        )
        shift, noise, global_reference, local_reference = (
            _SHIFT.fullmatch(text),
            _NOISE.fullmatch(text),
            _GLOBAL_REFERENCE.fullmatch(text),
            _LOCAL_REFERENCE.fullmatch(text),
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
                if key in left_censored_peaks:
                    raise AnnotationValidationError(
                        f"Duplicate LTSTDB unpaired peak marker at {annotation.sample}."
                    )
                left_censored_peaks[key] = annotation
                continue
            open_events[key].append(annotation)
        elif end:
            key = (
                int(end["lead"]),
                "heart_rate_related" if end["rate"] else "ischemic",
            )
            sequence = open_events.pop(key, None)
            if sequence is None or len(sequence) != 2:
                left_censored_peak = left_censored_peaks.pop(key, None)
                if left_censored_peak is not None:
                    warnings.append(
                        f"{record.record_id} has a left-censored LTSTDB {key[1]} "
                        f"episode on lead {key[0]} ending at {annotation.sample}."
                    )
                    continue
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
            peak_match = _PEAK.fullmatch(_text(peak_annotation))
            peak_uv = float(peak_match["value"]) if peak_match else None
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
            lead = _lead(unreadable_start)
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
                    _lead(shift),
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
        elif global_reference or local_reference:
            reference = global_reference or local_reference
            markers.append(
                AnnotationMarker(
                    record.record_id,
                    record.subject_id,
                    _lead(reference),
                    annotation.sample,
                    "reference",
                    "global" if global_reference else "local",
                    annotation_set,
                    annotation,
                )
            )
    if left_censored_peaks:
        raise AnnotationValidationError(
            f"Unmatched LTSTDB peak markers in {record.record_id}: "
            f"{sorted(left_censored_peaks)}."
        )
    if open_events or open_unreadable:
        warnings.append(
            f"{record.record_id} ends with source-censored LTSTDB annotations."
        )
    return ParsedAnnotations(
        tuple(events),
        tuple(intervals),
        tuple(markers),
        classifications,
        tuple(warnings),
    )


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
