"""European ST-T Database v1.0.0 adapter and case-sensitive annotation parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Iterable

from cardiosentinel.data.models import (
    AnnotationClassification,
    AnnotationSample,
    AnnotationValidationError,
    DatasetRecord,
    ParsedAnnotations,
    SignalQualityInterval,
    STEvent,
)

DATASET_ID: Final = "edb"
DATASET_VERSION: Final = "1.0.0"
EXPECTED_RECORD_COUNT: Final = 90
EXPECTED_SUBJECT_COUNT: Final = 79
PRIMARY_ANNOTATION: Final = "atr"
EDB_RECORD_IDS: Final = (
    "e0103",
    "e0104",
    "e0105",
    "e0106",
    "e0107",
    "e0108",
    "e0110",
    "e0111",
    "e0112",
    "e0113",
    "e0114",
    "e0115",
    "e0116",
    "e0118",
    "e0119",
    "e0121",
    "e0122",
    "e0123",
    "e0124",
    "e0125",
    "e0126",
    "e0127",
    "e0129",
    "e0133",
    "e0136",
    "e0139",
    "e0147",
    "e0148",
    "e0151",
    "e0154",
    "e0155",
    "e0159",
    "e0161",
    "e0162",
    "e0163",
    "e0166",
    "e0170",
    "e0202",
    "e0203",
    "e0204",
    "e0205",
    "e0206",
    "e0207",
    "e0208",
    "e0210",
    "e0211",
    "e0212",
    "e0213",
    "e0302",
    "e0303",
    "e0304",
    "e0305",
    "e0306",
    "e0403",
    "e0404",
    "e0405",
    "e0406",
    "e0408",
    "e0409",
    "e0410",
    "e0411",
    "e0413",
    "e0415",
    "e0417",
    "e0418",
    "e0501",
    "e0509",
    "e0515",
    "e0601",
    "e0602",
    "e0603",
    "e0604",
    "e0605",
    "e0606",
    "e0607",
    "e0609",
    "e0610",
    "e0611",
    "e0612",
    "e0613",
    "e0614",
    "e0615",
    "e0704",
    "e0801",
    "e0808",
    "e0817",
    "e0818",
    "e1301",
    "e1302",
    "e1304",
)
EDB_SHARED_SUBJECT_GROUPS: Final = (
    ("e0118", "e0119", "e0121", "e0122"),
    ("e0123", "e0124", "e0125", "e0126"),
    ("e0129", "e0133"),
    ("e0136", "e0139"),
    ("e0147", "e0148"),
    ("e0154", "e0155"),
    ("e0162", "e0163"),
)
_START = re.compile(r"^\((?P<kind>ST|st)(?P<lead>[01])(?P<direction>[+-])$")
_PEAK = re.compile(
    r"^(?P<prefix>AST|ast)(?P<lead>[01])(?P<direction>[+-])(?P<value>\d+)$"
)
_END = re.compile(r"^(?P<kind>ST|st)(?P<lead>[01])(?P<direction>[+-])\)$")
_T_START = re.compile(r"^\((?P<kind>T|t)(?P<lead>[01])(?P<direction>\+{1,2}|-{1,2})$")
_T_PEAK = re.compile(
    r"^(?P<prefix>AT|at)(?P<lead>[01])(?P<direction>\+{1,2}|-{1,2})(?P<value>\d+)$"
)
_T_END = re.compile(r"^(?P<kind>T|t)(?P<lead>[01])(?P<direction>\+{1,2}|-{1,2})\)$")
_KNOWN_IGNORED_SYMBOLS: Final = frozenset(
    {"N", "a", "J", "S", "V", "F", "Q", "n", "|", "+"}
)
_KNOWN_IRRELEVANT_COMMENT_TEXT: Final = frozenset({"BUTTON", "TS"})
_QUALITY_SUBTYPE_MASK: Final = 0x33


def subject_id_for_record(record_id: str) -> str:
    """Return the documented EDB subject mapping; never infer from demographics."""
    if record_id not in EDB_RECORD_IDS:
        raise AnnotationValidationError(f"Unknown EDB record {record_id}.")
    for group in EDB_SHARED_SUBJECT_GROUPS:
        if record_id in group:
            return f"edb:{group[0]}"
    return f"edb:{record_id}"


def _event_fields(kind: str) -> tuple[str, str]:
    if kind == "ST":
        return "st_change", "reference_st_change"
    return "axis_shift", "apparent_st_change"


def _text(annotation: AnnotationSample) -> str:
    """Normalize fixed-width WFDB auxiliary-note padding."""
    return annotation.aux_note.strip(" \t\r\n\x00")


def _expected_st_symbol(kind: str) -> str:
    return "s" if kind == "ST" else '"'


def quality_states(subtype: int) -> tuple[str, str]:
    """Decode EDB's WFDB bitmask, with unreadable taking precedence over noise."""
    if subtype < 0 or subtype & ~_QUALITY_SUBTYPE_MASK:
        raise AnnotationValidationError(f"Unsupported EDB quality subtype {subtype}.")
    return tuple(
        "unreadable"
        if subtype & (0x10 << lead)
        else "noisy"
        if subtype & (1 << lead)
        else "clean"
        for lead in range(2)
    )


def classify_annotations(
    annotations: Iterable[AnnotationSample],
) -> tuple[AnnotationClassification, ...]:
    """Classify every EDB annotation before episode reconstruction."""
    classifications: list[AnnotationClassification] = []
    for annotation in annotations:
        text = _text(annotation)
        start, peak, end = (
            _START.fullmatch(text),
            _PEAK.fullmatch(text),
            _END.fullmatch(text),
        )
        if start or peak or end:
            kind = (
                start["kind"]
                if start
                else ("ST" if peak and peak["prefix"] == "AST" else "st")
                if peak
                else end["kind"]
            )
            expected = _expected_st_symbol(kind)
            if annotation.symbol == expected and annotation.channel == 0:
                classifications.append(
                    AnnotationClassification(
                        annotation, "recognized_relevant", "edb.st_episode"
                    )
                )
            else:
                classifications.append(
                    AnnotationClassification(
                        annotation,
                        "unknown",
                        f"EDB ST text requires WFDB symbol {expected!r} and channel 0",
                        True,
                    )
                )
        else:
            t_start, t_peak, t_end = (
                _T_START.fullmatch(text),
                _T_PEAK.fullmatch(text),
                _T_END.fullmatch(text),
            )
            if t_start or t_peak or t_end:
                kind = (
                    t_start["kind"]
                    if t_start
                    else ("T" if t_peak and t_peak["prefix"] == "AT" else "t")
                    if t_peak
                    else t_end["kind"]
                )
                expected = "T" if kind == "T" else '"'
                if annotation.symbol == expected and annotation.channel == 0:
                    classifications.append(
                        AnnotationClassification(
                            annotation, "recognized_irrelevant", "edb.t_episode"
                        )
                    )
                else:
                    classifications.append(
                        AnnotationClassification(
                            annotation,
                            "unknown",
                            "EDB T text requires its documented WFDB symbol and "
                            "channel 0",
                        )
                    )
            elif annotation.symbol == "T":
                # Some released T annotations retain non-NUL fixed-width padding.
                classifications.append(
                    AnnotationClassification(
                        annotation, "recognized_irrelevant", "edb.t_episode"
                    )
                )
            elif annotation.symbol == '"' and text in _KNOWN_IRRELEVANT_COMMENT_TEXT:
                classifications.append(
                    AnnotationClassification(
                        annotation,
                        "recognized_irrelevant",
                        "edb.comment",
                    )
                )
            elif annotation.symbol == "~":
                try:
                    quality_states(annotation.subtype)
                    valid_subtype = True
                except AnnotationValidationError:
                    valid_subtype = False
                if valid_subtype and annotation.channel == 0:
                    classifications.append(
                        AnnotationClassification(
                            annotation, "recognized_relevant", "edb.signal_quality"
                        )
                    )
                else:
                    classifications.append(
                        AnnotationClassification(
                            annotation,
                            "unknown",
                            "EDB quality annotation has unsupported subtype or channel",
                            True,
                        )
                    )
            elif annotation.symbol in _KNOWN_IGNORED_SYMBOLS:
                classifications.append(
                    AnnotationClassification(
                        annotation, "recognized_irrelevant", "edb.beat_or_rhythm"
                    )
                )
            else:
                classifications.append(
                    AnnotationClassification(
                        annotation,
                        "unknown",
                        "unrecognized EDB annotation",
                        annotation.symbol in {"s", '"', "~"}
                        or "st" in text.lower(),
                    )
                )
    return tuple(classifications)


def parse_annotations(
    record: DatasetRecord, annotations: Iterable[AnnotationSample]
) -> ParsedAnnotations:
    """Strictly reconstruct upper-case ST and lower-case axis-shift episodes."""
    annotations = tuple(annotations)
    classifications = classify_annotations(annotations)
    open_events: dict[tuple[str, int, str], list[AnnotationSample]] = {}
    events: list[STEvent] = []
    quality_samples: list[AnnotationSample] = []
    for annotation, classification in zip(annotations, classifications, strict=True):
        if classification.disposition != "recognized_relevant":
            continue
        text = _text(annotation)
        start = _START.fullmatch(text)
        peak = _PEAK.fullmatch(text)
        end = _END.fullmatch(text)
        if start:
            key = (start["kind"], int(start["lead"]), start["direction"])
            if key in open_events:
                raise AnnotationValidationError(
                    f"Duplicate EDB start marker at {annotation.sample}."
                )
            open_events[key] = [annotation]
        elif peak:
            kind = "ST" if peak["prefix"] == "AST" else "st"
            key = (kind, int(peak["lead"]), peak["direction"])
            if key not in open_events or len(open_events[key]) != 1:
                raise AnnotationValidationError(
                    f"Unmatched EDB peak marker at {annotation.sample}."
                )
            open_events[key].append(annotation)
        elif end:
            key = (end["kind"], int(end["lead"]), end["direction"])
            sequence = open_events.pop(key, None)
            if sequence is None or len(sequence) != 2:
                raise AnnotationValidationError(
                    f"Unmatched EDB end marker at {annotation.sample}."
                )
            onset, peak_annotation = sequence
            lead = key[1]
            if (
                lead >= record.signal_count
                or not onset.sample < peak_annotation.sample < annotation.sample
            ):
                raise AnnotationValidationError(
                    f"Invalid EDB episode timing in {record.record_id}."
                )
            family, subtype = _event_fields(key[0])
            peak_match = _PEAK.fullmatch(_text(peak_annotation))
            magnitude = int(peak_match["value"]) if peak_match else 0
            signed_value = magnitude if key[2] == "+" else -magnitude
            events.append(
                STEvent(
                    DATASET_ID,
                    record.record_id,
                    record.subject_id,
                    lead,
                    family,
                    subtype,
                    onset.sample,
                    peak_annotation.sample,
                    annotation.sample,
                    onset.sample / record.sampling_frequency_hz,
                    peak_annotation.sample / record.sampling_frequency_hz,
                    annotation.sample / record.sampling_frequency_hz,
                    signed_value,
                    "elevation" if key[2] == "+" else "depression",
                    "atr",
                    "edb.reference",
                    True,
                    (onset, peak_annotation, annotation),
                )
            )
        elif annotation.symbol == "~":
            quality_samples.append(annotation)
    warnings = ()
    if open_events:
        warnings = (
            f"{record.record_id} ends with source-censored EDB episodes: "
            f"{sorted(open_events)}.",
        )
    return ParsedAnnotations(
        tuple(events),
        _quality_intervals(record, quality_samples),
        (),
        classifications,
        warnings,
    )


def _quality_intervals(
    record: DatasetRecord, samples: list[AnnotationSample]
) -> tuple[SignalQualityInterval, ...]:
    intervals: list[SignalQualityInterval] = []
    for index, current in enumerate(samples):
        end = (
            samples[index + 1].sample
            if index + 1 < len(samples)
            else record.sample_count
        )
        if not current.sample < end <= record.sample_count:
            raise AnnotationValidationError(
                f"Invalid EDB quality timing in {record.record_id}."
            )
        for lead, state in enumerate(quality_states(current.subtype)):
            intervals.append(
                SignalQualityInterval(
                    record.record_id,
                    record.subject_id,
                    lead,
                    current.sample,
                    end,
                    state,
                    "atr",
                    (current,),
                )
            )
    return tuple(intervals)


def read_record(root: Path, record_id: str) -> DatasetRecord:
    """Read WFDB header metadata; samples and annotations are never guessed."""
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
        {"source": "PhysioNet EDB v1.0.0"},
    )


def read_annotations(root: Path, record: DatasetRecord) -> ParsedAnnotations:
    """Read the official `.atr` stream and preserve every parsed marker."""
    import wfdb

    raw = wfdb.rdann(str(root / record.record_id), "atr")
    annotations = tuple(
        AnnotationSample(int(sample), symbol, int(subtype), int(channel), aux)
        for sample, symbol, subtype, channel, aux in zip(
            raw.sample, raw.symbol, raw.subtype, raw.chan, raw.aux_note, strict=True
        )
    )
    return parse_annotations(record, annotations)
