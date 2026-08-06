"""WFDB remote metadata probes that never request ECG waveform files."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from cardiosentinel.data.manifest import dataset_module
from cardiosentinel.data.models import (
    AnnotationSample,
    AnnotationValidationError,
    DatasetRecord,
    ParsedAnnotations,
    annotation_pattern,
)
from cardiosentinel.data.validation import ValidationReport, validate_dataset


def _pn_dir(module: Any) -> str:
    return f"{module.DATASET_ID}/{module.DATASET_VERSION}"


def _annotation_set(dataset_id: str, requested: str | None) -> str:
    module = dataset_module(dataset_id)
    selected = requested or module.PRIMARY_ANNOTATION
    if dataset_id == "edb" and selected != "atr":
        raise ValueError("EDB supports only the documented atr annotation stream.")
    if dataset_id == "ltstdb" and selected not in module.ANNOTATION_SETS:
        raise ValueError("LTSTDB annotation set must be sta, stb, or stc.")
    return selected


def remote_record(dataset_id: str, record_id: str) -> DatasetRecord:
    """Read an official WFDB header through pn_dir without fetching signal samples."""
    import wfdb

    module = dataset_module(dataset_id)
    header = wfdb.rdheader(record_id, pn_dir=_pn_dir(module))
    signal_names = tuple(header.sig_name)
    return DatasetRecord(
        module.DATASET_ID,
        module.DATASET_VERSION,
        record_id,
        module.subject_id_for_record(record_id),
        float(header.fs),
        signal_names,
        tuple(signal_names),
        header.n_sig,
        header.sig_len,
        header.sig_len / float(header.fs),
        f"physionet:{_pn_dir(module)}/{record_id}",
        {"comments": tuple(header.comments)},
        {"source": f"PhysioNet {_pn_dir(module)}"},
    )


def remote_annotations(
    dataset_id: str, record_id: str, annotation_set: str | None = None
) -> tuple[AnnotationSample, ...]:
    """Read one remote annotation stream without requesting waveform data."""
    import wfdb

    module = dataset_module(dataset_id)
    selected = _annotation_set(dataset_id, annotation_set)
    raw = wfdb.rdann(record_id, selected, pn_dir=_pn_dir(module))
    return tuple(
        AnnotationSample(int(sample), symbol, int(subtype), int(channel), aux)
        for sample, symbol, subtype, channel, aux in zip(
            raw.sample, raw.symbol, raw.subtype, raw.chan, raw.aux_note, strict=True
        )
    )


def parse_remote_record(
    dataset_id: str, record_id: str, annotation_set: str | None = None
) -> tuple[DatasetRecord, ParsedAnnotations]:
    """Parse a remote header and annotation stream, never the corresponding .dat."""
    module = dataset_module(dataset_id)
    selected = _annotation_set(dataset_id, annotation_set)
    record = remote_record(dataset_id, record_id)
    annotations = remote_annotations(dataset_id, record_id, selected)
    parsed = (
        module.parse_annotations(record, annotations)
        if dataset_id == "edb"
        else module.parse_annotations(record, annotations, selected)
    )
    return record, parsed


def probe_remote(
    dataset_id: str, record_id: str, annotation_set: str | None = None
) -> dict[str, object]:
    """Summarize one remote annotation stream using non-identifying patterns."""
    module = dataset_module(dataset_id)
    selected = _annotation_set(dataset_id, annotation_set)
    record = remote_record(dataset_id, record_id)
    annotations = remote_annotations(dataset_id, record_id, selected)
    classifications = module.classify_annotations(annotations)
    parse_error: str | None = None
    try:
        parsed = (
            module.parse_annotations(record, annotations)
            if dataset_id == "edb"
            else module.parse_annotations(record, annotations, selected)
        )
        warnings = parsed.warnings
    except AnnotationValidationError as error:
        parse_error = str(error)
        warnings = ()
    examples = [
        {
            "sample": item.annotation.sample,
            "symbol": item.annotation.symbol,
            "aux_pattern": annotation_pattern(item.annotation.aux_note),
            "classification": item.disposition,
        }
        for item in classifications[:10]
    ]
    unknown = [item for item in classifications if item.disposition == "unknown"]
    return {
        "record_id": record.record_id,
        "annotation_set": selected,
        "sampling_frequency_hz": record.sampling_frequency_hz,
        "signal_names": record.signal_names,
        "annotation_count": len(annotations),
        "distinct_symbols": dict(
            sorted(Counter(item.symbol for item in annotations).items())
        ),
        "recognized_annotation_count": sum(
            item.disposition == "recognized_relevant" for item in classifications
        ),
        "ignored_known_annotation_count": sum(
            item.disposition == "recognized_irrelevant" for item in classifications
        ),
        "unknown_annotation_count": len(unknown),
        "unknown_aux_patterns": sorted(
            {annotation_pattern(item.annotation.aux_note) for item in unknown}
        )[:20],
        "examples": examples,
        "warnings": warnings,
        "parse_error": parse_error,
    }


def validate_remote_dataset(
    dataset_id: str, annotation_set: str | None = None
) -> ValidationReport:
    """Validate all official headers and one annotation stream without waveforms."""
    import wfdb

    module = dataset_module(dataset_id)
    selected = _annotation_set(dataset_id, annotation_set)
    record_ids = tuple(wfdb.get_record_list(_pn_dir(module)))
    if dataset_id == "edb" and set(record_ids) != set(module.EDB_RECORD_IDS):
        raise ValueError("Remote EDB RECORDS does not match the documented v1.0.0 set.")
    records: list[DatasetRecord] = []
    parsed_items: list[ParsedAnnotations] = []
    parse_errors: list[str] = []

    def inspect_one(
        record_id: str,
    ) -> tuple[DatasetRecord, ParsedAnnotations, str | None]:
        record = remote_record(dataset_id, record_id)
        annotations = remote_annotations(dataset_id, record_id, selected)
        try:
            parsed = (
                module.parse_annotations(record, annotations)
                if dataset_id == "edb"
                else module.parse_annotations(record, annotations, selected)
            )
            return record, parsed, None
        except AnnotationValidationError as error:
            return (
                record,
                ParsedAnnotations(
                    (), (), (), module.classify_annotations(annotations), ()
                ),
                f"{record_id}: {error}",
            )

    with ThreadPoolExecutor(max_workers=6) as executor:
        for record, parsed, error in executor.map(inspect_one, record_ids):
            records.append(record)
            parsed_items.append(parsed)
            if error:
                parse_errors.append(error)
    events = tuple(event for item in parsed_items for event in item.events)
    intervals = tuple(
        interval for item in parsed_items for interval in item.quality_intervals
    )
    markers = tuple(marker for item in parsed_items for marker in item.markers)
    primary = (
        "edb.reference"
        if dataset_id == "edb"
        else f"ltstdb.{module.PRIMARY_ANNOTATION}"
    )
    report = validate_dataset(
        records,
        events,
        intervals,
        module.EXPECTED_RECORD_COUNT,
        module.EXPECTED_SUBJECT_COUNT,
        primary,
        markers,
        parsed_items,
    )
    return replace(report, errors=tuple(parse_errors) + report.errors)
