"""Dataset and split validation that makes leakage and malformed data explicit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from cardiosentinel.data.models import DatasetRecord, SignalQualityInterval, STEvent


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable findings from a strict validation pass."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    summary: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether validation found no blocking errors."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise a concise error for command-line callers."""
        if self.errors:
            raise ValueError("; ".join(self.errors))


def validate_subject_partitions(
    subject_to_partition: Mapping[str, str],
    record_to_subject: Mapping[str, str],
    record_to_partition: Mapping[str, str] | None = None,
) -> None:
    """Reject subject overlap and record assignments inconsistent with subjects."""
    allowed = {"train", "validation", "test"}
    invalid = {
        subject: partition
        for subject, partition in subject_to_partition.items()
        if partition not in allowed
    }
    if invalid:
        raise ValueError(f"Unknown partition assignments: {sorted(invalid.items())}.")
    if record_to_partition is not None:
        for record, subject in record_to_subject.items():
            if record not in record_to_partition:
                raise ValueError(f"Record {record} has no partition assignment.")
            expected = subject_to_partition.get(subject)
            if expected is None:
                raise ValueError(f"Subject {subject} has no partition assignment.")
            if record_to_partition[record] != expected:
                actual = record_to_partition[record]
                raise ValueError(
                    f"Leakage risk: record {record} is {actual} but "
                    f"subject {subject} is {expected}."
                )


def validate_dataset(
    records: Iterable[DatasetRecord],
    events: Iterable[STEvent],
    quality_intervals: Iterable[SignalQualityInterval],
    expected_records: int,
    expected_subjects: int,
    primary_definition: str | None,
) -> ValidationReport:
    """Validate record, event, channel, and quality invariants for a full dataset."""
    records = tuple(records)
    events = tuple(events)
    quality_intervals = tuple(quality_intervals)
    errors: list[str] = []
    by_record = {record.record_id: record for record in records}
    if len(records) != expected_records:
        errors.append(f"Expected {expected_records} records, found {len(records)}.")
    subject_count = len({record.subject_id for record in records})
    if subject_count != expected_subjects:
        errors.append(f"Expected {expected_subjects} subjects, found {subject_count}.")
    seen_events: set[tuple[object, ...]] = set()
    for event in events:
        record = by_record.get(event.record_id)
        if record is None:
            errors.append(f"Event references unknown record {event.record_id}.")
            continue
        if event.lead < 0 or event.lead >= record.signal_count:
            errors.append(f"Event has invalid lead {event.lead} for {event.record_id}.")
        if not 0 <= event.onset_sample < event.end_sample <= record.sample_count:
            errors.append(f"Event has invalid onset/end timing in {event.record_id}.")
        if event.peak_sample is not None and not (
            event.onset_sample < event.peak_sample < event.end_sample
        ):
            errors.append(f"Event has invalid peak timing in {event.record_id}.")
        if primary_definition and event.is_primary_definition != (
            event.annotation_definition == primary_definition
        ):
            errors.append(f"Primary definition flag is wrong for {event.record_id}.")
        identity = (
            event.record_id,
            event.lead,
            event.annotation_definition,
            event.onset_sample,
            event.end_sample,
            event.event_subtype,
        )
        if identity in seen_events:
            errors.append(f"Duplicate event detected in {event.record_id}.")
        seen_events.add(identity)
    for interval in quality_intervals:
        record = by_record.get(interval.record_id)
        if (
            record is None
            or not 0
            <= interval.start_sample
            < interval.end_sample
            <= record.sample_count
        ):
            errors.append(f"Invalid signal-quality interval in {interval.record_id}.")
        elif interval.lead is not None and not 0 <= interval.lead < record.signal_count:
            errors.append(
                f"Signal-quality interval has invalid lead in {interval.record_id}."
            )
    return ValidationReport(
        errors=tuple(errors),
        summary={
            "record_count": len(records),
            "subject_count": subject_count,
            "event_count": len(events),
            "signal_quality_interval_count": len(quality_intervals),
        },
    )
