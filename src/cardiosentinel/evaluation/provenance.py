"""Authoritative EDB/LTSTDB overlap registry; no demographic inference is used."""

from __future__ import annotations

from typing import Final

from cardiosentinel.data import edb
from cardiosentinel.evaluation.models import (
    CrossDatasetProvenance,
    EDBSecondaryCohort,
    EDBSecondaryCohortName,
)

_PHYSIONET_LTSTDB = "https://physionet.org/content/ltstdb/1.0.0/"
_PISA_SOURCE = "Pisa group collection used in the European ST-T Database"
_VERIFIED_PAIRS: Final = (
    ("s20021", "e0113"),
    ("s20151", "e0103"),
    ("s20161", "e0105"),
    ("s20171", "e0127"),
    ("s20181", "e0162"),
    ("s20291", "e0104"),
    ("s20301", "e0125"),
    ("s20311", "e0129"),
    ("s20581", "e0603"),
    ("s20591", "e0604"),
)


def verified_overlap_registry() -> tuple[CrossDatasetProvenance, ...]:
    """Return only direct correspondences stated in official LTSTDB headers."""
    return tuple(
        CrossDatasetProvenance(
            dataset="ltstdb",
            record=ltstdb_record,
            source_collection=_PISA_SOURCE,
            overlap_risk="direct_redigitized_excerpt_overlap",
            verified_corresponding_dataset="edb",
            verified_corresponding_record=edb_record,
            evidence_source=(
                "https://physionet.org/files/ltstdb/1.0.0/"
                f"{ltstdb_record}.hea"
            ),
            confidence="verified",
        )
        for ltstdb_record, edb_record in _VERIFIED_PAIRS
    )


def general_overlap_evidence_source() -> str:
    """Return the official release page describing the ten-record source overlap."""
    return _PHYSIONET_LTSTDB


def conservative_edb_overlap_exclusions() -> tuple[str, ...]:
    """Exclude verified excerpts and all records from their mapped EDB subjects."""
    paired_records = {edb_record for _, edb_record in _VERIFIED_PAIRS}
    affected_subjects = {
        edb.subject_id_for_record(record_id) for record_id in paired_records
    }
    return tuple(
        record_id
        for record_id in edb.EDB_RECORD_IDS
        if edb.subject_id_for_record(record_id) in affected_subjects
    )


def edb_secondary_cohort(name: EDBSecondaryCohortName) -> EDBSecondaryCohort:
    """Return an explicit descriptive or overlap-clean secondary EDB cohort."""
    exclusions = conservative_edb_overlap_exclusions()
    if name == "full":
        return EDBSecondaryCohort(
            name=name,
            record_ids=tuple(edb.EDB_RECORD_IDS),
            known_overlap_record_ids=exclusions,
            contains_known_source_overlap=True,
        )
    if name == "overlap_clean":
        excluded = set(exclusions)
        return EDBSecondaryCohort(
            name=name,
            record_ids=tuple(
                record_id
                for record_id in edb.EDB_RECORD_IDS
                if record_id not in excluded
            ),
            known_overlap_record_ids=exclusions,
            contains_known_source_overlap=False,
        )
    raise ValueError(f"Unknown EDB secondary cohort: {name}")


def recommended_edb_secondary_cohort(
    training_datasets: tuple[str, ...],
) -> EDBSecondaryCohort:
    """Require overlap-clean EDB evaluation after any LTSTDB model training."""
    normalized = {dataset.lower() for dataset in training_datasets}
    name: EDBSecondaryCohortName = (
        "overlap_clean" if "ltstdb" in normalized else "full"
    )
    return edb_secondary_cohort(name)


def validate_edb_secondary_evaluation_policy(
    cohort_name: EDBSecondaryCohortName,
    training_datasets: tuple[str, ...],
) -> None:
    """Reject known-overlap EDB evaluation for models trained with LTSTDB."""
    recommended = recommended_edb_secondary_cohort(training_datasets)
    if recommended.name == "overlap_clean" and cohort_name != "overlap_clean":
        raise ValueError(
            "LTSTDB-trained models must use the overlap-clean EDB secondary cohort."
        )
