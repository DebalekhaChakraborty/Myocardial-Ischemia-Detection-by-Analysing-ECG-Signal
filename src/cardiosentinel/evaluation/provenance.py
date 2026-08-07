"""Authoritative EDB/LTSTDB overlap registry; no demographic inference is used."""

from __future__ import annotations

from typing import Final

from cardiosentinel.data import edb
from cardiosentinel.evaluation.models import CrossDatasetProvenance

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
