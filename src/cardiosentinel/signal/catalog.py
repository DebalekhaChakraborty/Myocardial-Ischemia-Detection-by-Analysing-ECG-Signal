"""Minimal waveform dataset identity without importing annotation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from cardiosentinel.signal.models import SignalValidationError


@dataclass(frozen=True)
class WaveformDatasetSpec:
    dataset_id: str
    version: str

    @property
    def pn_dir(self) -> str:
        return f"{self.dataset_id}/{self.version}"


DATASETS: Final = {
    "edb": WaveformDatasetSpec("edb", "1.0.0"),
    "ltstdb": WaveformDatasetSpec("ltstdb", "1.0.0"),
}
_EDB_SHARED_SUBJECTS: Final = (
    ("e0118", "e0119", "e0121", "e0122"),
    ("e0123", "e0124", "e0125", "e0126"),
    ("e0129", "e0133"),
    ("e0136", "e0139"),
    ("e0147", "e0148"),
    ("e0154", "e0155"),
    ("e0162", "e0163"),
)


def dataset_spec(dataset_id: str) -> WaveformDatasetSpec:
    try:
        return DATASETS[dataset_id]
    except KeyError as error:
        raise SignalValidationError(
            f"Unsupported waveform dataset {dataset_id!r}; use edb or ltstdb."
        ) from error


def subject_id_for_record(dataset_id: str, record_id: str) -> str:
    """Apply the Phase 1 subject identity contract without reading annotations."""
    if dataset_id == "edb":
        for group in _EDB_SHARED_SUBJECTS:
            if record_id in group:
                return f"edb:{group[0]}"
        if (
            len(record_id) != 5
            or not record_id.startswith("e")
            or not record_id[1:].isdigit()
        ):
            raise SignalValidationError(f"Invalid EDB record identifier {record_id!r}.")
        return f"edb:{record_id}"
    if dataset_id == "ltstdb":
        if len(record_id) != 6 or record_id[0] != "s" or not record_id[1:].isdigit():
            raise SignalValidationError(
                f"Invalid LTSTDB record identifier {record_id!r}."
            )
        return f"ltstdb:{record_id[:-1]}"
    dataset_spec(dataset_id)
    raise AssertionError("unreachable")
