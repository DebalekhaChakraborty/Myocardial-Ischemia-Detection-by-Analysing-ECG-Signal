"""Per-row U1 OOF evidence: typed arrays with a hashed manifest.

Following the M2 convention exactly (`m2_evidence_store`), and for the same
reason: 473,897 PRIMARY rows plus 8,137 CHALLENGE rows of per-row calibration
evidence belong in a typed binary store with a digest, not inflated into a JSON
document. The claim-bearing JSON results bind this store by SHA-256 rather than
restating it.

**Both fitted families are persisted, side by side.** §6.1 requires the
comparator to be reported whichever family is selected, and storing only the
selected family would make the selection unauditable after the fact. The
selected family is recorded in the manifest and the selected probability is
DERIVED on read, so there is exactly one source of truth for it.

**PRIMARY and CHALLENGE never share a row group.** They are separate arrays
with separate digests, because merging them into one denominator is precisely
what the frozen protocol forbids.

Everything here is float64 and lossless. Nothing here fits a calibrator,
chooses a threshold or opens a partition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.u1_calibration import U1_FAMILIES, require_family

U1_EVIDENCE_STORE_SCHEMA: Final = "u1_v1_oof_evidence_store/1"
U1_STORE_MANIFEST_NAME: Final = "U1_OOF_EVIDENCE_STORE.json"
U1_PRIMARY_ROWS_NAME: Final = "u1_oof_primary_evidence.npz"
U1_CHALLENGE_ROWS_NAME: Final = "u1_oof_challenge_evidence.npz"

U1_PRIMARY_COLUMNS: Final = (
    "stable_id",
    "subject_id",
    "fold_index",
    "label",
    "score",
    "recovered_logit",
    "oof_probability_platt",
    "oof_probability_temperature",
    "frozen_decision",
    "cold_start_bin",
)

U1_CHALLENGE_COLUMNS: Final = (
    "stable_id",
    "subject_id",
    "fold_index",
    "target_family",
    "score",
    "recovered_logit",
    "oof_probability_platt",
    "oof_probability_temperature",
    "frozen_decision",
)


class U1EvidenceStoreError(RuntimeError):
    """Raised when per-row U1 evidence cannot be persisted or read faithfully."""


def _family_column(family: str) -> str:
    require_family(family)
    return (
        "oof_probability_platt"
        if family == U1_FAMILIES[0]
        else "oof_probability_temperature"
    )


@dataclass(frozen=True, slots=True)
class U1RowGroup:
    """One population's per-row evidence as row-aligned typed arrays."""

    name: str
    columns: tuple[str, ...]
    arrays: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        missing = [column for column in self.columns if column not in self.arrays]
        extra = [column for column in self.arrays if column not in self.columns]
        if missing or extra:
            raise U1EvidenceStoreError(
                f"{self.name} row group schema mismatch; missing {missing}, "
                f"unexpected {extra}."
            )
        lengths = {int(values.shape[0]) for values in self.arrays.values()}
        if len(lengths) != 1:
            raise U1EvidenceStoreError(f"{self.name} columns are not row-aligned.")
        identities = self.arrays["stable_id"]
        if len(set(identities.tolist())) != int(identities.shape[0]):
            raise U1EvidenceStoreError(
                f"{self.name} carries duplicate stable IDs; the frozen ordering "
                "would be ambiguous and they are never deduplicated."
            )

    @property
    def row_count(self) -> int:
        return int(self.arrays["stable_id"].shape[0])


def _as_group(name: str, columns: Sequence[str], data: dict[str, Any]) -> U1RowGroup:
    dtypes: dict[str, Any] = {
        "stable_id": np.str_,
        "subject_id": np.str_,
        "target_family": np.str_,
        "cold_start_bin": np.str_,
        "fold_index": np.int64,
        "label": np.int64,
        "frozen_decision": np.bool_,
    }
    arrays = {
        column: np.asarray(data[column], dtype=dtypes.get(column, np.float64))
        for column in columns
    }
    return U1RowGroup(name=name, columns=tuple(columns), arrays=arrays)


def write_u1_evidence_store(
    root: Path,
    *,
    primary: dict[str, Any],
    challenge: dict[str, Any],
    selected_family: str,
    fold_assignment_sha256: str,
    clamp_delta: float,
    classification_threshold: float,
) -> dict[str, Any]:
    """Persist both row groups and bind them with a self-digesting manifest."""
    require_family(selected_family)
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)

    groups = {
        "primary_metric": (
            _as_group("primary_metric", U1_PRIMARY_COLUMNS, primary),
            U1_PRIMARY_ROWS_NAME,
        ),
        "challenge_metric": (
            _as_group("challenge_metric", U1_CHALLENGE_COLUMNS, challenge),
            U1_CHALLENGE_ROWS_NAME,
        ),
    }
    manifest_groups: dict[str, Any] = {}
    for key, (group, filename) in groups.items():
        path = directory / filename
        if path.exists():
            raise U1EvidenceStoreError(
                f"{path} already exists; U1 evidence is never overwritten or reused."
            )
        with path.open("wb") as handle:
            np.savez(handle, **group.arrays)
        manifest_groups[key] = {
            "file": filename,
            "row_count": group.row_count,
            "columns": list(group.columns),
            "dtypes": {
                name: str(values.dtype) for name, values in sorted(group.arrays.items())
            },
            "sha256": sha256_file(path),
        }

    manifest: dict[str, Any] = {
        "schema": U1_EVIDENCE_STORE_SCHEMA,
        "row_groups": manifest_groups,
        "selected_family": selected_family,
        "selected_probability_column": _family_column(selected_family),
        "families_persisted": list(U1_FAMILIES),
        "fold_assignment_sha256": str(fold_assignment_sha256),
        "clamp_delta": float(clamp_delta),
        "classification_threshold": float(classification_threshold),
        "probability_dtype": "float64",
        "lossy_conversion_applied": False,
        "out_of_fold_only": True,
        "primary_and_challenge_merged": False,
        "deployment_calibrator_probabilities_present": False,
        "test_rows_present": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    (directory / U1_STORE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def validate_u1_evidence_store(
    manifest: dict[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Re-verify the manifest's self-digest, its schema and its file digests."""
    recorded = manifest.get("content_sha256")
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise U1EvidenceStoreError("The U1 evidence manifest failed digest validation.")
    if manifest.get("schema") != U1_EVIDENCE_STORE_SCHEMA:
        raise U1EvidenceStoreError(
            f"Unknown U1 evidence schema {manifest.get('schema')!r}."
        )
    for flag in (
        "lossy_conversion_applied",
        "primary_and_challenge_merged",
        "deployment_calibrator_probabilities_present",
        "test_rows_present",
    ):
        if manifest.get(flag) is not False:
            raise U1EvidenceStoreError(
                f"The U1 evidence store must record {flag}=false."
            )
    if manifest.get("out_of_fold_only") is not True:
        raise U1EvidenceStoreError(
            "Per-row U1 DEVELOPMENT evidence is out-of-fold only."
        )
    if manifest.get("probability_dtype") != "float64":
        raise U1EvidenceStoreError("Calibrated probabilities must be float64.")
    require_family(str(manifest.get("selected_family")))
    if manifest.get("selected_probability_column") != _family_column(
        str(manifest["selected_family"])
    ):
        raise U1EvidenceStoreError(
            "The manifest's selected probability column disagrees with its "
            "selected family."
        )
    if root is not None:
        base = Path(root)
        for key, entry in manifest["row_groups"].items():
            path = base / entry["file"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise U1EvidenceStoreError(
                    f"Persisted U1 row group {key} does not match its digest."
                )
    return dict(manifest)


def read_u1_row_group(root: Path, manifest: dict[str, Any], group: str) -> U1RowGroup:
    """Read one row group back at full precision, digest-checked first."""
    validate_u1_evidence_store(manifest, root=Path(root))
    entry = manifest["row_groups"][group]
    columns = tuple(entry["columns"])
    with np.load(Path(root) / entry["file"], allow_pickle=False) as payload:
        arrays = {column: np.asarray(payload[column]) for column in columns}
    return U1RowGroup(name=group, columns=columns, arrays=arrays)


def selected_probabilities(manifest: dict[str, Any], group: U1RowGroup) -> np.ndarray:
    """The OOF probabilities of the SELECTED family, derived, never duplicated."""
    column = manifest["selected_probability_column"]
    if column not in group.arrays:
        raise U1EvidenceStoreError(
            f"Row group {group.name} carries no {column!r} column."
        )
    return np.asarray(group.arrays[column], dtype=np.float64)
